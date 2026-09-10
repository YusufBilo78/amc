"""
sink_vs_embedding.py -- is the sink the nearest class in the model's OWN
learned representation, with the circularity broken?

sink_vs_geometry.py tried a hand-built geometric signature and it was too weak:
top-3 12/24, and it ranked two same-family QAMs far apart. The hand-drawn
taxonomy (18/24) actually beat my automatic metric. The lesson was that a crude
pairwise-distance histogram does not capture constellation structure well.

Rather than tune that metric to the answer (which would be re-scoring), ask the
model what it finds similar. But naively that is circular: the sink comes from
the model's decision and the embedding from the same model.

The circularity is broken by using two different models:

  - The **sinks** come from the leave-one-out runs: for class X, a model
    trained on the other 23. Twenty-four different models.
  - The **embedding geometry** comes from ONE model trained on ALL 24 classes.
    "Which classes sit near each other" is answered by a model that never had
    any class removed, so it is independent of any particular sink.

Prototype = mean embedding (the 256-d pre-logit feature) of a class's frames.
Distance between classes = Euclidean distance between prototypes. Then, for each
held-out class, the fixed question: is the observed sink the nearest prototype?
Rank reported for top-1/2/3.

Still one honest caveat, stated up front: both models share an architecture and
a training set, so "the representation the family lives in" is a property of
this model class, not of modulation per se. A different architecture could
carve the space differently. That is a limitation, not a defeater.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import cnn
import radioml

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
MODEL_PATH = ROOT / "model_all24.pt"

TRAIN_SNRS = list(range(-20, 31, 2))
PROTO_SNRS = list(range(10, 31, 2))   # clean frames for prototypes
FRAMES_TRAIN = 384
FRAMES_PROTO = 400
EPOCHS = 20
SEED = 0


@torch.no_grad()
def embed(model, X, batch=512):
    """256-d pre-logit features (the input to the final linear layer)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(cnn.DEVICE)
        with torch.amp.autocast("cuda", enabled=cnn.DEVICE.type == "cuda"):
            feat = model.features(xb)
            feat = feat.mean(dim=-1)  # global average pool -> (B, C)
        out.append(feat.float().cpu().numpy())
    return np.concatenate(out)


def load_cell(ds, ci, snrs, k, rng):
    rows = np.sort(np.concatenate([
        rng.choice(ds.cell_indices(ci, s), k, replace=False) for s in snrs]))
    X = np.transpose(ds.X[rows], (0, 2, 1)).astype(np.float32)
    p = np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1, keepdims=True)
    return X / (np.sqrt(p)[:, None] + 1e-12)


def main() -> None:
    classes = list(radioml.CLASSES)
    n = len(classes)
    ds = radioml.RadioML()
    print(f"backend: {ds.backend}")
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------ train the all-24 model
    if MODEL_PATH.exists():
        model = cnn.IQNet(n)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=cnn.DEVICE))
        model = model.to(cnn.DEVICE)
        print(f"loaded {MODEL_PATH.name}")
    else:
        print("training all-24 model (defines the embedding geometry) ...")
        t0 = time.perf_counter()
        Xr, yr, zr = [], [], []
        for ci in range(n):
            for s in TRAIN_SNRS:
                rows = np.sort(rng.choice(ds.cell_indices(ci, s),
                                          FRAMES_TRAIN, replace=False))
                Xr.append(rows); yr.append(np.full(FRAMES_TRAIN, ci, np.int64))
                zr.append(np.full(FRAMES_TRAIN, s, np.int16))
        rows = np.concatenate(Xr); order = np.argsort(rows); rows = rows[order]
        y = np.concatenate(yr)[order]; z = np.concatenate(zr)[order]
        X = np.transpose(ds.X[rows], (0, 2, 1)).astype(np.float32)
        p = np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1, keepdims=True)
        X /= np.sqrt(p)[:, None] + 1e-12
        torch.manual_seed(SEED); np.random.seed(SEED)
        tr, te = cnn.split(X, y, z, test_fraction=0.2, seed=SEED)
        model = cnn.IQNet(n)
        model = cnn.train_model(model, X[tr], y[tr], X[te], y[te], epochs=EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  trained in {time.perf_counter()-t0:.0f}s")

    # ------------------------------------------------ class prototypes
    print("\ncomputing class prototypes in embedding space ...")
    protos = np.zeros((n, 256))
    for ci in range(n):
        X = load_cell(ds, ci, PROTO_SNRS, FRAMES_PROTO, rng)
        protos[ci] = embed(model, X).mean(axis=0)
    ds.close()

    # pairwise prototype distance
    D = np.linalg.norm(protos[:, None] - protos[None, :], axis=2)

    # ------------------------------------------------ compare with sinks
    sinks = json.loads((ROOT / "sink_class_24_partial.json").read_text())
    idx = {c: i for i, c in enumerate(classes)}

    print(f"\n{'held out':<12} {'observed sink':<13} {'emb. rank':>9}  "
          f"nearest 3 in embedding")
    print("-" * 74)
    ranks, rows = [], []
    for held in classes:
        if held not in sinks:
            continue
        i = idx[held]
        order = [classes[j] for j in np.argsort(D[i]) if classes[j] != held]
        sink = sinks[held]["sink"]
        rank = order.index(sink) + 1
        ranks.append(rank)
        rows.append((held, sink, rank))
        flag = "" if rank <= 3 else "  <-- far"
        print(f"{held:<12} {sink:<13} {rank:>9}  {', '.join(order[:3]):<32}{flag}")

    ranks = np.array(ranks)
    print("\n" + "=" * 60)
    for k in (1, 2, 3):
        print(f"sink within embedding top-{k}: {int((ranks<=k).sum())}/{len(ranks)}"
              f"   (chance {len(ranks)*k/23:.1f})")
    print(f"median rank {np.median(ranks):.0f}, mean {ranks.mean():.1f}")
    print("=" * 60)

    # compare all three notions side by side
    geo = np.load(ROOT / "sink_vs_geometry.npz", allow_pickle=True) \
        if (ROOT / "sink_vs_geometry.npz").exists() else None
    print("\ntop-3 by method:")
    print(f"  hand-built family taxonomy : 18/24")
    if geo is not None:
        gr = geo["ranks"]
        print(f"  hand-built geometry metric : {int((gr<=3).sum())}/{len(gr)}")
    print(f"  model's own embedding      : {int((ranks<=3).sum())}/{len(ranks)}")

    # ------------------------------------------------------------------ plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    im = ax1.imshow(D, cmap="viridis_r")
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(classes, rotation=90, fontsize=7)
    ax1.set_yticklabels(classes, fontsize=7)
    ax1.set_title("Prototype distance in the model's embedding\n"
                  "(all-24 model; sinks from leave-one-out models)",
                  fontsize=11, fontweight="bold")
    for held, sink, rank in rows:
        ax1.scatter(idx[sink], idx[held], marker="s", s=40,
                    edgecolor="red", facecolor="none", linewidth=1.5)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ax2.hist(ranks, bins=np.arange(0.5, 12, 1), color="#8e44ad", edgecolor="white")
    ax2.axvline(1.5, color="#27ae60", lw=2, ls="--", label="top-1")
    ax2.axvline(3.5, color="#f39c12", lw=2, ls="--", label="top-3")
    ax2.set_xlabel("embedding rank of the observed sink")
    ax2.set_ylabel("number of held-out classes")
    ax2.set_title(f"top-1: {int((ranks<=1).sum())}/{len(ranks)}, "
                  f"top-3: {int((ranks<=3).sum())}/{len(ranks)} "
                  f"(chance top-3 ≈ {len(ranks)*3/23:.0f})",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("Does the unseen-class sink match the nearest class in the "
                 "model's learned space?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "25_sink_vs_embedding.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "sink_vs_embedding.npz", classes=np.array(classes),
             distance=D, ranks=ranks)
    print("\nwrote figures/25_sink_vs_embedding.png")


if __name__ == "__main__":
    main()

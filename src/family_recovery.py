"""
family_recovery.py -- turning the sink phenomenon into something usable.

The sink experiments established that an unseen modulation lands on a
structurally similar class. That is an observation. The obvious reviewer
question is "so what?" -- a predictable failure is only interesting if it can
be acted on.

The claim tested here: **even when the model cannot identify an unseen
modulation, its prediction still carries correct family-level information.**
The exact class is unrecoverable (the class was never trained), but the family
may not be.

Two ways to read the family off a leave-one-out model probed with the held-out
class:

  1. top-1 family    the family of the single most-predicted class
  2. family mass     sum the prediction probabilities over all members of each
                     family, take the largest -- uses the whole distribution,
                     not just the mode

Family mass should win: an unseen 128QAM may split its votes across
32/64/256QAM without any single one dominating, yet the QAM family as a whole
still gets most of the mass.

Baseline for "is this useful": a random unseen modulation assigned to a random
known class hits the right family with probability
(family_size - 1)/23, averaged over classes -- about 0.15 here. Anything well
above that means the family is genuinely recoverable.

Full prediction distributions are stored this time (earlier runs kept only the
top 3), so the family-mass computation is exact.
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
from sink_class_24 import FAMILIES, family_of

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
CKPT = ROOT / "family_recovery.json"

TRAIN_SNRS = list(range(-20, 31, 2))
PROBE_SNRS = list(range(10, 31, 2))
FRAMES_TRAIN = 256
FRAMES_PROBE = 300
EPOCHS = 15
SEED = 0

# Only the digital families carry a clean physical taxonomy; analog/other are
# reported but excluded from the headline number, stated up front.
DIGITAL = ["ASK", "PSK", "APSK", "QAM"]


def load_probe(ds, ci, rng):
    rows = np.sort(np.concatenate([
        rng.choice(ds.cell_indices(ci, s), FRAMES_PROBE, replace=False)
        for s in PROBE_SNRS]))
    X = np.transpose(ds.X[rows], (0, 2, 1)).astype(np.float32)
    p = np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1, keepdims=True)
    return X / (np.sqrt(p)[:, None] + 1e-12)


@torch.no_grad()
def predict_proba(model, X, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(cnn.DEVICE)
        with torch.amp.autocast("cuda", enabled=cnn.DEVICE.type == "cuda"):
            out.append(torch.softmax(model(xb), 1).float().cpu().numpy())
    return np.concatenate(out).mean(axis=0)   # mean posterior over probe frames


def main() -> None:
    classes = list(radioml.CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    ds = radioml.RadioML()
    print(f"backend: {ds.backend}")

    results = json.loads(CKPT.read_text()) if CKPT.exists() else {}

    print(f"loading all 24 classes ({FRAMES_TRAIN}/cell) ...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    rows, ys, zs = [], [], []
    for ci in range(len(classes)):
        for s in TRAIN_SNRS:
            r = np.sort(rng.choice(ds.cell_indices(ci, s), FRAMES_TRAIN, replace=False))
            rows.append(r); ys.append(np.full(FRAMES_TRAIN, ci, np.int64))
            zs.append(np.full(FRAMES_TRAIN, s, np.int16))
    rows = np.concatenate(rows); order = np.argsort(rows); rows = rows[order]
    y_all = np.concatenate(ys)[order]; z_all = np.concatenate(zs)[order]
    X_all = np.transpose(ds.X[rows], (0, 2, 1)).astype(np.float32)
    p = np.mean(X_all[:, 0] ** 2 + X_all[:, 1] ** 2, axis=1, keepdims=True)
    X_all /= np.sqrt(p)[:, None] + 1e-12
    print(f"  {X_all.shape} in {time.perf_counter()-t0:.0f}s")

    probes = {c: load_probe(ds, idx[c], rng) for c in classes}
    ds.close()

    t_start = time.perf_counter()
    for held in classes:
        if held in results:
            continue
        held_i = idx[held]
        keep = y_all != held_i
        kept = [c for c in classes if c != held]
        remap = {idx[c]: j for j, c in enumerate(kept)}
        X = X_all[keep]
        y = np.array([remap[v] for v in y_all[keep]], dtype=np.int64)
        z = z_all[keep]

        torch.manual_seed(SEED); np.random.seed(SEED)
        tr, te = cnn.split(X, y, z, test_fraction=0.25, seed=SEED)
        model = cnn.IQNet(len(kept))
        model = cnn.train_model(model, X[tr], y[tr], X[te], y[te], epochs=EPOCHS)

        proba = predict_proba(model, probes[held])   # over kept classes
        # family mass
        mass = {}
        for fam, members in FAMILIES.items():
            mass[fam] = float(sum(proba[remap[idx[m]]]
                                  for m in members if m != held and m in kept))
        top1_fam = family_of(kept[int(proba.argmax())])
        massfam = max(mass, key=mass.get)

        results[held] = dict(
            true_family=family_of(held),
            top1_family=top1_fam, top1_correct=top1_fam == family_of(held),
            mass_family=massfam, mass_correct=massfam == family_of(held),
            correct_family_mass=mass.get(family_of(held), 0.0),
            all_mass=mass)
        CKPT.write_text(json.dumps(results, indent=2))
        print(f"{held:<11} true {family_of(held):<7} | top1 {top1_fam:<7}"
              f"{'OK' if top1_fam==family_of(held) else '--'} | "
              f"mass {massfam:<7}{'OK' if massfam==family_of(held) else '--'} "
              f"| correct-family mass {mass.get(family_of(held),0):.2f}")

    print(f"\ntotal {(time.perf_counter()-t_start)/60:.1f} min")

    # ------------------------------------------------------------- summary
    dig = [c for c in classes if family_of(c) in DIGITAL]
    top1 = sum(results[c]["top1_correct"] for c in dig)
    massc = sum(results[c]["mass_correct"] for c in dig)
    avg_mass = np.mean([results[c]["correct_family_mass"] for c in dig])
    chance = np.mean([(len(FAMILIES[family_of(c)]) - 1) / 23 for c in dig])

    print("\n" + "=" * 70)
    print(f"DIGITAL classes only ({len(dig)}): ASK/PSK/APSK/QAM")
    print(f"  family via top-1 prediction : {top1}/{len(dig)} "
          f"= {top1/len(dig):.0%}")
    print(f"  family via prediction mass  : {massc}/{len(dig)} "
          f"= {massc/len(dig):.0%}")
    print(f"  mean prob mass on correct family : {avg_mass:.2f}")
    print(f"  chance (random known class)      : {chance:.2f}")
    print("=" * 70)
    print("Reading: an unseen digital modulation's FAMILY is recoverable from")
    print("the prediction distribution, though its exact class is not.")

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(12, 6))
    order_c = [c for fam in DIGITAL for c in FAMILIES[fam]]
    x = np.arange(len(order_c))
    correct = [results[c]["correct_family_mass"] for c in order_c]
    other = [1 - m for m in correct]
    fam_color = {"ASK": "#e67e22", "PSK": "#2980b9",
                 "APSK": "#8e44ad", "QAM": "#27ae60"}
    colors = [fam_color[family_of(c)] for c in order_c]
    ax.bar(x, correct, color=colors, label="prob. mass on correct family")
    ax.bar(x, other, bottom=correct, color="#d5d8dc",
           label="mass elsewhere")
    ax.axhline(chance, ls="--", color="red", lw=1.5,
               label=f"chance ≈ {chance:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(order_c, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("prediction probability mass")
    ax.set_ylim(0, 1)
    ax.set_title("Recovering the family of an unseen modulation from the "
                 "prediction distribution\n(each bar = one held-out class, "
                 "leave-one-out)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "27_family_recovery.png", dpi=140)
    plt.close(fig)
    print("\nwrote figures/27_family_recovery.png")


if __name__ == "__main__":
    main()

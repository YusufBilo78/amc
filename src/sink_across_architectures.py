"""
sink_across_architectures.py -- does the sink stay family-structured off the CNN?

The finding was measured on IQNet (1-D CNN). The open caveat: it might be a
property of convolutional locality rather than of modulation. Four architectures
with different inductive biases are compared -- CNN, deeper residual CNN,
recurrent GRU, attention-only Transformer.

Cost control. A full 24-way leave-one-out per architecture is 96 trainings.
Instead a subset of six held-out classes is fixed **in advance** by a stated
rule -- the highest-order member of each of the four digital families, plus the
two non-digital hard cases (FM, OQPSK). Six is enough to see whether the
same-family behaviour survives the architecture change; the rule is stated
before any result so there is no cherry-picking. FM and OQPSK are deliberately
included even though they were the misses on IQNet -- stacking the deck the
other way would be dishonest.

For each (architecture, held-out class): train on the other 23, probe with the
held-out class from RadioML, record the sink and whether it is in the same
hand-drawn family. Then compare same-family rates across architectures.

ICRNNA, the current backbone, is in the list alongside the four it was written
for: the finding has to hold on the model the rest of the repository reports,
not only on the models chosen to differ from it.

Protocol matches sink_class_24.py: 70/15/15 split, early stopping on the
validation slice, stopping epoch recorded per run.

    cd src && python sink_across_architectures.py --out-dir <drive> --tag e60
"""

from __future__ import annotations

import argparse
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
import model_zoo
import radioml
from sink_class_24 import FAMILIES, family_of

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# Rule fixed in advance: top of each digital family + the two hard non-digital
# cases. Not chosen by looking at any architecture's behaviour.
HELD_OUT = ["8ASK", "32PSK", "128APSK", "256QAM", "FM", "OQPSK"]

ARCHS = ["IQNet (1D CNN)", "ResNet1D", "GRU", "Transformer", "ICRNNA"]

TRAIN_SNRS = list(range(-20, 31, 2))
PROBE_SNRS = list(range(10, 31, 2))
FRAMES_TRAIN = 256
FRAMES_PROBE = 300
EPOCHS = 60        # a ceiling; early stopping at PATIENCE decides the stop
PATIENCE = 10
SEED = 0


def build(name, n):
    if name == "IQNet (1D CNN)":
        return cnn.IQNet(n)
    return model_zoo.ARCHITECTURES[name](n)


def load_probe(ds, ci, rng):
    rows = np.sort(np.concatenate([
        rng.choice(ds.cell_indices(ci, s), FRAMES_PROBE, replace=False)
        for s in PROBE_SNRS]))
    X = np.transpose(ds.X[rows], (0, 2, 1)).astype(np.float32)
    p = np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1, keepdims=True)
    return X / (np.sqrt(p)[:, None] + 1e-12)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None,
                   help="where the checkpoint and the figure go, and where a "
                        "partial sweep is resumed from. Defaults to the "
                        "repository root")
    p.add_argument("--tag", default="",
                   help="appended to every output name, so a run under other "
                        "settings neither overwrites nor resumes this one")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else ROOT
    fig_dir = out_dir / "figures" if args.out_dir else FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    CKPT = out_dir / f"sink_across_arch{suffix}.json"

    classes = list(radioml.CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    ds = radioml.RadioML()
    print(f"backend: {ds.backend}")

    results = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    if results:
        print(f"resuming: {len(results)} of {len(ARCHS) * len(HELD_OUT)} "
              f"already done")

    # Load the full training pool once (all 24 classes); mask per run.
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

    probes = {h: load_probe(ds, idx[h], rng) for h in HELD_OUT}
    ds.close()

    t_start = time.perf_counter()
    for arch in ARCHS:
        for held in HELD_OUT:
            key = f"{arch} | {held}"
            if key in results:
                continue
            held_i = idx[held]
            keep = y_all != held_i
            kept = [c for c in classes if c != held]
            remap = {idx[c]: j for j, c in enumerate(kept)}
            X = X_all[keep]
            y = np.array([remap[v] for v in y_all[keep]], dtype=np.int64)
            z = z_all[keep]

            torch.manual_seed(SEED); np.random.seed(SEED)
            tr, va, te = cnn.split(X, y, z, test_fraction=0.15,
                                   val_fraction=0.15, seed=SEED)
            model = build(arch, len(kept))
            t0 = time.perf_counter()
            model = cnn.train_model(model, X[tr], y[tr], X[va], y[va],
                                    epochs=args.epochs, patience=args.patience)
            in_dist = float((cnn.predict(model, X[te]) == y[te]).mean())
            best_epoch = int(getattr(model, "best_epoch", args.epochs))
            converged = best_epoch + args.patience <= args.epochs

            pred = cnn.predict(model, probes[held])
            share = np.bincount(pred, minlength=len(kept)).astype(float)
            share /= share.sum()
            sink = kept[int(share.argmax())]
            same = family_of(sink) == family_of(held)
            results[key] = dict(arch=arch, held=held, sink=sink,
                                share=float(share.max()), same_family=same,
                                in_dist=in_dist, best_epoch=best_epoch,
                                converged=bool(converged), ceiling=args.epochs)
            CKPT.write_text(json.dumps(results, indent=2))
            print(f"{arch:<16} hold {held:<9} -> {sink:<10} "
                  f"({share.max():.0%})  {'SAME' if same else 'diff':<4} family"
                  f"   [acc {in_dist:.3f}, ep {best_epoch}"
                  f"{'' if converged else ' NOT CONVERGED'}, "
                  f"{time.perf_counter()-t0:.0f}s]")
        print()

    print(f"total {(time.perf_counter()-t_start)/60:.1f} min\n")

    # ------------------------------------------------------------- summary
    print("=" * 78)
    print(f"{'held out':<11} " + " ".join(f"{a.split()[0]:<12}" for a in ARCHS))
    print("-" * 78)
    for held in HELD_OUT:
        cells = []
        for arch in ARCHS:
            r = results[f"{arch} | {held}"]
            tag = "OK" if r["same_family"] else "--"
            cells.append(f"{r['sink']:<9}{tag}")
        print(f"{held:<11} " + " ".join(f"{c:<12}" for c in cells))
    print("-" * 78)
    rates = {}
    for arch in ARCHS:
        s = sum(results[f"{arch} | {h}"]["same_family"] for h in HELD_OUT)
        rates[arch] = s
        print(f"{arch:<16} same-family: {s}/{len(HELD_OUT)}")
    print("=" * 78)
    short = [k for k, r in results.items() if not r.get("converged", True)]
    if short:
        print(f"\nWARNING: {len(short)} run(s) did not converge: "
              f"{', '.join(short)}. Sinks stand; in-dist accuracies are floors.")

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = [a.split()[0] for a in ARCHS]
    vals = [rates[a] for a in ARCHS]
    bars = ax.bar(names, vals,
                  color=["#2980b9", "#16a085", "#8e44ad", "#c0392b", "#d35400"][:len(ARCHS)])
    ax.axhline(len(HELD_OUT), ls=":", color="gray", label="all same-family")
    # chance line: expected same-family for these 6 classes
    exp = sum((len(FAMILIES[family_of(h)]) - 1) / 23 for h in HELD_OUT)
    ax.axhline(exp, ls="--", color="red", lw=1.5, label=f"chance ≈ {exp:.1f}")
    ax.set_ylabel(f"same-family sinks (of {len(HELD_OUT)})")
    ax.set_ylim(0, len(HELD_OUT) + 0.3)
    ax.set_title("Does the same-family sink survive across architectures?\n"
                 "CNN / residual CNN / recurrent / attention / ICRNNA",
                 fontsize=12, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                ha="center", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / f"26_sink_across_architectures{suffix}.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {fig_dir / f'26_sink_across_architectures{suffix}.png'}")


if __name__ == "__main__":
    main()

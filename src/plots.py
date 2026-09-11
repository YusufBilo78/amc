"""
plots.py -- the two figures a classification run produces, and a way to rebuild
them from its saved .npz.

Why this is its own module. A run done on Colab leaves its `.npz` behind and
its figures on whatever machine produced them. The `.npz` is the record worth
committing -- a few kilobytes, and it is what anyone checking the numbers reads
-- and the figures belong in `figures/` next to it. Rebuilding them should not
require retraining, and it should not require torch either, which importing
`train_backbone` would. Nothing here needs more than numpy and matplotlib.

`train_backbone.py` calls `write_figures` at the end of a run and this file's
CLI calls the same function on a saved result, so a figure rebuilt here is the
figure that run produced rather than a second implementation of it.

    cd src && python plots.py ../train_backbone_rml2018_f512_colab.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def write_figures(curves, confusions, snrs, class_names, dataset,
                  high_acc, stem, fig_dir, n_seeds,
                  high_snr_threshold=10):
    """
    The accuracy curve and the confusion matrix, written to `fig_dir`.

    Split out of `main` so that `replot.py` can rebuild both from a saved .npz
    without retraining -- a run done on Colab leaves its .npz behind, and the
    figures belong in the repository next to it. Two copies of this code would
    drift apart and quietly produce two different-looking plots of the same
    numbers.
    """
    n = len(class_names)
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs, mean_curve, "o-", lw=2, color="tab:blue", label="ICRNNA")
    if n_seeds > 1:
        ax.fill_between(snrs, mean_curve - std_curve, mean_curve + std_curve,
                        color="tab:blue", alpha=0.15)
    ax.axhline(1 / n, ls=":", color="red", lw=1, label=f"chance ({1/n:.3f})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"ICRNNA on {dataset}, {n} classes, {n_seeds} seed(s)"
                 + (" (band = 1 s.d.)" if n_seeds > 1 else ""))
    fig.tight_layout()
    fig.savefig(fig_dir / f"29_{stem}_accuracy.png", dpi=140)
    plt.close(fig)

    cm = confusions.sum(axis=0).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = max(7, n * 0.55)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"ICRNNA on {dataset}, SNR >= {high_snr_threshold} dB "
                 f"(acc {high_acc:.3f})")
    if n <= 14:
        for r in range(n):
            for c in range(n):
                if cm[r, c] > 0.02:
                    ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if cm[r, c] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / f"30_{stem}_confusion.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+",
                   help="result file(s) written by train_backbone.py")
    p.add_argument("--out-dir", default=None,
                   help="where the figures go (default: <repo>/figures)")
    args = p.parse_args()

    fig_dir = pathlib.Path(args.out_dir) if args.out_dir else ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for path in args.npz:
        path = pathlib.Path(path)
        d = np.load(path, allow_pickle=True)
        curves, confusions = d["curves"], d["confusions"]
        names = [str(c) for c in d["class_names"]]
        n_seeds = int(curves.shape[0])
        threshold = int(d["high_snr_threshold"])

        print(f"{path.name}: {len(names)} classes, {n_seeds} seed(s), "
              f"{curves.shape[1]} SNR levels")
        print(f"  overall {np.nanmean(d['overall']):.4f}   "
              f"SNR >= {threshold} dB {np.nanmean(d['high']):.4f}")

        write_figures(curves, confusions, d["snrs"], names, str(d["dataset"]),
                      float(np.nanmean(d["high"])), path.stem, fig_dir,
                      n_seeds, threshold)
        print(f"  wrote {fig_dir / f'29_{path.stem}_accuracy.png'}")
        print(f"  wrote {fig_dir / f'30_{path.stem}_confusion.png'}")


if __name__ == "__main__":
    main()

"""Our run of the paper-faithful ICRNNA, drawn the way the paper draws its own
results, for the paper deck:

    python deck/paper_figs.py [results.json ...]  -> deck/paper/ours_*.png

Defaults to icrnna_faithful_e150_results.json and icrnna_faithful_results.json
in the repository root (copied from Drive/RadioML/ICRNNA_faithful/ after the
Colab task faithful_paper). Matrices are pooled over the seeds that have them
and shown as percent of each row, like the paper's Figs. 11 and 12; the
training curve is one seed, train and validation accuracy per epoch, like its
Fig. 7, with the paper's 58-epoch budget marked.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).parent / "paper"
NAVY, ORANGE, MUTED = "#0F1B2D", "#F08A24", "#5B6B7F"


def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    seeds = [r for r in d["per_seed"] if "confusions_by_snr" in r]
    return d, seeds


def matrix(ax, m, classes, title):
    pct = 100 * m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = pct[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.1f}" if v < 99.95 else "100", ha="center", va="center", fontsize=7,
                        color="white" if v > 60 else "#1E293B")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("said", fontsize=9)
    ax.set_ylabel("sent", fontsize=9)


def pooled(seeds, snr):
    snrs = seeds[0]["snrs"]
    return np.sum([np.asarray(r["confusions_by_snr"])[snrs.index(snr)] for r in seeds], axis=0)


def main(paths):
    OUT.mkdir(exist_ok=True)
    d, seeds = load(paths[0])
    classes = d.get("classes")
    tag = "e150" if d["config"]["epochs"] != 58 else "e58"
    n = len(seeds)
    if not seeds:
        raise SystemExit(f"{paths[0]}: no seed has confusion matrices yet")

    # Fig. 11 counterpart: 0 dB
    m0 = pooled(seeds, 0)
    acc0 = 100 * np.trace(m0) / m0.sum()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    matrix(ax, m0, classes, f"Ours, 0 dB, {n} run{'s' * (n > 1)}: {acc0:.2f}%  ({int(m0.sum()):,} test frames)")
    fig.tight_layout()
    fig.savefig(OUT / f"ours_{tag}_0db.png", dpi=170)
    plt.close(fig)

    # Fig. 12 counterpart: -4, -2, 2, 4 dB
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    for ax, snr in zip(axs.flat, (-4, -2, 2, 4)):
        m = pooled(seeds, snr)
        matrix(ax, m, classes, f"{snr:+d} dB: {100 * np.trace(m) / m.sum():.2f}%")
    fig.tight_layout()
    fig.savefig(OUT / f"ours_{tag}_4snr.png", dpi=150)
    plt.close(fig)

    # Fig. 7 counterpart: accuracy over epochs, first seed that has a history
    hist = next((r for r in seeds if r.get("history")), None)
    if hist:
        h = hist["history"]
        ep = [x["epoch"] for x in h]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(ep, [x["train_acc"] for x in h], color=NAVY, lw=2, label="training accuracy")
        ax.plot(ep, [x["val_acc"] for x in h], color=ORANGE, lw=2, label="validation accuracy")
        ax.axvline(58, color=MUTED, ls="--", lw=1.2)
        ax.text(59, ax.get_ylim()[0] + 0.5, "paper stops: 58", color=MUTED, fontsize=10)
        ax.axvline(hist["best_epoch"], color=ORANGE, ls=":", lw=1.2)
        ax.text(hist["best_epoch"] + 1, ax.get_ylim()[1] - 1.5, f"best: {hist['best_epoch']}", color=ORANGE, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"Ours, seed {hist['seed']}", fontsize=11)
        ax.legend(frameon=False, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"ours_{tag}_curve.png", dpi=170)
        plt.close(fig)

    accs = [r["test_acc"] for r in d["per_seed"]]
    print(f"{paths[0]}: {len(accs)} seeds, {np.mean(accs):.2f} +- {np.std(accs):.2f}; "
          f"matrices from {n}; 0 dB {acc0:.2f}%; history: {'seed ' + str(hist['seed']) if hist else 'none'}")


if __name__ == "__main__":
    args = sys.argv[1:] or [str(ROOT / "icrnna_faithful_e150_results.json")]
    for a in args:
        main([a])

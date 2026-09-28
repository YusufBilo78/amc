"""Figure 34: whitening on RML2016.10a against the envelope smoothing width.

    python tools/smoothing_figure.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    z = np.load(ROOT / "whitening_smoothing_rml2016_e100.npz")
    cm = np.load(ROOT / "compare_methods_ICRNNA_es_rml2016_e100.npz")
    bins = z["bins"][1:]
    n = int(z["frame_len"])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for k, (key, label) in enumerate((("in_domain", "in-domain"),
                                      ("cross", "cross-domain"))):
        v = 100 * z[key][1:]
        ax.errorbar(bins, v.mean(1), yerr=v.std(1), color=SLOT[k],
                    marker="o", markersize=6, capsize=3, linewidth=2)
        ax.axhline(100 * z[key][0].mean(), color=SLOT[k], linewidth=1,
                   linestyle="--")
        ax.annotate(f"{label}, whitened", (bins[-1], v.mean(1)[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
        ax.text(2.6, 100 * z[key][0].mean() + 0.3, f"{label}, no whitening",
                color=INK2, fontsize=8.5)
    aug = 100 * cm["cross"][1][:5].mean()
    ax.axhline(aug, color=SLOT[2], linewidth=1, linestyle=":")
    ax.text(2.6, aug + 0.3, "cross-domain, standard augmentation", color=INK2,
            fontsize=8.5)
    ax.axvline(5, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(5.2, 80.6, "5 bins: the width used\nin the method table",
            color=INK2, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bins, [f"{b}\n{100 * b / n:.0f}%" for b in bins])
    ax.set_xlim(2.5, 160)
    ax.set_ylim(78, 100)
    ax.set_xlabel(f"envelope smoothing width, bins of the {n}-point spectrum "
                  "(and share of the band)")
    ax.set_ylabel("accuracy at SNR ≥ 10 dB (%)")
    ax.set_title("Whitening on RML2016.10a (α = 0.75) against smoothing width\n"
                 "5 seeds per point, bars = 1 s.d.; in-domain 3,750 and "
                 "cross-domain 10,000 test frames per seed",
                 loc="left", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "34_whitening_smoothing_rml2016.png", dpi=150)


if __name__ == "__main__":
    main()

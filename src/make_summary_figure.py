"""Compact two-panel figure for the one-page summary sent to faculty."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
d = np.load(ROOT / "compare_methods.npz", allow_pickle=True)
methods = [str(m) for m in d["methods"]]
snrs, curves = d["snrs"], d["curves"]      # curves: (method, seed, snr)
in_dom, cross = d["in_domain"], d["cross"]
gap = in_dom - cross

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ---- Panel A: the gap and its closure -----------------------------------
none_i, best_i = 0, 3
ax1.plot(snrs, np.nanmean(curves[none_i], axis=0), "s-", lw=2, ms=4,
         color="#c0392b", label="cross-domain, no fix")
ax1.plot(snrs, np.nanmean(curves[best_i], axis=0), "o-", lw=2, ms=4,
         color="#27ae60", label="cross-domain, with fix")
ax1.axhline(in_dom[none_i].mean(), ls="--", lw=1.3, color="#2c3e50",
            label="in-domain (lab performance)")
ax1.fill_between(snrs, np.nanmean(curves[none_i], axis=0),
                 np.nanmean(curves[best_i], axis=0), color="#27ae60", alpha=0.12)
ax1.set_xlabel("SNR (dB)", fontsize=9)
ax1.set_ylabel("classification accuracy", fontsize=9)
ax1.set_title("A. The gap, and how much of it closes", fontsize=10, fontweight="bold")
ax1.set_ylim(0, 1.03); ax1.grid(alpha=0.25); ax1.tick_params(labelsize=8)
ax1.legend(fontsize=7.5, loc="upper left")

# ---- Panel B: against the literature baseline ---------------------------
labels = ["none", "standard\naugmentation", "whitening", "whitening +\nstandard"]
y = np.arange(len(labels))
ax2.barh(y, gap.mean(1), 0.6, xerr=gap.std(1), capsize=3,
         color=["#95a5a6", "#e67e22", "#2980b9", "#27ae60"])
for i in range(len(labels)):
    ax2.text(gap[i].mean() + gap[i].std() + 0.012, i, f"{gap[i].mean():.3f}",
             va="center", fontsize=8)
ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8)
ax2.invert_yaxis(); ax2.set_xlim(0, 0.235)
ax2.set_xlabel("generalization gap  (lower is better)", fontsize=9)
ax2.set_title("B. Against the standard AMC\naugmentation set", fontsize=10, fontweight="bold")
ax2.grid(alpha=0.25, axis="x"); ax2.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(ROOT / "figures" / "summary_panel.png", dpi=200, bbox_inches="tight")
print("wrote figures/summary_panel.png")

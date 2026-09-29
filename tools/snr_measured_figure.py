"""Figure 35: the SNR measured in RML2016.10a against its label.

    python tools/snr_measured_figure.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ed"
LOW, HIGH = -13, 18          # outside this the estimator cannot read this file


def main():
    d = json.loads((ROOT / "rml2016_measured_snr.json").read_text())
    lab = np.array(d["labels"])
    m = {c: np.array([d["measured_snr_db"][c][str(s)] for s in lab])
         for c in d["measured_snr_db"]}
    series = [("PSK (BPSK, QPSK, 8PSK)", np.mean([m["BPSK"], m["QPSK"], m["8PSK"]], 0)),
              ("PAM4", m["PAM4"]), ("QAM16", m["QAM16"]), ("QAM64", m["QAM64"])]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhspan(HIGH, 26, color=BAND, zorder=0)
    ax.axhspan(-40, LOW, color=BAND, zorder=0)
    ax.text(-19.6, 24.2, "above ~18 dB the file's own out-of-band floor hides "
            "the noise: not readable", color=INK2, fontsize=8.5)
    ax.text(-19.6, -24.5, "below about −13 dB: not readable", color=INK2, fontsize=8.5)
    ax.plot(lab, 2 * lab + 2.4, color=INK2, linewidth=1.2, linestyle="--")
    ax.text(-19.6, -35.5, "published code, rebuilt: 2 × label + 2.4 dB",
            color=INK2, fontsize=8.5)
    for k, (name, v) in enumerate(series):
        ax.plot(lab, v, color=SLOT[k], marker="o", markersize=4, linewidth=2,
                label=name)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(-20.5, 18.5)
    ax.set_ylim(-38, 26)
    ax.set_xticks(range(-20, 19, 2))
    ax.set_xlabel("SNR label in the dataset (dB)")
    ax.set_ylabel("SNR measured in the frames (dB, per sample, full band)")
    ax.set_title("RML2016.10a: the SNR label against the SNR in the frames\n"
                 "1,000 frames per point; noise read from the spectrum outside "
                 "the signal's band",
                 loc="left", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "35_rml2016_snr_measured.png", dpi=150)


if __name__ == "__main__":
    main()

"""
diagnose_16qam.py -- why does our 16QAM read as 64QAM in every configuration?

sweep.py ruled out the easy answer. 16QAM sits between 0.05 and 0.37 across
every value of beta and sps tried, while the other four classes reach 1.00 at
the right settings. So this is not a mistuned parameter; something about the
two 16QAM signals differs structurally.

Four measurements, cheapest and most decisive first:

1. Constellation. What the CNN actually sees, raw, no matched filter -- because
   the CNN has no matched filter either.

2. Amplitude histogram. This is the discriminator that matters: 16QAM has three
   distinct symbol amplitudes, 64QAM has nine. If our 16QAM shows more distinct
   levels than RadioML's, the model is right to call it a denser constellation
   and the bug is in our generator, not the model.

3. Cyclostationary symbol-rate estimate. For a linearly modulated signal the
   instantaneous power |x|^2 is periodic at the symbol rate, so its spectrum has
   a line at 1/sps. This measures RadioML's undocumented oversampling factor
   directly instead of inferring it from where the model happens to transfer.

4. Mean power spectrum. Occupied bandwidth follows from symbol rate and
   roll-off; a mismatch here corroborates whatever (3) says.
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import domains
import features

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

PAIR = ["16QAM", "64QAM"]
SNR = 30
FRAMES = 300


def symbol_rate_estimate(frames: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate samples-per-symbol from the cyclostationarity of |x|^2.

    Returns (sps_estimate, freqs, mean_spectrum_db). The search ignores very
    low frequencies, where the slow envelope variation of the message sits.
    """
    power = np.abs(frames) ** 2
    power = power - power.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(power, axis=1)) ** 2
    mean_spec = spec.mean(axis=0)

    freqs = np.fft.rfftfreq(frames.shape[1])
    db = 10 * np.log10(mean_spec + 1e-20)
    db = db - db.max()

    # Symbol rate must be below Nyquist and above 1/64 of the sample rate.
    band = (freqs > 1 / 64) & (freqs < 0.5)
    idx = np.flatnonzero(band)
    peak_i = idx[np.argmax(db[idx])]

    # A bare argmax over a monotonically falling curve always "finds" a peak at
    # the low edge of the search band and reports a confident, meaningless
    # number. Require actual prominence: the candidate must stand clear of a
    # smoothed local baseline, otherwise report that no line exists.
    window = max(9, len(db) // 64) | 1
    kernel = np.ones(window) / window
    baseline = np.convolve(db, kernel, mode="same")
    prominence = db[peak_i] - baseline[peak_i]

    if prominence < 6.0:
        return np.nan, freqs, db
    return 1.0 / freqs[peak_i], freqs, db


def main() -> None:
    rml = domains.RadioMLDomain()
    syn = domains.SyntheticDomain(sps=8, beta=0.35)

    sets: dict[str, dict[str, np.ndarray]] = {}
    for label, dom in [("RadioML", rml), ("ours", syn)]:
        data = dom.load(PAIR, [SNR], frames_per_cell=FRAMES, seed=3)
        z = data["X"][:, 0] + 1j * data["X"][:, 1]
        sets[label] = {name: z[data["y"] == i] for i, name in enumerate(PAIR)}
    rml.close()

    # ------------------------------------------------------- symbol rate
    print(f"{'source':<10} {'class':<8} {'est. sps':>9}")
    print("-" * 30)
    sps_est = {}
    spectra = {}
    for label in ("RadioML", "ours"):
        for name in PAIR:
            est, freqs, db = symbol_rate_estimate(sets[label][name])
            sps_est[(label, name)] = est
            spectra[(label, name)] = (freqs, db)
            shown = f"{est:>9.2f}" if np.isfinite(est) else f"{'no line':>9}"
            print(f"{label:<10} {name:<8} {shown}")

    # ---------------------------------------------------------- features
    print(f"\n{'source':<10} {'class':<8} " +
          " ".join(f"{n:>9}" for n in ("|C20|", "|C40|", "|C42|", "sigma_aa", "kurt_amp")))
    print("-" * 62)
    for label in ("RadioML", "ours"):
        for name in PAIR:
            f = dict(zip(features.FEATURE_NAMES,
                         features.extract_batch(sets[label][name]).mean(axis=0)))
            print(f"{label:<10} {name:<8} " + " ".join(
                f"{f[k]:>9.3f}" for k in ("|C20|", "|C40|", "|C42|", "sigma_aa", "kurt_amp")))

    # ------------------------------------------------------------- plot
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    # Row 1: constellations, raw samples (what the CNN sees)
    for col, (label, name) in enumerate(
            [(l, n) for l in ("RadioML", "ours") for n in PAIR]):
        ax = axes[0, col]
        z = sets[label][name][:60].ravel()
        ax.plot(z.real, z.imag, ",", alpha=0.25, color="tab:blue")
        ax.set_title(f"{label} — {name}", fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("raw IQ (no matched filter)", fontsize=10)

    # Row 2: amplitude histograms, overlaid per class
    for col, name in enumerate(PAIR):
        ax = axes[1, col * 2]
        for label, color in [("RadioML", "tab:blue"), ("ours", "tab:red")]:
            amp = np.abs(sets[label][name]).ravel()
            amp = amp / amp.mean()
            ax.hist(amp, bins=200, range=(0, 3), density=True, histtype="step",
                    lw=1.8, color=color, label=label)
        ax.set_title(f"|x| distribution — {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("normalized amplitude")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        axes[1, col * 2 + 1].axis("off")

    # Row 3: cyclostationary spectra and mean PSD
    ax = axes[2, 0]
    for (label, name), (freqs, db) in spectra.items():
        style = "-" if label == "RadioML" else "--"
        ax.plot(freqs, db, style, lw=1.3, label=f"{label} {name}")
    ax.set_title("spectrum of $|x|^2$ — peak sits at the symbol rate",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("normalized frequency")
    ax.set_ylabel("dB")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-45, 3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[2, 1]
    for label, color in [("RadioML", "tab:blue"), ("ours", "tab:red")]:
        for name, style in zip(PAIR, ("-", "--")):
            frames = sets[label][name]
            psd = np.mean(np.abs(np.fft.fftshift(
                np.fft.fft(frames, axis=1), axes=1)) ** 2, axis=0)
            psd_db = 10 * np.log10(psd + 1e-12)
            ax.plot(np.fft.fftshift(np.fft.fftfreq(frames.shape[1])),
                    psd_db - psd_db.max(), style, lw=1.3, color=color,
                    label=f"{label} {name}")
    ax.set_title("mean power spectrum", fontsize=11, fontweight="bold")
    ax.set_xlabel("normalized frequency")
    ax.set_ylabel("dB")
    ax.set_ylim(-50, 3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Estimated sps as a bar chart, so the mismatch is unmissable
    ax = axes[2, 2]
    keys = list(sps_est.keys())
    values = [sps_est[k] if np.isfinite(sps_est[k]) else 0.0 for k in keys]
    ax.bar(range(len(keys)), values,
           color=["tab:blue", "tab:blue", "tab:red", "tab:red"])
    for i, k in enumerate(keys):
        if not np.isfinite(sps_est[k]):
            ax.text(i, 0.4, "no\nsymbol-rate\nline", ha="center", va="bottom",
                    fontsize=8, color="tab:blue", fontweight="bold")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{a}\n{b}" for a, b in keys], fontsize=8)
    ax.set_ylabel("estimated samples per symbol")
    ax.set_title("cyclostationary symbol-rate line", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.grid(alpha=0.2, axis="y")
    axes[2, 3].axis("off")

    fig.suptitle(
        "Why our 16QAM is read as 64QAM: structural comparison at SNR = 30 dB",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "11_diagnose_16qam.png", dpi=130)
    plt.close(fig)
    print("\nwrote figures/11_diagnose_16qam.png")


if __name__ == "__main__":
    main()

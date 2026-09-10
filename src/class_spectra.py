"""
class_spectra.py -- average power spectrum per class, to settle the AM-SSB question.

verify_classes.py found that classes 17 and 18 (assumed AM-SSB-WC / AM-SSB-SC)
have |C20| ~ 0.78, i.e. they look real-valued, whereas an ideal single-sideband
signal is analytic and therefore proper complex with |C20| = 0.

Cumulants are a summary statistic and can be fooled. The spectrum cannot: a
single-sideband signal puts its energy on one side of DC and nothing on the
other. This plots the mean spectrum of every class so the answer is visual.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import radioml

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

FRAMES = 256
SNR = 30


def main() -> None:
    with radioml.RadioML() as ds:
        print(f"loading {FRAMES} frames/class at SNR = {SNR} dB ...")
        data = ds.load(snrs=[SNR], frames_per_cell=FRAMES, test_fraction=0.0)
        X = radioml.RadioML.to_complex(data["X_train"])
        y = data["y_train"]

        freqs = np.fft.fftshift(np.fft.fftfreq(X.shape[1]))
        fig, axes = plt.subplots(4, 6, figsize=(20, 12), sharex=True)

        print(f"\n{'idx':>3} {'assumed name':<12} {'upper/total':>12}  shape")
        print("-" * 46)

        for class_id, ax in enumerate(axes.ravel()):
            frames = X[y == class_id]
            psd = np.mean(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2,
                          axis=0)
            psd_db = 10 * np.log10(psd + 1e-12)
            psd_db -= psd_db.max()

            # Energy above DC as a fraction of the total. ~0.5 means symmetric,
            # near 0 or 1 means one-sided (i.e. genuinely single-sideband).
            mid = len(psd) // 2
            frac_upper = float(psd[mid:].sum() / psd.sum())
            shape = ("ONE-SIDED" if frac_upper > 0.85 or frac_upper < 0.15
                     else "symmetric")

            name = radioml.CLASSES[class_id]
            print(f"{class_id:>3} {name:<12} {frac_upper:>12.3f}  {shape}")

            ax.plot(freqs, psd_db, lw=0.8,
                    color="tab:red" if shape == "ONE-SIDED" else "tab:blue")
            ax.axvline(0, color="gray", lw=0.5, ls="--")
            ax.set_title(f"{class_id}: {name}", fontsize=9, fontweight="bold")
            ax.set_ylim(-60, 3)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)

        fig.suptitle(
            f"Mean power spectrum per class, RadioML 2018.01A at SNR = {SNR} dB\n"
            "red = one-sided (single-sideband), blue = symmetric about DC",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(FIGURES / "08_radioml_class_spectra.png", dpi=130)
        plt.close(fig)
        print("\nwrote figures/08_radioml_class_spectra.png")


if __name__ == "__main__":
    main()

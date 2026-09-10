"""
spectral_occlusion.py -- show the shortcut, don't just infer it.

Everything so far argues *indirectly* that the unwhitened model reads
modulation order off the spectral envelope: substituting RadioML's magnitude
spectrum fixed the failure, and whitening the envelope away closed 79% of the
gap. Neither observation looks inside the network.

This does. Sweep a notch across the spectrum, null one narrow band at a time,
and measure how much accuracy that costs. The resulting curve is a direct map
of which frequencies each model actually depends on.

PREDICTION, recorded before running:

    The alpha=0 model should depend disproportionately on the band edges and
    the out-of-band region, because that is where envelope shape lives -- the
    roll-off skirts are what differ between 16QAM and 64QAM in its training
    data. The alpha=0.75 model should depend on in-band content instead, since
    whitening has flattened the envelope and left only constellation structure
    to key on.

    If both models show the same dependence profile, the shortcut story is
    wrong no matter how well whitening worked, and the mechanism needs
    rethinking.

Occlusion is used rather than input gradients because gradients at a single
point are noisy and can highlight directions the model never actually moves
along. Nulling a band and watching the prediction fail is a blunter question
with a less ambiguous answer.
"""

from __future__ import annotations

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
import domains
from whitening_sweep import spectral_whiten, to_model_input

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

CLASSES = list(domains.SHARED_CLASSES)
TRAIN_SNRS = list(range(-20, 31, 2))
EVAL_SNRS = list(range(10, 31, 2))
EPOCHS = 15
SEED = 0

NOTCH_WIDTH = 24        # FFT bins nulled at a time
NOTCH_STEP = 12         # slide in half-width steps


def notch(frames_iq: np.ndarray, centre_bin: int, width: int) -> np.ndarray:
    """
    Zero a band of the spectrum, symmetric about DC.

    Both the positive and negative frequency sides are nulled together: a real
    receiver cannot see one without the other, and nulling only one side would
    additionally destroy the conjugate structure, confounding the measurement.
    """
    spec = np.fft.fftshift(np.fft.fft(frames_iq, axis=1), axes=1)
    n = spec.shape[1]
    mid = n // 2
    for sign in (+1, -1):
        c = mid + sign * centre_bin
        lo, hi = max(0, c - width // 2), min(n, c + width // 2)
        spec[:, lo:hi] = 0
    out = np.fft.ifft(np.fft.ifftshift(spec, axes=1), axis=1)
    power = np.mean(np.abs(out) ** 2, axis=1, keepdims=True)
    return out / (np.sqrt(power) + 1e-12)


def train_one(alpha: float, a, a_iq, tr, te):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X = to_model_input(spectral_whiten(a_iq, alpha))
    model = model_zoo.backbone(len(CLASSES))
    model = cnn.train_model(model, X[tr], a["y"][tr], X[te], a["y"][te],
                            epochs=EPOCHS)
    return model


def main() -> None:
    src = domains.RadioMLDomain()
    a = src.load(CLASSES, TRAIN_SNRS, frames_per_cell=768, seed=0)
    ev = src.load(CLASSES, EVAL_SNRS, frames_per_cell=300, seed=9)
    src.close()

    a_iq = (a["X"][:, 0] + 1j * a["X"][:, 1]).astype(np.complex64)
    ev_iq = (ev["X"][:, 0] + 1j * ev["X"][:, 1]).astype(np.complex64)
    tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3, seed=SEED)

    n = a_iq.shape[1]
    centres = list(range(0, n // 2, NOTCH_STEP))
    freqs = np.array(centres) / n
    print(f"{len(centres)} notch positions, width {NOTCH_WIDTH} bins "
          f"({NOTCH_WIDTH / n:.4f} normalized)\n")

    results = {}
    for alpha in (0.0, 0.75):
        print(f"--- alpha = {alpha} ---")
        model = train_one(alpha, a, a_iq, tr, te)

        # Whitening is part of the model's input pipeline, so it is applied
        # AFTER the notch: the notch models a real spectral obstruction, and a
        # deployed receiver would whiten whatever reaches it.
        base_X = to_model_input(spectral_whiten(ev_iq, alpha))
        base = float((cnn.predict(model, base_X) == ev["y"]).mean())
        print(f"unoccluded accuracy: {base:.4f}")

        drops = []
        t0 = time.time()
        for c in centres:
            occluded = notch(ev_iq, c, NOTCH_WIDTH)
            X = to_model_input(spectral_whiten(occluded, alpha))
            acc = float((cnn.predict(model, X) == ev["y"]).mean())
            drops.append(base - acc)
        print(f"swept in {time.time()-t0:.0f}s   "
              f"max drop {max(drops):.3f} at f={freqs[int(np.argmax(drops))]:.3f}\n")
        results[alpha] = {"base": base, "drops": np.array(drops)}

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    # Mean signal spectrum for context: where is the signal actually located?
    psd = np.mean(np.abs(np.fft.fftshift(
        np.fft.fft(ev_iq, axis=1), axes=1)) ** 2, axis=0)
    psd_db = 10 * np.log10(psd + 1e-12)
    psd_db -= psd_db.max()
    half = len(psd_db) // 2
    ax.plot(np.arange(half) / n, psd_db[half:], color="gray", lw=1.2,
            label="mean signal spectrum (dB, right axis)")
    ax.set_ylabel("dB")
    ax.set_ylim(-55, 5)
    ax.set_xlabel("normalized frequency")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Where the signal is", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for alpha, color, label in [
        (0.0, "tab:red", "alpha = 0 (shortcut model)"),
        (0.75, "tab:blue", "alpha = 0.75 (whitened)"),
    ]:
        d = results[alpha]["drops"]
        ax.plot(freqs, d / (d.max() + 1e-12), "o-", lw=2, ms=4, color=color,
                label=f"{label}, base {results[alpha]['base']:.3f}")
    ax.axvline(0.06, ls="--", color="gray", lw=1, alpha=0.7)
    ax.text(0.062, 0.95, "edge of\noccupied band", fontsize=8, color="gray")
    ax.set_xlabel("notch centre frequency (normalized)")
    ax.set_ylabel("accuracy drop, normalized to each model's max")
    ax.set_title("Which frequencies each model depends on",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Spectral occlusion: nulling one band at a time", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "18_spectral_occlusion.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "spectral_occlusion.npz", freqs=freqs,
             drops_a0=results[0.0]["drops"], drops_a75=results[0.75]["drops"],
             base_a0=results[0.0]["base"], base_a75=results[0.75]["base"])

    # ------------------------------------------------------------- verdict
    band_edge = 0.06
    print("=" * 66)
    for alpha in (0.0, 0.75):
        d = results[alpha]["drops"]
        inside = d[freqs <= band_edge].sum()
        outside = d[freqs > band_edge].sum()
        print(f"alpha={alpha:<5} dependence in-band {inside/(inside+outside):.1%}"
              f"   out-of-band {outside/(inside+outside):.1%}"
              f"   peak at f={freqs[int(np.argmax(d))]:.3f}")
    print("=" * 66)
    print("\nwrote figures/18_spectral_occlusion.png")


if __name__ == "__main__":
    main()

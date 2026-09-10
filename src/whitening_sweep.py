"""
whitening_sweep.py -- remove the shortcut instead of randomising it.

Round 1 of the fix failed. Randomising the spectral envelope during training
(`augment.spectral_reshape`) made the cross-domain gap *worse*: 0.825 -> 0.783,
while the ablation without it landed at 0.815. Training loss plateaued at 0.64
against 0.20 unaugmented, i.e. the transform was destroying information rather
than teaching invariance.

Re-reading the evidence: the intervention that worked in narrow_the_search.py
*matched* the two spectra, it did not randomise them. So the right move is not
augmentation but preprocessing -- reduce every frame's spectral envelope to a
canonical form, in both domains, so the envelope carries no information for
anyone.

Spectral whitening does that:

    X_white(f) = X(f) / ( smooth(|X(f)|) + eps )^alpha

Divide out a smoothed version of the frame's own magnitude envelope, keep the
phase untouched. alpha = 0 is a no-op, alpha = 1 is full whitening. Partial
whitening is likely to win, because full whitening also amplifies out-of-band
noise, so the sweep covers the range rather than assuming an endpoint.

Applied identically to train and test. This is a preprocessing step, not a
domain adaptation method -- it needs no target-domain data at fit time, which
is what makes it usable on a real receiver.
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
from scipy.ndimage import uniform_filter1d

import cnn
import domains

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

CLASSES = list(domains.SHARED_CLASSES)
SNRS = list(range(-20, 31, 2))
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]
EPOCHS = 15          # lower than elsewhere: five models to train
SMOOTH_BINS = 33     # envelope smoothing width, in FFT bins


def spectral_whiten(frames_iq: np.ndarray, alpha: float,
                    smooth_bins: int = SMOOTH_BINS) -> np.ndarray:
    """
    Flatten each frame's spectral envelope by `alpha`, preserving phase.

    The envelope estimate is a circular moving average of |X(f)|. Frequency
    wraps around, so the smoothing has to wrap too -- a non-circular filter
    would fabricate an artificial roll-off at the band edges, which is exactly
    the kind of artifact this is meant to remove.
    """
    if alpha <= 0:
        return frames_iq

    spec = np.fft.fft(frames_iq, axis=1)
    env = uniform_filter1d(np.abs(spec), size=smooth_bins, axis=1, mode="wrap")
    out = np.fft.ifft(spec / (env**alpha + 1e-12), axis=1)
    power = np.mean(np.abs(out) ** 2, axis=1, keepdims=True)
    return out / (np.sqrt(power) + 1e-12)


def to_model_input(frames_iq: np.ndarray) -> np.ndarray:
    return np.stack([frames_iq.real, frames_iq.imag], axis=1).astype(np.float32)


def accuracy_by_snr(pred, y, z, snrs):
    return np.array([
        float((pred[z == s] == y[z == s]).mean()) if (z == s).any() else np.nan
        for s in snrs
    ])


def main() -> None:
    print("loading domains once; whitening is applied per alpha\n")
    src = domains.RadioMLDomain()
    a = src.load(CLASSES, SNRS, frames_per_cell=768, seed=0)
    src.close()
    dst = domains.SyntheticDomain()
    b = dst.load(CLASSES, SNRS, frames_per_cell=400, seed=1)

    a_iq = (a["X"][:, 0] + 1j * a["X"][:, 1]).astype(np.complex64)
    b_iq = (b["X"][:, 0] + 1j * b["X"][:, 1]).astype(np.complex64)
    tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3, seed=0)
    print(f"{len(tr)} train / {len(te)} in-domain test / {len(b_iq)} cross-domain\n")

    high = np.array(SNRS) >= 10
    rows = []

    header = (f"{'alpha':>6} {'in-domain':>10} {'cross':>8} {'gap':>8} "
              f"{'16QAM cross':>12} {'time':>7}")
    print(header)
    print("-" * len(header))

    for alpha in ALPHAS:
        t0 = time.time()
        Xa = to_model_input(spectral_whiten(a_iq, alpha))
        Xb = to_model_input(spectral_whiten(b_iq, alpha))

        model = cnn.IQNet(len(CLASSES))
        model = cnn.train_model(model, Xa[tr], a["y"][tr], Xa[te], a["y"][te],
                                epochs=EPOCHS)

        pred_in = cnn.predict(model, Xa[te])
        pred_cross = cnn.predict(model, Xb)
        acc_in = accuracy_by_snr(pred_in, a["y"][te], a["z"][te], SNRS)
        acc_cross = accuracy_by_snr(pred_cross, b["y"], b["z"], SNRS)

        q = CLASSES.index("16QAM")
        mask = (b["z"] >= 10) & (b["y"] == q)
        qam_cross = float((pred_cross[mask] == q).mean())

        in_hi = float(np.nanmean(acc_in[high]))
        cr_hi = float(np.nanmean(acc_cross[high]))
        rows.append((alpha, in_hi, cr_hi, acc_in, acc_cross, qam_cross))
        print(f"{alpha:>6.2f} {in_hi:>10.3f} {cr_hi:>8.3f} {in_hi-cr_hi:>+8.3f} "
              f"{qam_cross:>12.3f} {time.time()-t0:>6.0f}s")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    alphas = [r[0] for r in rows]
    ax.plot(alphas, [r[1] for r in rows], "o-", lw=2, color="tab:blue",
            label="in-domain (RadioML held out)")
    ax.plot(alphas, [r[2] for r in rows], "s-", lw=2, color="tab:red",
            label="cross-domain (synthetic)")
    ax.plot(alphas, [r[5] for r in rows], "^--", lw=1.8, color="tab:green",
            label="16QAM only, cross-domain")
    ax.fill_between(alphas, [r[2] for r in rows], [r[1] for r in rows],
                    color="tab:red", alpha=0.10)
    ax.set_xlabel(r"whitening strength  $\alpha$")
    ax.set_ylabel("accuracy at SNR >= 10 dB")
    ax.set_title("Spectral whitening as preprocessing, both domains",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))
    for (alpha, _, _, _, acc_cross, _), c in zip(rows, colors):
        ax.plot(SNRS, acc_cross, "-", lw=1.8, color=c, label=f"alpha={alpha}")
    ax.axhline(0.825, ls="--", color="gray", lw=1.2,
               label="no augmentation, no whitening (0.825)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("cross-domain accuracy")
    ax.set_title("Cross-domain accuracy vs SNR", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Removing the spectral shortcut instead of randomising it",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "15_whitening_sweep.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "whitening_sweep.npz", alphas=np.array(alphas),
             in_domain=np.array([r[1] for r in rows]),
             cross_domain=np.array([r[2] for r in rows]),
             qam16=np.array([r[5] for r in rows]),
             snrs=np.array(SNRS),
             curves=np.array([r[4] for r in rows]))

    best = max(rows, key=lambda r: r[2])
    print("\n" + "=" * 62)
    print(f"baseline (alpha=0):   cross {rows[0][2]:.3f}   gap {rows[0][1]-rows[0][2]:+.3f}")
    print(f"best alpha={best[0]:<4}      cross {best[2]:.3f}   gap {best[1]-best[2]:+.3f}")
    print(f"16QAM cross-domain:   {rows[0][5]:.3f} -> {best[5]:.3f}")
    print("=" * 62)
    print("\nwrote figures/15_whitening_sweep.png")


if __name__ == "__main__":
    main()

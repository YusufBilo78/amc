"""
explore_signals.py -- Week 1 deliverable: look at the data before modelling it.

Produces three figures in ../figures:

  01_constellations.png  what each modulation looks like in the IQ plane
  02_spectrograms.png    the same signals as time-frequency images
  03_snr_sweep.png       how 16QAM dissolves into noise as SNR drops

The third one is the important one. It shows, by eye, where the classification
problem stops being easy -- and that boundary is what the whole project is
about. Run this, then open the figures and stare at them for ten minutes.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # write files, no interactive window needed

import matplotlib.pyplot as plt
import numpy as np

import modem

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

SPS = 8  # samples per symbol used throughout


# --------------------------------------------------------------------------
# Small DSP helpers
# --------------------------------------------------------------------------


def matched_filter_symbols(x: np.ndarray, sps: int = SPS, beta: float = 0.35, span: int = 10):
    """
    Apply the receive matched filter and pick one sample per symbol.

    A root-raised-cosine pulse is NOT by itself intersymbol-interference free --
    only RRC(tx) * RRC(rx) = raised cosine is. That is why a clean constellation
    only appears after matched filtering. Skipping this step is the single most
    common reason a student's constellation plot looks like a smear.

    The sampling phase is chosen as the offset with the largest mean magnitude,
    which is a crude but reliable stand-in for timing recovery.

    Output is renormalized to unit average power. Matched filtering multiplies
    the amplitude by roughly sqrt(sps) (energy that was spread across sps
    samples gets collected back into one), so without this step the outer
    constellation points of 16QAM/64QAM land far outside any sensible axis.
    """
    h = modem.rrc_filter(span, sps, beta)
    y = np.convolve(x, h, mode="same")
    best_phase = max(range(sps), key=lambda p: np.mean(np.abs(y[p::sps])))
    s = y[best_phase::sps]
    return s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-12)


def spectrogram_db(x: np.ndarray, nperseg: int = 128, noverlap: int = 112) -> np.ndarray:
    """
    Complex-input spectrogram in dB, as a (frequency x time) image.

    Written out longhand rather than calling scipy so you can see there is no
    magic here: window a slice, FFT it, take the magnitude, slide, repeat.
    fftshift puts DC in the middle, which is the convention you want when the
    signal occupies both negative and positive frequencies.
    """
    step = nperseg - noverlap
    window = np.hanning(nperseg)
    starts = range(0, len(x) - nperseg + 1, step)
    frames = np.stack([x[i : i + nperseg] * window for i in starts])
    spec = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    return 20 * np.log10(np.abs(spec) + 1e-12).T


# --------------------------------------------------------------------------
# Figure 1 -- constellations
# --------------------------------------------------------------------------


def figure_constellations(snr_db: float = 20.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))

    for ax, name in zip(axes.ravel(), modem.MODULATIONS):
        x = modem.generate(name, n_samples=8192, snr_db=snr_db, sps=SPS, rng=rng)

        # Faint trajectory: every sample, including the pulse-shaping path
        # between symbols. This is closer to what a CNN on raw IQ actually sees.
        ax.plot(np.real(x), np.imag(x), lw=0.3, alpha=0.15, color="tab:blue")

        # Bright dots: decision instants, for the schemes where that concept
        # applies. CPFSK/GFSK/analog have no symbol grid to recover.
        if name in modem.CONSTELLATIONS:
            s = matched_filter_symbols(x)
            ax.scatter(np.real(s), np.imag(s), s=3, alpha=0.5, color="tab:orange")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.axhline(0, lw=0.5, color="gray", alpha=0.5)
        ax.axvline(0, lw=0.5, color="gray", alpha=0.5)
        lim = 2.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    for ax in axes.ravel()[len(modem.MODULATIONS) :]:
        ax.axis("off")

    fig.suptitle(
        f"IQ constellations at SNR = {snr_db:.0f} dB\n"
        "faint = all samples (pulse-shaped trajectory), orange = matched-filtered symbol instants",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "01_constellations.png", dpi=140)
    plt.close(fig)
    print("wrote figures/01_constellations.png")


# --------------------------------------------------------------------------
# Figure 2 -- spectrograms
# --------------------------------------------------------------------------


def figure_spectrograms(snr_db: float = 20.0, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))

    for ax, name in zip(axes.ravel(), modem.MODULATIONS):
        x = modem.generate(name, n_samples=4096, snr_db=snr_db, sps=SPS, rng=rng)
        img = spectrogram_db(x)
        ax.imshow(
            img,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=(0, len(x), -0.5, 0.5),
            vmin=np.percentile(img, 5),
            vmax=np.percentile(img, 99.5),
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("sample", fontsize=8)
        ax.set_ylabel("normalized freq", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(modem.MODULATIONS) :]:
        ax.axis("off")

    fig.suptitle(
        f"Spectrograms at SNR = {snr_db:.0f} dB -- this is the 'image' a CNN can be fed",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "02_spectrograms.png", dpi=140)
    plt.close(fig)
    print("wrote figures/02_spectrograms.png")


# --------------------------------------------------------------------------
# Figure 3 -- the one that matters: SNR sweep
# --------------------------------------------------------------------------


def figure_snr_sweep(scheme: str = "16QAM", snrs=(20, 10, 0, -10), seed: int = 2) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, len(snrs), figsize=(3.4 * len(snrs), 7))

    for col, snr in enumerate(snrs):
        x = modem.generate(scheme, n_samples=8192, snr_db=snr, sps=SPS, rng=rng)

        ax = axes[0, col]
        s = matched_filter_symbols(x)
        ax.scatter(np.real(s), np.imag(s), s=3, alpha=0.4, color="tab:orange")
        ax.set_title(f"SNR = {snr} dB", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(alpha=0.2)

        ax = axes[1, col]
        img = spectrogram_db(x[:4096])
        ax.imshow(
            img,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=(0, 4096, -0.5, 0.5),
            vmin=np.percentile(img, 5),
            vmax=np.percentile(img, 99.5),
        )
        ax.set_xlabel("sample", fontsize=8)
        if col == 0:
            ax.set_ylabel("normalized freq", fontsize=9)
        ax.tick_params(labelsize=7)

    axes[0, 0].set_ylabel("constellation", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"{scheme} as SNR falls -- where does the information actually disappear?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "03_snr_sweep.png", dpi=140)
    plt.close(fig)
    print("wrote figures/03_snr_sweep.png")


if __name__ == "__main__":
    figure_constellations()
    figure_spectrograms()
    figure_snr_sweep()
    print(f"\ndone -- figures are in {FIGURES}")

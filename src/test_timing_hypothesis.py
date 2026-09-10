"""
test_timing_hypothesis.py -- the decisive test, with the model held frozen.

Hypothesis from diagnose_16qam.py:

    Our generator has exact, fixed symbol timing, which puts a sharp
    cyclostationary line at f = 1/sps and makes the constellation grid visible
    in the raw samples. RadioML applies a per-frame symbol rate offset, so it
    has neither. The RadioML-trained model has therefore never seen a signal
    with an exact symbol-rate line, and our 16QAM -- whose grid is coarse
    enough to be plainly visible -- falls outside its training distribution.

Prediction: destroy the timing regularity and 16QAM accuracy recovers, with
*no change to the model at all*.

This is the strong form of the test. Retraining would confound the mechanism
with whatever else training changes; here the only thing that varies is one
property of the test signal. Sweeping the jitter amount also gives a
dose-response curve rather than a single before/after pair.

Two quantities are tracked together at each jitter level:

    line prominence  how far the symbol-rate spike stands above the local
                     spectral baseline -- the mechanism
    accuracy         per class, frozen model -- the effect

If the mechanism is real, the accuracy curve rises as the prominence curve
falls, and the crossing happens where the line stops being detectable.
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import resample

import cnn
import domains
import modem

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
MODEL_PATH = ROOT / "model_radioml_5class.pt"

CLASSES = list(domains.SHARED_CLASSES)
SNRS = list(range(10, 31, 2))
FRAMES = 200
FRAME_LEN = 1024
GEN_LEN = 1536  # generate long, crop after resampling -- avoids edge padding

# Fractional symbol-rate offset, +/- this much, drawn independently per frame.
DELTAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

PROMINENCE_THRESHOLD_DB = 6.0


def line_prominence_db(frames: np.ndarray) -> float:
    """
    How far the symbol-rate spike in the spectrum of |x|^2 stands above a
    smoothed local baseline. Same measure diagnose_16qam.py uses to decide
    whether a symbol-rate line exists at all.
    """
    power = np.abs(frames) ** 2
    power = power - power.mean(axis=1, keepdims=True)
    spec = (np.abs(np.fft.rfft(power, axis=1)) ** 2).mean(axis=0)
    db = 10 * np.log10(spec + 1e-20)
    db -= db.max()

    freqs = np.fft.rfftfreq(frames.shape[1])
    band = np.flatnonzero((freqs > 1 / 64) & (freqs < 0.5))
    peak_i = band[np.argmax(db[band])]

    window = max(9, len(db) // 64) | 1
    baseline = np.convolve(db, np.ones(window) / window, mode="same")
    return float(db[peak_i] - baseline[peak_i])


def jittered_frames(scheme: str, n: int, snr_db: float, delta: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    n frames of `scheme`, each independently resampled by a factor drawn from
    [1-delta, 1+delta]. delta=0 reproduces the original fixed-timing generator.
    """
    out = np.empty((n, FRAME_LEN), dtype=np.complex128)
    for i in range(n):
        x = modem.generate(scheme, GEN_LEN, snr_db=snr_db, sps=8, rng=rng)
        if delta > 0:
            factor = rng.uniform(1 - delta, 1 + delta)
            x = resample(x, int(round(GEN_LEN * factor)))
        start = (len(x) - FRAME_LEN) // 2
        x = x[start : start + FRAME_LEN]
        out[i] = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    return out


def build(delta: float, rng: np.random.Generator):
    """Returns (X for the model, labels, per-class frames for the spectral measure)."""
    X, y = [], []
    per_class: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    for i, name in enumerate(CLASSES):
        for snr in SNRS:
            frames = jittered_frames(name, FRAMES, float(snr), delta, rng)
            per_class[name].append(frames)
            X.append(np.stack([frames.real, frames.imag], axis=1).astype(np.float32))
            y.append(np.full(FRAMES, i, dtype=np.int64))
    return (np.concatenate(X), np.concatenate(y),
            {c: np.concatenate(v) for c, v in per_class.items()})


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH.name} not found -- run sweep.py first.")

    model = cnn.IQNet(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cnn.DEVICE))
    model = model.to(cnn.DEVICE)
    print(f"loaded frozen model: {MODEL_PATH.name}")
    print("the model is NOT retrained at any point below\n")

    # RadioML reference: what the prominence looks like in the training domain.
    rml = domains.RadioMLDomain()
    ref = rml.load(["16QAM"], [30], frames_per_cell=300, seed=3)
    rml.close()
    ref_frames = ref["X"][:, 0] + 1j * ref["X"][:, 1]
    ref_prom = line_prominence_db(ref_frames)
    print(f"RadioML 16QAM symbol-rate line prominence: {ref_prom:.2f} dB "
          f"(threshold {PROMINENCE_THRESHOLD_DB} dB)\n")

    header = (f"{'delta':>7} {'prom(dB)':>9} " +
              " ".join(f"{c:>8}" for c in CLASSES) + f" {'overall':>8}")
    print(header)
    print("-" * len(header))

    proms, accs = [], []
    for delta in DELTAS:
        rng = np.random.default_rng(11)
        X, y, per_class = build(delta, rng)
        prom = line_prominence_db(per_class["16QAM"])
        pred = cnn.predict(model, X)
        acc = np.array([float((pred[y == i] == i).mean()) for i in range(len(CLASSES))])
        proms.append(prom)
        accs.append(acc)
        print(f"{delta:>7.3f} {prom:>9.2f} " +
              " ".join(f"{a:>8.3f}" for a in acc) + f" {acc.mean():>8.3f}")

    proms = np.array(proms)
    accs = np.array(accs)

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(DELTAS, proms, "o-", lw=2, color="tab:purple")
    ax.axhline(PROMINENCE_THRESHOLD_DB, ls="--", color="gray", lw=1,
               label=f"detection threshold ({PROMINENCE_THRESHOLD_DB} dB)")
    ax.axhline(ref_prom, ls=":", color="tab:blue", lw=1.5,
               label=f"RadioML ({ref_prom:.1f} dB)")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlabel("per-frame symbol rate offset  ±delta")
    ax.set_ylabel("symbol-rate line prominence (dB)")
    ax.set_title("mechanism: the cyclostationary line disappears",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(CLASSES)))
    for i, name in enumerate(CLASSES):
        ax.plot(DELTAS, accs[:, i], "o-", lw=2, color=colors[i], label=name)
    ax.plot(DELTAS, accs.mean(axis=1), "k--", lw=1.5, alpha=0.6, label="mean")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlabel("per-frame symbol rate offset  ±delta")
    ax.set_ylabel("accuracy at SNR >= 10 dB")
    ax.set_title("effect: frozen model, only the test signal changed",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Testing the timing-regularity hypothesis without touching the model",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "12_timing_hypothesis.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "timing_hypothesis.npz", deltas=np.array(DELTAS),
             prominence=proms, accuracies=accs, classes=np.array(CLASSES),
             radioml_prominence=ref_prom)

    q = CLASSES.index("16QAM")
    print("\n" + "=" * 60)
    print(f"16QAM: {accs[0, q]:.3f} at delta=0  ->  {accs[:, q].max():.3f} "
          f"at delta={DELTAS[int(accs[:, q].argmax())]}")
    print(f"overall: {accs[0].mean():.3f}  ->  {accs.mean(1).max():.3f}")
    print(f"line prominence: {proms[0]:.1f} dB  ->  {proms[-1]:.1f} dB "
          f"(RadioML {ref_prom:.1f} dB)")
    print("=" * 60)
    print("\nwrote figures/12_timing_hypothesis.png")


if __name__ == "__main__":
    main()

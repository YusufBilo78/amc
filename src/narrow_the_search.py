"""
narrow_the_search.py -- two tests that split the remaining search space in half.

Channel impairments are ruled out (ablate_impairments.py: best recovery +0.045
against a 0.83 shortfall). So the difference is in how the signal itself is
made, not what happens to it afterwards.

Two things remain, and they are separable:

  A. SPECTRUM   our excess bandwidth is wrong. Evidence: sweep.py stopped at
     beta=0.15, which already doubled 16QAM accuracy (0.168 -> 0.370), and
     diagnose_16qam.py showed our spectral skirts sitting ~7 dB higher than
     RadioML's. The trend was pointing somewhere and the sweep stopped short.

  B. TIME/PHASE STRUCTURE   something about the waveform that survives having
     the correct spectrum.

Test 1 extends the beta sweep down to nearly zero roll-off.

Test 2 separates A from B directly: force our frames to have RadioML's average
magnitude spectrum, frame by frame, while keeping our own phase. If accuracy
recovers, the cause is entirely in A and B is irrelevant. If it does not, A is
a red herring no matter what the beta trend suggests.
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

import cnn
import model_zoo
import domains

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
MODEL_PATH = ROOT / "model_radioml_5class.pt"

CLASSES = list(domains.SHARED_CLASSES)
SNRS = list(range(10, 31, 2))
FRAMES = 200

BETAS = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.25, 0.35]


def per_class_accuracy(model, X, y) -> np.ndarray:
    pred = cnn.predict(model, X)
    return np.array([float((pred[y == i] == i).mean()) for i in range(len(CLASSES))])


def mean_magnitude_spectrum(frames_iq: np.ndarray) -> np.ndarray:
    """Average |FFT| across frames -- the target for spectral matching."""
    return np.abs(np.fft.fft(frames_iq, axis=1)).mean(axis=0)


def spectral_match(frames_iq: np.ndarray, target_mag: np.ndarray) -> np.ndarray:
    """
    Reshape each frame so the ensemble average magnitude spectrum equals
    `target_mag`, leaving the phase spectrum untouched.

    Phase is what carries the symbol sequence and the constellation geometry,
    so this isolates "wrong spectral envelope" from "wrong waveform structure".
    """
    spec = np.fft.fft(frames_iq, axis=1)
    current = np.abs(spec).mean(axis=0)
    gain = target_mag / (current + 1e-12)
    out = np.fft.ifft(spec * gain[None, :], axis=1)
    power = np.mean(np.abs(out) ** 2, axis=1, keepdims=True)
    return out / (np.sqrt(power) + 1e-12)


def to_model_input(frames_iq: np.ndarray) -> np.ndarray:
    return np.stack([frames_iq.real, frames_iq.imag], axis=1).astype(np.float32)


def main() -> None:
    model = model_zoo.backbone(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cnn.DEVICE))
    model = model.to(cnn.DEVICE)
    print("frozen model loaded\n")

    # ---------------------------------------------------- RadioML reference
    rml = domains.RadioMLDomain()
    ref = rml.load(CLASSES, SNRS, frames_per_cell=FRAMES, seed=5)
    rml.close()
    ref_iq = ref["X"][:, 0] + 1j * ref["X"][:, 1]
    ref_acc = per_class_accuracy(model, ref["X"], ref["y"])
    ref_targets = {name: mean_magnitude_spectrum(ref_iq[ref["y"] == i])
                   for i, name in enumerate(CLASSES)}
    print("RadioML in-domain  " + " ".join(f"{a:.3f}" for a in ref_acc)
          + f"   overall {ref_acc.mean():.3f}\n")

    # ================================================== TEST 1: beta -> 0
    print("TEST 1 -- roll-off sweep down to near-zero excess bandwidth")
    header = f"{'beta':>6}  " + " ".join(f"{c:>8}" for c in CLASSES) + f" {'overall':>8}"
    print(header)
    print("-" * len(header))

    beta_acc = []
    for beta in BETAS:
        dom = domains.SyntheticDomain(beta=beta, sps=8)
        data = dom.load(CLASSES, SNRS, frames_per_cell=FRAMES, seed=11)
        acc = per_class_accuracy(model, data["X"], data["y"])
        beta_acc.append(acc)
        print(f"{beta:>6.2f}  " + " ".join(f"{a:>8.3f}" for a in acc)
              + f" {acc.mean():>8.3f}")
    beta_acc = np.array(beta_acc)

    # ======================================= TEST 2: impose RadioML spectrum
    print("\nTEST 2 -- our waveform, RadioML's average magnitude spectrum")
    dom = domains.SyntheticDomain(beta=0.35, sps=8)
    data = dom.load(CLASSES, SNRS, frames_per_cell=FRAMES, seed=11)
    ours_iq = data["X"][:, 0] + 1j * data["X"][:, 1]

    matched = np.empty_like(ours_iq)
    for i, name in enumerate(CLASSES):
        mask = data["y"] == i
        matched[mask] = spectral_match(ours_iq[mask], ref_targets[name])

    before = per_class_accuracy(model, data["X"], data["y"])
    after = per_class_accuracy(model, to_model_input(matched), data["y"])

    print(f"{'':<22}" + " ".join(f"{c:>8}" for c in CLASSES) + f" {'overall':>8}")
    print("-" * (22 + 9 * (len(CLASSES) + 1)))
    print(f"{'ours, beta=0.35':<22}" + " ".join(f"{a:>8.3f}" for a in before)
          + f" {before.mean():>8.3f}")
    print(f"{'+ RadioML spectrum':<22}" + " ".join(f"{a:>8.3f}" for a in after)
          + f" {after.mean():>8.3f}")
    print(f"{'RadioML itself':<22}" + " ".join(f"{a:>8.3f}" for a in ref_acc)
          + f" {ref_acc.mean():>8.3f}")

    # ------------------------------------------------------------------ plot
    q = CLASSES.index("16QAM")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(CLASSES)))
    for i, name in enumerate(CLASSES):
        ax.plot(BETAS, beta_acc[:, i], "o-", lw=2, color=colors[i], label=name)
    ax.axhline(ref_acc[q], ls="--", color="gray", lw=1,
               label="RadioML 16QAM in-domain")
    ax.set_xscale("log")
    ax.set_xlabel("RRC roll-off  beta")
    ax.set_ylabel("accuracy at SNR >= 10 dB")
    ax.set_title("Test 1: does near-zero excess bandwidth help?",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="center left")

    ax = axes[1]
    width = 0.27
    idx = np.arange(len(CLASSES))
    ax.bar(idx - width, before, width, label="ours (beta=0.35)", color="tab:red")
    ax.bar(idx, after, width, label="+ RadioML spectrum", color="tab:orange")
    ax.bar(idx + width, ref_acc, width, label="RadioML itself", color="tab:blue")
    ax.set_xticks(idx)
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test 2: impose RadioML's magnitude spectrum",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    for label, frames, color in [
        ("RadioML 16QAM", ref_iq[ref["y"] == q], "tab:blue"),
        ("ours beta=0.35", ours_iq[data["y"] == q], "tab:red"),
        ("ours + spectrum", matched[data["y"] == q], "tab:orange"),
    ]:
        psd = np.mean(np.abs(np.fft.fftshift(
            np.fft.fft(frames, axis=1), axes=1)) ** 2, axis=0)
        db = 10 * np.log10(psd + 1e-12)
        ax.plot(np.fft.fftshift(np.fft.fftfreq(frames.shape[1])), db - db.max(),
                lw=1.4, color=color, label=label)
    ax.set_xlabel("normalized frequency")
    ax.set_ylabel("dB")
    ax.set_ylim(-55, 3)
    ax.set_title("16QAM spectra, for reference", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Narrowing the search: spectrum or waveform structure?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "14_narrow_the_search.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "narrow_search.npz", betas=np.array(BETAS),
             beta_acc=beta_acc, before=before, after=after, radioml=ref_acc,
             classes=np.array(CLASSES))

    print("\n" + "=" * 70)
    best_b = int(beta_acc[:, q].argmax())
    print(f"Test 1 best 16QAM: {beta_acc[best_b, q]:.3f} at beta={BETAS[best_b]}"
          f"   (was 0.168 at beta=0.35)")
    print(f"Test 2 16QAM: {before[q]:.3f} -> {after[q]:.3f} after spectral match")
    print(f"RadioML reference: {ref_acc[q]:.3f}")
    print("=" * 70)
    print("\nwrote figures/14_narrow_the_search.png")


if __name__ == "__main__":
    main()

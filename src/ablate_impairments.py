"""
ablate_impairments.py -- stop guessing, start eliminating.

Three hypotheses about the 16QAM failure have now been killed by measurement:
pulse-shaping mismatch (sweep.py), apparent constellation density
(diagnose_16qam.py), and symbol-timing regularity
(test_timing_hypothesis.py). Rather than invent a fourth, this enumerates
every impairment RadioML documents but our generator lacks, and adds them one
at a time.

The model stays frozen throughout. Only the test signal changes, so any
recovery is attributable to the impairment that was switched on and nothing
else.

    documented by DeepSig     ours          status
    ----------------------------------------------------------------
    symbol rate offset        added         ruled out already
    carrier frequency offset  available     test
    delay spread              added now     test
    thermal noise             have          n/a
    (absolute phase)          missing       test -- not in their list,
                                            but our generator is
                                            deterministic in phase and a
                                            real receiver never is

If none of these recovers 16QAM, the cause is outside this list and the next
step is to look at what the network itself is keying on.
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
GEN_LEN = 1536

# name -> impairment settings. Deliberately one knob at a time, then combined.
CONFIGS: list[tuple[str, dict]] = [
    ("none (baseline)",        {}),
    ("random phase",           {"phase": True}),
    ("CFO 1e-4",               {"cfo": 1e-4}),
    ("CFO 1e-3",               {"cfo": 1e-3}),
    ("CFO 1e-2",               {"cfo": 1e-2}),
    ("multipath ds=0.5",       {"multipath": 0.5}),
    ("multipath ds=2",         {"multipath": 2.0}),
    ("multipath ds=4",         {"multipath": 4.0}),
    ("rate offset 5% (ctrl)",  {"rate": 0.05}),
    ("phase+CFO+multipath",    {"phase": True, "cfo": 1e-3, "multipath": 2.0}),
]


def make_frames(scheme: str, n: int, snr_db: float, cfg: dict,
                rng: np.random.Generator) -> np.ndarray:
    """
    One cell of frames with the requested impairments.

    Order matters and follows the physical chain: the transmitter shapes the
    signal, the channel smears it, the receiver adds frequency and phase error.
    Noise is added inside modem.generate, so multipath here acts on an already
    noisy signal -- a simplification worth stating, though at SNR >= 10 dB the
    effect on the conclusion is small.
    """
    out = np.empty((n, FRAME_LEN), dtype=np.complex128)
    for i in range(n):
        x = modem.generate(scheme, GEN_LEN, snr_db=snr_db, sps=8, rng=rng)

        if cfg.get("multipath"):
            x = modem.multipath(x, rng, delay_spread=cfg["multipath"])
        if cfg.get("rate"):
            factor = rng.uniform(1 - cfg["rate"], 1 + cfg["rate"])
            x = resample(x, int(round(GEN_LEN * factor)))
        if cfg.get("cfo"):
            x = modem.carrier_offset(x, rng.uniform(-cfg["cfo"], cfg["cfo"]))
        if cfg.get("phase"):
            x = modem.random_phase(x, rng)

        start = (len(x) - FRAME_LEN) // 2
        x = x[start : start + FRAME_LEN]
        out[i] = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    return out


def evaluate(model, cfg: dict, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, name in enumerate(CLASSES):
        for snr in SNRS:
            frames = make_frames(name, FRAMES, float(snr), cfg, rng)
            X.append(np.stack([frames.real, frames.imag], axis=1).astype(np.float32))
            y.append(np.full(FRAMES, i, dtype=np.int64))
    X = np.concatenate(X)
    y = np.concatenate(y)
    pred = cnn.predict(model, X)
    return np.array([float((pred[y == i] == i).mean()) for i in range(len(CLASSES))])


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH.name} not found -- run sweep.py first.")

    model = cnn.IQNet(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cnn.DEVICE))
    model = model.to(cnn.DEVICE)
    print("frozen model loaded; nothing below retrains it\n")

    # In-domain reference: what the model achieves on RadioML's own frames.
    rml = domains.RadioMLDomain()
    ref = rml.load(CLASSES, SNRS, frames_per_cell=200, seed=5)
    rml.close()
    ref_pred = cnn.predict(model, ref["X"])
    ref_acc = np.array([float((ref_pred[ref["y"] == i] == i).mean())
                        for i in range(len(CLASSES))])
    print("RadioML (in-domain) " + " ".join(f"{a:.3f}" for a in ref_acc)
          + f"   overall {ref_acc.mean():.3f}\n")

    header = (f"{'impairment':<24} " + " ".join(f"{c:>8}" for c in CLASSES)
              + f" {'overall':>8}")
    print(header)
    print("-" * len(header))

    names, results = [], []
    for name, cfg in CONFIGS:
        acc = evaluate(model, cfg)
        names.append(name)
        results.append(acc)
        print(f"{name:<24} " + " ".join(f"{a:>8.3f}" for a in acc)
              + f" {acc.mean():>8.3f}")

    results = np.array(results)

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                             gridspec_kw={"width_ratios": [2.2, 1]})

    ax = axes[0]
    im = ax.imshow(results, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for r in range(results.shape[0]):
        for c in range(results.shape[1]):
            ax.text(c, r, f"{results[r, c]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="black" if 0.25 < results[r, c] < 0.85 else "white")
    ax.set_title("Frozen model, one impairment added at a time\n"
                 "accuracy at SNR >= 10 dB", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03)

    q = CLASSES.index("16QAM")
    ax = axes[1]
    ypos = np.arange(len(names))
    ax.barh(ypos, results[:, q], color="tab:red", alpha=0.8)
    ax.axvline(ref_acc[q], ls="--", color="tab:blue", lw=2,
               label=f"RadioML in-domain ({ref_acc[q]:.2f})")
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("16QAM accuracy")
    ax.set_title("16QAM only", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    axes[0].invert_yaxis()
    fig.suptitle("Impairment ablation: which missing effect explains the 16QAM failure?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "13_impairment_ablation.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "impairment_ablation.npz", names=np.array(names),
             results=results, classes=np.array(CLASSES), radioml=ref_acc)

    base = results[0, q]
    best_i = int(results[:, q].argmax())
    print("\n" + "=" * 70)
    print(f"16QAM baseline (no impairment): {base:.3f}")
    print(f"best:  {names[best_i]}  ->  {results[best_i, q]:.3f} "
          f"({results[best_i, q] - base:+.3f})")
    print(f"RadioML in-domain reference:    {ref_acc[q]:.3f}")
    if results[best_i, q] - base < 0.15:
        print("\nNo impairment recovers 16QAM. The cause is not on this list.")
    print("=" * 70)
    print("\nwrote figures/13_impairment_ablation.png")


if __name__ == "__main__":
    main()

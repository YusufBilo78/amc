"""
sweep.py -- does the diagnosis hold? Dose-response over transmitter parameters.

crossdomain.py found a 16.7-point gap whose entire mass sits in one cell:
16QAM read as 64QAM, 75% of the time. The proposed mechanism was pulse-shaping
and sampling mismatch making constellations *look* denser than they are.

That is a hypothesis, not a finding. This tests it the way it should be tested:
train once on RadioML, then evaluate the frozen model against a family of
synthetic domains that differ only in one transmitter parameter at a time.

If the mechanism is real, 16QAM accuracy should vary monotonically with excess
bandwidth (beta) and with oversampling (sps), while BPSK/QPSK stay flat. If
16QAM is flat too, the diagnosis was wrong and the fix would have been guesswork.

Training happens once and is cached; the sweep itself is seconds per point.

    python sweep.py            # reuse cached model if present
    python sweep.py --retrain
"""

from __future__ import annotations

import argparse
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
MODEL_PATH = ROOT / "model_radioml_5class.pt"

CLASSES = list(domains.SHARED_CLASSES)          # BPSK QPSK 8PSK 16QAM 64QAM
TRAIN_SNRS = list(range(-20, 31, 2))
EVAL_SNRS = list(range(10, 31, 2))              # the regime where the gap lives

BETAS = [0.15, 0.25, 0.35, 0.50, 0.70, 0.90]
SPS_VALUES = [4, 6, 8, 10, 12, 16]


def get_model(retrain: bool) -> torch.nn.Module:
    model = model_zoo.backbone(len(CLASSES))
    if MODEL_PATH.exists() and not retrain:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=cnn.DEVICE))
        print(f"loaded cached model from {MODEL_PATH.name}\n")
        return model.to(cnn.DEVICE)

    print("training on RadioML (5 shared classes) ...")
    src = domains.RadioMLDomain()
    a = src.load(CLASSES, TRAIN_SNRS, frames_per_cell=768, seed=0)
    src.close()
    tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3, seed=0)
    model = cnn.train_model(model, a["X"][tr], a["y"][tr],
                            a["X"][te], a["y"][te], epochs=25)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH.name}\n")
    return model


def evaluate(model, domain: domains.SyntheticDomain) -> np.ndarray:
    """Per-class accuracy at high SNR. Returns array of len(CLASSES)."""
    data = domain.load(CLASSES, EVAL_SNRS, frames_per_cell=200, seed=7)
    pred = cnn.predict(model, data["X"])
    return np.array([
        float((pred[data["y"] == i] == i).mean()) for i in range(len(CLASSES))
    ])


def run_sweep(model, values, build, label: str) -> np.ndarray:
    """Returns (len(values), len(CLASSES)) accuracy matrix."""
    out = []
    print(f"\nsweeping {label}")
    print(f"  {label:>6}  " + "  ".join(f"{c:>7}" for c in CLASSES) + "   overall")
    for v in values:
        t0 = time.time()
        acc = evaluate(model, build(v))
        out.append(acc)
        print(f"  {v:>6}  " + "  ".join(f"{a:>7.3f}" for a in acc)
              + f"   {acc.mean():.3f}   ({time.time()-t0:.0f}s)")
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true")
    args = p.parse_args()

    model = get_model(args.retrain)

    beta_acc = run_sweep(
        model, BETAS,
        lambda b: domains.SyntheticDomain(beta=b, sps=8),
        "beta",
    )
    sps_acc = run_sweep(
        model, SPS_VALUES,
        lambda s: domains.SyntheticDomain(beta=0.35, sps=s),
        "sps",
    )

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(CLASSES)))

    for ax, values, acc, xlabel, fixed in [
        (axes[0], BETAS, beta_acc, "RRC roll-off  beta  (excess bandwidth)", "sps = 8"),
        (axes[1], SPS_VALUES, sps_acc, "samples per symbol  sps", "beta = 0.35"),
    ]:
        for i, name in enumerate(CLASSES):
            ax.plot(values, acc[:, i], "o-", lw=2, color=colors[i], label=name)
        ax.plot(values, acc.mean(axis=1), "k--", lw=1.5, alpha=0.6, label="mean")
        ax.set_xlabel(xlabel)
        ax.set_title(f"({fixed})", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("accuracy at SNR >= 10 dB")
    axes[1].legend(fontsize=9, loc="lower left")
    fig.suptitle(
        "Frozen RadioML model, evaluated against synthetic domains that differ "
        "in one transmitter parameter\n"
        "flat lines = the model is invariant to that parameter; sloped lines = "
        "it is not",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURES / "10_parameter_sweep.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "sweep_results.npz", betas=np.array(BETAS),
             beta_acc=beta_acc, sps_values=np.array(SPS_VALUES),
             sps_acc=sps_acc, classes=np.array(CLASSES))

    # -------------------------------------------------------------- verdict
    # np.ptp(arr), not arr.ptp() -- the method was removed in NumPy 2.0.
    print("\n" + "=" * 68)
    print(f"{'class':<8} {'beta: min..max':>20} {'range':>7} "
          f"{'sps: min..max':>20} {'range':>7}")
    for i, name in enumerate(CLASSES):
        b, s = beta_acc[:, i], sps_acc[:, i]
        print(f"{name:<8} {b.min():>9.3f} ..{b.max():>7.3f} {np.ptp(b):>7.3f} "
              f"{s.min():>9.3f} ..{s.max():>7.3f} {np.ptp(s):>7.3f}")

    stable = [c for i, c in enumerate(CLASSES)
              if max(np.ptp(beta_acc[:, i]), np.ptp(sps_acc[:, i])) < 0.1]
    print(f"\ninvariant to both parameters (<0.10 range): "
          f"{', '.join(stable) if stable else 'none'}")
    print(f"best overall: beta={BETAS[beta_acc.mean(1).argmax()]} "
          f"({beta_acc.mean(1).max():.3f}), "
          f"sps={SPS_VALUES[sps_acc.mean(1).argmax()]} "
          f"({sps_acc.mean(1).max():.3f})")
    print("=" * 68)
    print("\nwrote figures/10_parameter_sweep.png")


if __name__ == "__main__":
    main()

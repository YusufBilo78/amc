"""
match_radioml_params.py -- generate the way the paper says they generated.

O'Shea, Roy & Clancy (arXiv:1712.04578), Table I, "Random Variable
Initialization", drawn independently per example:

    alpha (RRC roll-off)      U(0.1, 0.4)
    delta_t (timing)          U(0, 16)
    delta_fs (symbol rate)    N(0, sigma_clk)
    theta_c (carrier phase)   U(0, 2pi)
    delta_fc (carrier freq)   N(0, sigma_clk)
    H (multipath)             Rayleigh, tau in [0, 0.5, 1.0, 2.0]

Two things follow immediately.

1. Our generator used a FIXED beta = 0.35, which is inside their range. So
   "we picked the wrong roll-off" cannot be the explanation. What we lacked was
   roll-off *diversity*, and -- more importantly -- the right symbol rate.

2. Occupied bandwidth goes as (1 + alpha) / sps. Ours at beta=0.35, sps=8 is
   0.169. Measured RadioML occupancy is about 0.12, which at their mean alpha
   of 0.25 implies sps ~ 10.4, not 8. And beta=0.01 at sps=8 gives 0.126 --
   which is why sweeping beta to an unphysical 0.01 helped: it was compensating
   for the wrong symbol rate by shrinking excess bandwidth.

Prediction, recorded before running: matching their stated parameters should
close most of the gap with no whitening at all. If it does not, the
bandwidth account is incomplete.

The model is the frozen 5-class RadioML network used everywhere else, so only
the generator changes.
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
import modem

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
MODEL_PATH = ROOT / "model_radioml_5class.pt"

CLASSES = list(domains.SHARED_CLASSES)
SNRS = list(range(10, 31, 2))
FRAMES = 200
FRAME_LEN = 1024
GEN_LEN = 1600

PAPER_ALPHA = (0.1, 0.4)      # Table I
PAPER_TIMING = 16             # Table I, delta_t ~ U(0, 16) samples

CONFIGS: list[tuple[str, dict]] = [
    ("ours, as built (beta=0.35, sps=8)",
     dict(beta=0.35, sps=8)),
    ("bandwidth-matched by beta (beta=0.01, sps=8)",
     dict(beta=0.01, sps=8)),
    ("paper roll-off only (a~U(.1,.4), sps=8)",
     dict(beta_range=PAPER_ALPHA, sps=8)),
    ("paper symbol rate only (beta=0.35, sps=10)",
     dict(beta=0.35, sps=10)),
    ("PAPER-MATCHED (a~U(.1,.4), sps=10)",
     dict(beta_range=PAPER_ALPHA, sps=10)),
    ("PAPER-MATCHED + timing + CFO + multipath",
     dict(beta_range=PAPER_ALPHA, sps=10, timing=PAPER_TIMING,
          cfo=1e-3, multipath=1.0)),
]


def occupied_bw(cfg: dict) -> float:
    """(1 + alpha) / sps, using the mean alpha when it is a range."""
    a = (np.mean(cfg["beta_range"]) if "beta_range" in cfg else cfg["beta"])
    return (1.0 + a) / cfg["sps"]


def make_cell(scheme: str, snr: float, cfg: dict, rng) -> np.ndarray:
    """FRAMES frames, each with its own independently drawn parameters."""
    out = np.empty((FRAMES, FRAME_LEN), dtype=np.complex128)
    for i in range(FRAMES):
        beta = (rng.uniform(*cfg["beta_range"]) if "beta_range" in cfg
                else cfg["beta"])
        x = modem.generate(scheme, GEN_LEN, snr_db=snr, sps=cfg["sps"],
                           rng=rng, beta=beta)
        if cfg.get("multipath"):
            x = modem.multipath(x, rng, delay_spread=cfg["multipath"])
        if cfg.get("cfo"):
            x = modem.carrier_offset(x, rng.uniform(-cfg["cfo"], cfg["cfo"]))
        if cfg.get("timing"):
            x = np.roll(x, int(rng.integers(0, cfg["timing"] + 1)))
        x = modem.random_phase(x, rng)  # Table I: theta_c ~ U(0, 2pi)

        start = (len(x) - FRAME_LEN) // 2
        x = x[start:start + FRAME_LEN]
        out[i] = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    return out


def evaluate(model, cfg: dict, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, name in enumerate(CLASSES):
        for snr in SNRS:
            f = make_cell(name, float(snr), cfg, rng)
            X.append(np.stack([f.real, f.imag], axis=1).astype(np.float32))
            y.append(np.full(FRAMES, i, dtype=np.int64))
    X, y = np.concatenate(X), np.concatenate(y)
    pred = cnn.predict(model, X)
    return np.array([float((pred[y == i] == i).mean()) for i in range(len(CLASSES))])


def main() -> None:
    model = model_zoo.backbone(len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cnn.DEVICE))
    model = model.to(cnn.DEVICE)
    print("frozen 5-class RadioML model; only the generator changes\n")

    # h5py is currently blocked by a Windows Application Control policy on this
    # machine, so RadioML cannot be re-read. The in-domain ceiling for this exact
    # frozen model was measured earlier (narrow_the_search.py) and is reused as
    # the reference line. Nothing here depends on re-reading the dataset --
    # only the generator changes.
    try:
        rml = domains.RadioMLDomain()
        ref = rml.load(CLASSES, SNRS, frames_per_cell=FRAMES, seed=5)
        rml.close()
        ref_pred = cnn.predict(model, ref["X"])
        ref_acc = np.array([float((ref_pred[ref["y"] == i] == i).mean())
                            for i in range(len(CLASSES))])
        src = "measured now"
    except Exception as exc:
        ref_acc = np.array([1.000, 1.000, 0.999, 0.993, 0.978])
        src = f"cached from earlier run; {type(exc).__name__}"
    print(f"RadioML itself (in-domain ceiling, {src})  "
          + " ".join(f"{a:.3f}" for a in ref_acc)
          + f"   overall {ref_acc.mean():.3f}\n")

    header = (f"{'generator configuration':<44} {'BW':>6}  "
              + " ".join(f"{c:>7}" for c in CLASSES) + f" {'overall':>8}")
    print(header)
    print("-" * len(header))

    names, rows, bws = [], [], []
    t0 = time.time()
    for name, cfg in CONFIGS:
        acc = evaluate(model, cfg)
        bw = occupied_bw(cfg)
        names.append(name)
        rows.append(acc)
        bws.append(bw)
        print(f"{name:<44} {bw:>6.3f}  "
              + " ".join(f"{a:>7.3f}" for a in acc) + f" {acc.mean():>8.3f}")
    rows = np.array(rows)
    print(f"\ngenerated and evaluated in {time.time()-t0:.0f}s")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    y = np.arange(len(names))
    colors = ["#c0392b", "#e67e22", "#e67e22", "#e67e22", "#27ae60", "#1e8449"]
    ax.barh(y, rows.mean(axis=1), 0.62, color=colors)
    ax.axvline(ref_acc.mean(), ls="--", lw=2, color="#2c3e50",
               label=f"RadioML in-domain ({ref_acc.mean():.3f})")
    for i, v in enumerate(rows.mean(axis=1)):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("cross-domain accuracy at SNR >= 10 dB")
    ax.set_title("Matching the paper's stated generation parameters",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    ax.scatter(bws, rows.mean(axis=1), s=90, c=colors, zorder=3)
    for b, v, n in zip(bws, rows.mean(axis=1), names):
        ax.annotate(n.split("(")[0].strip(), (b, v), fontsize=7,
                    xytext=(0, 9), textcoords="offset points", ha="center")
    ax.axvline(0.12, ls=":", color="gray", lw=1.5)
    ax.text(0.122, 0.35, "RadioML measured\noccupancy ~0.12", fontsize=8,
            color="gray")
    ax.set_xlabel("occupied bandwidth  (1 + alpha) / sps")
    ax.set_ylabel("cross-domain accuracy")
    ax.set_title("Accuracy against occupied bandwidth", fontsize=12,
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Does generating the way the paper describes close the gap?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "21_paper_matched_generation.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "paper_matched.npz", names=np.array(names), acc=rows,
             bandwidths=np.array(bws), radioml=ref_acc,
             classes=np.array(CLASSES))

    base, matched = rows[0].mean(), rows[4].mean()
    full = rows[5].mean()
    q = CLASSES.index("16QAM")
    print("\n" + "=" * 74)
    print(f"as built            {base:.3f}   (16QAM {rows[0][q]:.3f})")
    print(f"paper-matched       {matched:.3f}   (16QAM {rows[4][q]:.3f})   "
          f"{matched - base:+.3f}")
    print(f"+ their impairments {full:.3f}   (16QAM {rows[5][q]:.3f})   "
          f"{full - base:+.3f}")
    print(f"RadioML ceiling     {ref_acc.mean():.3f}")
    print(f"\ngap closed by parameter matching alone: "
          f"{(matched - base) / (ref_acc.mean() - base):.0%}")
    print("=" * 74)
    print("\nwrote figures/21_paper_matched_generation.png")


if __name__ == "__main__":
    main()

"""
compare_methods.py -- spectral whitening against what the field already does.

"Whitening beats doing nothing" is not a result anybody has to take seriously.
The comparison that matters is against the augmentation set the AMC literature
already uses: rotation, conjugate flip, and additive Gaussian noise
(arXiv:1912.03026).

Four configurations, identical data, identical schedule, several seeds:

    none                    the number papers report
    standard augmentation   the literature baseline
    whitening (alpha=0.75)  this project's proposal
    whitening + standard    do they compose, or overlap?

The last row is the one worth thinking about in advance. Rotation and flip
teach invariance to *phase and mirror symmetry*; whitening removes the
*spectral envelope*. Those are different nuisance parameters, so if the
reasoning is right the two should add rather than substitute. If combining them
gains nothing over whitening alone, that would say the literature transforms
were never addressing this failure mode at all -- also a finding.
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

import augment
import cnn
import domains
from whitening_sweep import accuracy_by_snr, spectral_whiten, to_model_input

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

CLASSES = list(domains.SHARED_CLASSES)
SNRS = list(range(-20, 31, 2))
SEEDS = [0, 1, 2, 3]
EPOCHS = 15
BEST_ALPHA = 0.75

# name -> (whitening alpha, augmenter factory)
METHODS: list[tuple[str, float, object]] = [
    ("none",                     0.0,        None),
    ("standard augmentation",    0.0,        augment.StandardAMCAugmenter),
    ("whitening a=0.75",         BEST_ALPHA, None),
    ("whitening + standard",     BEST_ALPHA, augment.StandardAMCAugmenter),
]


def build_model(arch: str, n_classes: int):
    """
    Backbone selection.

    IQNet stays the default so every number already written up reproduces
    untouched. ICRNNA is the alternative measured in overfit_2x2.py: on this
    same data IQNet reaches train 1.000 / test 0.675 by epoch 30 -- it
    memorises the training set from about epoch 20 -- while ICRNNA holds a
    train-test gap of +0.009 and scores higher on test with fewer parameters.
    That makes IQNet a poor instrument for measurements whose whole point is a
    difference between two accuracies, which is what this file computes.

    ICRNNA here is transcribed from a peer's reproduction and is **not** the
    published architecture: comparison against the paper (El-Haryqy et al.,
    Results in Engineering 26, 2025) found six differences, five of them real
    -- conv1 kernel 5 vs 3, two max-pools vs one, one BatchNorm after the LSTM
    stack vs one after each layer, no attention dropout or LayerNorm, and one
    dense layer of 128 vs two of 128 and 64. Any write-up has to say so.
    """
    if arch == "IQNet":
        return cnn.IQNet(n_classes)
    import model_zoo
    return model_zoo.ICRNNA(n_classes)


def main() -> None:
    global SEEDS, EPOCHS
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=("IQNet", "ICRNNA"), default="IQNet")
    p.add_argument("--seeds", type=int, default=len(SEEDS))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=None,
                   help="switch to the 70/15/15 split with early stopping and "
                        "best-validation checkpointing; without it the "
                        "original 70/30 fixed-length recipe is used")
    args = p.parse_args()
    SEEDS = list(range(args.seeds))
    EPOCHS = args.epochs

    # The default configuration keeps the original output filenames so the
    # numbers in the README are not silently overwritten by a different setup.
    tag = ("" if args.arch == "IQNet" and args.patience is None
           else f"_{args.arch}" + ("_es" if args.patience else ""))
    npz_path = ROOT / f"compare_methods{tag}.npz"
    fig_path = FIGURES / f"17_method_comparison{tag}.png"

    print(f"{len(METHODS)} methods x {len(SEEDS)} seeds = "
          f"{len(METHODS) * len(SEEDS)} models")
    print(f"backbone {args.arch}, max {EPOCHS} epochs, "
          f"protocol {'70/15/15 + early stopping' if args.patience else '70/30 fixed'}")
    print(f"-> {npz_path.name}\n")

    src = domains.RadioMLDomain()
    a = src.load(CLASSES, SNRS, frames_per_cell=768, seed=0)
    src.close()
    dst = domains.SyntheticDomain()
    b = dst.load(CLASSES, SNRS, frames_per_cell=400, seed=1)

    a_iq = (a["X"][:, 0] + 1j * a["X"][:, 1]).astype(np.complex64)
    b_iq = (b["X"][:, 0] + 1j * b["X"][:, 1]).astype(np.complex64)

    # Whitening depends only on alpha, so cache the two variants.
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in {m[1] for m in METHODS}:
        cache[alpha] = (to_model_input(spectral_whiten(a_iq, alpha)),
                        to_model_input(spectral_whiten(b_iq, alpha)))

    high = np.array(SNRS) >= 10
    q = CLASSES.index("16QAM")

    in_domain = np.full((len(METHODS), len(SEEDS)), np.nan)
    cross = np.full((len(METHODS), len(SEEDS)), np.nan)
    qam16 = np.full((len(METHODS), len(SEEDS)), np.nan)
    curves = np.full((len(METHODS), len(SEEDS), len(SNRS)), np.nan)

    # Resume. Every cell is written to disk as soon as it finishes, so a run
    # cut short by a flat battery or a reboot picks up where it stopped rather
    # than repeating hours of training. A cell counts as done when its
    # in-domain entry is no longer NaN; the shape check guards against
    # resuming into a file written by a differently-shaped configuration.
    if npz_path.exists():
        old = np.load(npz_path, allow_pickle=True)
        if old["in_domain"].shape == in_domain.shape:
            in_domain, cross = old["in_domain"], old["cross"]
            qam16, curves = old["qam16"], old["curves"]
            done = int(np.count_nonzero(~np.isnan(in_domain)))
            if done:
                print(f"resuming: {done}/{in_domain.size} models already done\n")
        else:
            print(f"{npz_path.name} has a different shape; starting over\n")

    header = (f"{'method':<24} {'seed':>5} {'in-dom':>8} {'cross':>8} "
              f"{'gap':>8} {'16QAM':>8} {'time':>7}")
    print(header)
    print("-" * len(header))

    t_start = time.time()
    for i, (name, alpha, aug_cls) in enumerate(METHODS):
        Xa, Xb = cache[alpha]
        for j, seed in enumerate(SEEDS):
            if not np.isnan(in_domain[i, j]):
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)

            # With early stopping the monitored set must be a validation set:
            # the test set is what gets reported, and selecting the stopping
            # epoch on it would be selecting on the number being reported.
            if args.patience:
                tr, va, te = cnn.split(a["X"], a["y"], a["z"],
                                       test_fraction=0.15, val_fraction=0.15,
                                       seed=seed)
                mon_X, mon_y = Xa[va], a["y"][va]
            else:
                tr, te = cnn.split(a["X"], a["y"], a["z"],
                                   test_fraction=0.3, seed=seed)
                mon_X, mon_y = Xa[te], a["y"][te]

            model = build_model(args.arch, len(CLASSES))
            model = cnn.train_model(model, Xa[tr], a["y"][tr], mon_X, mon_y,
                                    epochs=EPOCHS, patience=args.patience,
                                    grad_clip=5.0 if args.patience else None,
                                    augment=aug_cls() if aug_cls else None)

            pred_in = cnn.predict(model, Xa[te])
            pred_cross = cnn.predict(model, Xb)
            acc_in = accuracy_by_snr(pred_in, a["y"][te], a["z"][te], SNRS)
            acc_cross = accuracy_by_snr(pred_cross, b["y"], b["z"], SNRS)

            mask = (b["z"] >= 10) & (b["y"] == q)
            in_domain[i, j] = float(np.nanmean(acc_in[high]))
            cross[i, j] = float(np.nanmean(acc_cross[high]))
            qam16[i, j] = float((pred_cross[mask] == q).mean())
            curves[i, j] = acc_cross

            print(f"{name:<24} {seed:>5} {in_domain[i,j]:>8.3f} "
                  f"{cross[i,j]:>8.3f} {in_domain[i,j]-cross[i,j]:>+8.3f} "
                  f"{qam16[i,j]:>8.3f} {time.time()-t0:>6.0f}s")

            np.savez(npz_path,
                     methods=np.array([m[0] for m in METHODS]),
                     seeds=np.array(SEEDS), in_domain=in_domain,
                     cross=cross, qam16=qam16, curves=curves,
                     snrs=np.array(SNRS))
        print()

    print(f"total {(time.time()-t_start)/60:.1f} min\n")

    gap = in_domain - cross
    print(f"{'method':<24} {'in-domain':>15} {'cross-domain':>15} "
          f"{'gap':>15} {'16QAM':>15}")
    print("-" * 87)
    for i, (name, _, _) in enumerate(METHODS):
        print(f"{name:<24} "
              f"{in_domain[i].mean():>8.3f} ±{in_domain[i].std():<5.3f} "
              f"{cross[i].mean():>8.3f} ±{cross[i].std():<5.3f} "
              f"{gap[i].mean():>+8.3f} ±{gap[i].std():<5.3f} "
              f"{qam16[i].mean():>8.3f} ±{qam16[i].std():<5.3f}")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    names = [m[0] for m in METHODS]
    colors = ["tab:gray", "tab:orange", "tab:blue", "tab:green"]
    ypos = np.arange(len(METHODS))

    ax = axes[0]
    ax.barh(ypos - 0.19, in_domain.mean(1), 0.36, xerr=in_domain.std(1),
            color="lightsteelblue", label="in-domain", capsize=3)
    ax.barh(ypos + 0.19, cross.mean(1), 0.36, xerr=cross.std(1),
            color="tab:red", label="cross-domain", capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy at SNR >= 10 dB")
    ax.set_title("In-domain vs cross-domain", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    ax.barh(ypos, gap.mean(1), 0.55, xerr=gap.std(1), color="tab:purple",
            capsize=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("generalization gap")
    ax.set_title("Gap (lower is better)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    for i in range(len(METHODS)):
        ax.text(gap[i].mean() + 0.008, i, f"{gap[i].mean():.3f}",
                va="center", fontsize=9)

    ax = axes[2]
    for i, (name, color) in enumerate(zip(names, colors)):
        mean = np.nanmean(curves[i], axis=0)
        std = np.nanstd(curves[i], axis=0)
        ax.plot(SNRS, mean, "-", lw=2, color=color, label=name)
        ax.fill_between(SNRS, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("cross-domain accuracy")
    ax.set_title("Cross-domain accuracy vs SNR", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"Spectral whitening vs the standard AMC augmentation set "
        f"({args.arch}, {len(SEEDS)} seeds, bars = 1 s.d.)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------- verdict
    def pooled(x, y):
        return np.sqrt((x.std() ** 2 + y.std() ** 2) / 2) + 1e-9

    print("\n" + "=" * 72)
    for i in (1, 2, 3):
        d = cross[i].mean() - cross[0].mean()
        print(f"{names[i]:<24} vs none: {d:+.3f}  "
              f"({d / pooled(cross[i], cross[0]):+.1f} s.d.)")
    d = cross[2].mean() - cross[1].mean()
    print(f"\nwhitening vs standard augmentation: {d:+.3f}  "
          f"({d / pooled(cross[2], cross[1]):+.1f} s.d.)")
    d = cross[3].mean() - cross[2].mean()
    print(f"combining adds over whitening alone: {d:+.3f}  "
          f"({d / pooled(cross[3], cross[2]):+.1f} s.d.)")
    print("=" * 72)
    print(f"\nwrote figures/{fig_path.name}")


if __name__ == "__main__":
    main()

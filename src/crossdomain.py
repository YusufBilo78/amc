"""
crossdomain.py -- the central experiment: train on one domain, test on another.

    python crossdomain.py --train radioml --test synthetic

Trains on domain A, then evaluates on
  (a) held-out frames from A  -- the number papers report
  (b) every frame from B      -- the number that says whether it transfers

The difference between those two curves is the generalization gap, and closing
it is what the project is about. Everything is deliberately matched where it
can be (1024-sample frames, unit average power, identical SNR grid, identical
class set) so that whatever gap appears is attributable to the transmitter and
channel, not to bookkeeping.

Adding a real SDR capture later needs no change here: `--test capture` already
works the moment files exist in the layout CaptureDomain documents.
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
from sklearn.metrics import confusion_matrix

import cnn
import domains

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def make_domain(spec: str) -> domains.Domain:
    if spec == "radioml":
        return domains.RadioMLDomain()
    if spec == "synthetic":
        return domains.SyntheticDomain()
    if spec == "capture":
        return domains.CaptureDomain()
    raise ValueError(f"unknown domain: {spec}")


def accuracy_by_snr(pred, y, z, snrs):
    return np.array([
        float((pred[z == s] == y[z == s]).mean()) if (z == s).any() else np.nan
        for s in snrs
    ])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="radioml")
    p.add_argument("--test", default="synthetic")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--train-frames", type=int, default=768)
    p.add_argument("--test-frames", type=int, default=400)
    p.add_argument("--include-aliases", action="store_true",
                   help="widen from 5 exact matches to 10 loose ones "
                        "(declare the mismatch in the report)")
    p.add_argument("--augment", action="store_true",
                   help="train with domain augmentation (the 'B' model)")
    p.add_argument("--ablate", choices=("none", "spectral", "resample"),
                   default="none",
                   help="switch one transform off, to attribute the improvement")
    args = p.parse_args()

    src = make_domain(args.train)
    dst = make_domain(args.test)

    classes = domains.shared_classes(src, dst, include_aliases=args.include_aliases)
    snrs = list(range(-20, 31, 2))
    print(f"train domain: {src.name}")
    print(f"test  domain: {dst.name}")
    print(f"shared classes ({len(classes)}): {', '.join(classes)}")
    print(f"SNR grid: {snrs[0]} .. {snrs[-1]} dB, {len(snrs)} levels\n")

    t0 = time.time()
    print(f"loading {src.name} ...")
    a = src.load(classes, snrs, args.train_frames, seed=0)
    print(f"  {len(a['X'])} frames ({time.time() - t0:.1f}s)")

    print(f"loading {dst.name} ...")
    b = dst.load(classes, snrs, args.test_frames, seed=1)
    print(f"  {len(b['X'])} frames ({time.time() - t0:.1f}s)\n")

    tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3, seed=0)
    print(f"{len(tr)} train / {len(te)} in-domain test / {len(b['X'])} cross-domain test\n")

    augmenter = None
    if args.augment:
        import augment as augment_mod

        off = {"spectral": {"p_spectral_reshape": 0.0},
               "resample": {"p_resample": 0.0}}.get(args.ablate, {})
        augmenter = augment_mod.Augmenter(**off)
        if off:
            print(f"ABLATION: {args.ablate} transform disabled")
        print(f"augmentation: {augmenter.describe()}")

    model = cnn.IQNet(len(classes))
    print("training ...")
    model = cnn.train_model(model, a["X"][tr], a["y"][tr],
                            a["X"][te], a["y"][te], epochs=args.epochs,
                            augment=augmenter)

    pred_in = cnn.predict(model, a["X"][te])
    pred_cross = cnn.predict(model, b["X"])

    acc_in = accuracy_by_snr(pred_in, a["y"][te], a["z"][te], snrs)
    acc_cross = accuracy_by_snr(pred_cross, b["y"], b["z"], snrs)

    print(f"\n  SNR (dB)   in-domain   cross-domain   gap")
    for s, ai, ac in zip(snrs, acc_in, acc_cross):
        print(f"  {s:>7}      {ai:.3f}        {ac:.3f}      {ai - ac:+.3f}")

    high = np.array(snrs) >= 10
    print(f"\nmean over SNR >= 10 dB:  in-domain {np.nanmean(acc_in[high]):.3f}"
          f"   cross-domain {np.nanmean(acc_cross[high]):.3f}"
          f"   gap {np.nanmean(acc_in[high] - acc_cross[high]):+.3f}")

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(snrs, acc_in, "o-", lw=2, color="tab:blue",
            label=f"in-domain ({src.name} held out)")
    ax.plot(snrs, acc_cross, "s-", lw=2, color="tab:red",
            label=f"cross-domain (tested on {dst.name})")
    ax.fill_between(snrs, acc_cross, acc_in, where=acc_in >= acc_cross,
                    color="tab:red", alpha=0.12, label="generalization gap")
    ax.axhline(1 / len(classes), ls=":", color="gray", lw=1,
               label=f"chance ({1/len(classes):.2f})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Trained on {src.name}, tested on {dst.name}\n"
                 f"{len(classes)} shared classes: {', '.join(classes)}",
                 fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    suffix = ""
    if args.augment:
        suffix = "_augmented"
        if args.ablate != "none":
            suffix += f"_no_{args.ablate}"
    name = f"09_crossdomain_{src.name}_to_{dst.name}{suffix}"
    fig.savefig(FIGURES / f"{name}_accuracy.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- confusion pair
    fig, axes = plt.subplots(1, 2, figsize=(6 * 2 + 1, 6))
    for ax, (pred, y, z, title) in zip(axes, [
        (pred_in, a["y"][te], a["z"][te], f"in-domain ({src.name})"),
        (pred_cross, b["y"], b["z"], f"cross-domain ({dst.name})"),
    ]):
        mask = z >= 10
        cm = confusion_matrix(y[mask], pred[mask], labels=range(len(classes)))
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}, SNR >= 10 dB\n"
                     f"acc {(pred[mask] == y[mask]).mean():.3f}", fontsize=10)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                if cm[r, c] > 0.02:
                    ax.text(c, r, f"{cm[r,c]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if cm[r, c] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Where the model breaks when the transmitter changes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / f"{name}_confusion.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / f"{name}.npz", snrs=np.array(snrs), acc_in=acc_in,
             acc_cross=acc_cross, classes=np.array(classes))
    print(f"\nwrote figures/{name}_accuracy.png")
    print(f"wrote figures/{name}_confusion.png")

    if hasattr(src, "close"):
        src.close()


if __name__ == "__main__":
    main()

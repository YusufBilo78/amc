"""
train_backbone.py -- plain modulation classification on RadioML, current backbone.

Why this exists
---------------
Every measurement in this project is a *difference* between two accuracies --
in-domain minus cross-domain, with-whitening minus without. None of the scripts
that produce those differences reports the thing a reader asks for first: how
well does the backbone actually classify modulation on the benchmark itself?

`cnn.py --data radioml` does not answer it either. It still builds `cnn.IQNet`,
the superseded backbone, on a two-way split. So the number does not exist for
`model_zoo.backbone()` on either dataset, and this file is the entry point that
produces it.

Two datasets, one protocol:

    rml2018   RadioML 2018.01A, 24 classes, 1024-sample frames, 26 SNR levels
    rml2016   RML2016.10a,      11 classes,  128-sample frames, 20 SNR levels

`model_zoo.ICRNNA` is length-agnostic -- conv, pool, LSTM over time, attention
pooling over time -- so the same module takes both at an identical parameter
count. **2018 is run at its native 1024 samples here.** The peer's reference
trainer crops 2018 to 128 so that a 2016-shaped architecture applies unchanged,
and its own docstring puts the expected accuracy near 44% for that reason; that
is a deliberately handicapped setup and not what this file measures. Numbers
from here are not comparable to it.

Protocol, matching the rest of the repository rather than the reference
trainers:

  - 70/15/15 stratified over (class, SNR) cells, via `cnn.split`
  - early stopping and the LR schedule read **validation only**; the test set
    is touched once, at the end
  - per-frame unit average power (`cnn.normalize_frames`). The reference
    trainers divide by `sqrt(mean(iq**2))` instead, which leaves average power
    at 2 rather than 1 -- a constant factor of sqrt(2), absorbed by the first
    BatchNorm, but worth knowing if a number here is compared against theirs.
  - one training run per seed, results written after each seed and skipped on
    restart

Cost, measured on the RTX 3070 Laptop rather than reasoned about. The BiLSTM
runs over L/4 timesteps and recurrence does not parallelise over time, so a
1024-sample frame was expected to cost 8x a 128-sample one. It costs **4.2x**:
7.7 ms per batch of 256 at 128 samples against 32.2 ms at 1024. The conv front
end and the dense head do not scale with sequence length, and a 256-step LSTM
keeps the GPU better fed than a 32-step one, which amortises kernel launch.

Those are ceiling numbers -- data already resident on the GPU, no evaluation
pass. Real training measures about half that throughput (~3,900 frames/s at
1024 samples), so double any estimate taken from a synthetic benchmark:

    24 classes, 512 frames/cell, 1024 samples   223,641 train frames
                                                ~1 min/epoch, ~1 h for 60 epochs
    11 classes, all 1000/cell, 128 samples      154,000 train frames
                                                ~0.2 min/epoch, ~12 min for 60

Both of those are the laptop. Measured on a Colab GPU the 2018 configuration
came in at 9-13 minutes per seed rather than the hour above, so treat the table
as an upper bound tied to that machine and not as a property of the run. Which
of the two -- a faster GPU or early stopping well before epoch 60 -- accounts
for the difference was not separated.

`--frames-per-cell` is the lever, and 512 is affordable, which is why it is the
default. 1024 doubles it (~2 min/epoch) and is an overnight run. Keep the
machine on mains power -- on battery the GPU is capped near 35 W and runs about
5x slower.

Peak VRAM at 1024 samples and batch 256 is 1.4 GB, so the 8 GB budget is not
the binding constraint; the host-side array is (24 x 26 x fpc, 2, 1024) float32,
2.6 GB at fpc=512.

Run:
    cd src && python train_backbone.py --data rml2018
    cd src && python train_backbone.py --data rml2016 --data-path /content/drive/MyDrive
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# "High SNR" for the headline number, matching compare_methods.py so the two
# are read on the same axis. The per-SNR curve is reported in full regardless.
HIGH_SNR = 10

# Context only, printed next to the 2016 result. It is the target for
# `colab/icrnna_faithful_2016.py`, which is built from the paper; the backbone
# here is transcribed from a peer's reproduction and differs from the paper in
# five places (see model_zoo.ICRNNA). Hitting or missing 63.24% from this file
# neither validates nor refutes the paper -- that is what the faithful build
# is for.
PAPER_2016 = 63.24


def load_dataset(name: str, frames_per_cell: int, data_path: str | None,
                 seed: int = 0):
    """
    (X, y, z, class_names) with X as (n, 2, L) float32 at unit average power.

    Both loaders return (n, L, 2); the transpose to channel-first happens here
    so the two paths stay identical from this point on. Intermediates are freed
    explicitly -- at 2018's full frame length each copy of the array is a
    couple of gigabytes.
    """
    if name == "rml2018":
        import radioml

        with radioml.RadioML(search_dir=data_path) as ds:
            data = ds.load(frames_per_cell=frames_per_cell,
                           test_fraction=0.0, seed=seed)
        class_names = radioml.CLASSES
    else:
        import rml2016

        with rml2016.RML2016(data_path) as ds:
            data = ds.load(frames_per_cell=frames_per_cell,
                           test_fraction=0.0, seed=seed)
            class_names = ds.classes

    X = np.transpose(data["X_train"], (0, 2, 1)).astype(np.float32)
    y = data["y_train"].astype(np.int64)
    z = data["z_train"].astype(np.int16)
    del data

    X = cnn.normalize_frames(X)
    return X, y, z, class_names


def run_seed(X, y, z, class_names, seed, args):
    """One training run. Returns (per-SNR accuracy, confusion counts, model)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr, va, te = cnn.split(X, y, z, test_fraction=0.15, val_fraction=0.15,
                           seed=seed)
    print(f"  {len(tr):,} train / {len(va):,} val / {len(te):,} test")

    # `cnn.split` allocates per (class, SNR) cell and truncates, so a small
    # --frames-per-cell empties the validation set entirely: at 4 frames per
    # cell, int(4 * 0.15) == 0. Early stopping would then be monitoring nothing.
    # Fail here with the arithmetic rather than inside the training loop.
    if len(va) == 0:
        raise SystemExit(
            f"validation set is empty: --frames-per-cell "
            f"{args.frames_per_cell} leaves int({args.frames_per_cell} * 0.15)"
            f" = 0 frames per (class, SNR) cell.\n"
            "Early stopping and the LR schedule read validation only, so "
            "there is nothing to train against. Use at least 32."
        )

    model = model_zoo.backbone(len(class_names))
    model = cnn.train_model(model, X[tr], y[tr], X[va], y[va],
                            epochs=args.epochs, batch_size=args.batch_size,
                            lr=args.lr, patience=args.patience, grad_clip=5.0)

    pred = cnn.predict(model, X[te])
    snrs = np.unique(z)
    curve = np.array([
        float((pred[z[te] == s] == y[te][z[te] == s]).mean())
        if (z[te] == s).any() else np.nan
        for s in snrs
    ])

    n = len(class_names)
    high = z[te] >= HIGH_SNR
    conf = np.zeros((n, n), dtype=np.int64)
    np.add.at(conf, (y[te][high], pred[high]), 1)

    overall = float((pred == y[te]).mean())
    high_acc = float((pred[high] == y[te][high]).mean())
    return curve, conf, overall, high_acc, model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=("rml2018", "rml2016"), default="rml2018")
    p.add_argument("--data-path", default=None,
                   help="directory holding the dataset, tried before the "
                        "loader's own search path. Needed wherever the data is "
                        "not where the loader expects -- a mounted Drive in "
                        "Colab, a staging directory on a runner's local disk. "
                        "For rml2018 the directory must hold the radioml_X/y/z "
                        ".npy triple; for rml2016, the .pkl")
    p.add_argument("--frames-per-cell", type=int, default=512,
                   help="frames drawn per (class, SNR) cell. 2018 has 4096 "
                        "available and 2016 has 1000. 512 measures at about "
                        "1 min/epoch on 2018; see the cost note in the module "
                        "docstring")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--tag", default="",
                   help="appended to the output filenames, so a run with "
                        "different settings does not overwrite an earlier one")
    p.add_argument("--out-dir", default=None,
                   help="where results, checkpoints and figures are written. "
                        "Defaults to the repository root. Point it at durable "
                        "storage when the machine running this is not durable "
                        "-- a mounted Drive in Colab, where the runtime is "
                        "reclaimed after 12 hours and the per-seed .npz is "
                        "what makes a restart a resume rather than a restart")
    args = p.parse_args()

    seeds = list(range(args.seeds))
    suffix = f"_{args.tag}" if args.tag else ""
    stem = f"train_backbone_{args.data}_f{args.frames_per_cell}{suffix}"

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else ROOT
    fig_dir = out_dir / "figures" if args.out_dir else FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{stem}.npz"

    print(f"loading {args.data} ({args.frames_per_cell} frames per cell) ...")
    t0 = time.time()
    X, y, z, class_names = load_dataset(args.data, args.frames_per_cell,
                                        args.data_path)
    snrs = np.unique(z)
    print(f"  {X.shape[0]:,} frames  {X.shape}  "
          f"({X.nbytes / 1e9:.1f} GB)  in {time.time() - t0:.0f}s")
    print(f"  {len(class_names)} classes, SNR {snrs.min()}..{snrs.max()} dB "
          f"({len(snrs)} levels)")
    print(f"-> {npz_path.name}\n")

    n = len(class_names)
    overall = np.full(len(seeds), np.nan)
    high_snr = np.full(len(seeds), np.nan)
    curves = np.full((len(seeds), len(snrs)), np.nan)
    confusions = np.zeros((len(seeds), n, n), dtype=np.int64)

    # Resume. A seed counts as done when its overall entry is no longer NaN.
    # The shape check guards against resuming into a file written by a
    # differently-shaped configuration.
    if npz_path.exists():
        old = np.load(npz_path, allow_pickle=True)
        if old["overall"].shape == overall.shape and \
                old["curves"].shape == curves.shape:
            overall, high_snr = old["overall"], old["high"]
            curves, confusions = old["curves"], old["confusions"]
            done = int(np.count_nonzero(~np.isnan(overall)))
            if done:
                print(f"resuming: {done}/{len(seeds)} seeds already done\n")
        else:
            print(f"{npz_path.name} has a different shape; starting over\n")

    def save():
        np.savez(npz_path, overall=overall, high=high_snr, curves=curves,
                 confusions=confusions, snrs=snrs,
                 class_names=np.array(class_names), dataset=args.data,
                 frames_per_cell=args.frames_per_cell, epochs=args.epochs,
                 patience=args.patience, high_snr_threshold=HIGH_SNR)

    for j, seed in enumerate(seeds):
        if not np.isnan(overall[j]):
            continue
        print(f"{'=' * 66}\n seed {seed}\n{'=' * 66}")
        t_seed = time.time()
        curve, conf, acc, acc_high, model = run_seed(X, y, z, class_names,
                                                     seed, args)
        curves[j], confusions[j] = curve, conf
        overall[j], high_snr[j] = acc, acc_high
        print(f"  test {acc:.4f} overall, {acc_high:.4f} at SNR >= {HIGH_SNR} dB"
              f"   ({(time.time() - t_seed) / 60:.1f} min)\n")
        save()
        torch.save(model.state_dict(), out_dir / f"{stem}_seed{seed}.pt")
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    print(f"{'=' * 66}")
    print(f"{len(seeds)} seed(s): overall {np.nanmean(overall):.4f} "
          f"+- {np.nanstd(overall):.4f}   |   "
          f"SNR >= {HIGH_SNR} dB {np.nanmean(high_snr):.4f} "
          f"+- {np.nanstd(high_snr):.4f}")
    if args.data == "rml2016":
        print(f"context: El-Haryqy et al. report {PAPER_2016}% on this dataset "
              f"for ICRNNA. This backbone is a reproduction of a reproduction "
              f"and\n         differs from the paper in five places, so treat "
              f"that as context, not as a target hit or missed.\n"
              f"         colab/icrnna_faithful_2016.py is the build that can "
              f"answer it.")

    print(f"\n  SNR (dB)   accuracy")
    for s, a in zip(snrs, np.nanmean(curves, axis=0)):
        print(f"  {s:>7}     {a:.3f}")

    # -------------------------------------------------------------- figures
    mean_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs, mean_curve, "o-", lw=2, color="tab:blue", label="ICRNNA")
    if len(seeds) > 1:
        ax.fill_between(snrs, mean_curve - std_curve, mean_curve + std_curve,
                        color="tab:blue", alpha=0.15)
    ax.axhline(1 / n, ls=":", color="red", lw=1, label=f"chance ({1/n:.3f})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"ICRNNA on {args.data}, {n} classes, {len(seeds)} seed(s)"
                 + (" (band = 1 s.d.)" if len(seeds) > 1 else ""))
    fig.tight_layout()
    fig.savefig(fig_dir / f"29_{stem}_accuracy.png", dpi=140)
    plt.close(fig)

    cm = confusions.sum(axis=0).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = max(7, n * 0.55)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"ICRNNA on {args.data}, SNR >= {HIGH_SNR} dB "
                 f"(acc {np.nanmean(high_snr):.3f})")
    if n <= 14:
        for r in range(n):
            for c in range(n):
                if cm[r, c] > 0.02:
                    ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if cm[r, c] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / f"30_{stem}_confusion.png", dpi=140)
    plt.close(fig)

    print(f"\nwrote {npz_path}")
    print(f"wrote {fig_dir / f'29_{stem}_accuracy.png'}")
    print(f"wrote {fig_dir / f'30_{stem}_confusion.png'}")


if __name__ == "__main__":
    main()

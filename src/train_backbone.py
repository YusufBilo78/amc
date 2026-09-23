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
import plots

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


def resolve_classes(wanted: str | None, available: list[str], data: str):
    """
    (class_ids, class_names) for a --classes string, or (None, available).

    The ids index the dataset's own class list and are what the loader is
    given; the names come back in the order they were asked for, because that
    is the order the confusion matrix rows are printed in and reading a table
    whose rows are in a different order from the flag that produced it is a
    reliable way to misread it.
    """
    if not wanted:
        return None, available
    names = [c.strip() for c in wanted.split(",")]
    missing = [c for c in names if c not in available]
    if missing:
        raise SystemExit(
            f"--classes: not in {data}: {', '.join(missing)}\n"
            f"available: {', '.join(available)}")
    if len(names) != len(set(names)):
        raise SystemExit(f"--classes: repeated class in {wanted!r}")
    if len(names) < 2:
        raise SystemExit("--classes needs at least two classes")
    return [available.index(c) for c in names], names


def load_dataset(name: str, frames_per_cell: int, data_path: str | None,
                 seed: int = 0, classes: str | None = None):
    """
    (X, y, z, class_names) with X as (n, 2, L) float32 at unit average power.

    Both loaders return (n, L, 2); the transpose to channel-first happens here
    so the two paths stay identical from this point on. Intermediates are freed
    explicitly -- at 2018's full frame length each copy of the array is a
    couple of gigabytes.

    `classes` restricts the run to a subset by name. The selection is pushed
    down into the loader rather than applied to the loaded arrays, which is
    what makes a four-class run cheap: only those cells are read, so the same
    memory buys eight times the frames per class that a 24-class run can
    afford. The labels come back remapped to 0..k-1 in the order asked for, so
    the model is built with k outputs and is choosing between k answers.

    A subset run therefore does not train on the same frames as the full run
    would have given those classes -- both loaders draw each cell from one
    generator walking the classes in order, so iterating four consumes it
    differently from iterating twenty-four. That is fine, because a subset run
    is its own experiment and is written to its own file; it is not fine to
    quote one as if it were a slice of the other.
    """
    if name == "rml2018":
        import radioml

        available = list(radioml.CLASSES)
        ids, class_names = resolve_classes(classes, available, name)
        with radioml.RadioML(search_dir=data_path) as ds:
            data = ds.load(classes=ids, frames_per_cell=frames_per_cell,
                           test_fraction=0.0, seed=seed)
    else:
        import rml2016

        with rml2016.RML2016(data_path) as ds:
            available = list(ds.classes)
            ids, class_names = resolve_classes(classes, available, name)
            data = ds.load(classes=ids, frames_per_cell=frames_per_cell,
                           test_fraction=0.0, seed=seed)

    X = np.transpose(data["X_train"], (0, 2, 1)).astype(np.float32)
    y = data["y_train"].astype(np.int64)
    z = data["z_train"].astype(np.int16)
    del data

    if ids is not None:
        # The loader returns the dataset's own label ids; the model needs
        # 0..k-1, in the order the names were given.
        remap = np.full(len(available), -1, dtype=np.int64)
        for new_id, old_id in enumerate(ids):
            remap[old_id] = new_id
        y = remap[y]
        assert y.min() >= 0, "loader returned a class that was not requested"

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

    # One confusion matrix per SNR level. The pooled matrix above the
    # threshold is the headline table, but on an easy problem it saturates --
    # four classes at 1024 samples make two errors in forty thousand decisions
    # above 10 dB -- and the table that shows where decisions actually go is
    # the one at 0 dB. Storing every level costs nothing and means the run
    # never has to be repeated to ask a different SNR.
    n = len(class_names)
    by_snr = np.zeros((len(snrs), n, n), dtype=np.int64)
    for k, s in enumerate(snrs):
        at = z[te] == s
        np.add.at(by_snr[k], (y[te][at], pred[at]), 1)
    conf = by_snr[snrs >= HIGH_SNR].sum(axis=0)

    high = z[te] >= HIGH_SNR
    overall = float((pred == y[te]).mean())
    high_acc = float((pred[high] == y[te][high]).mean())

    if getattr(model, "stopped_early", False):
        print(f"  converged: best epoch {model.best_epoch} of "
              f"{model.epoch_ceiling}, then {args.patience} without improvement")
    else:
        print(f"  NOT converged: best epoch {model.best_epoch} of "
              f"{model.epoch_ceiling} -- the ceiling stopped this run, not the "
              f"model.\n  The accuracy below is a floor. Raise --epochs.")
    return curve, conf, by_snr, overall, high_acc, model


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
    p.add_argument("--classes", default=None,
                   help="comma-separated class names to train on, e.g. "
                        "BPSK,QPSK,16QAM,64QAM. The model is built with that "
                        "many outputs, so this is a genuinely smaller problem "
                        "rather than a slice of a larger one -- a four-class "
                        "run gives the model four answers to choose between, "
                        "and its confusion matrix rows sum over four columns. "
                        "Slicing four rows out of a finished 24-class table is "
                        "a different thing; confusion_table.py --classes does "
                        "that and says so")
    p.add_argument("--frame-len", type=int, default=None,
                   help="keep only the first N samples of every frame. The "
                        "frame-length experiment: how much of the decision "
                        "table survives when the model sees 64 or 32 samples "
                        "instead of 2016's 128. Cropping is the same for "
                        "train and test, the model is built for the cropped "
                        "length, and the output name carries _L<N>, so a "
                        "cropped run is its own file")
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
    # A subset run is a different experiment from the full one, so it has to
    # land in a different file even when --tag is not given.
    subset = f"_c{len(args.classes.split(','))}" if args.classes else ""
    crop = f"_L{args.frame_len}" if args.frame_len else ""
    stem = (f"train_backbone_{args.data}_f{args.frames_per_cell}"
            f"{subset}{crop}{suffix}")

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else ROOT
    fig_dir = out_dir / "figures" if args.out_dir else FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{stem}.npz"

    print(f"loading {args.data} ({args.frames_per_cell} frames per cell) ...")
    t0 = time.time()
    X, y, z, class_names = load_dataset(args.data, args.frames_per_cell,
                                        args.data_path, classes=args.classes)
    if args.frame_len:
        if args.frame_len >= X.shape[-1]:
            raise SystemExit(f"--frame-len {args.frame_len}: frames are only "
                             f"{X.shape[-1]} samples long")
        # The first N samples of every frame, then renormalised to unit
        # power over what is kept. Same crop for every frame, so no frame
        # has more information than another.
        X = cnn.normalize_frames(np.ascontiguousarray(X[:, :, :args.frame_len]))
        print(f"  cropped every frame to its first {args.frame_len} samples")
    snrs = np.unique(z)
    print(f"  {X.shape[0]:,} frames  {X.shape}  "
          f"({X.nbytes / 1e9:.1f} GB)  in {time.time() - t0:.0f}s")
    print(f"  {len(class_names)} classes, SNR {snrs.min()}..{snrs.max()} dB "
          f"({len(snrs)} levels)")
    print(f"-> {npz_path.name}\n")

    n = len(class_names)
    overall = np.full(len(seeds), np.nan)
    high_snr = np.full(len(seeds), np.nan)
    best_epochs = np.full(len(seeds), np.nan)
    curves = np.full((len(seeds), len(snrs)), np.nan)
    confusions = np.zeros((len(seeds), n, n), dtype=np.int64)
    confusions_by_snr = np.zeros((len(seeds), len(snrs), n, n), dtype=np.int64)

    # Resume. A seed counts as done when its overall entry is no longer NaN.
    # The shape check guards against resuming into a file written by a
    # differently-shaped configuration.
    if npz_path.exists():
        old = np.load(npz_path, allow_pickle=True)
        if old["overall"].shape == overall.shape and \
                old["curves"].shape == curves.shape:
            overall, high_snr = old["overall"], old["high"]
            curves, confusions = old["curves"], old["confusions"]
            if "best_epochs" in old.files:
                best_epochs = old["best_epochs"]
            if "confusions_by_snr" in old.files:
                confusions_by_snr = old["confusions_by_snr"]
            else:
                # Seeds finished before this key existed have no per-SNR
                # record here; eval_by_snr.py rebuilds it from their
                # checkpoints. Seeds run from now on fill it directly.
                print("  (no per-SNR confusions in this file for the seeds "
                      "already done -- see eval_by_snr.py)")
            done = int(np.count_nonzero(~np.isnan(overall)))
            if done:
                print(f"resuming: {done}/{len(seeds)} seeds already done\n")
        else:
            print(f"{npz_path.name} has a different shape; starting over\n")

    def save():
        np.savez(npz_path, overall=overall, high=high_snr, curves=curves,
                 confusions=confusions, confusions_by_snr=confusions_by_snr,
                 snrs=snrs, best_epochs=best_epochs,
                 class_names=np.array(class_names), dataset=args.data,
                 frames_per_cell=args.frames_per_cell, epochs=args.epochs,
                 patience=args.patience, high_snr_threshold=HIGH_SNR,
                 frame_len=int(X.shape[-1]))

    for j, seed in enumerate(seeds):
        if not np.isnan(overall[j]):
            continue
        print(f"{'=' * 66}\n seed {seed}\n{'=' * 66}")
        t_seed = time.time()
        curve, conf, by_snr, acc, acc_high, model = run_seed(
            X, y, z, class_names, seed, args)
        curves[j], confusions[j], confusions_by_snr[j] = curve, conf, by_snr
        overall[j], high_snr[j] = acc, acc_high
        best_epochs[j] = model.best_epoch
        print(f"  test {acc:.4f} overall, {acc_high:.4f} at SNR >= {HIGH_SNR} dB"
              f"   ({(time.time() - t_seed) / 60:.1f} min)\n")
        save()
        torch.save(model.state_dict(), out_dir / f"{stem}_seed{seed}.pt")
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    print(f"{'=' * 66}")
    # A run has converged only when early stopping fired, which means
    # `best_epoch + patience <= ceiling`. Testing `best_epoch >= ceiling` instead is
    # too weak and was wrong here: three cells peaked at epochs 53, 37 and 50 under a
    # 60-epoch ceiling with patience 20, so two of them would have needed to reach
    # epoch 73 and 70 before stopping and instead ran out of budget at 60 -- while
    # `53 >= 60` and `50 >= 60` are both False and the warning stayed silent.
    if args.patience:
        unconverged = best_epochs + args.patience > args.epochs
        n = int(np.count_nonzero(unconverged & ~np.isnan(best_epochs)))
        if n:
            worst = int(np.nanmax(best_epochs))
            print(f"\n  WARNING: {n} of {len(seeds)} seed(s) did not converge. "
                  f"The budget ran out at the\n  {args.epochs}-epoch ceiling "
                  f"before early stopping fired: the latest peak was epoch "
                  f"{worst},\n  which needs {worst + args.patience} to stop. "
                  f"Those accuracies are floors.\n")

    print(f"{len(seeds)} seed(s): overall {np.nanmean(overall):.4f} "
          f"+- {np.nanstd(overall):.4f}   |   "
          f"SNR >= {HIGH_SNR} dB {np.nanmean(high_snr):.4f} "
          f"+- {np.nanstd(high_snr):.4f}")
    # The paper's number is for all eleven classes. Printing it next to a
    # subset run invites exactly the comparison it cannot support -- four
    # classes is an easier problem and a higher number there means nothing
    # about the paper.
    if args.data == "rml2016" and not args.classes:
        print(f"context: El-Haryqy et al. report {PAPER_2016}% on this dataset "
              f"for ICRNNA. This backbone is a reproduction of a reproduction "
              f"and\n         differs from the paper in five places, so treat "
              f"that as context, not as a target hit or missed.\n"
              f"         colab/icrnna_faithful_2016.py is the build that can "
              f"answer it.")

    print(f"\n  SNR (dB)   accuracy")
    for s, a in zip(snrs, np.nanmean(curves, axis=0)):
        print(f"  {s:>7}     {a:.3f}")

    plots.write_figures(curves, confusions, snrs, class_names,
                        args.data, float(np.nanmean(high_snr)), stem,
                        fig_dir, len(seeds), HIGH_SNR)

    print(f"\nwrote {npz_path}")
    print(f"wrote {fig_dir / f'29_{stem}_accuracy.png'}")
    print(f"wrote {fig_dir / f'30_{stem}_confusion.png'}")


if __name__ == "__main__":
    main()

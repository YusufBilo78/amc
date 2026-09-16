"""
eval_by_snr.py -- rebuild per-SNR confusion matrices from a finished run.

`train_backbone.py` stores a confusion matrix for every SNR level in
`confusions_by_snr`. Runs made before that key existed stored only the pooled
matrix above the threshold, and on an easy problem that one is a diagonal:
the four-class 2018 run makes two errors in forty thousand decisions above
10 dB, while at 0 dB it is at 80%. The table that says where decisions go is
the low-SNR one, and it was never written.

This regenerates it without retraining. The checkpoints are on disk, the data
draw is seeded, and the split is seeded, so the test set each seed was scored
on can be rebuilt exactly. It then has to *prove* that: the pooled matrix
rebuilt here is compared against the one stored in the file, count for count,
and nothing is written unless they agree. A silent mismatch here would mean a
table attributed to a run it did not come from.

    cd src && python eval_by_snr.py \\
        /content/drive/MyDrive/amc-results/train_backbone_rml2018_f2048_c4_colab.npz \\
        --data-path /content/amc-shared5

Everything else -- dataset, frames per cell, class subset, seed count -- is
read from the file, so the command cannot disagree with the run it is scoring.
The `.pt` checkpoints are expected next to the `.npz`, which is where
`train_backbone.py` puts them. Needs the GPU and the staged data; a few
minutes per seed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

import cnn
import model_zoo
import train_backbone as tb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="a .npz written by train_backbone.py")
    p.add_argument("--data-path", default=None,
                   help="where the data is; same meaning as in "
                        "train_backbone.py")
    args = p.parse_args()

    path = pathlib.Path(args.npz)
    z = dict(np.load(path, allow_pickle=True))
    names = [str(c) for c in z["class_names"]]
    dataset = str(z["dataset"])
    fpc = int(z["frames_per_cell"])
    threshold = int(z["high_snr_threshold"])
    n_seeds = len(z["overall"])
    stem = path.stem

    # The run's own class list decides whether this was a subset. Passing the
    # full list through --classes would be harmless but would also change the
    # draw (the loader iterates the classes it is given), so give the loader
    # exactly what train_backbone gave it.
    if dataset == "rml2018":
        import radioml
        full = list(radioml.CLASSES)
    else:
        import rml2016
        with rml2016.RML2016(args.data_path) as ds:
            full = list(ds.classes)
    classes = None if names == full else ",".join(names)

    print(f"{path.name}: {dataset}, {len(names)} classes, {fpc} frames/cell, "
          f"{n_seeds} seeds")
    X, y, zsnr, class_names = tb.load_dataset(dataset, fpc, args.data_path,
                                              classes=classes)
    assert class_names == names, (class_names, names)
    snrs = np.unique(zsnr)
    assert np.array_equal(snrs, z["snrs"]), "SNR levels differ from the file"
    n = len(names)

    by_snr = np.zeros((n_seeds, len(snrs), n, n), dtype=np.int64)
    for j in range(n_seeds):
        seed = j
        ckpt = path.with_name(f"{stem}_seed{seed}.pt")
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint {ckpt}")

        # Same calls, same order, as train_backbone.run_seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        _, _, te = cnn.split(X, y, zsnr, test_fraction=0.15,
                             val_fraction=0.15, seed=seed)

        model = model_zoo.backbone(n).to(cnn.DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=cnn.DEVICE))
        pred = cnn.predict(model, X[te])

        for k, s in enumerate(snrs):
            at = zsnr[te] == s
            np.add.at(by_snr[j, k], (y[te][at], pred[at]), 1)

        pooled = by_snr[j][snrs >= threshold].sum(axis=0)
        stored = z["confusions"][j]
        if not np.array_equal(pooled, stored):
            print(f"seed {seed}: rebuilt pooled matrix does NOT match the "
                  f"stored one\n{pooled}\nvs\n{stored}")
            raise SystemExit(
                "refusing to write: the test set or the checkpoint is not the "
                "one this file was scored on")
        acc = (pred == y[te]).mean()
        print(f"  seed {seed}: pooled matrix matches the file  "
              f"(overall {acc:.4f}, stored {z['overall'][j]:.4f})")

    z["confusions_by_snr"] = by_snr
    np.savez(path, **z)
    print(f"wrote confusions_by_snr ({by_snr.shape}) into {path}")
    print("now: python confusion_table.py", path, "--snr 0")


if __name__ == "__main__":
    main()

"""
whitening_smoothing.py -- why does whitening hurt on RML2016.10a?

On RadioML 2018 (1024 samples) whitening at alpha=0.75 closes the domain gap
at no in-domain cost. On RML2016.10a (128 samples) it costs 8 points in-domain
(0.948 -> 0.864) and does not close the gap. The hypothesis in README is the
envelope estimate: whitening divides each frame's spectrum by a smoothed copy
of its own magnitude, and at 128 samples that copy is made from 128 points
smoothed over 5 bins -- the same fraction of the band as 33 bins of 1024, but
a far noisier estimate.

The premise is measured (no training involved, synthetic 16QAM at 18 dB,
3000 frames, relative error of the per-frame envelope against the mean one):

    1024 samples, 33 bins (3.2% of the band)    0.106
     128 samples,  5 bins (3.9%)                0.277
     128 samples, 17 bins (13%)                 0.193
     128 samples, 65 bins (51%)                 0.118

So at 128 samples no smoothing width gets the estimate as clean as 2018's
while still resolving the shape of the band. Whether that is what costs the
accuracy is what this script measures, two ways:

  --source rml2016 --bins 3,5,9,17,33,65
        the smoothing sweep. If estimator noise is the cause, the in-domain
        cost should shrink as the smoothing widens. On its own this cannot
        separate "less noise" from "less whitening" -- at 65 bins whitening
        is close to doing nothing -- which is why the second arm exists.

  --source rml2018 --frame-len 128 --bins 5
        the crop. RadioML 2018 frames cut to their first 128 samples, where
        the estimate is exactly as noisy as on 2016. If whitening now costs
        in-domain accuracy on 2018 too, frame length is the cause. If it
        still costs nothing, the cause is something about RML2016.10a itself
        (its channel model), not the length.

Every run has a `none` row. Seeds, split, data selection and training are
those of compare_methods.py, so on rml2016 the `none` row and the 5-bin row
must reproduce seeds 0..4 of compare_methods_ICRNNA_es_rml2016_e100.npz
exactly -- a built-in check that this is the same experiment.

Output: whitening_smoothing_<source>[_L<n>]_e<epochs>.npz, written after
every cell; a rerun resumes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

import cnn
import domains
from compare_methods import BEST_ALPHA, CLASSES, build_model
from whitening_sweep import accuracy_by_snr, spectral_whiten, to_model_input

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("rml2016", "rml2018"), required=True)
    p.add_argument("--data-path", default=None,
                   help="rml2016: the pickle or its directory")
    p.add_argument("--frame-len", type=int, default=None,
                   help="crop every source frame to its first N samples and "
                        "generate the synthetic domain at N")
    p.add_argument("--bins", default="3,5,9,17,33,65",
                   help="comma-separated smoothing widths, in FFT bins of the "
                        "frame actually used (odd)")
    p.add_argument("--alpha", type=float, default=BEST_ALPHA)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    bins = [int(b) for b in args.bins.split(",")]
    if any(b % 2 == 0 for b in bins):
        raise SystemExit("--bins must be odd, so the window is centred")
    rows = [("none", 0)] + [(f"whitening a={args.alpha:g}, {b} bins", b)
                            for b in bins]
    seeds = list(range(args.seeds))

    if args.source == "rml2016":
        snrs = list(domains.RML2016Domain.SNRS)
        src = domains.RML2016Domain(args.data_path)
        a = src.load(CLASSES, snrs, frames_per_cell=1000, seed=0)
    else:
        snrs = list(range(-20, 31, 2))
        src = domains.RadioMLDomain()
        a = src.load(CLASSES, snrs, frames_per_cell=768, seed=0)
    src.close()
    n = src.frame_len
    if args.frame_len:
        if args.frame_len >= n:
            raise SystemExit(f"--frame-len {args.frame_len}: frames are {n}")
        a["X"] = cnn.normalize_frames(
            np.ascontiguousarray(a["X"][:, :, :args.frame_len]))
        n = args.frame_len
    b = domains.SyntheticDomain(n_samples=n).load(CLASSES, snrs,
                                                  frames_per_cell=400, seed=1)
    assert a["X"].shape[-1] == b["X"].shape[-1] == n

    stem = f"whitening_smoothing_{args.source}"
    stem += f"_L{args.frame_len}" if args.frame_len else ""
    stem += f"_e{args.epochs}"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{stem}.npz"

    print(f"source {args.source}: {len(a['X'])} frames of {n} samples, "
          f"SNR {snrs[0]}..{snrs[-1]} dB; synthetic {len(b['X'])} frames")
    print(f"{len(rows)} rows x {len(seeds)} seeds, alpha {args.alpha}, "
          f"ceiling {args.epochs}, patience {args.patience}")
    print(f"-> {npz_path}\n")

    a_iq = (a["X"][:, 0] + 1j * a["X"][:, 1]).astype(np.complex64)
    b_iq = (b["X"][:, 0] + 1j * b["X"][:, 1]).astype(np.complex64)

    high = np.array(snrs) >= 10
    q = CLASSES.index("16QAM")
    shape = (len(rows), len(seeds))
    in_domain, cross, qam16 = (np.full(shape, np.nan) for _ in range(3))
    best_epochs, ceilings = np.full(shape, np.nan), np.full(shape, np.nan)
    curves_in = np.full(shape + (len(snrs),), np.nan)
    curves_cross = np.full(shape + (len(snrs),), np.nan)
    if npz_path.exists():
        old = np.load(npz_path, allow_pickle=True)
        if old["in_domain"].shape == shape:
            in_domain, cross, qam16 = old["in_domain"], old["cross"], old["qam16"]
            best_epochs, ceilings = old["best_epochs"], old["ceilings"]
            curves_in, curves_cross = old["curves_in"], old["curves_cross"]
            print(f"resuming: {int(np.count_nonzero(~np.isnan(best_epochs)))}"
                  f"/{best_epochs.size} cells done\n")

    for i, (name, nb) in enumerate(rows):
        # a cell is done once its stopping epoch is recorded
        if np.isnan(best_epochs[i]).any():
            alpha = args.alpha if nb else 0.0
            Xa = to_model_input(spectral_whiten(a_iq, alpha, smooth_bins=nb or None))
            Xb = to_model_input(spectral_whiten(b_iq, alpha, smooth_bins=nb or None))
        for j, seed in enumerate(seeds):
            if not np.isnan(best_epochs[i, j]):
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            tr, va, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.15,
                                   val_fraction=0.15, seed=seed)
            model = build_model("ICRNNA", len(CLASSES))
            model = cnn.train_model(model, Xa[tr], a["y"][tr], Xa[va],
                                    a["y"][va], epochs=args.epochs,
                                    patience=args.patience, grad_clip=5.0)
            pred_in = cnn.predict(model, Xa[te])
            pred_cross = cnn.predict(model, Xb)
            curves_in[i, j] = accuracy_by_snr(pred_in, a["y"][te], a["z"][te], snrs)
            curves_cross[i, j] = accuracy_by_snr(pred_cross, b["y"], b["z"], snrs)
            in_domain[i, j] = float(np.nanmean(curves_in[i, j][high]))
            cross[i, j] = float(np.nanmean(curves_cross[i, j][high]))
            mask = (b["z"] >= 10) & (b["y"] == q)
            qam16[i, j] = float((pred_cross[mask] == q).mean())
            best_epochs[i, j] = model.best_epoch
            ceilings[i, j] = args.epochs
            flag = "" if model.best_epoch + args.patience <= args.epochs \
                else "  NOT converged"
            print(f"{name:<32} seed {seed}  in {in_domain[i, j]:.4f}  "
                  f"cross {cross[i, j]:.4f}  16QAM {qam16[i, j]:.3f}  "
                  f"best {model.best_epoch}  {time.time() - t0:.0f}s{flag}")
            np.savez(npz_path, rows=np.array([r[0] for r in rows]),
                     bins=np.array([r[1] for r in rows]), alpha=args.alpha,
                     seeds=np.array(seeds), snrs=np.array(snrs),
                     frame_len=n, source=args.source,
                     in_domain=in_domain, cross=cross, qam16=qam16,
                     curves_in=curves_in, curves_cross=curves_cross,
                     best_epochs=best_epochs, ceilings=ceilings)

    unconverged = best_epochs + args.patience > ceilings
    if unconverged.any():
        print(f"\nWARNING: {int(unconverged.sum())} cell(s) ran out of budget; "
              f"rerun with a higher --epochs after deleting those cells.")
    print(f"\n{'row':<32} {'in-domain':>16} {'cross':>16} {'cost vs none':>13}")
    for i, (name, _) in enumerate(rows):
        print(f"{name:<32} {in_domain[i].mean():>8.4f} ±{in_domain[i].std():.4f} "
              f"{cross[i].mean():>8.4f} ±{cross[i].std():.4f} "
              f"{in_domain[i].mean() - in_domain[0].mean():>+13.4f}")
    print(f"\nwrote {npz_path}")


if __name__ == "__main__":
    main()

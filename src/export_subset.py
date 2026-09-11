"""
export_subset.py -- carve an uploadable subset out of the 2018 dataset.

Why this exists
---------------
RadioML 2018.01A is 21 GB as HDF5 and about 21 GB again as the .npy memmaps
converted from it. That does not go into git -- `.gitignore` refuses `*.npy`
and `*.hdf5` on purpose, and the dataset is not ours to redistribute -- and it
does not fit in a free Google Drive either, which holds 15 GB in total.

But no run uses all of it. `train_backbone.py --frames-per-cell 512` reads 512
of the 4096 frames in each (class, SNR) cell, which is 319,488 frames, 2.6 GB.
That fits in Drive with room to spare, and it is not a compromise: it is the
same data the run would have selected anyway.

This writes that subset as a `radioml_X/y/z.npy` triple, the exact layout
`radioml.py` looks for, so on the far side nothing special is needed --
`--data-path <upload dir>` and the loader finds it.

The selection is `radioml.select_cell_balanced`, the same function the Colab
cell uses when it stages from HDF5, at the same fixed seed. So a subset
exported here and a subset staged there contain the same frames, and results
from the two machines are comparable rather than merely similar.

Sizes, so the choice is made with numbers rather than by feel:

    frames/cell    frames     float32     float16
            256   159,744      1.3 GB      0.7 GB
            512   319,488      2.6 GB      1.3 GB
           1024   638,976      5.2 GB      2.6 GB

Run:
    cd src && python export_subset.py --frames-per-cell 512
    cd src && python export_subset.py --frames-per-cell 512 --dtype float16
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

import radioml

CHUNK = 8192   # rows per copy; 67 MB at float32, comfortably inside RAM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-per-cell", type=int, default=512,
                   help="frames kept from each (class, SNR) cell. Match this "
                        "to the --frames-per-cell the run will use; a larger "
                        "export is wasted upload and a smaller one silently "
                        "caps the run")
    p.add_argument("--out", default=None,
                   help="output directory (default: <data dir>/upload_f<N>)")
    p.add_argument("--dtype", choices=("float32", "float16"), default="float32",
                   help="float16 halves the upload. The stored values lose "
                        "relative precision of about 1e-3, i.e. quantisation "
                        "noise some 60 dB below the signal -- far below even "
                        "the 30 dB top of this dataset's SNR range, and the "
                        "training loop runs in fp16 autocast anyway. It is "
                        "still a change to the data, so it is not the default")
    p.add_argument("--seed", type=int, default=42,
                   help="selection seed. Leave it alone unless you intend the "
                        "export to differ from what the Colab cell stages")
    p.add_argument("--classes", default=None,
                   help="comma-separated class names to keep, or 'shared' for "
                        "the five that mean the same thing in every domain "
                        "(BPSK, QPSK, 8PSK, 16QAM, 64QAM). Default: all 24")
    p.add_argument("--all-frames", action="store_true",
                   help="keep every frame of the selected classes rather than "
                        "subsampling. Required for anything that must match a "
                        "run done against the full file -- see the note below")
    args = p.parse_args()

    ds = radioml.RadioML()
    print(f"source   : {ds.path}")
    print(f"backend  : {ds.backend}")
    print(f"X        : {ds.X.shape}  {ds.X.dtype}")
    print(f"classes  : {ds.n_classes}   SNR levels: {len(ds.unique_snrs)}")

    cell = len(ds.cell_indices(0, int(ds.unique_snrs[0])))
    print(f"frames per cell available: {cell}")
    if args.frames_per_cell > cell:
        print(f"\nNOTE: asked for {args.frames_per_cell} per cell but only "
              f"{cell} exist. Every cell will be taken whole.")

    # Class filter. Ids stay in the file's own 24-class numbering so that
    # `RadioML.load(classes=[...])` keeps working against the export unchanged.
    if args.classes is None:
        class_ids = None
        tag = "all"
    else:
        import domains

        wanted = (list(domains.SHARED_CLASSES) if args.classes == "shared"
                  else [c.strip() for c in args.classes.split(",")])
        unknown = [c for c in wanted if c not in radioml.CLASSES]
        if unknown:
            raise SystemExit(f"not RadioML 2018 class names: {unknown}\n"
                             f"known: {', '.join(radioml.CLASSES)}")
        class_ids = [radioml.CLASSES.index(c) for c in wanted]
        tag = "-".join(wanted)
        print(f"classes  : {len(wanted)} of {len(radioml.CLASSES)}  "
              f"({', '.join(wanted)})  ids {class_ids}")

    if args.out:
        out_dir = pathlib.Path(args.out)
    elif args.all_frames:
        out_dir = ds.path.parent / f"upload_{tag}_allframes"
    else:
        out_dir = ds.path.parent / f"upload_f{args.frames_per_cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_frames:
        # Every frame of the selected classes, in the file's own order.
        #
        # This is what makes an export usable by an experiment whose other
        # cells were computed against the full file. `RadioML.load` picks its
        # frames with `rng.choice` over the positions within each (class, SNR)
        # cell, so as long as the cell still holds all of them in the same
        # order, the same call selects the same frames here as there. Export
        # fewer and it silently selects different ones.
        if class_ids is None:
            raise SystemExit("--all-frames without --classes is the whole "
                             "21 GB file; there is nothing to export.")
        sel = np.flatnonzero(np.isin(ds.labels, class_ids))
        print(f"keeping every frame of those classes (no subsampling)")
    else:
        sel = radioml.select_cell_balanced(ds.labels, ds.snrs,
                                           args.frames_per_cell, seed=args.seed)
        if class_ids is not None:
            sel = sel[np.isin(ds.labels[sel], class_ids)]
    dtype = np.dtype(args.dtype)
    need = len(sel) * 1024 * 2 * dtype.itemsize
    free = shutil.disk_usage(out_dir).free
    print(f"\nselected : {len(sel):,} of {ds.X.shape[0]:,} frames "
          f"({len(sel) / ds.X.shape[0]:.1%})")
    print(f"output   : {out_dir}")
    print(f"size     : {need / 1e9:.2f} GB as {args.dtype}   "
          f"(free: {free / 1e9:.0f} GB)")
    if free < need * 1.05:
        raise SystemExit("not enough free space")

    # Written through a memmap so the array never has to fit in RAM twice, and
    # read in sorted chunks, which is the access pattern both a memmap and an
    # HDF5 file are fastest at.
    x_out = np.lib.format.open_memmap(
        out_dir / "radioml_X.npy", mode="w+", dtype=dtype,
        shape=(len(sel), 1024, 2))
    t0 = time.perf_counter()
    for start in range(0, len(sel), CHUNK):
        stop = min(start + CHUNK, len(sel))
        x_out[start:stop] = np.asarray(ds.X[sel[start:stop]]).astype(dtype)
        done = stop / len(sel)
        el = time.perf_counter() - t0
        print(f"\r  {done:6.1%}  {stop:>8,}/{len(sel):,}  {el:5.0f}s elapsed, "
              f"~{el / max(done, 1e-9) - el:5.0f}s left", end="")
    x_out.flush()
    del x_out
    print(f"\n  written in {time.perf_counter() - t0:.0f}s")

    np.save(out_dir / "radioml_y.npy", ds.labels[sel].astype(np.int8))
    np.save(out_dir / "radioml_z.npy", ds.snrs[sel].astype(np.int16))

    # Verify against the source rather than trusting the write. float16 is
    # checked for closeness, not equality -- losing precision is the point of
    # that option, and an equality check would fail by design.
    print("\nverifying ...")
    check = np.load(out_dir / "radioml_X.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    for i in rng.integers(0, len(sel), 5):
        i = int(i)
        a = np.asarray(ds.X[sel[i]]).astype(np.float32)
        b = np.asarray(check[i]).astype(np.float32)
        if dtype == np.float32:
            ok = np.array_equal(a, b)
            how = "identical"
        else:
            ok = np.allclose(a, b, rtol=2e-3, atol=1e-7)
            how = "within float16 precision"
        print(f"  row {i:>7,} (source {sel[i]:>9,})  {how}: {ok}")
        if not ok:
            raise SystemExit("verification failed")

    counts = np.bincount(ds.labels[sel].astype(np.int64) * 1000
                         + (ds.snrs[sel].astype(np.int64) + 20) // 2)
    nz = counts[counts > 0]
    n_classes_out = len(np.unique(ds.labels[sel]))
    print(f"\ncells: {len(nz)} (expect "
          f"{n_classes_out * len(ds.unique_snrs)}), "
          f"each holding {nz.min()}..{nz.max()} frames")

    total = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"\nOK. {out_dir} is {total / 1e9:.2f} GB across "
          f"{len(list(out_dir.iterdir()))} files.")
    print("\nUpload the whole directory to Drive, then in the Colab cell set")
    print("    NPY_HINTS = ('<the directory name you uploaded>',) + NPY_HINTS")
    print(f"or just drop it at MyDrive/amc-data, which is already searched.")
    ds.close()


if __name__ == "__main__":
    main()

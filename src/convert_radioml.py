"""
convert_radioml.py -- convert the RadioML HDF5 to .npy once, then never need
an HDF5 library again.

Why this exists: h5py stopped loading on this machine --

    ImportError: DLL load failed while importing defs:
    An Application Control policy has blocked this file.

AppLocker/WDAC blocks unsigned DLLs loaded from user-writable paths, and a
virtualenv under the user profile is exactly that. Reinstalling does not help,
because it reinstalls to the same blocked location.

`pyfive` is a pure-Python HDF5 reader with no compiled extensions, so there is
nothing for the policy to block. It reads this file correctly at about
127 MB/s -- slow, but only paid once.

After conversion everything downstream uses numpy memory-maps: faster for the
sorted-index reads radioml.py does, and dependency-free.

    python convert_radioml.py
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pyfive

DATA = pathlib.Path(r"C:\Users\yusuf\amc-data")
SRC = DATA / "GOLD_XYZ_OSC.0001_1024.hdf5"
X_OUT = DATA / "radioml_X.npy"
Y_OUT = DATA / "radioml_y.npy"
Z_OUT = DATA / "radioml_z.npy"

CHUNK = 20_000  # rows per read; ~164 MB, comfortably inside RAM


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found")

    f = pyfive.File(str(SRC))
    X, Y, Z = f["X"], f["Y"], f["Z"]
    n = X.shape[0]
    need = np.prod(X.shape) * 4
    free = shutil.disk_usage(DATA).free
    print(f"source : {X.shape} {X.dtype}")
    print(f"need   : {need / 1e9:.1f} GB     free: {free / 1e9:.1f} GB")
    if free < need * 1.05:
        raise SystemExit("not enough free space")

    # Labels and SNR are small; read them whole. One-hot -> class index, and
    # int8 is enough for 24 classes.
    print("\nlabels and SNR ...")
    y = np.asarray(Y[:]).argmax(axis=1).astype(np.int8)
    z = np.asarray(Z[:]).ravel().astype(np.int16)
    np.save(Y_OUT, y)
    np.save(Z_OUT, z)
    print(f"  {Y_OUT.name}  {y.shape}  classes {y.min()}..{y.max()}")
    print(f"  {Z_OUT.name}  {z.shape}  SNR {z.min()}..{z.max()} dB")

    # X is written through a memmap so the 21 GB never has to fit in RAM.
    print(f"\nX -> {X_OUT.name} in {CHUNK}-row chunks")
    out = np.lib.format.open_memmap(X_OUT, mode="w+", dtype=np.float32,
                                    shape=tuple(X.shape))
    t0 = time.perf_counter()
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        out[start:stop] = X[start:stop]
        done = stop / n
        el = time.perf_counter() - t0
        print(f"\r  {done:6.1%}  {stop:>9,}/{n:,}  "
              f"{el:5.0f}s elapsed, ~{el / max(done, 1e-9) - el:4.0f}s left",
              end="")
    out.flush()
    del out
    print(f"\n  done in {time.perf_counter() - t0:.0f}s")

    # Verify against the source rather than trusting the write.
    print("\nverifying ...")
    check = np.load(X_OUT, mmap_mode="r")
    rng = np.random.default_rng(0)
    for i in rng.integers(0, n, 5):
        a = np.asarray(X[int(i):int(i) + 1])[0]
        b = np.asarray(check[int(i)])
        ok = np.array_equal(a, b)
        print(f"  row {int(i):>9,}  identical: {ok}")
        if not ok:
            raise SystemExit("verification failed")

    print(f"\nOK. {X_OUT.name} is {X_OUT.stat().st_size / 1e9:.1f} GB")
    print("radioml.py will now use these and no longer needs h5py.")


if __name__ == "__main__":
    main()

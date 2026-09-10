"""
radioml.py -- loader for DeepSig RadioML 2018.01A (GOLD_XYZ_OSC.0001_1024.hdf5).

The file is ~21 GB uncompressed:

    X  (2555904, 1024, 2) float32   I/Q, last axis is [I, Q]
    Y  (2555904, 24)      one-hot class labels
    Z  (2555904, 1)       SNR in dB

2555904 = 24 classes x 26 SNR levels x 4096 frames.

Design notes
------------
This machine has 64 GB of RAM, so the whole array *would* fit -- but loading it
for every experiment wastes minutes per run and leaves nothing for the GPU
pipeline. The loader therefore reads Y and Z in full (about 250 MB, cheap) to
build an index, then pulls only the requested frames from X.

Subsetting is by (class, SNR) cell, and the train/test split is applied
*within* each cell so both sides cover the whole SNR range. Splitting any other
way silently changes the question you are answering.
"""

from __future__ import annotations

import pathlib

import numpy as np

# h5py is deliberately NOT imported at module level. On this machine an
# Application Control policy blocks its DLLs (they live in a user-writable
# virtualenv, which is what AppLocker/WDAC refuses), so importing it here would
# break every script that touches this module. The .npy path below needs no
# HDF5 library at all; see convert_radioml.py.

# Where to look for the file, in order.
SEARCH_DIRS = (
    pathlib.Path(r"C:\Users\yusuf\amc-data"),
    pathlib.Path.home() / "Downloads",
    pathlib.Path(r"C:\Users\yusuf\OneDrive\Masaüstü\Claude Code\amc-data"),
)

FILENAME = "GOLD_XYZ_OSC.0001_1024.hdf5"

# Canonical class order from O'Shea et al., "Over-the-Air Deep Learning Based
# Radio Signal Classification" (2018). The file stores labels as one-hot only,
# with no embedded names, so this list supplies them -- verify it against the
# paper before quoting class names in the report.
CLASSES: tuple[str, ...] = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
)


def find_file() -> pathlib.Path:
    """Locate the HDF5 file, with a useful error if it is not there yet."""
    for directory in SEARCH_DIRS:
        candidate = directory / FILENAME
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(d) for d in SEARCH_DIRS)
    raise FileNotFoundError(
        f"{FILENAME} not found. Looked in:\n  {searched}\n"
        "Extract the downloaded .zip and put the .hdf5 in one of these."
    )


class RadioML:
    """Indexed access to the dataset without loading all 21 GB."""

    def __init__(self, path: pathlib.Path | None = None,
                 search_dir: pathlib.Path | str | None = None):
        """
        Prefers the converted .npy files, which are memory-mapped and need no
        HDF5 library. Falls back to reading the original HDF5 through pyfive
        (pure Python, so nothing for the security policy to block) and finally
        to h5py if it happens to work.

        `search_dir` is tried before SEARCH_DIRS. It exists so the dataset can
        live somewhere this module has no business knowing about -- a mounted
        Drive in Colab, a staging directory on a runner's local disk -- without
        editing a module-level constant per environment. Everything else is
        unchanged, so passing nothing behaves exactly as before.
        """
        self._file = None
        npy = self._find_npy(search_dir)
        if npy is not None:
            x_path, y_path, z_path = npy
            self.path = x_path
            self.X = np.load(x_path, mmap_mode="r")
            self.labels = np.load(y_path).astype(np.int64)
            self.snrs = np.load(z_path).astype(np.int16)
            self.backend = "npy (memmap)"
        else:
            self.path = path or find_file()
            self._file, self.backend = self._open_hdf5(self.path)
            self.X = self._file["X"]
            self.labels = np.asarray(self._file["Y"]).argmax(axis=1)
            self.snrs = np.asarray(self._file["Z"]).ravel().astype(np.int16)

        self.unique_snrs = np.unique(self.snrs)
        self.n_classes = int(self.labels.max()) + 1

    @staticmethod
    def _find_npy(search_dir=None):
        directories = list(SEARCH_DIRS)
        if search_dir is not None:
            directories.insert(0, pathlib.Path(search_dir))
        for directory in directories:
            x = directory / "radioml_X.npy"
            y = directory / "radioml_y.npy"
            z = directory / "radioml_z.npy"
            if x.exists() and y.exists() and z.exists():
                return x, y, z
        return None

    @staticmethod
    def _open_hdf5(path):
        try:
            import pyfive

            return pyfive.File(str(path)), "pyfive (pure python, slow)"
        except Exception:
            import h5py

            return h5py.File(str(path), "r"), "h5py"

    def close(self) -> None:
        if self._file is not None and hasattr(self._file, "close"):
            self._file.close()

    def __enter__(self) -> "RadioML":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def cell_indices(self, class_id: int, snr: int) -> np.ndarray:
        """Row indices for one (class, SNR) cell."""
        return np.flatnonzero((self.labels == class_id) & (self.snrs == snr))

    def load(
        self,
        classes: list[int] | None = None,
        snrs: list[int] | None = None,
        frames_per_cell: int | None = None,
        test_fraction: float = 0.3,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """
        Pull a subset into memory, already split into train and test.

        frames_per_cell=None uses all 4096 frames per cell. During development
        use something like 512: the pipeline behaves identically and epochs
        take seconds instead of minutes.

        Returns arrays shaped (n, 1024, 2) float32 plus labels and SNRs.
        """
        rng = np.random.default_rng(seed)
        classes = list(range(self.n_classes)) if classes is None else classes
        snrs = list(self.unique_snrs) if snrs is None else snrs

        train_idx, test_idx = [], []
        for class_id in classes:
            for snr in snrs:
                idx = self.cell_indices(class_id, int(snr))
                if frames_per_cell is not None and len(idx) > frames_per_cell:
                    idx = rng.choice(idx, frames_per_cell, replace=False)
                idx = rng.permutation(idx)
                cut = int(len(idx) * (1.0 - test_fraction))
                train_idx.append(idx[:cut])
                test_idx.append(idx[cut:])

        train_idx = np.sort(np.concatenate(train_idx))  # sorted -> fast h5py reads
        test_idx = np.sort(np.concatenate(test_idx))

        return {
            "X_train": self.X[train_idx],
            "y_train": self.labels[train_idx],
            "z_train": self.snrs[train_idx],
            "X_test": self.X[test_idx],
            "y_test": self.labels[test_idx],
            "z_test": self.snrs[test_idx],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def to_complex(frames: np.ndarray) -> np.ndarray:
        """(n, 1024, 2) real -> (n, 1024) complex, for the feature pipeline."""
        return frames[..., 0] + 1j * frames[..., 1]


if __name__ == "__main__":
    with RadioML() as ds:
        print(f"file:     {ds.path}")
        print(f"X shape:  {ds.X.shape}  dtype {ds.X.dtype}")
        print(f"classes:  {ds.n_classes}")
        print(f"SNRs:     {ds.unique_snrs.min()} .. {ds.unique_snrs.max()} dB "
              f"({len(ds.unique_snrs)} levels, step "
              f"{np.diff(ds.unique_snrs)[0] if len(ds.unique_snrs) > 1 else '?'})")

        counts = np.bincount(ds.labels)
        print(f"\nframes per class: min {counts.min()}, max {counts.max()}")
        cell = len(ds.cell_indices(0, int(ds.unique_snrs[0])))
        print(f"frames per (class, SNR) cell: {cell}")

        expected = ds.n_classes * len(ds.unique_snrs) * cell
        print(f"\n{ds.n_classes} x {len(ds.unique_snrs)} x {cell} = {expected}"
              f"   (actual {ds.X.shape[0]})  "
              f"{'OK' if expected == ds.X.shape[0] else 'MISMATCH'}")

        if ds.n_classes == len(CLASSES):
            print(f"\nclass names assumed: {', '.join(CLASSES)}")
        else:
            print(f"\nWARNING: file has {ds.n_classes} classes but CLASSES lists "
                  f"{len(CLASSES)} -- do not trust the names.")

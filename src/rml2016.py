"""
rml2016.py -- loader for DeepSig RML2016.10a (RML2016.10a_dict.pkl).

The 2016 file is a pickled dict, not HDF5:

    {(modulation, snr): ndarray (1000, 2, 128) float32}

    11 modulations x 20 SNR levels (-20..18 dB, step 2) x 1000 frames
    = 220,000 frames, about 225 MB in float32.

Design notes
------------
Two things differ from `radioml.py` and both matter downstream:

  1. **Frame length is 128, not 1024.** `model_zoo.ICRNNA` is length-agnostic
     -- conv, pool, LSTM over time and attention pooling over time all are --
     so the same module takes both with an identical parameter count. Nothing
     else in the pipeline assumes 1024 either, except `cnn.FRAME_LEN`, which is
     only used by the synthetic generator.

  2. **The pickle stores (frames, 2, 128), already channel-first**, where the
     2018 HDF5 stores (frames, 1024, 2). `load()` below transposes to the 2018
     convention, (n, 128, 2), so that this class is interface-compatible with
     `radioml.RadioML` and callers do not have to know which dataset they got.
     Every caller then applies the same `np.transpose(X, (0, 2, 1))` it already
     applies to 2018 data.

The whole file fits in RAM comfortably, so there is no memmap path and no
conversion step -- pickle needs no HDF5 library, which is also why the
Application Control policy that blocks h5py on this machine is irrelevant here.

Class names are read from the file rather than assumed. EXPECTED_CLASSES is a
cross-check, not the source of truth.
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np

# Where to look, in order. The Colab Drive path is included because the .pkl
# currently lives there rather than on the local machine.
SEARCH_DIRS = (
    pathlib.Path(r"C:\Users\yusuf\amc-data"),
    pathlib.Path.home() / "Downloads",
    pathlib.Path("/content/drive/MyDrive"),
    pathlib.Path("/content/drive/MyDrive/RadioML"),
)

FILENAME = "RML2016.10a_dict.pkl"

# What the standard 10a file contains, sorted as `sorted(set(...))` returns it.
# Checked against the file at load time; a mismatch is reported rather than
# silently accepted, because a wrong name list means wrong class labels in
# every figure downstream.
EXPECTED_CLASSES: tuple[str, ...] = (
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4",
    "QAM16", "QAM64", "QPSK", "WBFM",
)

# 2016 spells several classes differently from 2018, and one class has no 2018
# counterpart at all. Kept here so that anyone putting the two datasets on the
# same axis has to look at the gaps rather than discover them by accident.
#
#   - QAM16/QAM64 are 16QAM/64QAM under another name: exact correspondence.
#   - PAM4 vs 4ASK, GFSK vs GMSK, WBFM vs FM are *same family, not the same
#     modulation* -- usable only with the mismatch declared. `domains.ALIASES`
#     carries the same caveat for the synthetic domain.
#   - CPFSK has no 2018 class. There is nothing to map it to.
TO_2018: dict[str, str | None] = {
    "8PSK": "8PSK",
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "QAM16": "16QAM",
    "QAM64": "64QAM",
    "PAM4": "4ASK",          # both 4-level amplitude, different level spacing
    "GFSK": "GMSK",          # both Gaussian-filtered CPM, different mod index
    "WBFM": "FM",            # analog FM, different deviation
    "AM-DSB": "AM-DSB-SC",
    "AM-SSB": "AM-SSB-SC",
    "CPFSK": None,           # no counterpart in 2018.01A
}

# The five that mean exactly the same thing in 2016 and 2018. Anything wider
# than this needs the mismatch stated in the write-up.
EXACT_SHARED: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "QAM16", "QAM64")


def find_file(path: pathlib.Path | str | None = None) -> pathlib.Path:
    """Locate the pickle, with a useful error if it is not there yet."""
    if path is not None:
        candidate = pathlib.Path(path)
        if candidate.is_dir():
            candidate = candidate / FILENAME
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{candidate} does not exist")

    for directory in SEARCH_DIRS:
        candidate = directory / FILENAME
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(d) for d in SEARCH_DIRS)
    raise FileNotFoundError(
        f"{FILENAME} not found. Looked in:\n  {searched}\n"
        "Download it from DeepSig (RML2016.10a.tar.bz2, ~600 MB) and put the "
        ".pkl in one of these, or pass --data-path."
    )


class RML2016:
    """
    Indexed access to RML2016.10a, interface-compatible with radioml.RadioML.

    The whole array is held in memory (225 MB), so unlike the 2018 loader there
    is nothing to stream and `X` is a plain ndarray rather than a memmap.
    """

    def __init__(self, path: pathlib.Path | str | None = None):
        self.path = find_file(path)

        with open(self.path, "rb") as handle:
            raw = pickle.load(handle, encoding="latin1")

        self.classes: tuple[str, ...] = tuple(sorted({k[0] for k in raw}))
        index = {name: i for i, name in enumerate(self.classes)}

        frames, labels, snrs = [], [], []
        for (mod, snr), block in sorted(raw.items()):
            block = np.asarray(block, dtype=np.float32)   # (n, 2, 128)
            frames.append(np.transpose(block, (0, 2, 1)))  # -> (n, 128, 2)
            labels.append(np.full(len(block), index[mod], dtype=np.int64))
            snrs.append(np.full(len(block), snr, dtype=np.int16))

        self.X = np.concatenate(frames)
        self.labels = np.concatenate(labels)
        self.snrs = np.concatenate(snrs)
        self.backend = "pickle (in memory)"

        self.unique_snrs = np.unique(self.snrs)
        self.n_classes = len(self.classes)
        self.frame_len = self.X.shape[1]

    # ------------------------------------------------------------------
    # Present for symmetry with radioml.RadioML, which holds an open handle.
    def close(self) -> None:
        return None

    def __enter__(self) -> "RML2016":
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
        Pull a subset, already split into train and test.

        Signature and return keys match `radioml.RadioML.load` exactly, so a
        caller can swap one loader for the other. Subsetting is by (class, SNR)
        cell and the split is applied *within* each cell, so both sides cover
        the whole SNR range -- splitting any other way silently changes the
        question being answered.

        Returns arrays shaped (n, 128, 2) float32 plus labels and SNRs.
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

        train_idx = np.sort(np.concatenate(train_idx))
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
        """(n, 128, 2) real -> (n, 128) complex, for the feature pipeline."""
        return frames[..., 0] + 1j * frames[..., 1]


if __name__ == "__main__":
    with RML2016() as ds:
        print(f"file:     {ds.path}")
        print(f"X shape:  {ds.X.shape}  dtype {ds.X.dtype}  "
              f"({ds.X.nbytes / 1e6:.0f} MB in RAM)")
        print(f"classes:  {ds.n_classes}: {', '.join(ds.classes)}")
        print(f"SNRs:     {ds.unique_snrs.min()} .. {ds.unique_snrs.max()} dB "
              f"({len(ds.unique_snrs)} levels)")

        cell = len(ds.cell_indices(0, int(ds.unique_snrs[0])))
        expected = ds.n_classes * len(ds.unique_snrs) * cell
        print(f"frames per (class, SNR) cell: {cell}")
        print(f"\n{ds.n_classes} x {len(ds.unique_snrs)} x {cell} = {expected}"
              f"   (actual {ds.X.shape[0]})  "
              f"{'OK' if expected == ds.X.shape[0] else 'MISMATCH'}")

        if ds.classes == EXPECTED_CLASSES:
            print("\nclass names match EXPECTED_CLASSES.")
        else:
            print(f"\nWARNING: file lists {ds.classes},\n"
                  f"         expected {EXPECTED_CLASSES}.\n"
                  "         Check TO_2018 before comparing against 2018.01A.")

        unmapped = [c for c in ds.classes if TO_2018.get(c) is None]
        print(f"no 2018 counterpart: {unmapped or 'none'}")

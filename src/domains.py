"""
domains.py -- one interface for every source of labelled IQ.

The project's question is "how much accuracy is lost when the test data comes
from somewhere else?", so *source* has to be a first-class concept rather than
something hardcoded in each script.

A Domain supplies labelled 1024-sample complex baseband frames tagged with SNR.
Three exist:

    RadioMLDomain     the DeepSig benchmark file
    SyntheticDomain   modem.py, parameterised -- several distinct domains
                      can be made by varying pulse shaping and impairments
    CaptureDomain     .npz files written by an SDR capture session

CaptureDomain is written now, before any hardware exists, so that the on-disk
format is fixed. When lab access happens the capture script only has to write
files in this layout and every downstream experiment runs unchanged.

Class vocabulary
----------------
Domains disagree about names. Five modulations map exactly across all of them
(BPSK, QPSK, 8PSK, 16QAM, 64QAM); those are the default shared set. ALIASES
holds looser correspondences that are usable with a stated caveat -- e.g.
GFSK and GMSK are both Gaussian-filtered CPM but not the same modulation, so
including them means saying so in the report.
"""

from __future__ import annotations

import abc
import pathlib

import numpy as np

# Modulations that mean the same thing in every domain. Safe to compare.
SHARED_CLASSES: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM")

# Looser correspondences: same family, not identical. Using these widens the
# experiment from 5 to 10 classes but the mismatch must be declared.
ALIASES: dict[str, str] = {
    "PAM4": "4ASK",          # both 4-level amplitude, different level spacing
    "GFSK": "GMSK",          # both Gaussian-filtered CPM, different mod index
    "WBFM": "FM",            # analog FM, different deviation
    "AM-DSB": "AM-DSB-SC",   # ours is suppressed-carrier
    "AM-SSB": "AM-SSB-SC",
}


class Domain(abc.ABC):
    """A named source of labelled IQ frames."""

    name: str

    @property
    @abc.abstractmethod
    def available_classes(self) -> tuple[str, ...]:
        """Canonical class names this domain can produce."""

    @abc.abstractmethod
    def load(
        self,
        classes: list[str],
        snrs: list[int],
        frames_per_cell: int,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """
        Returns:
            X : (n, 2, 1024) float32, unit average power per frame
            y : (n,) int64, indices into `classes`
            z : (n,) int16, SNR in dB
        """

    # ------------------------------------------------------------------
    @staticmethod
    def _finalize(frames: np.ndarray) -> np.ndarray:
        """(n, 1024) complex -> (n, 2, 1024) float32 at unit average power."""
        X = np.stack([frames.real, frames.imag], axis=1).astype(np.float32)
        power = np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1, keepdims=True)
        return X / (np.sqrt(power)[:, None] + 1e-12)


# ==========================================================================


class RadioMLDomain(Domain):
    """DeepSig RadioML 2018.01A."""

    name = "radioml"

    def __init__(self):
        import radioml

        self._module = radioml
        self._ds = radioml.RadioML()
        self._index = {name: i for i, name in enumerate(radioml.CLASSES)}

    @property
    def available_classes(self) -> tuple[str, ...]:
        return self._module.CLASSES

    def load(self, classes, snrs, frames_per_cell, seed=0):
        ids = [self._index[self._to_native(c)] for c in classes]
        data = self._ds.load(
            classes=ids, snrs=snrs, frames_per_cell=frames_per_cell,
            test_fraction=0.0, seed=seed,
        )
        frames = self._module.RadioML.to_complex(data["X_train"])

        # Remap RadioML's global class ids onto positions in `classes`.
        remap = {native_id: pos for pos, native_id in enumerate(ids)}
        y = np.array([remap[v] for v in data["y_train"]], dtype=np.int64)

        return {"X": self._finalize(frames), "y": y,
                "z": data["z_train"].astype(np.int16)}

    @staticmethod
    def _to_native(canonical: str) -> str:
        return ALIASES.get(canonical, canonical)

    def close(self):
        self._ds.close()


class SyntheticDomain(Domain):
    """
    modem.py, parameterised.

    Varying the constructor arguments produces genuinely *different* domains,
    which is the cheap way to study generalization with no hardware: train on
    one set of transmitter parameters, test on another.
    """

    def __init__(self, name="synthetic", sps=8, beta=0.35,
                 cfo_norm=0.0, iq_gain_db=0.0, iq_phase_deg=0.0):
        import modem

        self._modem = modem
        self.name = name
        self.sps = sps
        self.beta = beta
        self.cfo_norm = cfo_norm
        self.iq_gain_db = iq_gain_db
        self.iq_phase_deg = iq_phase_deg

    @property
    def available_classes(self) -> tuple[str, ...]:
        return self._modem.MODULATIONS

    def load(self, classes, snrs, frames_per_cell, seed=0):
        rng = np.random.default_rng(seed)
        X, y, z = [], [], []

        for pos, canonical in enumerate(classes):
            native = self._to_native(canonical)
            for snr in snrs:
                frames = np.stack([
                    self._generate(native, float(snr), rng)
                    for _ in range(frames_per_cell)
                ])
                X.append(self._finalize(frames))
                y.append(np.full(frames_per_cell, pos, dtype=np.int64))
                z.append(np.full(frames_per_cell, snr, dtype=np.int16))

        return {"X": np.concatenate(X), "y": np.concatenate(y),
                "z": np.concatenate(z)}

    def _generate(self, native: str, snr_db: float, rng) -> np.ndarray:
        x = self._modem.generate(
            native, 1024, snr_db=snr_db, sps=self.sps, rng=rng,
            cfo_norm=self.cfo_norm, beta=self.beta,
        )
        if self.iq_gain_db or self.iq_phase_deg:
            x = self._modem.iq_imbalance(x, self.iq_gain_db, self.iq_phase_deg)
        return x

    @staticmethod
    def _to_native(canonical: str) -> str:
        reverse = {v: k for k, v in ALIASES.items()}
        return reverse.get(canonical, canonical)


class CaptureDomain(Domain):
    """
    Real SDR captures. No hardware yet -- this fixes the format in advance.

    Expected layout, one directory per capture session::

        <root>/
            2026-09-03_pluto_cable/
                BPSK_10dB.npz
                BPSK_04dB.npz
                QPSK_10dB.npz
                ...
            2026-09-10_pluto_cable/
                ...

    Each .npz holds a single array ``iq`` of shape (n_frames, 1024), complex64.

    **The directory is the session, and the session is the split unit.** Frames
    from one recording share thermal state, cable position and oscillator
    drift; splitting inside a session leaks that state into the test set and
    inflates accuracy for free. `sessions()` exists so experiments can hold out
    whole sessions.
    """

    name = "capture"

    def __init__(self, root: pathlib.Path | str = r"C:\Users\yusuf\amc-data\captures"):
        self.root = pathlib.Path(root)

    def sessions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    @property
    def available_classes(self) -> tuple[str, ...]:
        names = {f.stem.rsplit("_", 1)[0]
                 for s in self.sessions()
                 for f in (self.root / s).glob("*.npz")}
        return tuple(sorted(names))

    def load(self, classes, snrs, frames_per_cell, seed=0, sessions=None):
        available = self.sessions()
        if not available:
            raise FileNotFoundError(
                f"No capture sessions under {self.root}.\n"
                "Expected <root>/<session>/<MODULATION>_<snr>dB.npz, each "
                "holding an 'iq' array of shape (n, 1024) complex64.\n"
                "Until hardware exists, use SyntheticDomain as the second domain."
            )
        sessions = sessions or available
        rng = np.random.default_rng(seed)

        X, y, z = [], [], []
        for session in sessions:
            for pos, name in enumerate(classes):
                for snr in snrs:
                    path = self.root / session / f"{name}_{int(snr):02d}dB.npz"
                    if not path.exists():
                        continue
                    frames = np.load(path)["iq"]
                    if len(frames) > frames_per_cell:
                        frames = frames[rng.choice(len(frames), frames_per_cell,
                                                   replace=False)]
                    X.append(self._finalize(frames))
                    y.append(np.full(len(frames), pos, dtype=np.int64))
                    z.append(np.full(len(frames), snr, dtype=np.int16))

        if not X:
            raise ValueError(f"No frames matched classes={classes} snrs={snrs}")
        return {"X": np.concatenate(X), "y": np.concatenate(y),
                "z": np.concatenate(z)}


# ==========================================================================


def shared_classes(*domains: Domain, include_aliases: bool = False) -> list[str]:
    """Canonical class names every given domain can produce."""
    candidates = list(SHARED_CLASSES)
    if include_aliases:
        candidates += list(ALIASES.keys())

    out = []
    for canonical in candidates:
        if all(
            canonical in d.available_classes
            or ALIASES.get(canonical, "") in d.available_classes
            or {v: k for k, v in ALIASES.items()}.get(canonical, "") in d.available_classes
            for d in domains
        ):
            out.append(canonical)
    return out


if __name__ == "__main__":
    synth = SyntheticDomain()
    print(f"{'domain':<12} classes")
    print(f"{synth.name:<12} {len(synth.available_classes)}: "
          f"{', '.join(synth.available_classes)}")

    cap = CaptureDomain()
    print(f"{cap.name:<12} sessions found: {cap.sessions() or 'none yet'}")

    try:
        rml = RadioMLDomain()
        print(f"{rml.name:<12} {len(rml.available_classes)}: "
              f"{', '.join(rml.available_classes[:8])}, ...")
        print(f"\nshared (strict):  {shared_classes(rml, synth)}")
        print(f"shared (aliases): {shared_classes(rml, synth, include_aliases=True)}")
        rml.close()
    except FileNotFoundError as exc:
        print(f"radioml     unavailable: {exc}")

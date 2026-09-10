"""
features.py -- hand-designed features for the classical AMC baseline.

This is what the field used before deep learning, and it is the yardstick the
CNN has to beat. Two families:

  1. Higher-order cumulants (Swami & Sadler, IEEE Trans. Comm. 2000).
     Cumulants of order >2 vanish for Gaussian processes, so they suppress
     AWGN while still separating constellations. They are the reason a
     hand-crafted classifier works at all.

  2. Instantaneous amplitude / phase / frequency statistics
     (Azzouz & Nandi). These carry the analog-vs-digital and
     constant-envelope-vs-not distinctions that cumulants handle poorly.

Everything here is scale invariant: cumulants are normalized by C21^(p/2) and
the instantaneous features work on mean-normalized quantities. A classifier
must not be able to win by reading absolute power.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Higher-order cumulants
# --------------------------------------------------------------------------


def _moments(x: np.ndarray) -> dict[str, complex]:
    """
    Sample moments M_pq = E[ x^(p-q) * conj(x)^q ].

    The signal is centered first: cumulant formulas below assume zero mean,
    and AM-DSB in particular has a large DC term that would otherwise leak
    into every higher-order statistic.
    """
    x = x - np.mean(x)
    xc = np.conj(x)
    return {
        "M20": np.mean(x**2),
        "M21": np.mean(x * xc),  # = E[|x|^2], real and positive
        "M40": np.mean(x**4),
        "M41": np.mean(x**3 * xc),
        "M42": np.mean(np.abs(x) ** 4),
        "M60": np.mean(x**6),
    }


def cumulants(x: np.ndarray) -> dict[str, float]:
    """
    Normalized cumulant magnitudes.

    Normalizing C_pq by C21^(p/2) makes each value independent of signal
    amplitude, so the same feature vector describes a signal at any gain.
    Magnitudes are taken because a constant phase rotation of the whole frame
    (which any real receiver introduces) rotates the complex cumulants but
    leaves their magnitudes alone.
    """
    m = _moments(x)

    c20 = m["M20"]
    c21 = np.real(m["M21"])  # real by construction
    c40 = m["M40"] - 3.0 * m["M20"] ** 2
    c41 = m["M41"] - 3.0 * m["M20"] * m["M21"]
    c42 = m["M42"] - np.abs(m["M20"]) ** 2 - 2.0 * m["M21"] ** 2
    c60 = m["M60"] - 15.0 * m["M20"] * m["M40"] + 30.0 * m["M20"] ** 3

    p = c21 + 1e-12
    return {
        "|C20|": float(np.abs(c20) / p),
        "|C40|": float(np.abs(c40) / p**2),
        "|C41|": float(np.abs(c41) / p**2),
        "|C42|": float(np.abs(c42) / p**2),
        "|C60|": float(np.abs(c60) / p**3),
    }


# --------------------------------------------------------------------------
# Instantaneous amplitude / phase / frequency statistics
# --------------------------------------------------------------------------


def instantaneous(x: np.ndarray) -> dict[str, float]:
    """
    Classical Azzouz-Nandi style features.

    The intuition worth keeping:
      - constant-envelope schemes (FSK, FM) have near-zero amplitude variance
      - analog schemes have smooth, slowly-varying phase
      - digital schemes have phase that jumps at symbol boundaries
    """
    amp = np.abs(x)
    mean_amp = np.mean(amp) + 1e-12

    # Normalized-centered instantaneous amplitude
    a_cn = amp / mean_amp - 1.0

    # gamma_max: peak of the spectral density of a_cn. Large when the envelope
    # carries a periodic message (AM), small for constant-envelope schemes.
    spec = np.abs(np.fft.fft(a_cn)) ** 2
    gamma_max = float(np.max(spec[1:]) / len(a_cn))  # skip DC

    # Phase, with the linear (carrier-offset) trend removed so that a residual
    # frequency error does not masquerade as modulation structure.
    phase = np.unwrap(np.angle(x))
    n = np.arange(len(phase))
    slope, intercept = np.polyfit(n, phase, 1)
    phase_nl = phase - (slope * n + intercept)

    # Instantaneous frequency, normalized
    freq = np.diff(phase) / (2.0 * np.pi)
    freq = freq - np.mean(freq)
    freq_n = freq / (np.std(freq) + 1e-12)

    def kurtosis(v: np.ndarray) -> float:
        v = v - np.mean(v)
        var = np.mean(v**2) + 1e-12
        return float(np.mean(v**4) / var**2)

    return {
        "gamma_max": gamma_max,
        "sigma_aa": float(np.std(np.abs(a_cn))),
        "sigma_dp": float(np.std(phase_nl)),
        "sigma_ap": float(np.std(np.abs(phase_nl))),
        "sigma_af": float(np.std(np.abs(freq_n))),
        "kurt_amp": kurtosis(a_cn),
        "kurt_freq": kurtosis(freq),
        # Fraction of energy in the upper half of the spectrum minus the lower
        # half -- a direct handle on sideband asymmetry, which is what makes
        # AM-SSB different from AM-DSB.
        "spectral_asym": _spectral_asymmetry(x),
    }


def _spectral_asymmetry(x: np.ndarray) -> float:
    """Signed imbalance between positive- and negative-frequency energy."""
    spec = np.abs(np.fft.fft(x)) ** 2
    half = len(spec) // 2
    upper = np.sum(spec[1:half])
    lower = np.sum(spec[half:])
    return float((upper - lower) / (upper + lower + 1e-12))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

FEATURE_NAMES: tuple[str, ...] = (
    "|C20|",
    "|C40|",
    "|C41|",
    "|C42|",
    "|C60|",
    "gamma_max",
    "sigma_aa",
    "sigma_dp",
    "sigma_ap",
    "sigma_af",
    "kurt_amp",
    "kurt_freq",
    "spectral_asym",
)


def extract(x: np.ndarray) -> np.ndarray:
    """Feature vector for one complex baseband frame, ordered by FEATURE_NAMES."""
    values = {**cumulants(x), **instantaneous(x)}
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract_batch(frames: np.ndarray) -> np.ndarray:
    """Feature matrix for an (n_frames, n_samples) complex array."""
    return np.stack([extract(f) for f in frames])


if __name__ == "__main__":
    import modem

    # Theory check: |C40| should be near 1.0 for BPSK, 1.0 for QPSK
    # (real-valued negative), and clearly smaller for 16QAM/64QAM. Constant
    # envelope schemes should show sigma_aa near zero.
    rng = np.random.default_rng(0)
    print(f"{'modulation':<9} " + " ".join(f"{n:>10}" for n in FEATURE_NAMES[:6]))
    for name in modem.MODULATIONS:
        frames = np.stack(
            [modem.generate(name, 4096, snr_db=25, rng=rng) for _ in range(30)]
        )
        f = extract_batch(frames).mean(axis=0)
        print(f"{name:<9} " + " ".join(f"{v:>10.3f}" for v in f[:6]))

"""
modem.py -- baseband IQ generator for Automatic Modulation Classification (AMC).

Why this exists
---------------
Week 1 goal is to *see* what the modulation classes look like before touching
any classifier. Generating the signals ourselves (instead of only downloading
RadioML) has two payoffs:

  1. Intuition. You control SNR, pulse shaping and impairments one at a time,
     so you learn which visual feature comes from which physical effect.
  2. Later, this becomes a second, independent "domain". The core project
     question is how badly a model trained on one domain fails on another --
     and for that you need a domain you fully control.

Everything here is complex baseband: x[n] = I[n] + j*Q[n].

Conventions
-----------
sps   : samples per symbol (oversampling factor)
beta  : root-raised-cosine roll-off
snr_db: Es/N0 in dB, applied over the whole complex signal
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Constellations (unit average energy)
# --------------------------------------------------------------------------


def _normalize(points: np.ndarray) -> np.ndarray:
    """Scale a constellation so its average symbol energy is 1."""
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def _qam_square(order: int) -> np.ndarray:
    """Square QAM constellation, e.g. 16 -> 4x4 grid, 64 -> 8x8 grid."""
    side = int(np.sqrt(order))
    if side * side != order:
        raise ValueError(f"{order}-QAM is not a square constellation")
    levels = np.arange(-(side - 1), side, 2)  # e.g. [-3, -1, 1, 3]
    re, im = np.meshgrid(levels, levels)
    return _normalize((re + 1j * im).ravel())


def _psk(order: int) -> np.ndarray:
    k = np.arange(order)
    return np.exp(1j * 2 * np.pi * k / order)


CONSTELLATIONS: dict[str, np.ndarray] = {
    "BPSK": _psk(2),
    "QPSK": _psk(4) * np.exp(1j * np.pi / 4),  # rotated to the classic diamond
    "8PSK": _psk(8),
    "16QAM": _qam_square(16),
    "64QAM": _qam_square(64),
    "PAM4": _normalize(np.array([-3.0, -1.0, 1.0, 3.0], dtype=complex)),
}


# --------------------------------------------------------------------------
# Pulse shaping
# --------------------------------------------------------------------------


def rrc_filter(span: int, sps: int, beta: float) -> np.ndarray:
    """
    Root-raised-cosine FIR taps, unit energy.

    span : filter length in symbol periods (use 8-12 in practice)
    sps  : samples per symbol
    beta : roll-off in [0, 1]. beta=0 is a brick wall (long ringing),
           beta=1 is very smooth but uses twice the bandwidth.

    The closed form has removable singularities at t=0 and t=+-1/(4*beta),
    so those taps are filled in with their analytic limits.
    """
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps  # time axis in symbol periods
    h = np.zeros_like(t)

    at_zero = np.isclose(t, 0.0)
    h[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)

    if beta > 0:
        at_sing = np.isclose(np.abs(t), 1.0 / (4.0 * beta))
        h[at_sing] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
    else:
        at_sing = np.zeros_like(t, dtype=bool)

    rest = ~(at_zero | at_sing)
    tr = t[rest]
    num = np.sin(np.pi * tr * (1 - beta)) + 4 * beta * tr * np.cos(np.pi * tr * (1 + beta))
    den = np.pi * tr * (1.0 - (4.0 * beta * tr) ** 2)
    h[rest] = num / den

    return h / np.sqrt(np.sum(h**2))


def _upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Insert sps-1 zeros between symbols (impulse train for the shaping filter)."""
    out = np.zeros(len(symbols) * sps, dtype=complex)
    out[::sps] = symbols
    return out


# --------------------------------------------------------------------------
# Linear (memoryless) digital modulations
# --------------------------------------------------------------------------


def linear_digital(
    scheme: str,
    n_samples: int,
    sps: int = 8,
    beta: float = 0.35,
    span: int = 10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """PSK / QAM / PAM with root-raised-cosine pulse shaping."""
    rng = rng or np.random.default_rng()
    points = CONSTELLATIONS[scheme]

    # Generate a few extra symbols so filter transients can be trimmed off.
    n_symbols = n_samples // sps + 2 * span
    symbols = points[rng.integers(0, len(points), n_symbols)]

    shaped = np.convolve(_upsample(symbols, sps), rrc_filter(span, sps, beta), mode="full")
    start = span * sps  # discard the filter ramp-up
    return shaped[start : start + n_samples]


# --------------------------------------------------------------------------
# Continuous-phase modulations (they have memory -- constellation plots of
# these look like a ring, which is exactly the point worth seeing)
# --------------------------------------------------------------------------


def cpfsk(
    n_samples: int,
    sps: int = 8,
    mod_index: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Binary continuous-phase FSK. Constant envelope, phase never jumps."""
    rng = rng or np.random.default_rng()
    n_symbols = n_samples // sps + 2
    bits = rng.integers(0, 2, n_symbols) * 2 - 1  # +-1
    freq = np.repeat(bits, sps) * (np.pi * mod_index / sps)
    phase = np.cumsum(freq)
    return np.exp(1j * phase)[:n_samples]


def gfsk(
    n_samples: int,
    sps: int = 8,
    mod_index: float = 0.5,
    bt: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Gaussian FSK (what Bluetooth uses). Same as CPFSK but the symbol stream is
    smoothed by a Gaussian pulse first, which narrows the spectrum.
    """
    rng = rng or np.random.default_rng()
    n_symbols = n_samples // sps + 4
    bits = rng.integers(0, 2, n_symbols) * 2 - 1

    # Gaussian pulse spanning ~3 symbols
    span = 3
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-(t**2) / (2 * sigma**2))
    g /= np.sum(g)

    smoothed = np.convolve(np.repeat(bits, sps), g, mode="same")
    phase = np.cumsum(smoothed * np.pi * mod_index / sps)
    return np.exp(1j * phase)[:n_samples]


# --------------------------------------------------------------------------
# Analog modulations -- these need a "message" signal. Lowpass-filtered noise
# stands in for audio: it has the right bandwidth and randomness, and unlike a
# real audio file it is reproducible from a seed.
# --------------------------------------------------------------------------


def _message(n_samples: int, bandwidth: float, rng: np.random.Generator) -> np.ndarray:
    """Real-valued, zero-mean, unit-variance lowpass process ("audio")."""
    noise = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples)
    spectrum[freqs > bandwidth] = 0.0
    msg = np.fft.irfft(spectrum, n=n_samples)
    msg -= np.mean(msg)
    return msg / (np.std(msg) + 1e-12)


def am_dsb(n_samples: int, depth: float = 0.5, rng=None) -> np.ndarray:
    """Double-sideband AM with carrier. Envelope carries the message."""
    rng = rng or np.random.default_rng()
    return (1.0 + depth * _message(n_samples, 0.02, rng)).astype(complex)


def am_ssb(n_samples: int, depth: float = 0.5, rng=None) -> np.ndarray:
    """
    Single-sideband AM. Built as msg + j*hilbert(msg), which cancels one
    sideband -- visible as an asymmetric spectrum.
    """
    from scipy.signal import hilbert

    rng = rng or np.random.default_rng()
    msg = _message(n_samples, 0.02, rng)
    analytic = hilbert(msg)  # msg + j*Hilbert{msg}
    return 1.0 + depth * analytic


def wbfm(n_samples: int, deviation: float = 0.02, rng=None) -> np.ndarray:
    """
    Wideband FM. Constant envelope; the message rides in the phase, so the
    instantaneous frequency is proportional to the message amplitude.

    `deviation` is the peak frequency deviation normalized to the sample rate.
    With a message bandwidth of 0.01 this gives a modulation index near 2,
    i.e. genuinely *wide*band -- the spectrum is much wider than the message.
    """
    rng = rng or np.random.default_rng()
    msg = _message(n_samples, 0.01, rng)
    phase = 2 * np.pi * deviation * np.cumsum(msg)
    return np.exp(1j * phase)


# --------------------------------------------------------------------------
# Channel and hardware impairments
# --------------------------------------------------------------------------


def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex white Gaussian noise at the requested SNR (dB)."""
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    )
    return x + noise


def carrier_offset(x: np.ndarray, cfo_norm: float, phase: float = 0.0) -> np.ndarray:
    """
    Residual carrier frequency offset, normalized to the sample rate.
    This is one of the main reasons a model trained on one radio fails on
    another -- keep it in your back pocket for the generalization study.
    """
    n = np.arange(len(x))
    return x * np.exp(1j * (2 * np.pi * cfo_norm * n + phase))


def random_phase(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Uniform random absolute phase for the whole frame.

    Our generator is otherwise deterministic in phase: every frame comes out
    with the constellation in the same orientation. A real receiver has no idea
    what the absolute phase is, so every captured frame lands at a different
    rotation. This is the cheapest impairment to be missing and the easiest to
    overlook.
    """
    rng = rng or np.random.default_rng()
    return x * np.exp(1j * rng.uniform(0, 2 * np.pi))


def multipath(
    x: np.ndarray,
    rng: np.random.Generator | None = None,
    n_taps: int = 8,
    delay_spread: float = 2.0,
) -> np.ndarray:
    """
    Frequency-selective fading with an exponential power delay profile.

    Taps are complex Gaussian (Rayleigh), with average power decaying as
    exp(-k / delay_spread) over k = 0 .. n_taps-1. `delay_spread` is in
    samples, so at sps=8 a value of 2 is a quarter of a symbol -- enough to
    smear adjacent symbols into each other without destroying the signal.

    This is the one impairment RadioML documents that our generator had no
    equivalent for at all.
    """
    rng = rng or np.random.default_rng()
    profile = np.exp(-np.arange(n_taps) / max(delay_spread, 1e-6))
    profile /= profile.sum()
    taps = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps))
    taps *= np.sqrt(profile / 2.0)
    return np.convolve(x, taps, mode="full")[: len(x)]


def iq_imbalance(x: np.ndarray, gain_db: float = 0.0, phase_deg: float = 0.0) -> np.ndarray:
    """Analog front-end imperfection: I and Q not perfectly matched."""
    g = 10 ** (gain_db / 20.0)
    p = np.deg2rad(phase_deg)
    i = np.real(x)
    q = np.imag(x)
    return (i + 1j * g * (q * np.cos(p) + i * np.sin(p))).astype(complex)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

MODULATIONS: tuple[str, ...] = (
    "BPSK",
    "QPSK",
    "8PSK",
    "PAM4",
    "16QAM",
    "64QAM",
    "CPFSK",
    "GFSK",
    "AM-DSB",
    "AM-SSB",
    "WBFM",
)


def generate(
    scheme: str,
    n_samples: int = 1024,
    snr_db: float = 20.0,
    sps: int = 8,
    rng: np.random.Generator | None = None,
    cfo_norm: float = 0.0,
    beta: float = 0.35,
) -> np.ndarray:
    """
    One clean call: pick a modulation, get noisy complex baseband back.

    The output is normalized to unit average power *after* noise is added,
    matching how RadioML frames are scaled -- so a classifier cannot cheat by
    reading absolute amplitude.

    `beta` only affects the linearly modulated schemes (PSK/QAM/PAM), which are
    the ones that carry a pulse-shaping filter. Varying it turns this generator
    into a family of related-but-distinct domains, which is what the
    excess-bandwidth sweep in sweep.py needs.
    """
    rng = rng or np.random.default_rng()

    if scheme in CONSTELLATIONS:
        x = linear_digital(scheme, n_samples, sps=sps, beta=beta, rng=rng)
    elif scheme == "CPFSK":
        x = cpfsk(n_samples, sps=sps, rng=rng)
    elif scheme == "GFSK":
        x = gfsk(n_samples, sps=sps, rng=rng)
    elif scheme == "AM-DSB":
        x = am_dsb(n_samples, rng=rng)
    elif scheme == "AM-SSB":
        x = am_ssb(n_samples, rng=rng)
    elif scheme == "WBFM":
        x = wbfm(n_samples, rng=rng)
    else:
        raise ValueError(f"unknown modulation: {scheme!r}")

    if cfo_norm:
        x = carrier_offset(x, cfo_norm)

    x = awgn(x, snr_db, rng=rng)
    return x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)


if __name__ == "__main__":
    # Smoke test: every modulation should produce finite, unit-power output.
    rng = np.random.default_rng(0)
    print(f"{'modulation':<10} {'len':>6} {'mean |x|^2':>12} {'finite':>8}")
    for name in MODULATIONS:
        x = generate(name, n_samples=1024, snr_db=20, rng=rng)
        print(f"{name:<10} {len(x):>6} {np.mean(np.abs(x)**2):>12.4f} {str(np.all(np.isfinite(x))):>8}")

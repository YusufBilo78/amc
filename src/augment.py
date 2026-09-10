"""
augment.py -- GPU-side augmentations aimed at the diagnosed failure.

The point is not "more augmentation is better". Each transform here exists to
make the model invariant to one specific way the test domain can differ from
the training domain:

    resample        symbol-rate / oversampling mismatch  -> the prime suspect
                    behind 16QAM being read as 64QAM
    freq_offset     free-running oscillators between TX and RX
    phase_rotation  arbitrary absolute phase at the receiver
    time_shift      no symbol-timing alignment between frames
    spectral_tilt   non-flat analog front-end response

Anything not on that list is left out on purpose: augmentation that models no
real mechanism costs accuracy without buying robustness, and makes the ablation
impossible to interpret.

Everything runs on the batch tensor already sitting on the GPU, so the cost is
negligible next to the forward pass.

Tensors are (batch, 2, n_samples) with channel 0 = I and channel 1 = Q.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def _from_complex(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=1)


def _renormalize(x: torch.Tensor) -> torch.Tensor:
    """Unit average power per frame -- augmentation must not leak gain cues."""
    power = (x[:, 0] ** 2 + x[:, 1] ** 2).mean(dim=1, keepdim=True)
    return x / (power.sqrt().unsqueeze(1) + 1e-12)


# --------------------------------------------------------------------------


def phase_and_freq(x: torch.Tensor, max_cfo: float = 0.01) -> torch.Tensor:
    """
    Random absolute phase plus a random carrier frequency offset.

    max_cfo is normalized to the sample rate; 0.01 over a 1024-sample frame is
    about 10 cycles of rotation, which is a realistic residual offset for
    consumer SDR hardware after coarse correction.
    """
    batch, _, n = x.shape
    device = x.device
    phase = torch.rand(batch, 1, device=device) * 2 * torch.pi
    cfo = (torch.rand(batch, 1, device=device) * 2 - 1) * max_cfo
    k = torch.arange(n, device=device, dtype=x.dtype).unsqueeze(0)
    angle = 2 * torch.pi * cfo * k + phase
    return _from_complex(_to_complex(x) * torch.exp(1j * angle.to(torch.complex64)))


def time_shift(x: torch.Tensor) -> torch.Tensor:
    """Per-sample circular shift: no frame is aligned to symbol boundaries."""
    batch, _, n = x.shape
    shifts = torch.randint(0, n, (batch,), device=x.device)
    idx = (torch.arange(n, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % n
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, 2, -1))


def resample(x: torch.Tensor, lo: float = 0.75, hi: float = 1.35) -> torch.Tensor:
    """
    Stretch or squeeze the waveform in time, then re-crop to the original length.

    This is the transform that targets the diagnosis directly. Changing the
    time scale changes how many symbols fall inside a 1024-sample frame and how
    the pulse-shaping tails overlap, which is precisely the cue the model was
    leaning on when it mistook 16QAM for 64QAM.

    One scale factor is drawn per batch rather than per frame -- per-frame would
    need a loop, and with hundreds of batches per epoch the model still sees
    plenty of distinct scales.
    """
    n = x.shape[-1]
    factor = float(torch.empty(1).uniform_(lo, hi))
    m = max(8, int(round(n * factor)))
    y = F.interpolate(x, size=m, mode="linear", align_corners=False)
    if m >= n:
        start = (m - n) // 2
        return y[..., start : start + n]
    pad = n - m
    return F.pad(y, (pad // 2, pad - pad // 2), mode="replicate")


def spectral_reshape(
    x: torch.Tensor,
    f_min: float = 0.05,
    f_max: float = 0.40,
    a_min: float = 0.05,
    a_max: float = 0.60,
) -> torch.Tensor:
    """
    Randomly re-shape the occupied bandwidth and skirt steepness of each frame.

    This is the transform the diagnosis actually calls for. narrow_the_search.py
    showed that substituting RadioML's average magnitude spectrum into our
    frames -- phase untouched -- lifted 16QAM from 0.191 to 0.972. The model had
    learned to read modulation order off the spectral envelope, which is a
    shortcut: envelope is set by symbol rate and pulse shaping, and carries no
    information about constellation order.

    Each frame gets an independent raised-cosine band-limiting response with
    random cutoff f_c and random transition width a. Varying both means no
    consistent envelope survives training, so the envelope stops being a usable
    cue and the network has to key on constellation structure instead.

    Cutoffs deliberately extend below the signal bandwidth, so sometimes the
    filter genuinely cuts into the signal. That is intended: a transmitter with
    a different symbol rate looks exactly like this, and modulation order must
    remain recoverable either way.
    """
    batch, _, n = x.shape
    device = x.device

    f_c = torch.empty(batch, 1, device=device).uniform_(f_min, f_max)
    a = torch.empty(batch, 1, device=device).uniform_(a_min, a_max)

    freqs = torch.fft.fftshift(torch.fft.fftfreq(n, device=device)).abs().unsqueeze(0)
    lower = f_c * (1 - a)
    upper = f_c * (1 + a)

    # Raised-cosine transition between passband and stopband.
    ramp = 0.5 * (1 + torch.cos(
        torch.pi * (freqs - lower) / (upper - lower).clamp(min=1e-6)
    ))
    gain = torch.where(freqs <= lower, torch.ones_like(ramp),
                       torch.where(freqs >= upper, torch.zeros_like(ramp), ramp))

    z = _to_complex(x)
    spec = torch.fft.fftshift(torch.fft.fft(z, dim=1), dim=1)
    spec = spec * gain.to(spec.dtype)
    return _from_complex(torch.fft.ifft(torch.fft.ifftshift(spec, dim=1), dim=1))


def spectral_tilt(x: torch.Tensor, max_db: float = 4.0) -> torch.Tensor:
    """
    Mild linear gain slope across the band, modelling a non-flat front end.

    Deliberately gentle: a strong filter would remove information rather than
    teach invariance to a nuisance parameter.
    """
    batch, _, n = x.shape
    z = _to_complex(x)
    spec = torch.fft.fftshift(torch.fft.fft(z, dim=1), dim=1)

    slope_db = (torch.rand(batch, 1, device=x.device) * 2 - 1) * max_db
    ramp = torch.linspace(-0.5, 0.5, n, device=x.device).unsqueeze(0)
    gain = 10 ** (slope_db * ramp / 20.0)

    spec = spec * gain.to(spec.dtype)
    return _from_complex(torch.fft.ifft(torch.fft.ifftshift(spec, dim=1), dim=1))


def add_noise(x: torch.Tensor, min_snr_db: float = 5.0,
              max_snr_db: float = 40.0) -> torch.Tensor:
    """
    Extra AWGN on top of whatever the frame already carries.

    Only ever lowers SNR, so a frame's label stays valid; it widens the
    effective SNR coverage of the training set without regenerating data.
    """
    batch = x.shape[0]
    snr = torch.empty(batch, 1, 1, device=x.device).uniform_(min_snr_db, max_snr_db)
    noise_power = 10 ** (-snr / 10.0)
    return x + torch.randn_like(x) * (noise_power / 2).sqrt()


# --------------------------------------------------------------------------


class Augmenter:
    """
    Composable augmentation pipeline.

    Each transform has an independent probability so that ablations are easy:
    switch one off, retrain, and the change in the cross-domain gap is
    attributable to that transform alone.

    Defaults are set by the diagnosis, not by taste:

      spectral_reshape  p=0.9  the measured cause of the gap -- primary
      phase_and_freq    p=0.8  cheap, always physically justified
      time_shift        p=0.8  cheap, always physically justified
      spectral_tilt     p=0.4  secondary envelope variation
      resample          p=0.2  deliberately low. The timing test drove the
                               symbol-rate line from 14.3 dB to 0.1 dB and
                               moved overall accuracy 0.806 -> 0.797, i.e. the
                               model was already invariant to it. Kept at low
                               probability for completeness, not because it
                               earns its place.
    """

    def __init__(
        self,
        p_spectral_reshape: float = 0.9,
        p_phase_freq: float = 0.8,
        p_time_shift: float = 0.8,
        p_spectral_tilt: float = 0.4,
        p_resample: float = 0.2,
        p_noise: float = 0.3,
        max_cfo: float = 0.01,
        resample_range: tuple[float, float] = (0.75, 1.35),
    ):
        self.p_spectral_reshape = p_spectral_reshape
        self.p_resample = p_resample
        self.p_phase_freq = p_phase_freq
        self.p_time_shift = p_time_shift
        self.p_spectral_tilt = p_spectral_tilt
        self.p_noise = p_noise
        self.max_cfo = max_cfo
        self.resample_range = resample_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_spectral_reshape:
            x = spectral_reshape(x)
        if torch.rand(1).item() < self.p_resample:
            x = resample(x, *self.resample_range)
        if torch.rand(1).item() < self.p_spectral_tilt:
            x = spectral_tilt(x)
        if torch.rand(1).item() < self.p_phase_freq:
            x = phase_and_freq(x, self.max_cfo)
        if torch.rand(1).item() < self.p_time_shift:
            x = time_shift(x)
        if torch.rand(1).item() < self.p_noise:
            x = add_noise(x)
        return _renormalize(x)

    def describe(self) -> str:
        return (f"spectral_reshape p={self.p_spectral_reshape}, "
                f"resample p={self.p_resample}, "
                f"phase/cfo p={self.p_phase_freq} max={self.max_cfo}, "
                f"shift p={self.p_time_shift}, tilt p={self.p_spectral_tilt}, "
                f"noise p={self.p_noise}")


class StandardAMCAugmenter:
    """
    The augmentation set used in the AMC literature, as a comparison baseline.

    Rotation, flip and additive Gaussian noise are the three transforms named in
    the standard reference on augmentation for deep-learning AMC
    (arXiv:1912.03026). Reproducing them here means any claim about spectral
    whitening is made against what the field already does, rather than against
    doing nothing.

    Label validity:
      - rotation is always safe. Absolute phase is arbitrary at the receiver,
        so no rotation can change which modulation a frame is.
      - conjugation mirrors the constellation *and* the spectrum. That is
        label-preserving for the symmetric constellations used here (BPSK,
        QPSK, 8PSK, 16QAM, 64QAM) but would NOT be for single-sideband or any
        spectrally asymmetric class. Extending the class set means revisiting
        this.
    """

    def __init__(self, p_rotate: float = 0.8, p_flip: float = 0.5,
                 p_noise: float = 0.5):
        self.p_rotate = p_rotate
        self.p_flip = p_flip
        self.p_noise = p_noise

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_rotate:
            x = rotate_90(x)
        if torch.rand(1).item() < self.p_flip:
            x = conjugate_flip(x)
        if torch.rand(1).item() < self.p_noise:
            x = add_noise(x)
        return _renormalize(x)

    def describe(self) -> str:
        return (f"[literature baseline] rotate90 p={self.p_rotate}, "
                f"flip p={self.p_flip}, noise p={self.p_noise}")


def rotate_90(x: torch.Tensor) -> torch.Tensor:
    """Per-sample rotation by a random multiple of 90 degrees."""
    batch = x.shape[0]
    k = torch.randint(0, 4, (batch,), device=x.device)
    i, q = x[:, 0], x[:, 1]
    # k=0 (I,Q)  k=1 (-Q,I)  k=2 (-I,-Q)  k=3 (Q,-I)
    ni = torch.where(k[:, None] == 0, i,
         torch.where(k[:, None] == 1, -q,
         torch.where(k[:, None] == 2, -i, q)))
    nq = torch.where(k[:, None] == 0, q,
         torch.where(k[:, None] == 1, i,
         torch.where(k[:, None] == 2, -q, -i)))
    return torch.stack([ni, nq], dim=1)


def conjugate_flip(x: torch.Tensor) -> torch.Tensor:
    """Per-sample conjugation: mirrors the constellation about the real axis."""
    batch = x.shape[0]
    flip = (torch.randint(0, 2, (batch,), device=x.device) * 2 - 1).to(x.dtype)
    return torch.stack([x[:, 0], x[:, 1] * flip[:, None]], dim=1)


NONE = None  # explicit "no augmentation" for ablation tables


if __name__ == "__main__":
    # Shape and power invariants must hold, otherwise the model would be able
    # to detect augmentation from gain alone.
    x = torch.randn(16, 2, 1024, device=cuda) if (cuda := "cuda" if
        torch.cuda.is_available() else "cpu") else torch.randn(16, 2, 1024)
    x = x / (x[:, 0] ** 2 + x[:, 1] ** 2).mean(1, keepdim=True).sqrt().unsqueeze(1)

    aug = Augmenter()
    print(aug.describe())
    print(f"\n{'transform':<18} {'shape':<18} {'mean |x|^2':>11}")
    for name, fn in [
        ("input", lambda t: t),
        ("spectral_reshape", spectral_reshape),
        ("resample", resample),
        ("spectral_tilt", spectral_tilt),
        ("phase_and_freq", phase_and_freq),
        ("time_shift", time_shift),
        ("add_noise", add_noise),
        ("full pipeline", aug),
    ]:
        y = _renormalize(fn(x))
        power = (y[:, 0] ** 2 + y[:, 1] ** 2).mean().item()
        print(f"{name:<18} {str(tuple(y.shape)):<18} {power:>11.4f}")

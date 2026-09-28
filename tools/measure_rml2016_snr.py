"""Measure the SNR actually present in RML2016.10a, from the frames alone.

The label is not a measurement: the generator sets the channel's noise
*amplitude* to 10**(-label/10), which `rml2016_channel_snr.py` shows gives a
true SNR of about 2*label + 2.4 dB. This checks that against the pickle
itself, with no assumption about how it was made.

Method. The digital classes are root-raised-cosine pulses at 8 samples per
symbol, roll-off 0.35, so the signal occupies |f| < 1.35/16 = 0.084 of the
sample rate and everything outside is noise, which is white. Per frame:
Blackman-Harris window (so the signal's own sidelobes do not leak into the
noise bins), 128-point FFT, noise level = mean power in the bins with
|f| >= 20/128, signal = total power minus that level times 128. Frames are
pooled per (class, label) after scaling each to unit power, so every frame
weighs the same; the ratio inside a frame does not depend on the scale.

Validated on frames with a known SNR (`--selftest`, needs GNU Radio): the
generator's chain rebuilt, noise added at a known power, estimate compared.

    python tools/measure_rml2016_snr.py --pkl RML2016.10a_dict.pkl
    /usr/bin/python3.12 tools/measure_rml2016_snr.py --selftest
"""
import argparse
import json
import pathlib
import pickle

import numpy as np

N = 128
GUARD = 20                                   # |bin| >= GUARD is noise only
CLASSES = ("BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64")


def blackman_harris(n):
    k = np.arange(n)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    return (a[0] - a[1] * np.cos(2 * np.pi * k / n) + a[2] * np.cos(4 * np.pi * k / n)
            - a[3] * np.cos(6 * np.pi * k / n))


WIN = blackman_harris(N)
BINS = np.fft.fftfreq(N) * N
NOISE = np.abs(BINS) >= GUARD


def snr_db(frames):
    """frames: (n, 128) complex. Pooled SNR in dB, noise measured out of band."""
    frames = frames / np.sqrt(np.mean(np.abs(frames) ** 2, axis=1, keepdims=True))
    P = np.abs(np.fft.fft(frames * WIN, axis=1)) ** 2
    n0 = P[:, NOISE].mean(axis=1)            # per-bin noise level, per frame
    signal = P.sum(axis=1) - n0 * N
    noise = n0 * N
    return 10 * np.log10(signal.sum() / noise.sum())


def from_pickle(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    labels = sorted({s for _, s in d})
    out = {}
    for c in CLASSES:
        out[c] = {}
        for s in labels:
            x = d[(c, s)]
            out[c][s] = round(float(snr_db(x[:, 0] + 1j * x[:, 1])), 2)
    return labels, out


def selftest():
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from rml2016_channel_snr import run, transmit

    rng = np.random.default_rng(1)
    print(f"{'class':<6} {'true SNR':>9} {'estimated':>10}")
    for name in ("BPSK", "QAM16"):
        tx, _ = transmit(name, rng)
        clean = run(tx, 0.0)[2000:]
        ps = np.mean(np.abs(clean) ** 2)
        for true in (-10, -5, 0, 5, 10, 20, 30):
            pn = ps / 10 ** (true / 10)
            noise = np.sqrt(pn / 2) * (rng.standard_normal(clean.size)
                                       + 1j * rng.standard_normal(clean.size))
            y = clean + noise
            frames = y[: (y.size // N) * N].reshape(-1, N)[:1000]
            print(f"{name:<6} {true:>7} dB {snr_db(frames):>7.1f} dB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", default="rml2016_measured_snr.json")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    labels, out = from_pickle(a.pkl)
    print(f"{'label':>6} " + " ".join(f"{c:>7}" for c in CLASSES)
          + "   2*label+2.4")
    for s in labels:
        print(f"{s:>6} " + " ".join(f"{out[c][s]:>7.1f}" for c in CLASSES)
              + f"   {2 * s + 2.4:>9.1f}")
    pathlib.Path(a.out).write_text(json.dumps(
        {"labels": labels, "measured_snr_db": out,
         "method": "out-of-band noise floor, Blackman-Harris, |bin|>=%d of %d"
                   % (GUARD, N)}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

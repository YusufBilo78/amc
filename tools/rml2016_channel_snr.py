"""What does "10 dB" mean in RML2016.10a? Rebuilt from the generator's own code.

DeepSig's generator (github.com/radioML/dataset, generate_RML2016.10a.py and
transmitters.py) is rerun here, block for block, in GNU Radio, with the one
number the dataset calls SNR:

    noise_amp = 10**(-snr/10.0)
    chan = channels.dynamic_channel_model(200e3, 0.01, 50, .01, 0.5e3, 8, fD=1,
               True, 4, [0.0, 0.9, 1.7], [1, 0.8, 0.3], 8, noise_amp, 0x1337)

and measures what that setting produces: the signal power leaving the
transmitter, the signal power leaving the fading channel, and the noise power
the channel adds. The transmitter is the generator's: gr-mapper
constellations (scaled to unit *mean magnitude*, as gr-mapper does, which is
unit power only for PSK) through a 32-filter polyphase root-raised-cosine
resampler at 8 samples per symbol, roll-off 0.35.

Needs GNU Radio (runs here with 3.10 under /usr/bin/python3.12). The dataset
was made with 3.7 in 2016; that the noise block's scaling did not change in
between is checked against the dataset itself in `measure_rml2016_snr.py`.

    /usr/bin/python3.12 tools/rml2016_channel_snr.py
"""
import numpy as np
from gnuradio import blocks, channels, filter, gr

SPS, BETA, NFILTS = 8, 0.35, 32
N_SYM = 200_000


def constellation(name):
    if name == "BPSK":
        pts = np.array([-1, 1], complex)
    elif name == "QPSK":
        pts = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    else:
        m = {"QAM16": 4, "QAM64": 8}[name]
        lv = np.arange(-(m - 1), m, 2)
        pts = (lv[:, None] + 1j * lv[None, :]).ravel()
    # gr-mapper, constellation.cc: "normalize constellation power to 1" --
    # but the loop averages |s|, not |s|^2, so mean magnitude becomes 1
    return pts * len(pts) / np.abs(pts).sum()


def run(src_vec, noise_amp, with_channel=True):
    tb = gr.top_block()
    src = blocks.vector_source_c(src_vec.astype(np.complex64).tolist(), False)
    snk = blocks.vector_sink_c()
    if with_channel:
        chan = channels.dynamic_channel_model(
            200e3, 0.01, 50, .01, 0.5e3, 8, 1, True, 4,
            [0.0, 0.9, 1.7], [1, 0.8, 0.3], 8, noise_amp, 0x1337)
        tb.connect(src, chan, snk)
    else:
        tb.connect(src, snk)
    tb.run()
    return np.array(snk.data())


def transmit(name, rng):
    pts = constellation(name)
    sym = pts[rng.integers(len(pts), size=N_SYM)]
    taps = filter.firdes.root_raised_cosine(NFILTS, NFILTS, 1.0, BETA,
                                            NFILTS * 11 * SPS)
    tb = gr.top_block()
    src = blocks.vector_source_c(sym.astype(np.complex64).tolist(), False)
    rs = filter.pfb_arb_resampler_ccf(SPS, taps)
    snk = blocks.vector_sink_c()
    tb.connect(src, rs, snk)
    tb.run()
    return np.array(snk.data()), sym


def db(x):
    return 10 * np.log10(x)


def main():
    rng = np.random.default_rng(0)
    skip = 2000                                  # channel start-up transient
    zeros = np.zeros(SPS * N_SYM, complex)
    pn = {}
    for label in (10, 0, -10):
        amp = 10 ** (-label / 10)
        pn[label] = np.mean(np.abs(run(zeros, amp)[skip:]) ** 2)
        print(f"label {label:+3d} dB: noise_amp {amp:g} -> noise power "
              f"{pn[label]:.4g} ({db(pn[label]):+.1f} dB), amp^2 = {amp**2:.4g}")
    print()
    print(f"{'class':<6} {'symbol P':>9} {'tx P':>7} {'after chan':>11} "
          f"{'true SNR at label 10':>21} {'in-band':>8} {'Es/N0':>7}")
    for name in ("BPSK", "QPSK", "QAM16", "QAM64"):
        tx, sym = transmit(name, rng)
        ps_tx = np.mean(np.abs(tx[skip:]) ** 2)
        ps_ch = np.mean(np.abs(run(tx, 0.0)[skip:]) ** 2)
        snr = ps_ch / pn[10]
        inband = snr * SPS / (1 + BETA)          # noise is white, signal is not
        esn0 = snr * SPS
        print(f"{name:<6} {np.mean(np.abs(sym) ** 2):>9.3f} {ps_tx:>7.4f} "
              f"{ps_ch:>11.4f} {db(snr):>18.1f} dB {db(inband):>6.1f} dB "
              f"{db(esn0):>5.1f} dB")


if __name__ == "__main__":
    main()

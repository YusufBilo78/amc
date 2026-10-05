"""What RML2016.10a's channel actually does, run by run. Needs GNU Radio.

generate_RML2016.10a.py builds a fresh dynamic_channel_model for every run,
always with noise_seed 0x1337, and GNU Radio 3.7.10 hands that one seed to
every part of the model: the clock-offset walk, the carrier-offset walk, the
fading, and the noise. This measures the consequences on the same block in
GNU Radio 3.10.

Read it as 3.10's behaviour, not the dataset's. In 3.7 the noise and both
walks pick from their seeded pools with the unseeded, process-wide
lrand48(), so they differ run to run; only the fading repeats. The pickle
agrees: tools/rml2016_noise_reuse.py finds no shared noise segments.
Measured here:

  - two runs with the noise off: bit-identical output?
  - two runs of the noise alone: the same sequence?
  - the carrier-offset walk: how far it gets in one run, and how much it
    rotates a 128-sample frame (its clip is 500 Hz; std is 0.01 Hz/sample)
  - the fading gain over one run

    /usr/bin/python3.12 tools/rml2016_channel_audit.py
"""
import numpy as np
from gnuradio import blocks, channels, gr

FS = 200e3


def run(x, amp=0.0):
    tb = gr.top_block()
    src = blocks.vector_source_c(x.tolist(), False)
    snk = blocks.vector_sink_c()
    ch = channels.dynamic_channel_model(FS, 0.01, 50, .01, 0.5e3, 8, 1, True, 4,
                                        [0.0, 0.9, 1.7], [1, 0.8, 0.3], 8, amp, 0x1337)
    tb.connect(src, ch, snk)
    tb.run()
    return np.array(snk.data())


def main():
    n = 80_000                       # the longest run: BPSK, 10k symbols x 8
    one = np.ones(n, complex)
    a, b = run(one), run(one)
    print(f"noise off, two runs identical: {np.array_equal(a, b)}")
    z = np.zeros(n, complex)
    print(f"noise alone, two runs identical: {np.array_equal(run(z, 1.0), run(z, 1.0))}")
    g = a[2000:]
    ph = np.unwrap(np.angle(g))
    f = np.diff(ph) * FS / (2 * np.pi)
    print(f"carrier offset reached: {np.abs(f[-4000:].mean()):.1f} Hz after "
          f"{n / FS:.1f} s (clip 500 Hz)")
    rot = [abs(ph[i + 127] - ph[i]) for i in range(0, len(ph) - 128, 500)]
    print(f"rotation inside one 128-sample frame: median {np.median(rot):.4f} rad, "
          f"max {np.max(rot):.4f} rad ({np.degrees(np.max(rot)):.1f} deg)")
    print(f"total carrier phase drift over the run: {ph[-1] - ph[0]:.1f} rad")
    print(f"fading gain |h| over the run: {np.abs(g).min():.2f} .. {np.abs(g).max():.2f}, "
          f"i.e. {20 * np.log10(np.abs(g).max() / np.abs(g).min()):.1f} dB of swing")


if __name__ == "__main__":
    main()

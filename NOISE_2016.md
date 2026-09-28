# The noise in RML2016.10a, defined

Moshe, 2026-09-22: *"10 dB of what? Who made the noise, and what are its
properties?"* This page answers that for RML2016.10a from three sources:

- **The dataset paper.** O'Shea & West, "Radio Machine Learning Dataset
  Generation with GNU Radio", Proc. 6th GNU Radio Conference, 2016. It names
  the channel model but gives no parameters and no definition of SNR.
- **The generator code DeepSig published.** github.com/radioML/dataset,
  `generate_RML2016.10a.py` and `transmitters.py`, commit `4ecf612` (May
  2017). The released pickle is from October 2016, so this is the code as
  published, not a guaranteed copy of what ran. Section 4 checks it against
  the pickle itself.
- **A measurement.** The generator's chain was rebuilt in GNU Radio 3.10
  (`tools/rml2016_channel_snr.py`) and run to see what each setting
  produces.

---

## 1. Who made the noise

DeepSig (O'Shea, then at Virginia Tech, and West). They generated it
synthetically in GNU Radio, not over the air. The paper, §2.3:

> "For channel simulation we use the GNU Radio Dynamic Channel Model
> hierarchical block. This includes a number of desired effects such as
> random processes for center frequency offset, sample rate offset, additive
> white Gaussian noise, multi-path, and fading."

And on the noise itself:

> "Noise Model (AWGN): The basic GNU Radio additive white Gaussian noise
> model which introduces thermal noise simulation at the receiver at a
> specific noise power level corresponding to the desired signal to noise
> ratio."

That is the whole definition in the paper. Nothing there says which power,
over which bandwidth, or how the "desired" ratio is set.

---

## 2. The chain, with its numbers (from the generator code)

```
bits / audio → modulator → dynamic_channel_model → sample 128 → scale → store
```

| stage | setting | what it means |
|---|---|---|
| sample rate | 200 kHz | a frame of 128 samples is 0.64 ms |
| digital modulators | gr-mapper constellation → root-raised-cosine, **8 samples per symbol**, roll-off 0.35 | 25 kbaud; **a frame is 16 symbols**; the signal occupies 33.75 kHz of the 200 |
| constellation scale | gr-mapper sets mean *magnitude* to 1 | unit power for PSK; 1.11 for 16-QAM and 1.13 for 64-QAM |
| sample-rate offset | random walk, std 0.01, clipped at 50 Hz | clock drift |
| carrier-frequency offset | random walk, std 0.01, clipped at **500 Hz** | at the limit, 2 rad (115°) of rotation across one frame |
| fading | sum of 8 sinusoids, **Rician K = 4**, Doppler 1 Hz | coherence time is about 0.4 s, so each frame sees one fixed random complex gain |
| multipath | 3 paths at delays 0, 0.9, 1.7 samples, gains 1, 0.8, 0.3 | delay spread 0.2 symbol: mild but real inter-symbol interference |
| **noise** | `noise_amp = 10**(-snr/10.0)`, seed `0x1337` | see §3 |
| framing | 128 consecutive samples at a random offset | |
| scaling | each frame divided by Σ\|x\| | the code uses the sum of magnitudes, not the "unit energy" the paper states; this repository rescales every frame to unit power anyway (`cnn.normalize_frames`) |

Every effect except the noise is present at every SNR, including +18 dB.

---

## 3. What "10 dB" is the ratio of

The label sets one number, `noise_amp = 10**(-label/10)`. In GNU Radio that
parameter is the noise **amplitude**. Measured on the rebuilt channel:

| label | noise_amp | noise power measured |
|---|---|---|
| +10 dB | 0.1 | 0.0098 (−20.1 dB) |
| 0 dB | 1 | 0.98 (−0.1 dB) |
| −10 dB | 10 | 98 (+19.9 dB) |

So the noise power is 10^(−label/5): **the label moves the noise twice as
fast as its name says.** Nothing in the chain normalises the signal either.
It leaves the transmitter at 1.0 (PSK) or 1.11–1.13 (QAM), and leaves the
fading channel at 1.75 (PSK) or 1.95–1.99 (QAM), because the three path gains
add power (1 + 0.64 + 0.09 = 1.73).

**What the ratio actually is**, at a label of 10 dB, measured:

| class | per-sample SNR, full 200 kHz band | in the signal's own band | Es/N0 |
|---|---|---|---|
| BPSK, QPSK | **22.5 dB** | 30.3 dB | 31.6 dB |
| 16-QAM | 23.0 dB | 30.7 dB | 32.0 dB |
| 64-QAM | 23.1 dB | 30.8 dB | 32.1 dB |

In general, for PSK, **true per-sample SNR ≈ 2 × label + 2.4 dB**. The
signal's own band is 7.7 dB higher than that, because the noise is white over
200 kHz and the signal sits in 33.75 kHz. Es/N0 is 9 dB higher still, the
factor of 8 samples per symbol.

| label | per-sample SNR | Es/N0 (PSK) |
|---|---|---|
| −20 | −37.6 dB | −28.6 dB |
| −10 | −17.6 dB | −8.6 dB |
| 0 | 2.4 dB | 11.4 dB |
| +2 | 6.4 dB | 15.4 dB |
| +6 | 14.4 dB | 23.4 dB |
| +10 | 22.5 dB | 31.6 dB |
| +18 | 38.4 dB | 47.4 dB |

**Properties of the noise.** It is complex, circular, white, and Gaussian.
It is added after the fading and multipath, so its level is fixed while the
signal's level varies from frame to frame with the fading draw. Its seed is
fixed (`0x1337`) and a new channel is built for every run of the generator,
so every run adds the same noise sequence, only scaled per SNR. Frames are
cut at random offsets, which keeps two frames from sharing noise sample for
sample except by coincidence of offset.

---

## 4. Is the pickle what the code says? — to be measured

Two things could make the table in §3 wrong for the file we train on:

- The published code dates from May 2017, and the pickle was released in
  October 2016.
- The chain was rebuilt with GNU Radio 3.10, and the pickle was made with
  3.7. The noise block's scaling could have changed in between.

`tools/measure_rml2016_snr.py` settles this from the frames alone, with no
assumption about how they were made. For the six digital classes the signal
occupies only |f| < 0.084 of the sample rate. Everything above 20/128 is
noise, and since the noise is white that out-of-band level gives the total
noise. Checked on frames of known SNR, the estimator is within 0.5 dB from
−10 to +30 dB. If §3 is right, the pickle should read about 2 × label +
2.4 dB, less the estimator's 0.4 dB bias.

**Result: pending** (Colab, no GPU needed).

---

## 5. What this changes in the results already reported

Nothing in any table changes: every number is indexed by label, and the label
is what everyone using this dataset reports. What changes is how to read the
axis:

- "Recall is flat from +2 dB up" means flat from a per-sample SNR of 6.4 dB,
  an Es/N0 of 15 dB. At +18 dB (Es/N0 47 dB) noise is negligible. The
  QAM16/QAM64 confusion that stays there (86–88% and 91–93% recall) comes from
  16 symbols, carrier offset, fading and multipath, not from noise. That
  strengthens the frame-length reading of `SNR_2016.md`.
- "The box breaks below 0 dB" means below 2.4 dB per sample, an Es/N0 of
  11 dB.
- At −20 dB the per-sample SNR is −37.6 dB: the signal is 1/5,800 of the
  noise. That the four rows of the −20 dB matrix are identical is what this
  predicts.
- Comparing SNR axes across datasets or papers is not safe unless both
  define SNR the same way. RadioML 2018 has its own generator and its own
  definition. Nothing here transfers to it.

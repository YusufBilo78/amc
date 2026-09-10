# Automatic Modulation Classification — Domain Generalization Study

## The claim this project is trying to establish

> A CNN that reaches ~90% accuracy on the RadioML benchmark collapses when
> tested on IQ captured with different hardware. Quantify that collapse, then
> recover as much of it as possible with preprocessing and augmentation.

Reproducing RadioML accuracy is not the contribution — that is a solved
benchmark. The contribution is measuring, explaining and partially closing the
**generalization gap** between the training domain and an independently
captured one.

---

## Setup

Python 3.12, virtual environment at `C:\Users\yusuf\.venvs\amc`
(kept outside OneDrive — packages are multi-GB and syncing them is painful).

```bash
C:\Users\yusuf\.venvs\amc\Scripts\python.exe src/explore_signals.py
```

Datasets live in `C:\Users\yusuf\amc-data` (also outside OneDrive).

GPU: RTX 3070 Laptop, 8 GB. PyTorch 2.11 + CUDA 12.8, verified working.

---

## Layout

```
src/modem.py            baseband IQ generator (11 modulation classes)
src/explore_signals.py  Week 1 figures
figures/                generated output
```

---

## Roadmap

| Week | Task | Status |
|------|------|--------|
| 1 | Generate signals, look at constellations and spectrograms | done |
| 2 | Classical baseline: higher-order cumulants + SVM. Accuracy vs SNR curve | done |
| 3–4 | CNN baseline. Compare raw-IQ input vs spectrogram input | done (synthetic) |
| 5 | Capture own IQ over a cabled (conducted) SDR loopback | |
| 6 | Test the trained model on own captures — document the collapse | |
| 7–8 | Diagnose and fix: normalization, resampling, augmentation, fine-tuning | |
| 9 | Explainability: saliency maps, confusion-matrix analysis | |
| 10 | Report and presentation | |

---

## Rules that keep this honest

1. **Always have a classical baseline.** "The CNN got 94%" means nothing on its
   own.
2. **No data leakage.** Never split train/test within a single capture — split
   by recording, device, or session. Splitting within a capture inflates
   accuracy dramatically and invalidates every number downstream.
3. **Report accuracy vs SNR**, not a single averaged number.
4. **Read the confusion matrix physically.** 16QAM vs 64QAM at low SNR is
   near-impossible in principle; a model failing there is not a bad model.
5. **Never transmit over the air.** Use a cabled TX → attenuator → RX loop.
   It is legal, and it makes experiments repeatable.

---

## Week 1 findings

From `figures/02_spectrograms.png`: BPSK, QPSK, 8PSK, PAM4, 16QAM and 64QAM
produce **visually near-identical spectrograms** — same pulse shaping means
same occupied bandwidth and same flat power profile. The spectrogram separates
the analog and continuous-phase classes (AM-DSB, AM-SSB, WBFM, CPFSK, GFSK) but
carries almost no information about constellation order.

Consequence for Week 3–4: a spectrogram-input CNN should be expected to fail
exactly on the linear digital classes, while a raw-IQ CNN should not. That is a
falsifiable prediction, and testing it is a better experiment than simply
picking whichever representation scores higher.

From `figures/03_snr_sweep.png`: for 16QAM the constellation is clean at 20 dB,
readable at 10 dB, and structurally gone at 0 dB — while the signal is still
plainly *visible* in the spectrogram at −10 dB. Detection and classification
break down at very different SNRs.

---

## Week 2 findings — the baseline the CNN must beat

13 features (5 normalized cumulant magnitudes + 8 instantaneous
amplitude/phase/frequency statistics), RBF SVM, 24 200 synthetic frames of
1024 samples, 11 classes. Chance = 0.091.

| SNR (dB) | −20 | −12 | −4 | 0 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.098 | 0.126 | 0.334 | 0.609 | 0.840 | 0.907 | 0.952 | 0.974 | 0.972 |

Sanity check against theory: measured |C40| ranks BPSK (1.60) > QPSK (0.83) >
16QAM (0.57) > 64QAM (0.51), and 8PSK gives 0.04 — the fourth-order cumulant of
8PSK vanishes analytically. Absolute values sit below the ideal symbol-level
figures because cumulants are measured on the pulse-shaped waveform rather than
at symbol instants.

**The confusion matrix is more useful than the accuracy number.** Three
distinct regimes:

1. **16QAM ↔ 64QAM** is the only meaningful error left at 16 dB (11–16%
   cross-confusion) and collapses to near-chance-between-the-pair at 0 dB.
   These two differ only in fine constellation density, which is exactly what
   noise destroys first. This is close to an information-theoretic limit, not
   a modelling failure.
2. **CPFSK / GFSK / WBFM** are perfectly separated at 16 dB but merge into one
   blob at 0 dB (0.36 / 0.29 / 0.36 on-diagonal). All three are
   constant-envelope, so every amplitude feature is blind to them and only
   fine phase statistics separate them.
3. **AM-DSB and AM-SSB stay at 1.00 even at 0 dB.** They are separated by
   coarse structural properties — realness (|C20|) and sideband asymmetry —
   which degrade far more slowly than density-based features.

**Prediction for Week 3–4:** a CNN cannot improve on regime 3, which is already
saturated. Any gain must come from regimes 1 and 2. So the right way to report
the CNN result is per-cell improvement in the confusion matrix, not a single
headline accuracy delta.

---

## Week 3–4 findings — representation decides the ceiling

Two CNNs, identical data and training schedule, differing only in what they are
allowed to see. `IQNet`: 895k parameters, 1D convolutions over raw IQ.
`SpecNet`: 95k parameters, 2D convolutions over a log-magnitude STFT.

| SNR (dB) | −20 | −12 | −4 | 0 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|---|
| cumulants + SVM | 0.098 | 0.126 | 0.334 | 0.609 | 0.907 | 0.974 | 0.972 |
| CNN, spectrogram | 0.099 | 0.284 | 0.661 | 0.734 | 0.782 | **0.793** | 0.785 |
| CNN, raw IQ | 0.199 | 0.491 | 0.832 | 0.925 | 0.968 | 0.988 | **0.990** |

**The Week 1 prediction holds, and sharpens.** The spectrogram CNN saturates
near 0.79 no matter how much SNR it is given — above roughly 4 dB it is *worse
than 13 hand-designed features*. Its confusion matrix at SNR ≥ 12 dB shows why:

- CPFSK, GFSK, AM-DSB, AM-SSB, WBFM: **1.00 each.** Perfect.
- QPSK, 8PSK, 16QAM, 64QAM: collapse into a single 4×4 block of mutual
  confusion (0.33 / 0.58 / 0.19 / 0.61 on-diagonal).

The refinement the experiment produced, which the Week 1 reasoning missed:
**BPSK (1.00) and PAM4 (0.98) survive.** Both are real-valued, so their spectra
are conjugate-symmetric — a structural property that survives the magnitude
operation in a spectrogram. The four classes that collapse are exactly the
*complex* linear digital ones, whose only distinguishing information is
constellation density, and taking |STFT| throws away the phase that encodes it.

So the ceiling is not a capacity problem. `SpecNet` is 9× smaller than
`IQNet`, but scaling it up cannot help: the information is not in its input.

**Honest caveat:** `IQNet` reaches a training loss of 0.003 while test accuracy
sits at 0.755 — it is memorizing. Augmentation (phase rotation, frequency
offset, time shift) is the obvious Week 7–8 lever, and `modem.carrier_offset`
and `modem.iq_imbalance` already exist for it.

---

## Week 4 — `IQNet` on real RadioML 2018.01A

24 classes, 512 of 4096 frames per (class, SNR) cell (12.5% of the dataset),
223 392 train / 96 096 test frames. 898 584 parameters, 25 epochs, 658 s on the
RTX 3070. Chance = 1/24 = 0.042.

| SNR (dB) | −20 | −12 | −6 | 0 | 4 | 8 | 12 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.044 | 0.077 | 0.240 | 0.522 | 0.709 | 0.851 | 0.879 | 0.883 | 0.878 |

At −20 dB it sits at chance to three decimals, which is the right sanity check:
nothing is being invented at the bottom of the curve. The plateau is ~0.88 from
12 dB upward. Published results on this dataset reach roughly 0.90–0.95 with
larger models and the full 2 M frames; matching them is not the goal here.

**The confusion structure is the result worth reporting.** Twelve classes are
essentially perfect at SNR ≥ 12 dB — OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16APSK,
32APSK, 16QAM, FM, GMSK, OQPSK. Every remaining error falls into one of five
tight blocks, and each block is a set of *adjacent members of the same family*:

| block | classes |
|---|---|
| PSK, high order | 16PSK ↔ 32PSK |
| APSK, high order | 64APSK ↔ 128APSK |
| QAM, high order | 32QAM ↔ 64QAM ↔ 128QAM ↔ 256QAM |
| analog SSB | AM-SSB-WC ↔ AM-SSB-SC |
| analog DSB | AM-DSB-WC ↔ AM-DSB-SC |

There is essentially no cross-family confusion. The model solves *which family*
completely, and fails only on *which order within a family*, and only above a
threshold order — PSK breaks after 8, APSK after 32, QAM after 16.

That is the correct behaviour, not a defect. Constellation points get
geometrically closer as order rises, so at fixed SNR the orders become
statistically indistinguishable. A model reporting high confidence between
128QAM and 256QAM at 12 dB would be a model that had learned an artifact.

Same lesson as the synthetic 16QAM/64QAM result in Week 2, now at scale: the
ceiling is set by the physics of the constellations, not by model capacity.

---

## The diagnosis, and how it was reached

`crossdomain.py` found a 16.7-point gap at SNR >= 10 dB (in-domain 0.991,
cross-domain 0.825) whose entire mass sat in one cell: our 16QAM read as
64QAM, 75% of the time.

Four rounds. Three of them negative, and the negatives are controlled results
worth reporting, not wasted effort.

| # | Hypothesis | Test | Outcome |
|---|---|---|---|
| 1 | pulse-shaping / spectrum mismatch | `sweep.py`, beta 0.15–0.90 | **wrongly dismissed** — see below |
| 2 | our constellation looks denser | `diagnose_16qam.py`, amplitude histograms | refuted: distributions overlap almost exactly |
| 3 | symbol-timing regularity | `test_timing_hypothesis.py` | refuted: line prominence driven 14.3 → 0.1 dB (RadioML 0.7 dB), 16QAM moved 0.168 → 0.202 |
| — | missing channel impairments | `ablate_impairments.py` | refuted: random phase, CFO to 1e-2, delay spread to 4 samples, and all combined — best recovery +0.045 |
| 4 | spectral envelope | `narrow_the_search.py` | **confirmed** |

**The answer.** Keeping our own phase spectrum and imposing RadioML's average
*magnitude* spectrum frame by frame:

| | BPSK | QPSK | 8PSK | 16QAM | 64QAM | overall |
|---|---|---|---|---|---|---|
| ours, beta=0.35 | 1.000 | 1.000 | 0.910 | **0.191** | 0.940 | 0.808 |
| + RadioML spectrum | 1.000 | 1.000 | 0.999 | **0.972** | 0.977 | **0.990** |
| RadioML itself | 1.000 | 1.000 | 0.999 | 0.993 | 0.978 | 0.994 |

The gap collapses from 16.7 points to 0.4 without touching the time structure,
the symbol sequence, or the phase. Corroborated by a monotonic roll-off sweep:
beta 0.35 → 0.01 takes 16QAM from 0.191 to 0.745 across eight points.

Measured difference: our out-of-band floor sits ~12 dB higher (−25 dB vs
−37 dB) and the main lobe is slightly wider (±0.075 vs ±0.06 normalized).

**Methodological note that belongs in the report.** Hypothesis 1 was correct
and was dismissed on an inadequate test: the sweep bottomed out at beta = 0.15,
where 16QAM only reached 0.37, and that was read as "broken at every setting".
Extending to 0.01 would have shown 0.745. Stopping at the edge of a sweep range
while the trend is still monotonic is a reliable way to miss a real effect.

That beta = 0.01 still only reaches 0.745, while full spectral matching reaches
0.972, says RadioML's pulse shape is not a plain RRC — no roll-off value
reproduces it exactly.

## The fix

### Attempt 1 — randomise the envelope. Failed.

`augment.spectral_reshape`: random raised-cosine band-limiting per frame, with
random cutoff and transition width, so no consistent envelope survives training.

| config | in-domain | cross-domain | gap |
|---|---|---|---|
| no augmentation | 0.991 | **0.825** | +0.167 |
| full augmentation | 0.998 | **0.783** | +0.214 |
| ablation, `spectral_reshape` off | 0.997 | **0.815** | +0.182 |

Worse than doing nothing, and the ablation identifies the new transform as the
cause. Training loss plateaued at 0.64 against 0.20 unaugmented: cutoffs were
drawn as low as 0.05 while the signal occupies ±0.06, so the filter was
routinely cutting into the signal. That destroys information; it does not teach
invariance.

### Attempt 2 — remove the envelope. Works.

The misreading was in the inference, not the diagnosis. The intervention that
worked in `narrow_the_search.py` *matched* the two spectra rather than
randomising them, which points at preprocessing rather than augmentation:
reduce every frame's envelope to a canonical form, in both domains, so it
carries no information for anyone.

    X_white(f) = X(f) / ( smooth(|X(f)|) + eps )^alpha

Phase untouched; envelope estimated as a circular moving average of |X(f)|
over 33 FFT bins.

Four seeds per point, varying both the train/test split and the weight
initialisation. Accuracy at SNR >= 10 dB, mean +- standard deviation.

| alpha | in-domain | cross-domain | gap | 16QAM cross-domain |
|---|---|---|---|---|
| 0.00 | 0.993 ±0.002 | 0.806 ±0.017 | +0.187 ±0.018 | 0.117 ±0.053 |
| 0.25 | 0.997 ±0.000 | 0.862 ±0.017 | +0.135 ±0.017 | 0.352 ±0.089 |
| 0.50 | 0.994 ±0.001 | 0.900 ±0.003 | +0.094 ±0.003 | 0.557 ±0.017 |
| **0.75** | 0.964 ±0.010 | **0.907 ±0.009** | **+0.056 ±0.004** | 0.691 ±0.048 |
| 1.00 | 0.879 ±0.007 | 0.827 ±0.003 | +0.052 ±0.005 | 0.737 ±0.010 |

At alpha = 0.75 the gap falls from 0.187 to 0.056 — **70% closed** — with
cross-domain accuracy rising 0.806 → 0.907. The improvement is **7.4 pooled
standard deviations**, so it is not run-to-run variation.

Partial whitening wins, as anticipated: alpha = 1.0 lifts out-of-band noise
along with the signal and in-domain accuracy collapses to 0.879. Sweeping
rather than assuming the endpoint was the right call.

**Single runs invent effects.** The first pass used one seed per alpha and
produced a non-monotonic dip at alpha = 0.25 (cross-domain 0.793, 16QAM 0.035)
that looked like a real phenomenon worth explaining. Across four seeds that
point is 0.862 ±0.017 — better than alpha = 0, and the curve is monotonic
throughout. The dip was one unlucky model. Worth keeping in the report as a
concrete argument for error bars.

**16QAM and the mean disagree.** 16QAM keeps improving all the way to
alpha = 1.0 (0.737) while overall cross-domain accuracy peaks at 0.75 and then
falls. Full whitening helps the class that was broken and hurts the rest;
alpha = 0.75 is a compromise, not any single class's optimum.

**Why this matters beyond the number.** Whitening is preprocessing, not domain
adaptation. It needs no target-domain data at fit time, so it is deployable on
a receiver that has never seen the transmitter it will meet. Methods like DANN
require target-domain samples; this does not.

### Against the literature baseline

"Better than nothing" is not a result. The comparison that counts is against
the augmentation set the AMC literature already uses — rotation, conjugate
flip, additive Gaussian noise (arXiv:1912.03026) — reproduced in
`augment.StandardAMCAugmenter`. Four seeds, identical data and schedule.

| method | in-domain | cross-domain | gap | 16QAM |
|---|---|---|---|---|
| none | 0.993 ±0.001 | 0.811 ±0.016 | +0.182 ±0.017 | 0.123 ±0.056 |
| standard augmentation | 0.996 ±0.001 | 0.854 ±0.022 | +0.141 ±0.023 | 0.309 ±0.123 |
| whitening a=0.75 | 0.965 ±0.010 | 0.909 ±0.007 | +0.056 ±0.004 | 0.706 ±0.023 |
| **whitening + standard** | **0.985 ±0.001** | **0.948 ±0.006** | **+0.038 ±0.005** | **0.965 ±0.007** |

- standard augmentation vs none: **+0.044 (2.3 s.d.)** — real but modest
- whitening vs none: **+0.098 (8.2 s.d.)**
- whitening vs standard augmentation: **+0.054 (3.3 s.d.)**
- combining vs whitening alone: **+0.039 (6.4 s.d.)**

**The transforms compose rather than substitute**, which was predicted in
advance from what each one addresses: rotation and flip teach invariance to
phase and mirror symmetry, whitening removes the spectral envelope. Different
nuisance parameters, different axes, additive effect.

**And the trade-off largely disappears in combination.** Whitening alone costs
in-domain accuracy (0.993 → 0.965); adding the standard transforms brings it
back to 0.985 while keeping the cross-domain gain. That was not anticipated.

End to end: gap 0.182 → 0.038, **79% closed**, with 16QAM going from 0.123 to
0.965.

### What the mechanism actually is

The working account through most of this project was shortcut learning: the
model reads modulation order off the spectral envelope, and whitening removes
the shortcut. Two experiments were designed to confirm it. Both killed it.

**Spectral occlusion** (`spectral_occlusion.py`). Sweep a notch across the
spectrum, measure the accuracy cost. Prediction was that the alpha=0 model
would lean on the band edges where envelope shape lives. It does not: in-band
dependence is 91.0% for alpha=0 and 86.3% for alpha=0.75, with near-identical
profiles. The test was also badly designed for the question — nulling a band
deletes envelope and constellation structure together, so it cannot separate
them. One real result survives: the maximum single-notch drop is 0.791 for
alpha=0 against 0.577 for alpha=0.75, so the whitened model spreads its
dependence more evenly.

**Envelope transplant** (`envelope_transplant.py`). Keep a frame's phase
spectrum, give it another class's average magnitude spectrum, ask the model.
This does separate envelope from structure.

| | follows donated envelope | keeps true class |
|---|---|---|
| alpha = 0 | 0.002 | 0.992 |
| alpha = 0.75 | 0.009 | 0.962 |

The matrices are diagonal. Handing 16QAM frames a 64QAM envelope leaves the
prediction at 16QAM 99% of the time. **The envelope does not drive the class
decision.** Within RadioML the model reads order from phase structure, exactly
where the information is.

**Revised account.** Envelope mismatch does not mislead the model; it pushes
frames off the training manifold entirely, and the model falls back on a
default class. Independent support was already in hand before the hypothesis
existed: at sps = 16 in `sweep.py`, 64QAM scored 0.999 while QPSK fell to
0.042 — far below the 0.20 chance level, which is the signature of collapse
onto a sink class rather than random error.

So the contribution is **distribution alignment**, not shortcut removal:

> Envelope mismatch moves inputs off the training manifold and the classifier
> collapses to a default class. Whitening maps both domains onto a shared
> canonical form, keeping inputs on-manifold.

The fix is unaffected — 79% of the gap still closes, still 3.3 s.d. ahead of
the literature baseline. Only the explanation changed. Any "shortcut learning"
framing has to be dropped; the project's own data refutes it.

### Against adversarial domain adaptation

DANN (Ganin et al., and the AMC line built on it) is the method this literature
actually rests on, so it was implemented with the standard recipe — same
feature extractor as everywhere else, gradient reversal, the published
2/(1+exp(-10p))−1 lambda ramp, a two-layer discriminator, a full batch from
each domain. Four seeds, same data and schedule as the rest of the table.

**DANN is the only method here that uses target-domain data during training.**
Unlabelled, but present. To keep that honest the target set was split in half:
DANN adapts on one half and is scored on the other. The other methods were
scored on the whole target set, which is equivalent for them — they never touch
it. The comparison is therefore deliberately generous to DANN.

| method | in-domain | cross-domain | gap | 16QAM |
|---|---|---|---|---|
| none | 0.993 ±0.001 | 0.811 ±0.016 | +0.182 ±0.017 | 0.123 ±0.056 |
| standard augmentation | 0.996 ±0.001 | 0.854 ±0.022 | +0.141 ±0.023 | 0.309 ±0.123 |
| whitening a=0.75 | 0.965 ±0.010 | 0.909 ±0.007 | +0.056 ±0.004 | 0.706 ±0.023 |
| **whitening + standard** | 0.985 ±0.001 | **0.948 ±0.006** | **+0.038 ±0.005** | **0.965 ±0.007** |
| DANN \* | 0.987 ±0.004 | 0.792 ±0.113 | +0.195 ±0.117 | 0.577 ±0.165 |
| DANN + whitening \* | 0.977 ±0.002 | 0.919 ±0.004 | +0.058 ±0.006 | 0.711 ±0.045 |

\* uses target-domain data in training.

**The result is in the variance, not the mean.** DANN's four seeds gave
cross-domain 0.908, 0.654, 0.900, 0.706 — same data, same schedule, seed
alone, and a 25-point swing. Bimodal: the adversarial game either holds or
collapses. Standard deviation ±0.113 against whitening's ±0.007.

**Whitening removes that instability.** DANN + whitening gives 0.916, 0.922,
0.923, 0.914 — ±0.004, a 28× reduction in spread. Reducing the inputs to a
canonical form appears to make the adversarial objective tractable, which is a
more interesting relationship than a win.

**What must not be claimed.** Not "we beat DANN". Three reasons:

1. Effect sizes are 1.5–1.9 s.d., and small *because DANN's variance is
   enormous*, not because the means are close.
2. This implementation may be under-tuned. A smoke test showed the domain loss
   falling from 0.69 toward 0.17 — the discriminator winning, i.e. the
   adversarial balance not holding. A different lambda schedule, discriminator
   capacity, or a separate learning rate for the discriminator might stabilise
   it. Tuning it properly is future work, not a result.
3. Adversarial instability is well known in general, so what was observed may
   belong to the method rather than to this implementation of it.

**The defensible claim** is narrower and stronger: the best result in the table
is whitening + standard augmentation at 0.948 ±0.006, and it uses **no
target-domain data at all**. For a receiver meeting a transmitter it has never
seen, that difference is not a technicality — there is no target data to adapt
to.

### Caveats that belong in the report

- **The second domain is still synthetic.** An SDR capture remains the
  strongest available validation, and is blocked until NJIT lab access.
- **No comparison against adversarial domain adaptation.** DANN-style methods
  are the other obvious reference. They need target-domain data at fit time
  and whitening does not, so the comparison is not like-for-like, but it should
  still be run and that asymmetry stated.
- **Conjugate flip is not label-safe in general.** It mirrors the spectrum as
  well as the constellation, which is harmless for the five symmetric classes
  used here but would corrupt AM-SSB or any spectrally asymmetric class.
  Extending the class set means revisiting this transform.
- **15 epochs, one architecture.** No claim is made that these rankings hold
  under longer training or a different network.

---

## Reading the generation paper — and a refuted arithmetic

O'Shea, Roy & Clancy (arXiv:1712.04578) Table I gives the generation
parameters, drawn independently per example:

| variable | distribution |
|---|---|
| alpha (RRC roll-off) | **U(0.1, 0.4)** |
| delta_t (timing) | U(0, 16) |
| delta_fs (symbol rate) | N(0, sigma_clk) |
| theta_c (carrier phase) | U(0, 2pi) |
| delta_fc (carrier freq) | N(0, sigma_clk) |
| H (multipath) | Rayleigh, tau in [0, 0.5, 1, 2] |

Two consequences. Our fixed beta = 0.35 is **inside** their range, so "we picked
the wrong roll-off" was never the explanation. And their table already
randomises timing, symbol rate, phase, frequency and multipath — which explains
why adding our versions of those in `ablate_impairments.py` recovered nothing.
They were already there.

The tempting arithmetic: occupied bandwidth is (1 + alpha) / sps. Ours at
beta = 0.35, sps = 8 is 0.169; RadioML measures about 0.12, implying sps ≈ 10.
And beta = 0.01 at sps = 8 gives 0.126 — seemingly explaining why an unphysical
roll-off transferred best.

**That arithmetic is wrong.** `match_radioml_params.py`, frozen model:

| generator | occupied BW | 16QAM | overall |
|---|---|---|---|
| ours as built (beta 0.35, sps 8) | 0.169 | 0.184 | 0.804 |
| **beta 0.01, sps 8** | **0.126** | **0.772** | **0.941** |
| paper roll-off only (a~U, sps 8) | 0.156 | 0.266 | 0.827 |
| paper symbol rate only (beta 0.35, sps 10) | 0.135 | 0.180 | 0.811 |
| **paper-matched (a~U, sps 10)** | **0.125** | **0.206** | **0.799** |
| paper-matched + timing + CFO + multipath | 0.125 | 0.208 | 0.651 |

The two configurations at 0.125 and 0.126 have **the same occupied bandwidth
and opposite outcomes** — 0.799 against 0.941. Matching the paper's stated
parameters closes nothing (−0.006); adding our approximations of their
impairments makes it worse, which says more about those approximations than
about the impairments.

So the envelope matters — the substitution test settles that — but it is not
captured by (1 + alpha) / sps. This confirms what `narrow_the_search.py` already
hinted and the bandwidth argument then overlooked: **RadioML's spectral shape is
not reachable by our RRC generator at any (alpha, sps).** Parametric matching
fails; direct spectral substitution works.

That is a sharper claim than "I matched their parameters", and it strengthens
the case for whitening: what cannot be matched parametrically is instead
removed from both sides.

## Note: h5py is blocked on this machine

`ImportError: DLL load failed ... An Application Control policy has blocked
this file.` AppLocker/WDAC refuses unsigned DLLs from user-writable paths, and
a virtualenv under the user profile is one. Reinstalling does not help.

Fixed permanently by `convert_radioml.py`: reads the HDF5 with **pyfive** (pure
Python, no DLLs) and writes `radioml_X.npy` / `_y` / `_z` once. `radioml.py`
now prefers those as memory-maps and imports no HDF5 library at all. Verified
byte-identical on random rows; loading is faster than the old path.

## Where an unseen modulation lands

Three observations in this project pointed the same way and were followed up:
under out-of-distribution input the classifier does not spread its predictions,
it lands on one class. What governs the choice took three experiments to pin
down, and the first answer was wrong.

### Round 1 — five classes (`sink_class.py`)

Leave-one-class-out on BPSK/QPSK/8PSK/16QAM/64QAM, probing with the removed
class drawn from RadioML itself, so it is in-distribution as a signal and
out-of-distribution only as a class.

- Nested removals: the sink matched the densest remaining class in **12/12**
  cells, at 94–100%.
- But those cells cannot discriminate: with a single density ordering, the
  densest remaining class is also the nearest one.
- **Discriminating control** — train on QPSK…64QAM, probe with BPSK. Nearest is
  QPSK, densest is 64QAM. Result: **QPSK at 94% and 99.9%**, zero to 64QAM.
  So it is adjacency, not density.
- **Permutation control** — shuffling which output index means which class and
  retraining left the sink on the same class 3/3. Not an output-layer artifact.
- Gaussian noise produces **no** sink (20–30% across five classes). "Unfamiliar
  modulation" and "no modulation" are different failure modes.

### Round 2 — all 24 classes (`sink_class_24.py`)

Five classes are a weak test of "adjacency", because there they form one
density chain in which nearest and next-densest coincide. RadioML's full set
has four digital families each carrying its own internal order, which separates
the two. Ten held-out classes, leave-one-class-out:

| held out | family | sink | share | verdict |
|---|---|---|---|---|
| 8ASK | ASK | 4ASK | 100.0% | order-adjacent |
| 16PSK | PSK | 32PSK | 90.2% | order-adjacent |
| 32PSK | PSK | 16PSK | 87.2% | order-adjacent |
| 32APSK | APSK | 128APSK | 52.3% | same family |
| 128APSK | APSK | 64APSK | 57.3% | order-adjacent |
| 32QAM | QAM | 128QAM | 65.2% | same family |
| 64QAM | QAM | 256QAM | 64.6% | same family |
| 256QAM | QAM | 64QAM | 72.2% | same family |
| OQPSK | other | 16APSK | 39.7% | **different family** |
| AM-DSB-SC | analog | AM-DSB-WC | 100.0% | order-adjacent |

**Same family 9/10. Order-adjacent only 5/10.**

The critical pair behaved as the family account predicts and the density
account does not: 32APSK and 32QAM have the same order in different families,
and **each stayed in its own** (→128APSK, →128QAM). Neither crossed.

So the Round 1 conclusion needs correcting. **Family membership is the rule;
order-adjacency is a weaker secondary tendency.** Inside QAM the sink routinely
skips the immediate neighbour (32QAM→128QAM, 64QAM→256QAM).

Concentration tracks how well-defined the family is — ASK and analog 100%,
PSK 87–90%, QAM/APSK 52–72%, and OQPSK, which has no real family here, 39.7%.
The rule accounts for its own exception.

### Round 2b — the complete sweep, and a selection bias

Round 2 used ten held-out classes that I chose. Running all 24 shows that
choice flattered the result:

| | chosen ten | **all 24** |
|---|---|---|
| same family | 9/10 (90%) | **18/24 (75%)** |
| order-adjacent | 5/10 | 11/24 |
| different family | 1/10 | **6/24** |

The effect is real but weaker than the subset suggested. Worth stating plainly:
I picked the cases I expected to be informative, and they were also the cases
where the rule held best.

**Against chance it is still large.** Under random assignment the probability of
landing in the same family is (family size − 1)/23 per class, giving an expected
3.5 of 24. Observed 18.

**The six departures are not random.** Each lands on a modulation my taxonomy
separates but physics does not:

| held out | sink | share | why |
|---|---|---|---|
| FM (analog) | GMSK | 96.7% | both constant-envelope frequency/phase modulation |
| GMSK (other) | 8PSK | 91.6% | GMSK is continuous-phase, close kin to PSK |
| BPSK (PSK) | OQPSK | 91.0% | OQPSK is offset QPSK |
| OOK (ASK) | 32APSK | 78.8% | on-off keying is an odd member of ASK |
| 16APSK (APSK) | 32QAM | 51.4% | APSK is ring-structured QAM |
| OQPSK (other) | 16APSK | 39.7% | diffuse, as it was on five classes |

So most of the misses are the taxonomy's fault, not the model's. The families
here were defined by hand and do not fully track structural similarity.

**Deliberately not re-scored.** Merging FM with GMSK, moving OQPSK into PSK and
treating APSK/QAM as one super-family would give 21/24 — but redrawing
categories after seeing the results is fitting the taxonomy to the data, and
the test stops being a test. The honest number is **18/24 under the taxonomy
fixed in advance**.

The right fix is to define the similarity structure independently — a distance
computed from constellation geometry, or a standard modulation hierarchy —
and re-run with categories fixed before looking. That is future work, not a
rescoring.

**Revised claim:** an unseen modulation lands on a **structurally similar** one,
not on a random class. The hand-built family taxonomy captures that 18 times in
24 against a chance expectation of 3.5, and its failures are all cases where
the taxonomy and the physics disagree.

### Round 2c — defining similarity without a hand-drawn taxonomy

The 18/24 result depends on families I drew by hand, and the six misses were all
taxonomy/physics disagreements. Redrawing families to fit would be re-scoring,
so instead the similarity structure was defined independently, twice.

**Hand-built geometric metric (`sink_vs_geometry.py`).** A rotation- and
scale-invariant constellation-shape signature (histogram of pairwise point
distances), distance = L1 between signatures. This was **worse**, top-3 12/24 —
the crude signature ranked two same-family QAMs far apart and confused an ASK
line with a QAM grid. Notable that hand-drawn families (18/24) beat this
automatic metric; the pairwise-distance histogram is too weak a shape
descriptor.

**The model's own embedding (`sink_vs_embedding.py`).** Ask the model what it
finds similar, with the circularity broken by using two different models: the
**sinks** come from the leave-one-out runs (24 models, each missing one class),
while the **embedding geometry** comes from one model trained on all 24 — so
"which classes sit near each other" is answered by a model that never removed
any class. Prototype = mean 256-d pre-logit feature per class; distance =
Euclidean between prototypes.

| similarity defined by | top-1 | top-3 |
|---|---|---|
| hand-built family taxonomy | — | 18/24 |
| hand-built geometry metric | — | 12/24 |
| **model's own embedding** | **16/24** | **19/24** |

Chance for top-3 is ≈ 3. In the model's learned space the sink is the single
nearest class 16 times in 24, median rank 1. So the sink is not "same family" by
my definition of family — it is **the nearest class in the representation the
model actually learned**, which is a stronger and taxonomy-free statement.

**Honest caveat.** Both models share an architecture and training set, so this
is a property of *this model class*, not of modulation in the abstract. A
different architecture might carve the space differently. The five remaining
outliers (OOK, QPSK→16PSK, GMSK→8PSK, OQPSK) are family-edge or
constant-envelope cases, and the model does place them far in its own space too
— so those sinks are genuinely surprising, not just taxonomy artifacts.

### Round 2d — is it an artifact of the architecture? (`sink_across_architectures.py`)

The remaining caveat was that all of this used IQNet, a 1-D CNN, so the family
structure might come from convolutional locality. Four architectures with
different inductive biases were compared: IQNet (CNN), ResNet1D (deeper residual
CNN), GRUNet (recurrent), TransformerNet (attention, no locality prior). Six
held-out classes, fixed in advance by rule — top of each digital family plus the
two hard non-digital cases (FM, OQPSK, which were the IQNet misses, included on
purpose).

| held out | IQNet | ResNet1D | GRU | Transformer |
|---|---|---|---|---|
| 8ASK | 4ASK ✓ | 4ASK ✓ | 4ASK ✓ | 4ASK ✓ |
| 32PSK | 16PSK ✓ | 16PSK ✓ | 16PSK ✓ | 16PSK ✓ |
| 128APSK | 64APSK ✓ | 64APSK ✓ | 256QAM ✗ | 64APSK ✓ |
| 256QAM | 64QAM ✓ | 64QAM ✓ | 128APSK ✗ | 64QAM ✓ |
| FM | GMSK ✗ | GMSK ✗ | GMSK ✗ | GMSK ✗ |
| OQPSK | 16APSK ✗ | 16APSK ✗ | QPSK ✗ | 16APSK ✗ |

same-family: IQNet 4/6, ResNet1D 4/6, Transformer 4/6, GRU 2/6.

**The Transformer matches the CNN cell for cell**, including the two misses.
Attention has no locality prior, so a family-structured sink cannot be coming
from convolutional locality. Three architecture families (conv, residual conv,
attention) produce the *same sinks* — and, tellingly, the **same mistakes**:
FM→GMSK and OQPSK→16APSK appear in three of four. If this were an architecture
artifact, different architectures would fail in different places; they fail in
the same place, so the effect lives in the modulations, not the model.

GRU is the one that drops (4/6 → 2/6), and consistently: its in-domain accuracy
was lower and its two misses are the highest-order classes (128APSK, 256QAM),
the hardest for a recurrent model to resolve over 1024 samples. Same pattern as
Round 3 — the sink is family-structured when the class is well learned and
diffuse when it is not.

**The architecture caveat is largely closed.** Structurally-similar landing of
an unseen modulation is not a property of convolutional locality; it survives
across conv, residual, and attention models.

### Making it useful — family recovery (`family_recovery.py`)

The sink experiments characterise a failure. The obvious question is "so what?"
A predictable failure is only interesting if it can be acted on. Claim tested:
even when the model cannot identify an unseen modulation, its prediction still
carries correct **family-level** information. Leave-one-out over all 24 classes,
full prediction distribution stored.

Digital classes only (ASK/PSK/APSK/QAM, 17 classes — the families with a clean
physical taxonomy):

| reading | result | chance |
|---|---|---|
| family via top-1 prediction | 14/17 = 82% | 0.15 |
| family via prediction mass | 12/17 = 71% | 0.15 |
| mean probability mass on correct family | **0.74** | 0.15 |

An unseen digital modulation puts, on average, 74% of its prediction mass on the
correct family — five times chance. The exact class is unrecoverable (never
trained); the family is not.

Two details:

- **top-1 beats mass** (82% vs 71%), against expectation, because of APSK:
  64APSK and 128APSK have the right top-1 (APSK) but their mass leaks to QAM
  (0.48, 0.49). APSK is ring-based QAM, so the votes split across the two — more
  evidence that the APSK/QAM boundary is genuinely blurred.
- **Analog is perfect** (0.99–1.00): the four AM classes put essentially all
  mass in the analog family, with no digital/analog cross-leakage.

**Honest limits.** BPSK (0.12), FM (0.09), OQPSK and GMSK (0.00) fail — the same
family-edge and "other" cases that were the sink outliers. Recovery is strong
exactly where the family structure is strong, which is why the headline is
digital-only. Framed as a usable result: **structured failure converts into
partial recognition — the family survives even when the class does not.**

### Round 3 — is it an artifact of weak training? (`sink_class_24_control.py`)

Round 2 used 256 frames/cell and 15 epochs so that ten models would fit in an
evening. Four cases repeated at 512 frames/cell and 25 epochs:

| held out | cheap sink | strong sink | agree | share |
|---|---|---|---|---|
| 32APSK | 128APSK | 128APSK | ✓ | 52.3% → **73.9%** |
| 32QAM | 128QAM | 128QAM | ✓ | 65.2% → **80.5%** |
| 64QAM | 256QAM | 256QAM | ✓ | 64.6% → **72.6%** |
| OQPSK | 16APSK | 128QAM | ✗ | 39.7% → 33.9% |

**3/4 unchanged, and the three family cases concentrate more, not less.** If
the family structure came from the model failing to learn finer features, more
training would weaken it. It strengthens. OQPSK moves, but it was always the
diffuse case and stays diffuse (33.9 / 17.9 / 14.2%) — a class with no family
has no stable home, which is what the rule predicts.

**A correction to my own caveat.** Round 2's in-domain accuracy of 0.53–0.55
looked alarmingly low, but that is the average over *all* SNRs from −20 to
+30 dB, including the region where the task is unsolvable. The earlier full
24-class run averaged 0.5625 the same way while reaching 0.879 at SNR ≥ 12 dB.
The models were never undertrained; I had compared an all-SNR average against a
high-SNR figure.

### What this supports

> A modulation classifier meeting a class it was never trained on does not err
> diffusely. It lands inside the **same modulation family** (9/10), the
> concentration tracking how cleanly that family is defined, and the effect
> grows with better training rather than washing out.

The practical consequence is that the errors are **systematically biased, not
random** — a monitoring system in unfamiliar conditions will over-report
specific classes in a predictable direction, which is a different and more
serious statement than "the model is sometimes wrong".

Searches turned up related ideas (neural collapse, minority collapse, per-class
coverage collapse) but nothing on *which* class a classifier defaults to under
distribution shift, or whether it is predictable from class structure. That is
weak evidence — a proper literature review has not been done.

## Next: remaining work

The same thing showed up three times today and was never followed up:

| where | what happened |
|---|---|
| cross-domain | 75% of our 16QAM predicted as **64QAM** |
| `sweep.py`, sps=16 | 64QAM **0.999**, QPSK **0.042** — far below the 0.20 chance level |
| spectrogram CNN | four classes collapsed into one block |

Under out-of-distribution input the classifier does not spread predictions
randomly. It falls onto one particular class, and each time it is the densest
constellation available.

**Hypothesis.** The default class under OOD input is the highest-entropy
constellation in the training set, because that is the most general explanation
of unfamiliar structure.

**Test.** Train on different class subsets and present OOD input. If 64QAM is
removed, does the default become 16QAM? The existing infrastructure covers
this — `domains.py` already supports arbitrary class subsets.

**Why it matters.** The failure is systematically biased, not random. A
spectrum monitor running in unfamiliar conditions would persistently
over-report high-order QAM. That is a different and more serious claim than
"the model sometimes errs".

A search for prior work found related ideas (neural collapse, minority
collapse, per-class coverage collapse) but nothing on *which* class a
classifier defaults to under OOD input, or whether it is predictable from class
structure. Absence of search results is weak evidence — this needs a proper
literature review before any novelty claim.

## Next: hardware (from mid-August, at NJIT)

A programmable RF impairment generator — the point being that the board should
*produce* the domain variation this project studies, in measured physical
units, rather than be generic RF hardware bolted on.

```
Pluto TX -> [PE4302 digital step attenuator] -> [SPF5189Z amplifier]
         -> [switchable filter bank] -> [fixed 40 dB pad] -> Pluto RX
                      controlled by a Pi Pico over USB
```

- **Drive axis**: the step attenuator sweeps the amplifier from linear into
  compression. With ~0 dBm from the Pluto and ~19 dB of gain, output runs
  +17.5 dBm down to −14 dBm, straddling the amplifier's P1dB.
- **Envelope axis**: filters of different *order* at the same nominal cutoff,
  not different bandwidths. Bandwidth variation at 433 MHz needs loaded Q in
  the hundreds, which lumped LC cannot deliver; order variation changes skirt
  steepness, which is the axis this project actually measured.

De-risking: prototype with off-the-shelf breakout modules first, integrate onto
a PCB only once the concept is verified. The board must remain an improvement,
never load-bearing.

Deliverables that form the hardware chapter: S21 per filter path, S21 vs
attenuator setting, and the amplifier's P_out vs P_in curve. The last one turns
the x-axis of the accuracy plot from a simulation knob into a measured physical
quantity.

---

## Data

**RadioML 2018.01A** — 24 classes, ~2M frames of 1024 IQ samples, 26 SNR
levels, ~20 GB HDF5, complex floating point.
From <https://www.deepsig.ai/datasets> (registration required); also mirrored
on Kaggle.

Note: DeepSig's own description says the released file contains **synthetic
simulated channel effects only**. The "over-the-air" phrase belongs to the
title of the associated paper, not to the contents of the download. The second
domain for the generalization study therefore has to come from elsewhere:
either an SDR capture (Week 5) or `modem.py`, which is an independent
generator we fully control.

`FRAME_LEN = 1024` in `baseline.py` already matches this dataset, so the
feature pipeline transfers without modification.

### Verified on download

`X (2555904, 1024, 2) float32`, `Y` one-hot over 24 classes, `Z` SNR from −20
to +30 dB in 2 dB steps. Perfectly balanced: 4096 frames in every
(class, SNR) cell, 106 496 per class, 24 × 26 × 4096 = 2 555 904 exactly.

**Class names are not stored in the file** — the ordering is an assumption
taken from the paper, so it was checked rather than trusted
(`verify_classes.py`, `class_spectra.py`):

- Index 5 gives |C40| = 0.104. The fourth-order cumulant of 8PSK vanishes
  analytically. Strong confirmation.
- Indices 0–3 (OOK, 4ASK, 8ASK, BPSK) all give |C20| ≈ 0.98 — real-valued, as
  ASK and BPSK must be. Indices 4–16 all give |C20| ≈ 0.1 — complex.
- Indices 19/20 show the symmetric twin-peak spectrum of double sideband;
  17/18 show a single narrow line. The DSB/SSB assignment is unambiguous.
- Index 21 (FM) has σ_aa = 0.005 and index 22 (GMSK) 0.051 — both effectively
  constant-envelope, as those schemes must be.
- Square vs cross constellations behave correctly: |C40| falls monotonically
  for square QAM (16QAM 0.482, 64QAM 0.437, 256QAM 0.425) while the cross
  constellations sit apart (32QAM 0.196, 128QAM 0.202).

Conclusion: the ordering in `radioml.CLASSES` is safe to cite.

**One methodological note worth keeping in the report.** The |C20| real/complex
test failed on AM-SSB-WC and AM-SSB-SC (0.78 instead of ≈0), which initially
looked like a labelling error. It was not — RadioML's SSB is extremely
narrowband and carrier-dominated, and over a 1024-sample frame a near-DC tone
does not decorrelate, so E[x²] never averages away. |C20| is only a valid
discriminator for signals occupying a reasonable fraction of the band. A
summary statistic can be fooled; the spectrum could not be.

### The Week 3–4 result should replicate, and get bigger

In `figures/08_radioml_class_spectra.png`, all fourteen of BPSK, QPSK, 8PSK,
16PSK, 32PSK, 16/32/64/128APSK, 16/32/64/128/256QAM plus GMSK and OQPSK have
visually identical symmetric flat-topped spectra. The same phenomenon found
synthetically, now on real data — and here it covers **16 of 24 classes**
rather than 4 of 11. Prediction: `SpecNet` on RadioML should saturate far
below `IQNet`, with the confusion mass concentrated in that block.

# Literature check

**Status: partial, not a systematic review.** What follows is a targeted search
against four specific claims, done with web search plus reading the full text
of the papers that looked closest. It is not exhaustive.

**Known blind spot, corrected.** An earlier version of this file said IEEE
Xplore "was not accessible here". That was a limit of the tooling that ran the
searches, not of this project: NJIT holds an Xplore subscription, and the three
papers added under Q4 below were downloaded through it. The accurate statement
is narrower and still worth keeping — **the search behind Q1-Q3 was
arXiv-weighted**, so coverage is good on recent ML-flavoured work and thin on
older or Xplore-only signal-processing work, and **no systematic Xplore pass
has been done yet**. "Not found" here means "not found with these search terms
in the sources actually searched", never "does not exist".

Four questions, one per claim this project makes.

---

## Q1 — Has anyone attributed an AMC domain gap to a specific signal property by controlled intervention?

**Not found.** The pattern across everything read is: measure a gap, propose a
method, report the improvement. Attribution of the gap to one property, by
intervening on that property and holding the rest fixed, did not appear.

Closest: **WhiteNet** (below) attributes its gap to the "channel envelope" and
supports that with a t-SNE separability measure — session separability drops
from 88.9% to 59.4% after whitening. That is correlational evidence about the
learned representation.

This project's phase-preserving magnitude substitution is an intervention on
the signal rather than an observation of the representation: keep the phase
spectrum exactly, swap only the magnitude, and see whether the prediction
follows. 16QAM went 0.191 → 0.972. Combined with the elimination of symbol-rate
offset, constellation density and channel impairments, this is a causal
argument rather than a correlational one.

**Assessment:** this looks like the most distinctive part of the work, with the
caveat that a methodological contribution is harder to place than a numerical
one.

---

## Q2 — Is spectral whitening already used for RF domain robustness?

**Yes. Found, and it is essentially the same operation.**

**WhiteNet: Robust Identification of Overlapping IEEE 802.11 Signals Across
Unseen Channels** — arXiv:2608.06581

Their transform:

    Y~[m] = Y[m] · sqrt(P̄) / sqrt(P̂[m])

with P̂ a circular moving-average smoothed PSD estimate, window W = 256, noise
floor δ = −30 dB. Dividing the spectrum by its own smoothed envelope is exactly
the operation used here, arrived at independently.

**So "spectral whitening for RF domain generalization" is not a novel proposal.
That claim has to be dropped.**

What differs, and the differences are real:

| | WhiteNet | this project |
|---|---|---|
| task | multi-label 802.11 protocol ID | modulation classification |
| domain shift | **channel variability** (frequency-selective fading, different capture sessions) | **transmitter mismatch** (pulse shaping / spectral envelope) |
| data | **real over-the-air captures**, 3 sessions | RadioML (synthetic) + own generator (synthetic) |
| whitening strength | full (exponent 1) | **swept 0→1; 0.75 beats 1.0** (0.907 vs 0.827) |
| attribution | t-SNE session separability | phase-preserving magnitude substitution |
| headline | 87.6% → 47.6% on unseen channels, recovered to 73.6% held-out | gap 0.182 → 0.038 |

Two things worth keeping:

1. **They whiten fully; the sweep here says that is suboptimal.** At alpha = 1
   in-domain accuracy collapses to 0.879 while 0.75 holds 0.964 with better
   cross-domain. That is a concrete, testable disagreement — on a different task,
   so not a contradiction, but worth reporting.
2. **Their domain shift is the channel; this one is the transmitter.** Related
   but not the same cause, and the fix working for both is itself informative.

Their result is stronger in the way that matters most: **their data is real.**

---

## Q3 — Is there work on *which* class a model defaults to under distribution shift?

**Adjacent field identified: open-set recognition (OSR).** That is the
established literature for what a classifier does when it meets a class it was
never trained on. It was not on my radar before this search and it belongs in
any write-up.

But OSR asks a different question. It is about **detecting** that a sample is
unknown — scoring, thresholding, rejection. Prototype methods such as Nearest
Class Mean assign a sample to the nearest class mean in feature space, which is
related in spirit, but the literature treats that assignment as machinery for
detection rather than as an object of study.

Also relevant: **Deep Open Set Identification for RF Devices** (arXiv:2112.02536)
— OSR applied to RF, though to device identification rather than modulation.

**Not found:** work asking whether the landing class is *predictable from the
taxonomic structure of the label set*. The finding here — that an unseen
modulation lands inside its own modulation family 9/10, verified with a control
where family and density predict opposite answers, and with a label-permutation
control ruling out an output-index artifact — did not turn up.

**Confidence: low.** OSR is a large field and a few searches do not cover it.
This needs a proper pass before any novelty claim, and it is the obvious
question to put to Prof. Abdi.

### Q3 follow-up — a closer read of OSR (second pass)

A more targeted pass sharpened the picture. The distinction that matters:

**OSR treats "unknown lands on similar known class" as a problem to prevent;
this project treats it as a phenomenon to measure and predict.**

A survey-level statement found in the search captures the field's stance
exactly: the core OSR challenge is *preventing* unknown classes — especially
those semantically similar — from being misclassified as their nearest known
neighbour. So the field is aware the confusion is similarity-structured; it
works to suppress it, not to characterise it.

Papers read on this pass:

- **OSNN (Open Set Nearest Neighbour).** Assigns a test sample to its nearest
  known class in feature space unless a distance ratio exceeds a threshold, in
  which case it is rejected. This is the closest prior mechanism: it confirms
  that "unknown resembles nearest known" is long established. But OSNN uses that
  as a *detection rule* — the nearest-class assignment is machinery for deciding
  known-vs-unknown, never studied as a structured, predictable phenomenon.

- **Liu et al., KTH 2026 (arXiv:2603.24268), OSR for unknown UAVs via RF
  semantics.** Same field as this project (RF), but the goal is detect / reject
  / cluster-then-learn. Its confusion matrix is used only to show knowns are
  right and unknowns are caught. It does **not** ask which known class an
  unknown resembles, nor predict it from signal structure.

- **Domain Feature Collapse (arXiv:2512.04034).** Name is close, content is not:
  an information-theoretic account of why OOD *detection* fails on single-domain
  models, on images. Not about which class OOD inputs land on.

**Where this leaves the claim.** "Unknown lands on nearest known class" is not
new — OSNN encodes it. What was not found:

1. Treating the landing class as a **structured, predictable** quantity rather
   than as detection machinery.
2. **Predicting it** from an independent notion of class similarity — here shown
   three ways: hand-drawn family (18/24), the model's own embedding
   (19/24 top-3, circularity broken), and stability across four architectures
   including attention (same sinks, same mistakes).
3. In **AMC specifically**, where the taxonomy (ASK/PSK/APSK/QAM families,
   constellation order) is unusually clean and physically grounded, so
   "structurally similar" has a precise meaning.

**Revised confidence: still cautious, but the gap looks narrower and more
specific than "nobody has looked".** The honest framing is not "a new
phenomenon" but "a known confusion, characterised and predicted rather than
suppressed, in a domain with a clean taxonomy." Whether that clears the bar for
novelty is exactly the judgement to put to Abdi — it is a positioning question,
not a measurement one, and he will know the OSR corner far better than these
searches reach.

---

## Q4 — What is the state of the art on cross-domain AMC?

Active, and more crowded than assumed at the start.

- **Adversarial adaptation.** DANN-style domain-adversarial training, e.g.
  arXiv:2508.06829 for AMC under channel variability. Requires unlabelled
  target data at fit time.
- **Metric learning / embeddings.** Henneke & Kurth, arXiv:2510.23186 — train
  on synthetic wireless protocols with an angular-margin loss, validate on
  **real** RF captures. Baselines include a 26-dimensional higher-order
  moment/cumulant vector, close to the classical baseline used here, and the
  spectral correlation function. No whitening, no gap attribution.
- **Augmentation.** arXiv:1912.03026 — rotation, flip, Gaussian noise, the set
  reproduced here as a baseline.
- **Whitening.** WhiteNet, above.
- **Dataset synthesis.** MDM, arXiv:2408.02714 — multi-domain distribution
  matching for AMC dataset synthesis.
- **Uncertainty / OOD.** arXiv:2608.00796 and others; an active 2025–26 line.

Two facts worth internalising:

1. **Synthetic-to-real is a recognised problem with real solutions already
   published.** Henneke & Kurth do the synthetic→real experiment this project
   cannot yet do.
2. **The published roll-off practice** of drawing alpha from a range, noted in
   an earlier search and confirmed in O'Shea's Table I as U(0.1, 0.4), means
   pulse-shaping variation is known to matter. That it *causes* a specific
   confusion between adjacent QAM orders is the part not seen elsewhere.

### Q4 follow-up — three Xplore papers read in full (September 2026)

Read end to end, not by abstract. Each one is a cross-domain AMC paper, and the
useful thing is not any single result but **which domain difference each one
varies**.

- **Wang, Xing, Wang, Zhou, Hou & Jiao — "SigDA: A Superimposed Domain
  Adaptation Framework for AMC".** *IEEE Trans. Wireless Communications* 23(10),
  Oct 2024. Code and data at github.com/HantongXING/SigDA. The strongest of the
  three and the most directly relevant. Its thesis is that prior work varies a
  *single* domain-difference factor while real shifts superimpose several, so it
  overlays them: **channel type, SNR, carrier frequency offset, sampling rate
  offset**, plus sampling duration and whether the data was energy-normalised.
  Adversarial alignment, so unlabelled target data is required at fit time.
  Two findings worth carrying:
  1. RML2016.10a -> RML2016.04c collapses to **11.58% on 11 classes** — chance.
     A total cross-domain collapse between two *published* RadioML sets, which
     is the cleanest available evidence that this failure is not an artifact of
     a home-made generator. Their DA recovers >0 dB accuracy from 12.84% to
     84.82%, and to 90.40% with their own backbone.
  2. They generate AWGN / Rician / Rayleigh variants using RadioML's own public
     generator (github.com/radioML/dataset) — a ready recipe for the third-domain
     experiment in README "Open work" item 6.

- **Deng, Li, Wang & Huang — "Cross-Domain AMC: A Multimodal-Information-Based
  Progressive Unsupervised Domain Adaptation Network" (PMSPDMC).** *IEEE
  Internet of Things Journal* 12(5), Mar 2025. Four modalities (IQ, amplitude-
  phase, spectrum, combined) treated as separate source domains, aligned to the
  target with a class-structure-aware MMD variant, then fused. **Its domain
  shift is symbol rate alone** — 25 kHz source, 50 kHz target, every other
  signal parameter held equal. Both domains simulated, Rician channel. Average
  accuracy 94.67% against ~69.5% source-only; a DANN-based competitor
  (CWDA-MR) already reaches ~93.2%, so the paper's own margin is 1.44 points.
  Also introduces **partial domain adaptation** — target has fewer classes than
  source, and the leftover "source outlier classes" cause negative transfer.
  That is a cousin of the sink work's setup and a term worth having.

- **Zhang, Yin, Yang, Wu & Zhao — "Unsupervised Domain Adaptation based
  Modulation Classification for Overlapped Signals".** *DSA 2022*, two pages.
  Applies DAN (Long et al., ICML 2015) to overlapped-signal AMC; **its domain
  shift is the channel** — trained on AWGN, tested on COST207 multipath. Treat
  with care: the method is off-the-shelf, the results section reports no
  numbers in the body, and the conclusion section describes a different paper
  entirely (jamming-signal classification with 15 CNN features). Cited here for
  the factor table below and for CORAL / MK-MMD as vocabulary, not as support
  for anything.

**What the three have in common, and what they do not:**

| paper | venue | domain difference varied |
|---|---|---|
| Zhang et al. 2022 | DSA (conf., 2 pp.) | channel (AWGN -> multipath) |
| Wang et al. 2024 | IEEE TWC | channel type, SNR, CFO, sampling rate, duration, energy norm. |
| Deng et al. 2025 | IEEE IoT J. | symbol rate |
| **this project** | — | **pulse-shaping roll-off / spectral envelope** |

All three require target-domain samples at fit time; whitening requires none.
And **none of the three varies pulse shaping** — including SigDA, which
explicitly sets out to superimpose the typical factors.

Two caveats that have to travel with that observation:

1. **Three papers are not a survey.** The honest phrasing is "I have not found
   the paper that isolates it", never "the field has not looked".
2. **SigDA's Table II was not verifiable here.** The tables in that PDF are
   images and did not extract as text, so the factor list above comes from the
   body prose. Table II lists the differences between the two RadioML sets and
   must be read by eye before the claim is repeated anywhere.

---

## Where this leaves the project

**Drop:** any claim that spectral whitening is a new idea for RF domain
robustness. It is published, on real data, with better results in its own
setting.

**Keep, with support:**

- The **attribution method** — causal intervention rather than correlational
  evidence. Q1 found nothing comparable.
- **Partial beats full whitening** — a concrete disagreement with WhiteNet's
  choice, measured over four seeds.
- The **family-structured failure** — Q3 found the adjacent field but not this
  question. Lowest confidence, highest potential.
- The **negative results**: symbol-rate offset, constellation density and
  channel impairments each ruled out by controlled test. Literature tends to
  attribute domain gaps to channel effects; this measured that they do not
  account for it here.

**Weakest point, unchanged:** both domains are synthetic. Henneke & Kurth and
WhiteNet both use real captures. That gap is the difference between a study and
a result, and it is what the SDR is for.

---

## What was searched

Terms covering: spectral whitening / spectrum flattening + AMC; domain
generalization + RF signal classification; domain adaptation + modulation
classification; sink class / default class / collapse under distribution shift;
open-set recognition + nearest class; RadioML generation parameters.

Full text read: O'Shea, Roy & Clancy (arXiv:1712.04578); Henneke & Kurth
(arXiv:2510.23186); WhiteNet (arXiv:2608.06581, HTML). Abstract or summary
only: the remainder.

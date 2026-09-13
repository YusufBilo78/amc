# Automatic Modulation Classification — Domain Generalization Study

## The claim this project is trying to establish

> A classifier that reaches ~99% on the RadioML benchmark loses a large part of
> that accuracy on IQ generated with a different transmitter. Quantify the
> collapse, attribute it to a specific signal property by intervention rather
> than correlation, then close it.

Reproducing RadioML accuracy is not the contribution — that is a solved
benchmark. The contribution is measuring, **explaining** and partially closing
the generalization gap between the training domain and an independent one.

---

## Status

The backbone is `model_zoo.ICRNNA` — convolutional front end, bidirectional
LSTM, additive attention, 786k parameters. Call it through
`model_zoo.backbone(n_classes)` rather than by name, so the default lives in one
place.

**Measurements are being redone on this backbone.** Everything below describes
methods, controls and findings that are established in approach but whose
numbers are being re-measured. Only the table immediately following is current.

### Currently measured

`compare_methods.py --arch ICRNNA --seeds 5 --epochs 60 --patience 20`,
17 of 20 cells complete (`compare_methods_ICRNNA_es.npz`):

| method | in-domain | cross-domain | gap | 16QAM |
|---|---|---|---|---|
| none | 0.999 | 0.804 | **+0.195** | 0.060 |
| standard augmentation | 1.000 | 0.806 | +0.194 | 0.036 |
| whitening α=0.75 | 1.000 | **0.993** | **+0.007** | **0.991** |
| whitening + standard (2/5 seeds) | 0.999 | 0.990 | +0.008 | 0.996 |

Three readings:

1. **The gap is real and large.** A model at 0.999 in-domain drops to 0.804 on
   an independently generated domain, and 16QAM — the class the whole diagnosis
   centres on — collapses to 0.060.
2. **The literature augmentation set does not address it.** Rotation, conjugate
   flip and additive Gaussian noise (arXiv:1912.03026) moved the gap by 0.001.
   Whatever these transforms are good for, this failure mode is not it.
3. **Whitening costs nothing in-domain.** In-domain accuracy stays at 1.000
   while the gap falls to +0.007, so the recovery is not bought by trading away
   benchmark performance.

A cross-domain accuracy of 0.993 sits close enough to the ceiling that it
deserves a sceptical pass of its own before it goes in a report.

### Plain classification on RML2016.10a

`train_backbone.py --data rml2016 --seeds 3 --epochs 60 --patience 20`, all
1000 frames per (class, SNR) cell, 70/15/15, run on Colab
(`train_backbone_rml2016_f1000_colab.npz`, 3/3 seeds):

| | accuracy |
|---|---|
| overall, all SNRs | **0.6223 ± 0.0019** (0.6248, 0.6200, 0.6222) |
| SNR ≥ 10 dB | **0.9148 ± 0.0013** |

The curve is the shape this dataset is supposed to give: chance (0.091, eleven
classes) below −16 dB, rising from −12 dB, past 0.90 at +2 dB, then flat at
0.915 to the top of the range. Seed spread never exceeds 0.014.

**What this establishes, and what it does not.** El-Haryqy et al. report 63.24%
for ICRNNA on this dataset and this lands 1.01 points below it. That is a
sanity check on the pipeline, *not* a reproduction of the paper: this backbone
differs from the published description in five places (see the provenance
caveat below) and the training protocol differs too — batch 256 against the
paper's 32, LR schedule on validation accuracy against validation loss. The
build that can actually answer the reproduction question is
`colab/icrnna_faithful_2016.py`, still unrun.

The high-SNR plateau is a property of the dataset rather than of the model.
WBFM and AM-DSB are both generated from voice recordings, and during the
silences in the source audio both degrade to an unmodulated carrier, which no
architecture can separate; ~0.92 is the practical ceiling this imposes and is
where the plateau sits.

Almost all of the remaining error is one cell. **WBFM is read as AM-DSB 60% of
the time**, and that single cell is 64% of every misclassified frame at SNR ≥ 10
dB; drop WBFM and the other ten classes sit at 0.968. This is the documented
dataset artifact rather than a property of the model — both classes are
generated from voice recordings, and during the silences in the source audio
both collapse to an unmodulated carrier. QAM16 ↔ QAM64 accounts for most of the
rest, at 0.06 and 0.08.

### Plain classification on RadioML 2018.01A

`train_backbone.py --data rml2018 --seeds 3 --epochs 60 --patience 20`, 512 of
the 4096 frames per (class, SNR) cell, native 1024-sample frames, 70/15/15, run
on Colab (`train_backbone_rml2018_f512_colab.npz`, 3/3 seeds):

| | accuracy |
|---|---|
| overall, all 26 SNRs | **0.5699 ± 0.0081** (0.5616, 0.5809, 0.5672) |
| SNR ≥ 10 dB | **0.8716 ± 0.0141** |

Chance is 0.042 with 24 classes, and the curve sits there below −14 dB. It
lifts from −12, passes 0.526 at 0 dB and 0.80 at +6, then flattens at 0.87 from
+8 dB to the top of the range, peaking at 0.880 at +12 dB.

Seed spread reaches 0.021, against 0.014 on 2016. Three seeds on 24 classes is
thin, and a difference of less than about 0.03 between two configurations
measured this way is not yet a difference.

**Against the literature.** Deep residual networks trained on the *whole*
dataset are reported at roughly 95% above 8 dB. This run is about 8 points
under that on 12.5% of the frames with a 786k-parameter recurrent model, and
two things differ at once — data volume and architecture — so the deficit is
not attributable to either. Closing it is not this project's goal; the number
exists so that the cross-domain differences measured elsewhere are known to
start from a competent classifier rather than a broken one.

Wall clock: about 9–13 minutes per seed on a Colab GPU, including a one-off
sequential copy of the 21 GB HDF5 to local disk and the staging of 512 frames
per cell from it.

#### Where the 2018 error goes, and why it is worth a look

Pooling the three seeds at SNR ≥ 10 dB, **84.2% of all error mass lands inside
the same modulation family** — using the taxonomy in `sink_class_24.py`, which
was written before any of this was measured. If misclassifications were spread
uniformly over the other 23 classes the share would be 16.6%, so this is 5.1×
that baseline. Family-level recall is 1.000 for ASK, PSK and analog, 0.988 for
QAM, and 0.896 for APSK.

Two groups carry three quarters of it. The **with-carrier / suppressed-carrier
analog pairs** (AM-DSB-WC ↔ AM-DSB-SC, AM-SSB-WC ↔ AM-SSB-SC) are 40.9%; drop
those four classes and the remaining twenty sit at 0.909. **Confusion inside the
QAM family** — mostly 64QAM ↔ 256QAM at 0.31 and 0.25 — is another 35.4%.

The exceptions are the interesting part. Every cross-family leak above 0.02 is
APSK → QAM *at matched constellation order*: 16APSK → 16QAM at 0.25, 64APSK →
64QAM at 0.10, 128QAM ↔ 128APSK at 0.03 and 0.02. Where the model leaves the
family it keeps the order.

**This is not the sink experiment rerun.** That one holds a class out of
training entirely and asks where an *unseen* modulation lands; this is ordinary
in-distribution error on classes the model was trained on. The two are
consistent, and the second is much weaker evidence than the first — errors
falling between similar classes is what any classifier does. What is worth
recording is the size of the effect and that the order-matched APSK → QAM leaks
survive it. Open work item 3 is still open.

### The faithful build, measured

`colab/icrnna_faithful_2016.py`, built from the paper rather than from a
reproduction, run under the paper's own hyperparameters — batch 32, 58 epochs,
LR on validation *loss* with patience 5 — across three seeds
(`icrnna_faithful_results.json`):

| | |
|---|---|
| test accuracy | **61.75% ± 0.07** (61.79, 61.65, 61.82) |
| paper (Table 3) | 63.24% |
| difference | **−1.49 points** |
| parameters | 794,827 (the paper states 0.79M) |

The parameter count matches the paper's figure, which says the architecture was
read correctly. The accuracy does not, and three things make that worth stating
carefully rather than waving through:

1. **The deficit is not seed noise.** Spread across seeds is 0.07 points; the
   gap to the paper is 21× that. Whatever causes it is systematic.
2. **The file's own verdict — "within ~1.5 points: the reproduction stands" —
   passes by 0.01 points.** A threshold a result squeaks under is not evidence
   that the result is fine. It is written here as a near miss, because that is
   what it is.
3. **Two of the three seeds peaked at the final epoch** (best epoch 58, 58, 55)
   and early stopping at patience 15 never fired. The model was still improving
   when the budget ran out, so the paper's "Number of Epochs 58" is binding
   here.

#### It was the epoch budget

Same build, same everything, ceiling raised to 150 so that early stopping at
patience 15 decides where to stop (`icrnna_faithful_e150_results.json`):

| | 58-epoch ceiling | 150-epoch ceiling |
|---|---|---|
| best epoch | 58, 58, 55 — at the ceiling | **107** — early stopping decided |
| test accuracy | 61.75% ± 0.07 | **63.21%** |
| against the paper's 63.24% | −1.49 | **−0.03** |

**The faithful build reproduces the paper.** It needed nearly twice the
published epoch count to do it, and that is the whole of the 1.49-point
deficit; nothing about the architecture was involved. The paper's "Number of
Epochs 58" is best read as a report of where its own early stopping landed than
as a recipe, and it does not transfer.

The gain is everywhere the signal is usable — +1.4 to +3.6 points at every SNR
from −8 dB up, +1.78 across the plateau — and nothing at all below −10 dB,
where there is nothing to learn. Trained to convergence it also passes the peer
reproduction, 63.21% against 62.23% overall and 92.55% against 91.48% at
SNR ≥ 10 dB.

#### What that implies for this project's own numbers

An under-trained run reports a floor, not a measurement, and this one was under
the ceiling by 49 epochs without anyone noticing. The same question applies
here, and the arithmetic is not reassuring:

| run | batch | epochs | gradient updates |
|---|---|---|---|
| faithful ICRNNA, converged | 32 | 107 | **515k** |
| `train_backbone --data rml2016` | 256 | 60 ceiling | 36k |
| `train_backbone --data rml2018` | 256 | 60 ceiling | 52k |

Fourteen times fewer updates on the same dataset. That is not evidence those
runs were under-trained — early stopping at patience 20 may well have fired
long before 60 — but **nothing in the saved results says which**, because the
stopping epoch was never recorded. It is now: `cnn.train_model` reports where
it stopped, `train_backbone.py` stores `best_epochs` in its `.npz` and says
loudly when a seed peaked at the ceiling.

#### 2016 converged — by one epoch

Rerun at a 300-epoch ceiling with the stopping epoch recorded
(`train_backbone_rml2016_f1000_colab_e300.npz`, one seed): **best epoch 39,
early stop at 59**. The committed 60-epoch run's ceiling was 60. The result is
bit-identical to that run's seed 0 — 0.624758 overall and 0.914909 at
SNR ≥ 10 dB in both — so nothing after epoch 39 improved on it and the
committed number is a measurement, not a floor.

The margin was **one epoch**. Had the trajectory peaked two epochs later, or
patience been 21, the committed run would have been cut off at the ceiling and
nothing in the file would have said so. Sixty epochs is not a comfortable
default for this configuration; it happened to be enough.

The batch-256 regime converges far faster per update than the faithful build's
batch 32, so the 14× arithmetic above turned out not to matter here. It does
not follow that it does not matter on 2018, which is a harder problem — 24
classes against 11, 1024-sample frames against 128 — and where more epochs, not
fewer, would be the expectation. **2018 has not been checked.**

**Being faithful did not move it toward the paper.** Against the peer
reproduction in `model_zoo.ICRNNA` on the same dataset: 61.75% versus 62.23%
overall, and 90.76% versus 91.48% at SNR ≥ 10 dB. Correcting the five
differences toward the published description moved the number half a point
*away* from the published number.

That comparison is suggestive and nothing more, because the two runs differ in
protocol as well as architecture — batch 32 against 256, LR scheduled on
validation loss against validation accuracy, 58 fixed epochs against 60 with
patience 20, no mixed precision against AMP. It is not an architecture
comparison and must not be reported as one. What it does rule out is the
comfortable assumption that the five differences were holding the number back.

The two curves are indistinguishable in the noise-limited regime — 33.19%
against 33.29% averaged below 0 dB — and separate only across the plateau.
Whatever differs between them acts where the signal is clean.

---

## Setup

Python 3.12, virtual environment at `C:\Users\yusuf\.venvs\amc` (outside
OneDrive — packages are multi-GB and syncing them is painful).

```bash
cd src && "C:\Users\yusuf\.venvs\amc\Scripts\python.exe" compare_methods.py --arch ICRNNA --seeds 5 --epochs 60 --patience 20
```

Datasets live in `C:\Users\yusuf\amc-data` (also outside OneDrive, ~40 GB,
untracked). GPU: RTX 3070 Laptop, 8 GB. **Keep the machine on mains power** —
on battery the GPU is capped near 35 W and runs about 5× slower, measured.

`h5py` is blocked by this machine's application-control policy, which refuses
unsigned DLLs in a user-writable venv. `radioml.py` reads `.npy` memmaps
converted once by `convert_radioml.py`, falling back to pure-Python pyfive.

Training can also run on Colab: `colab/run_training.py` is a single cell that
clones this branch, finds the dataset in Drive, stages it to local disk and runs
`train_backbone.py` with `--out-dir` pointing back at Drive, so results survive
the runtime being reclaimed and a rerun resumes rather than restarts. The 21 GB
file does not have to be in Drive, and does not fit in a free one: no run reads
all of it, so `export_subset.py` carves out exactly the frames a given
`--frames-per-cell` would use — 2.6 GB at 512 — and both sides select those
frames with the same function at the same seed, so the two subsets hold the same
frames rather than merely the same number of them.

---

## Layout

```
src/modem.py            baseband IQ generator, 11 modulation classes
src/radioml.py          RadioML 2018.01A loader
src/rml2016.py          RML2016.10a loader, same interface as radioml.py
src/export_subset.py    carve an uploadable subset out of 2018
src/domains.py          domain abstraction: RadioML, synthetic, capture
src/features.py         cumulants and instantaneous features
src/augment.py          augmentation transforms, including whitening
src/model_zoo.py        ICRNNA, ResNet1D, GRU, Transformer, backbone()
src/cnn.py              training loop, splits, evaluation
src/train_backbone.py   plain classification on 2018 or 2016, current backbone
src/plots.py            the run's two figures, rebuildable from a saved .npz
src/crossdomain.py      the central train-on-A / test-on-B experiment
src/whitening_*.py      whitening and its alpha sweep
src/compare_methods.py  whitening against the literature baseline
src/sink_*.py           where an unseen modulation lands
colab/run_training.py   run train_backbone.py on Colab
colab/run_experiment.py run the open-work sweeps on Colab
colab/icrnna_faithful_2016.py  paper-faithful ICRNNA, for calibration
```

---

## Rules that keep this honest

- **Predict before measuring.** A hypothesis written down after seeing the
  number is not a hypothesis.
- **Negative results are kept.** Four explanations for the gap were tested and
  refuted; they are part of the argument, not failed drafts.
- **Controls that can come out the other way.** A control which confirms
  whatever happens is decoration.
- **Three-way splits wherever anything is selected.** `cnn.split(...,
  val_fraction=0.15)` returns `(train, val, test)`; early stopping, the LR
  schedule and checkpoint choice read validation only. Passing the test set
  there means selecting on the number being reported.
- **Long runs checkpoint and resume.** Every sweep writes partial results after
  each cell and skips finished ones on restart. This is a laptop, and runs have
  been lost to a flat battery.

---

## The diagnosis — attribution by intervention

The gap concentrates almost entirely in one cell: 16QAM read as 64QAM. Four
explanations were tested and each refuted by controlled experiment:

| # | hypothesis | test | outcome |
|---|---|---|---|
| 1 | occupied bandwidth differs | `match_radioml_params.py` | refuted — matching bandwidth did not recover 16QAM |
| 2 | constellation density | `diagnose_16qam.py` | refuted — density is not what the model reads |
| 3 | symbol-timing regularity | `test_timing_hypothesis.py` | refuted — RadioML has **no** symbol-rate spectral line, and driving the line down did not move the prediction |
| 4 | channel impairments (CFO, multipath, IQ imbalance) | `ablate_impairments.py` | refuted — removing them one at a time changed little |

What did work is the piece of this project with the least precedent in the
literature: **phase-preserving magnitude substitution** (`narrow_the_search.py`).
Take a frame from the failing domain, keep its phase spectrum exactly, and
substitute only the magnitude spectrum of the training domain. Nothing else
changes — not the symbols, not the timing, not the channel.

The prediction follows the magnitude spectrum. That is an intervention on the
signal rather than an observation of the representation, so it supports a causal
claim rather than a correlational one. Combined with the four refutations, the
cause is the **transmitter's spectral envelope**, chiefly its pulse-shaping
roll-off.

The interventions themselves are signal-level and do not depend on the
classifier; the accuracies attached to them are pending re-measurement.

---

## The fix

**Attempt 1 — randomise the envelope during training. Failed.** Documented with
its ablation rather than dropped, because the failure is informative: teaching
invariance by sampling nuisance parameters did not transfer.

**Attempt 2 — remove the envelope. Works.** Spectral whitening: divide each
frame's spectrum by a smoothed estimate of its own magnitude envelope,
`X / smooth(|X|)^α`, with a circular moving average so the wrap-around is not a
discontinuity.

α is swept from 0 to 1. **Partial whitening beat full whitening** in the earlier
sweep (α=0.75 against α=1.0), which is a concrete, measurable disagreement with
WhiteNet's choice of full whitening — on a different task, so not a
contradiction, but worth reporting. **The sweep has not yet been rerun, so
α=0.75 is currently unverified for the present backbone.**

### What the mechanism is not

The working account through most of the project was shortcut learning: the model
reads modulation order off the spectral envelope. Two experiments were designed
to confirm it and both killed it — most directly the envelope transplant, where
handing 16QAM frames a 64QAM envelope left the prediction at 16QAM. The envelope
does not *carry* the class; removing it still helps. That distinction is
unresolved and is stated as unresolved.

---

## Where an unseen modulation lands

Train on 23 of RadioML's 24 classes, probe with the held-out one, and record
which known class absorbs it. The finding: **an unseen modulation lands inside
its own modulation family** — QAM into QAM, PSK into PSK — far above the ~0.15
chance rate.

It was checked five independent ways, each able to come out the other way:

1. **Hand-drawn taxonomy** — families defined before any result was seen.
2. **The model's own embedding** — similarity read from learned features rather
   than from a hand taxonomy, with the circularity broken.
3. **Four architectures** — CNN, residual CNN, recurrent, attention-only. Same
   sinks, including the same mistakes.
4. **A discriminating control** — a case where "nearest family" and "densest
   class" predict *opposite* answers. Family won.
5. **Label permutation** — ruling out an output-index artifact.

Made useful by family recovery (`family_recovery.py`): even when the exact class
is unrecoverable, summing prediction mass over each family recovers the correct
family well above chance.

**None of this has been rerun on the current backbone.** It is the most
expensive thread to redo — 24-class leave-one-out — and the most novel, so it is
also where re-measurement matters most.

---

## Caveats, stated rather than buried

- **Both domains are synthetic.** RadioML is simulated; so is the generator here.
  Until SDR capture is possible this is a study, not a result. It is the single
  largest weakness of the work.
- **Spectral whitening is not novel.** WhiteNet (arXiv:2608.06581) arrived at
  essentially the same operation, on *real* over-the-air captures, and the claim
  of novelty was dropped on finding it. See `LITERATURE.md`. What survives is the
  attribution method, the partial-vs-full disagreement, and the sink finding.
- **`model_zoo.ICRNNA` is not the published architecture.** It is transcribed
  from a peer's reproduction. Checked against El-Haryqy et al., *Results in
  Engineering* 26 (2025) 104783, it differs in five places: conv1 kernel 5 vs 3,
  two max-pools vs one, one BatchNorm after the LSTM stack vs one per layer, no
  attention dropout or LayerNorm, and one dense layer of 128 at dropout 0.5 vs
  two of 128 and 64 at 0.3. **Do not call the current model "the published
  architecture".** The faithful build has now been run — see below — and does
  not reproduce the paper either.
- **`dann.py` has not been ported.** It builds on a legacy feature extractor
  with no equivalent in the current backbone, so its numbers are not comparable
  to the rest until it is rewritten.
- **Open-set recognition is the adjacent field** for the sink work, and a few
  searches do not cover it. The novelty question there is genuinely open.

---

## Open work

1. **Check whether the 2018 classification run converged.** The 2016 one did,
   but with a single epoch of margin against the 60-epoch ceiling, and 2018 is
   the harder problem. Until this is answered, anything measured under the same
   ceiling — including items 2 and 3 below — may be comparing floors.
2. Finish the last 3 cells of `compare_methods --arch ICRNNA`
3. Rerun the α sweep — α=0.75 is unverified for the current backbone
4. Rerun the sink/family thread (24-class leave-one-out, the expensive one)
5. Port `dann.py` to the current backbone, or drop the comparison
6. Add RadioML 2016.10a as a third domain (still synthetic, so it only partly
   addresses the weakness above). `rml2016.py` loads it and
   `train_backbone.py` trains on it; what does not exist yet is an
   `RML2016Domain` for the cross-domain scripts, because 2016's 128-sample
   frames and 2018's 1024 have to be reconciled first and that is a decision,
   not a detail
7. Real SDR capture when hardware and lab access allow

**Closed.** Validate `colab/icrnna_faithful_2016.py` against 63.24% — at the
paper's 58-epoch ceiling it lands 1.49 points under; trained to convergence it
reaches 63.21%. Check whether the 2016 classification run converged — it did,
best epoch 39, one epoch inside the ceiling.

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

**The .npz is in Drive, not yet in this repository.** It has to be committed
before this row means anything to anyone else — a result nobody can look up
later is a result nobody can check.

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
src/crossdomain.py      the central train-on-A / test-on-B experiment
src/whitening_*.py      whitening and its alpha sweep
src/compare_methods.py  whitening against the literature baseline
src/sink_*.py           where an unseen modulation lands
colab/run_training.py   run train_backbone.py on Colab
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
  two of 128 and 64 at 0.3. A faithful build is in
  `colab/icrnna_faithful_2016.py`, unvalidated until run against the paper's
  63.24% on RML2016.10a. **Do not call the current model "the published
  architecture".**
- **`dann.py` has not been ported.** It builds on a legacy feature extractor
  with no equivalent in the current backbone, so its numbers are not comparable
  to the rest until it is rewritten.
- **Open-set recognition is the adjacent field** for the sink work, and a few
  searches do not cover it. The novelty question there is genuinely open.

---

## Open work

1. Finish the last 3 cells of `compare_methods --arch ICRNNA`
2. Rerun the α sweep — α=0.75 is unverified for the current backbone
3. Rerun the sink/family thread (24-class leave-one-out, the expensive one)
4. Port `dann.py` to the current backbone, or drop the comparison
5. Validate `colab/icrnna_faithful_2016.py` against 63.24% — needs
   `RML2016.10a_dict.pkl`, not on this machine
6. Add RadioML 2016.10a as a third domain (still synthetic, so it only partly
   addresses the weakness above). `rml2016.py` loads it and
   `train_backbone.py` trains on it; what does not exist yet is an
   `RML2016Domain` for the cross-domain scripts, because 2016's 128-sample
   frames and 2018's 1024 have to be reconciled first and that is a decision,
   not a detail
7. Real SDR capture when hardware and lab access allow

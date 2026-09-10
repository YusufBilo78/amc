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

## Status — read this before quoting any number

The project changed backbone. `cnn.IQNet`, a plain VGG-style 1-D CNN written
for this project, was replaced by `model_zoo.ICRNNA` (conv → BiLSTM → additive
attention) after a controlled comparison. On RadioML, five classes, 512 frames
per cell (`overfit_2x2.py`):

| backbone | final train | final test | gap | memorisation starts |
|---|---|---|---|---|
| IQNet | 0.9998 | 0.6750 | **+0.325** | epoch 21 |
| ICRNNA | 0.7188 | 0.7100 | **+0.009** | never |

IQNet carries one dropout, immediately before its output layer, and memorises
the training set from about epoch 20. ICRNNA regularises after every block, does
not memorise, and scores higher on test with fewer parameters (786k vs 894k).
The training recipe was varied on the same 2×2 — OneCycle at fixed length
against ReduceLROnPlateau with early stopping — and made no difference
(+0.325 vs +0.317), which is what makes this an architecture effect.

That matters beyond tidiness. Almost every number here is a **difference**
between two accuracies, and a backbone that memorises 30% of its training set is
a poor instrument for measuring one.

**Consequence: the measurements are being redone.** Everything below describes
methods, controls and findings whose numbers came from IQNet and are being
re-measured on ICRNNA. The IQNet-era results are preserved in
`archive_iqnet/` (untracked) rather than deleted, including the previous
917-line README with every table intact.

### What is currently measured on ICRNNA

`compare_methods.py --arch ICRNNA --seeds 5 --epochs 60 --patience 20`,
17 of 20 cells done (`compare_methods_ICRNNA_es.npz`):

| method | in-domain | cross-domain | gap | 16QAM |
|---|---|---|---|---|
| none | 0.999 | 0.804 | **+0.195** | 0.060 |
| standard augmentation | 1.000 | 0.806 | +0.194 | 0.036 |
| whitening α=0.75 | 1.000 | **0.993** | **+0.007** | **0.991** |
| whitening + standard (2/5 seeds) | 0.999 | 0.990 | +0.008 | 0.996 |

Three things follow, and the second was not expected:

1. **The gap is not an artifact of one network.** IQNet measured +0.182, ICRNNA
   measures +0.195 — on a model that is better in-domain and does not memorise.
2. **The literature augmentation set does nothing here.** Rotation, conjugate
   flip and additive noise (arXiv:1912.03026) moved the gap by 0.001. On IQNet
   the same transforms gave +0.182 → +0.141, a real if modest gain. The most
   likely reading is that their apparent benefit was compensating for IQNet's
   overfitting rather than addressing domain shift.
3. **Whitening costs nothing in-domain any more.** On IQNet it traded 0.993 →
   0.965; here in-domain stays at 1.000.

A cross-domain accuracy of 0.993 sits close enough to the ceiling that it
deserves a sceptical pass of its own before it goes in a report.

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

---

## Layout

```
src/modem.py            baseband IQ generator, 11 modulation classes
src/radioml.py          RadioML 2018.01A loader
src/domains.py          domain abstraction: RadioML, synthetic, capture
src/features.py         cumulants and instantaneous features
src/augment.py          augmentation transforms, including whitening
src/cnn.py              IQNet (superseded), training loop, splits
src/model_zoo.py        ResNet1D, GRU, Transformer, ICRNNA, backbone()
src/crossdomain.py      the central train-on-A / test-on-B experiment
src/whitening_*.py      whitening and its alpha sweep
src/compare_methods.py  whitening against the literature baseline
src/sink_*.py           where an unseen modulation lands
src/overfit_2x2.py      architecture vs recipe as a cause of memorisation
colab/                  paper-faithful ICRNNA, for calibration on 2016.10a
archive_iqnet/          superseded results and the IQNet-era README
```

Call `model_zoo.backbone(n_classes)` rather than naming a class, so the default
is one edit rather than twenty.

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

The gap concentrated almost entirely in one cell: 16QAM read as 64QAM. Four
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

*(The specific accuracies these conclusions rested on were measured with IQNet
and are pending re-measurement; the interventions themselves are signal-level
and do not depend on the classifier.)*

---

## The fix

**Attempt 1 — randomise the envelope during training. Failed.** Documented with
its ablation rather than dropped, because the failure is informative: teaching
invariance by sampling nuisance parameters did not transfer.

**Attempt 2 — remove the envelope. Works.** Spectral whitening: divide each
frame's spectrum by a smoothed estimate of its own magnitude envelope,
`X / smooth(|X|)^α`, with a circular moving average so the wrap-around is not a
discontinuity.

α is swept from 0 to 1. **Partial whitening beat full whitening on IQNet**
(α=0.75 against α=1.0), which is a concrete, measurable disagreement with
WhiteNet's choice of full whitening — on a different task, so not a
contradiction, but worth reporting. **The sweep has not yet been rerun on
ICRNNA, so α=0.75 is currently unverified for the present backbone.**

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

**All of it was measured on IQNet and none of it has been rerun on ICRNNA.**
This is the most expensive thread to redo — 24-class leave-one-out — and the
most novel, so it is also the one where re-measurement matters most.

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
  63.24% on RML2016.10a. **Do not call the current ICRNNA "the published
  architecture".**
- **DANN still runs on IQNet.** `dann.py` reaches into an IQNet-specific
  attribute and has not been ported, so its numbers are not comparable to the
  rest until it is.
- **Open-set recognition is the adjacent field** for the sink work, and a few
  searches do not cover it. The novelty question there is genuinely open.

---

## Open work

1. Finish the last 3 cells of `compare_methods --arch ICRNNA`
2. Rerun the α sweep on ICRNNA — α=0.75 is unverified for this backbone
3. Rerun the sink/family thread on ICRNNA (24-class leave-one-out, the expensive
   one), and add ICRNNA to `sink_across_architectures.py`
4. Port `dann.py` off IQNet, or drop the comparison
5. Validate `colab/icrnna_faithful_2016.py` against 63.24% — needs
   `RML2016.10a_dict.pkl`, not on this machine
6. Add RadioML 2016.10a as a third domain (still synthetic, so it only partly
   addresses the weakness above)
7. Real SDR capture when hardware and lab access allow

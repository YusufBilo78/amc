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
**all 20 cells** (`compare_methods_ICRNNA_es.npz`):

| method | in-domain | cross-domain | gap | 16QAM |
|---|---|---|---|---|
| none | 0.999 ±0.001 | 0.804 ±0.005 | **+0.195** ±0.005 | 0.060 ±0.021 |
| standard augmentation | 1.000 ±0.000 | 0.806 ±0.003 | +0.194 ±0.003 | 0.036 ±0.014 |
| whitening α=0.75 | 1.000 ±0.000 | **0.993** ±0.001 | **+0.007** ±0.001 | **0.991** ±0.006 |
| whitening + standard | 0.999 ±0.001 | 0.994 ±0.003 | +0.005 ±0.003 | 0.998 ±0.003 |

**Read this table with the convergence caveat below.**

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

Completing the table also settled the fourth row, which stood at 2 of 5 seeds.
Combining whitening with the literature augmentation adds **+0.001 (0.3 s.d.)**
over whitening alone. They do not compose; whitening is doing all of the work
and the rotation/flip/noise set contributes nothing on top of it. That is the
opposite of what this file's own docstring predicted, and the prediction was
written before the measurement — so it stays as a refuted hypothesis rather
than being quietly reworded.

### The convergence caveat on this table

The three cells run last recorded where training stopped: peaks at epochs
**53, 37 and 50**, under a 60-epoch ceiling with patience 20. Early stopping
fires 20 epochs after the peak, so two of the three needed to reach epoch 73
and 70 and instead **ran out of budget at 60**. Those two cells did not
converge and their accuracies are floors.

The other seventeen cells were run under the same ceiling before the stopping
epoch was recorded, so the same is likely true of them and unprovable either
way from the file.

How much this matters depends on which comparison you read:

- **The headline survives.** Whitening moves cross-domain accuracy from 0.804
  to 0.993, a difference of 0.19 against a seed spread of 0.005. Under-training
  depresses cells; it does not manufacture a nineteen-point gap, and the
  2016 and 2018 runs showed the penalty for stopping a few epochs early is
  fractions of a point.
- **The small comparisons do not.** Whitening against whitening-plus-standard
  is +0.001, and none against standard augmentation is +0.002. Differences that
  size, between cells that stopped at an arbitrary budget rather than at
  convergence, are not measurements. Neither is safe to report until the table
  is rerun with a ceiling that lets early stopping decide.

This was nearly missed. The warning added for exactly this case tested
`best_epoch >= ceiling`, which is too weak — a peak at 53 under a ceiling of 60
is not at the ceiling, but the run still stopped because the budget ended
rather than because it had converged. The test is now
`best_epoch + patience <= ceiling` in `compare_methods.py`,
`whitening_seeds.py` and `train_backbone.py`, and it names the epoch a rerun
would need.

#### The rerun at 100 epochs, and what it showed about convergence

`compare_methods_ICRNNA_es_e100.npz` is the same table under a 100-epoch
ceiling. It stands at **15 of 20 cells**; the whole fourth row, whitening plus
standard augmentation, is still to run. Every finished cell recorded its
stopping epoch, and those turned out to be the interesting part:

| method | stopping epochs | mean | gradient updates |
|---|---|---|---|
| none | 51, 41, 48, 36, 38 | 43 | 9.8k – 13.9k |
| standard augmentation | 89, 48, 83, 65, 82 | 73 | 13.1k – 24.3k |
| whitening α=0.75 | 26, 26, 33, 43, 25 | 31 | 6.8k – 11.7k |

**Convergence speed is a property of the method, not just of the dataset.**
Whitening converges 2.6× faster than the augmentation row and 1.5× faster than
training with no method at all. That is what the mechanism predicts: whitening
removes a nuisance dimension, so there is less to fit; augmentation adds
nuisance variation, so there is more. It also corrects the generalisation
drawn from the two classification runs — 23k updates was not a constant of the
optimizer, it was those two problems.

**Three cells still ran out of budget at 100**, all of them in the standard
augmentation row: peaks at 89, 83 and 82, which need 109, 103 and 102 to stop.
That row is the comparison baseline for the headline claim, so it is the worst
one to have under-trained, and the claim that the literature augmentation set
does not address this failure mode is not safe until those three are rerun.

**The two tables are independent samples, not one refined into the other.**
No cell reproduces its 60-epoch counterpart bit for bit — the `none` row reads
0.8066 against 0.8076 on the first seed, and so on down the table. The 2016 and
2018 convergence checks *were* bit-identical to the runs they re-ran, so this
is not how the code behaves on one machine; the 60-epoch table was measured on
the laptop and this one on Colab, and cuDNN and AMP do not promise the same
arithmetic across two GPUs. It costs nothing for the headline, which is a
nineteen-point difference, but it does mean a 60-vs-100 comparison at the cell
level confounds the ceiling with the hardware and should not be read as an
effect of the ceiling.

The one cell worth naming is whitening seed 2, which lands at 0.9750 here
against 0.9914 at the lower ceiling and drags that row's spread from ±0.0014 to
±0.0070. It peaked at epoch 33 under a ceiling of 100, so it converged with 47
epochs to spare — this is seed-level variation in the whitening row, not an
under-trained cell, and the row's honest spread is the wider one.

Fixing it does not cost another full table. **The ceiling only matters to a
cell that hits it**: a cell that peaked at epoch 36 peaks at 36 whatever
ceiling it ran under, so a table whose cells all converged is sound even
though they ran under different ceilings. `compare_methods.py` now stores the
ceiling per cell and takes `--redo-unconverged`, which on resume clears only
the cells that ran out of budget and reruns those. Ten cells instead of twenty,
and the per-cell ceilings are in the file so the claim can be checked rather
than taken on trust.

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

### The decision table

The number a review asks for first is not the average, it is the table the
average is computed over: N frames of a known modulation go in, and how often
did the classifier answer each of the classes available to it? Row =
transmitted, column = decided. `confusion_table.py` prints it from a finished
run's stored counts, so for a run that already exists it costs nothing.

RML2016.10a at SNR ≥ 10 dB, three seeds pooled, 2,250 decisions per row
(percentages of the row):

| transmitted \ decided | 8PSK | AM-DSB | AM-SSB | BPSK | CPFSK | GFSK | PAM4 | QAM16 | QAM64 | QPSK | WBFM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **8PSK** | **98.3** | | 0.8 | | | | | | 0.1 | 0.7 | |
| **AM-DSB** | | **100.0** | | | | | | | | | |
| **AM-SSB** | 1.0 | 2.9 | **89.5** | 0.8 | 0.8 | 1.7 | 1.0 | 0.3 | 0.1 | 0.9 | 1.0 |
| **BPSK** | | 0.1 | 0.8 | **98.8** | | | 0.2 | | | | |
| **CPFSK** | | | | | **100.0** | | | | | | |
| **GFSK** | | | | | | **99.9** | | | | | 0.1 |
| **PAM4** | | | 0.7 | 0.3 | 0.1 | | **98.8** | | | | |
| **QAM16** | 0.5 | | 0.9 | | | | 0.1 | **91.8** | 6.4 | 0.2 | |
| **QAM64** | 0.2 | | 0.8 | | | | 0.2 | 7.8 | **90.8** | 0.1 | |
| **QPSK** | | | 1.0 | | | | | 0.1 | 0.1 | **98.7** | |
| **WBFM** | | 60.3 | | | | | | | | | **39.6** |

22,641 of 24,750 correct, which is the 0.9148 reported above — the same number,
with its structure left in. Two cells carry nearly all of the error: WBFM
decided as AM-DSB 1,357 times out of 2,250, and the QAM16 ↔ QAM64 pair at 143
and 176. Everything else in the table is below 3%.

**A subset of a table is not a smaller experiment.** `confusion_table.py
--classes BPSK,QPSK,16QAM,64QAM` narrows which rows are printed, but the model
still had all of its classes available, so the rows do not sum to 100% across
the four columns and the remainder is reported in an `other` column rather than
renormalised away. On the 24-class 2018 run those four rows read 100.0, 100.0,
98.5 and 56.4 — and that last one is not a four-class result, it is 64QAM
losing 30.7% to 256QAM and 11.5% to 128QAM, neither of which is on the table.

The four-class *experiment* is `train_backbone.py --classes
BPSK,QPSK,16QAM,64QAM`, which builds the model with four outputs and so gives
it four answers to choose between. The two numbers answer different questions
and neither substitutes for the other.

The subset is pushed down into the loader rather than applied afterwards, and
that is what makes the run cheap: only those cells are read, so the same memory
buys four times the frames per class that the 24-class run can afford. At 2,048
frames per cell the test set holds **308 frames per (class, SNR) cell**, which
over the eleven SNR levels at or above 10 dB and three seeds is **about 10,200
decisions per class** — enough that a 1% confusion is a hundred events rather
than three. Staging is class-restricted too, and takes every frame of the four
rather than a subsample of all twenty-four: 3.5 GB against 10.5.

Because both loaders draw each cell from one generator walking the classes in
order, iterating four consumes it differently from iterating twenty-four, so a
subset run is **not** trained on the frames the 24-class run gave those
classes. It is its own experiment in its own file, which is the honest thing
for it to be; what it must not be is quoted as a slice of the other.

### The four-class decision table, measured

`train_backbone.py --data rml2016 --classes BPSK,QPSK,QAM16,QAM64 --seeds 3`,
all 1000 frames per (class, SNR) cell
(`train_backbone_rml2016_f1000_c4_colab.npz`). Overall **0.7086 ± 0.0076**
across all twenty SNRs, **0.9439 ± 0.0098** at SNR ≥ 10 dB, in 6.6 minutes on a
Colab T4. Converged: peaks at epochs 19, 23 and 33 under a 60-epoch ceiling
with patience 20, so the last of them stopped sixteen epochs inside the budget.

Decision counts, three seeds pooled, 2,250 frames per row:

| transmitted \ decided | BPSK | QPSK | QAM16 | QAM64 | recall |
|---|---|---|---|---|---|
| **BPSK** | **2235** | 14 | 0 | 1 | 99.3% |
| **QPSK** | 9 | **2232** | 5 | 4 | 99.2% |
| **QAM16** | 9 | 19 | **1978** | 244 | 87.9% |
| **QAM64** | 13 | 13 | 174 | **2050** | 91.1% |

8,495 of 9,000 correct, 94.39%. Two cells hold 83% of the error: QAM16 decided
as QAM64 244 times and the reverse 174 times. Everything outside the QAM pair
is at or below 0.8%.

#### Fewer classes did not make QAM16 easier

The obvious reading of a four-class table is that it is the easy case. Against
the eleven-class run on the same dataset it is not, and the direction is
consistent seed by seed:

| | four classes | eleven classes |
|---|---|---|
| BPSK | 99.33 ± 0.11 | 98.80 ± 0.19 |
| QPSK | 99.20 ± 0.58 | 98.71 ± 0.27 |
| **QAM16** | **87.91 ± 2.27** | **91.82 ± 0.88** |
| QAM64 | 91.11 ± 2.54 | 90.76 ± 0.44 |
| QAM16 → QAM64 | **10.84** | **6.36** |

BPSK and QPSK improve, which is what removing seven competitors is supposed to
do. QAM16 goes the other way, and it is not a seed picked out of three: its
four-class seeds are 84.8, 88.8 and 90.1 against 92.0, 90.7 and 92.8, so the
ranges do not overlap, and neither do the leak's — 8.9, 9.3, 14.3 against 4.9,
6.5, 7.6.

Part of that is arithmetic. With eleven classes QAM16's error also leaves for
8PSK, AM-SSB, QPSK and PAM4, 1.7 points in total; close those routes and the
mass has nowhere to go but QAM64. But the leak rose by 4.5 points, not 1.7, so
rerouting does not account for it. The seed spread also quadruples, 0.88 to
2.27.

**This is stated as an observation, not a mechanism.** Three seeds is thin for
a difference this size, and the two runs are not trained on the same frames:
both loaders draw each cell from one generator walking the classes in order, so
iterating four consumes it differently from iterating twenty-four. What can be
said is that the four-class table is not a magnified corner of the eleven-class
one, and that whichever way the effect is explained, it argues for reporting
the table rather than the average — an overall figure that went *up*, 91.48% to
94.39%, is hiding a class that went down.

#### And it converged four times faster

Best epochs 19, 23 and 33 at 219 steps per epoch is **4,156 to 7,218 gradient
updates**, against 23,439 for the eleven-class run on the same dataset. That is
a third of the updates for a problem with a third of the classes, and it is the
same lesson as the `compare_methods` rows: the update count is a property of
the problem, not a constant of the optimizer, so a ceiling that was safe for
one configuration says nothing about another.

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

#### 2018 converged too, with room to spare

Same check at a 200-epoch ceiling
(`train_backbone_rml2018_f512_colab_e200.npz`, one seed): **best epoch 26,
early stop at 46**, against the committed run's ceiling of 60 — fourteen epochs
of margin. Bit-identical to that run's seed 0 again, 0.561637 overall and
0.856546 at SNR ≥ 10 dB in both. **Both committed classification results are
measurements, not floors.**

2018 converged *sooner* in epochs than 2016 despite being the harder problem,
which is the wrong way round until the unit is fixed:

| | train frames | steps/epoch | best epoch | gradient updates |
|---|---|---|---|---|
| 2016, 11 classes, 128 samples | 154,000 | 601 | 39 | **23,439** |
| 2018, 24 classes, 1024 samples | 223,392 | 872 | 26 | **22,672** |

Two datasets differing in class count and in sequence length by a factor of
eight, converging within 3% of each other in *updates*. Under this optimizer
what the model needs is about 23k steps, and the epoch count is just that
number divided by however many batches the dataset happens to make.

#### Which makes the 60-epoch ceiling a problem for the sweeps

`compare_methods.py` and `whitening_seeds.py` train on the five shared classes:
99,840 frames, 69,888 after the split, **273 steps per epoch**. At 23k updates
that is **84 epochs** — above the 60 both are usually run with.

That is an extrapolation, not a measurement. Five classes is a much easier
problem than eleven or twenty-four and may well converge in far fewer updates.
But the consequence if it does not is worse here than for a single accuracy:
every number those files report is a *difference between two cells*, so one
cell stopped at the ceiling does not make the comparison low, it makes it
meaningless.

Both now record `best_epochs` and refuse to report quietly: a cell that peaked
at the ceiling is named as it happens and again in a warning above the summary
table. The three cells still missing from `compare_methods_ICRNNA_es.npz` will
therefore answer the question for the seventeen that are already in it — those
predate the recording and are stored as NaN.

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

1. **Finish `compare_methods_ICRNNA_es_e100.npz`**, which stands at 15 of 20 —
   the whitening + standard augmentation row is what is left — then rerun the
   three unconverged augmentation cells with `--epochs 150 --redo-unconverged`.
   Five cells plus three, not twenty.
2. **The four-class decision table** — `train_backbone.py --classes
   BPSK,QPSK,16QAM,64QAM`, asked for directly in the 2026-09-15 meeting. Four
   outputs, so the model chooses between four answers and the rows sum over
   four columns; the subset of the 24-class table is not the same measurement
   and says so where it is printed. Cheap: four classes against twenty-four
3. Rerun the α sweep — α=0.75 is unverified for the current backbone, and it
   needs the same ceiling for the same reason
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
reaches 63.21%. Check whether the two classification runs converged — both did,
best epoch 39 on 2016 and 26 on 2018.

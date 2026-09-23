# Orientation

Read this before quoting any number from this repository.

## What this project is

A study of **why** modulation classifiers lose accuracy across domains, not an
attempt to win a benchmark. Reproducing RadioML accuracy is solved; measuring,
attributing and closing the gap between the training domain and an
independently generated one is the contribution. `README.md` has the full
narrative and every caveat.

## The model

Call `model_zoo.backbone(n_classes)`. Never name a model class directly — the
point of the factory is that changing the default is one edit rather than
twenty. It currently returns `model_zoo.ICRNNA`: convolutional front end,
bidirectional LSTM, additive attention, 786k parameters.

`cnn.py` holds the training loop, the splits and the evaluation helpers — use
those. It also contains legacy scaffolding from an earlier backbone; do not
reach into it for models.

For plain modulation classification -- "how well does the backbone classify?",
with no domain gap involved -- use `train_backbone.py`, which runs both
datasets through one protocol: `--data rml2018` (24 classes, native 1024
samples) or `--data rml2016` (11 classes, 128 samples). Do **not** use
`cnn.py`'s own `main()` for this: it still builds `cnn.IQNet` on a two-way
split, so its numbers are neither the current backbone nor the current
protocol.

Two scripts instantiate models directly on purpose, because they compare
architectures: `overfit_2x2.py` and `sink_across_architectures.py`.
`compare_methods.py` exposes the choice through `--arch` and defaults correctly.
`dann.py` has not been ported and builds on the legacy extractor, so its numbers
are not comparable to anything else until it is rewritten.

**Provenance caveat.** `model_zoo.ICRNNA` is transcribed from a peer's
reproduction, not from the paper. Checked against El-Haryqy et al., *Results in
Engineering* 26 (2025) 104783, it differs in five places: conv1 kernel 5 vs 3,
two max-pools vs one, one BatchNorm after the LSTM stack vs one per layer, no
attention dropout or LayerNorm, and one dense layer of 128 at dropout 0.5 vs two
of 128 and 64 at 0.3. Do not describe the current model as "the published
architecture".

The faithful build in `colab/icrnna_faithful_2016.py` **does** reproduce the
paper, but only when trained to convergence. At the paper's own 58-epoch
ceiling it reaches 61.75% ± 0.07 against 63.24%; given a 150-epoch ceiling it
peaks at epoch 107 and reaches 63.21%. The whole 1.49-point deficit was the
epoch budget, not the architecture.

**Read that as a warning about this repository's own numbers.** An
under-trained run reports a floor. `train_backbone.py` uses batch 256 and a
60-epoch ceiling. The stopping epoch is now recorded — `cnn.train_model` sets
`best_epoch` / `stopped_early` on the model it returns, `train_backbone.py`
stores `best_epochs` and warns when a seed peaks at the ceiling — but the two
committed classification results predate that.

**Both have been checked and both converged.** 2016: best epoch 39, early stop
at 59, one epoch inside the ceiling. 2018: best epoch 26, early stop at 46,
fourteen epochs inside. Both bit-identical to the committed runs, so both
committed numbers are measurements rather than floors.

What that established is worth carrying: **convergence is set by gradient
updates, not epochs.** 2016 took 23,439 and 2018 took 22,672 — within 3% of
each other despite differing in class count and in sequence length by a factor
of eight. The epoch count is just that number divided by however many batches
the dataset makes.

That 23k is not a constant, though — it was those two problems. In
`compare_methods` the stopping epoch depends strongly on the *method*:
whitening converges at 26–33 epochs, no method at 36–51, and the standard
augmentation set at 48–89, which is 2.6× slower than whitening. Whitening
removes a nuisance dimension and augmentation adds one, so this is what the
mechanism predicts, but it means a single ceiling cannot be assumed safe
across a sweep whose cells differ in method.

So the 60-epoch ceiling is not safe everywhere. `compare_methods.py` and
`whitening_seeds.py` train on 69,888 frames, 273 steps per epoch, so 23k
updates would be ~84 epochs. Whether five classes need as many is unmeasured.
Both now record `best_epochs` and warn, which matters more there than for a
single accuracy: those files report *differences* between cells, and one cell
stopped by the ceiling makes the comparison meaningless rather than merely low.

**The test for "converged" is `best_epoch + patience <= ceiling`, not
`best_epoch >= ceiling`.** The second is too weak and got this wrong once
already: `compare_methods` cells peaked at epochs 53, 37 and 50 under a ceiling
of 60 with patience 20, so two of them needed to reach 73 and 70 before
stopping and instead ran out of budget — while `53 >= 60` is False and the
warning stayed silent.

## Which results are current

| file | status |
|---|---|
| `compare_methods_ICRNNA_es_e100.npz` | **complete and converged, 20/20 — quote this one for the method table.** Ceiling 100, except the three standard-augmentation cells rerun at 150 (bit-identical to their 100-ceiling runs, peaks 89/83/82). none 0.805, augmentation 0.807, whitening 0.989, whitening+augmentation 0.995; combining adds +0.006 at 1.1 s.d., not a measurement. It does not reproduce the 60-epoch table cell for cell — different machine, so the two are independent samples |
| `whitening_seeds_e100.npz` | **current** — the α sweep, 5 seeds × 5 alphas, all converged (peaks 22–62 under 100). Cross-domain 0.805 / 0.855 / 0.940 / 0.988 / 0.993 for α = 0 / 0.25 / 0.5 / 0.75 / 1.0. **Full whitening is at least as good as partial**; the old 'α=0.75 beats 1.0' claim is withdrawn. α=0 row bit-identical to compare_methods' none row |
| `compare_methods_ICRNNA_es.npz` | **complete at 20/20** — the finished copy is now the committed one. Two of the three cells with a recorded stopping epoch ran out of budget at the 60-epoch ceiling rather than converging. The 0.804 → 0.993 headline survives that; the small differences in the table do not. See README |
| `sink_class_24_e60.npz`, `sink_class_24_partial_e60.json` | **current** — the 24-class leave-one-out on the current backbone. 16 of 24 held-out classes sink into their own pre-registered family against a chance of 3.5; all 24 runs converged (best epochs 7–43 under 60, patience 10). Four of the eight misses are APSK ↔ QAM, two are FM ↔ GMSK — structured, but the count quoted is the pre-registered 16, not a rescored 20 |
| `sink_across_arch_e60.json` | **current** — the six-class architecture control, five architectures including ICRNNA: same-family 5/6, 4/6, 4/6, 3/6, 4/6 against a chance of 0.78. 29 of 30 runs converged; `Transformer \| 32PSK` peaked at 56 under 60, sink stands, in-dist is a floor. The ICRNNA rows are bit-identical trainings to the 24-class run's (same best epoch and accuracy to every digit) |
| `baseline_results.npz` | current — cumulants + SVM, no neural net involved |
| `sink_vs_geometry.npz` | current — signal geometry, no model involved |
| `overfit_2x2.json` | current — architecture comparison |
| `train_backbone_rml2016_f1000_colab.npz` | **current** — ICRNNA on RML2016.10a, 3 seeds, 0.6223 overall / 0.9148 at SNR ≥ 10 dB. Checked: converged, best epoch 39, one epoch inside the ceiling |
| `icrnna_faithful_results.json`, at the paper's 58-epoch ceiling | **current** — paper-faithful ICRNNA on RML2016.10a, 3 seeds, 61.75% ± 0.07 against the paper's 63.24%, at the paper's 58-epoch ceiling |
| `icrnna_faithful_e150_results.json` | **current** — the same build trained to convergence (peak at epoch 107): 63.21% against the paper's 63.24%. The deficit above was the epoch budget |
| `train_backbone_rml2016_f1000_c4_colab_s14.npz` | **current — quote this one for the four-class table on 2016.** BPSK/QPSK/QAM16/QAM64, all 1000 frames per cell, 14 seeds, ceiling 100, patience 20, all converged (peaks 16–46). 42,000 decisions above 10 dB, 2,421 errors (0.9424); 0.7080 overall; a matrix at every SNR, 2,100 decisions per cell. QAM16 recall is flat at 86–88% from +2 dB to +18 dB and QAM64 at 91–93%: the QAM pair does not separate with SNR at 128 samples, so this is a frame-length ceiling, not a noise one. BPSK and QPSK sit at 99–99.7%, never 100 |
| `train_backbone_rml2016_f1000_c4_colab.npz` | superseded by the row above, kept — the four-class decision table Moshe asked for. BPSK/QPSK/QAM16/QAM64 on RML2016.10a, 3 seeds, 0.7086 overall / 0.9439 at SNR ≥ 10 dB, 9,000 pooled decisions. Converged: peaks at 19, 23, 33 under a 60 ceiling. Note QAM16 is **worse** here than in the 11-class run (87.9 vs 91.8), seed ranges not overlapping |
| `train_backbone_rml2018_f2048_c4_colab_e150.npz` | **current** — the four-class table on 2018, converged: peaks 46, 48, 66 under a 150 ceiling, 0.7467 overall, 40,654 of 40,656 above 10 dB, a matrix at every SNR. Seeds 0 and 1 bit-identical to the 60-epoch file below; seed 2 within a point at every level. Quote this one |
| `train_backbone_rml2018_f2048_c4_colab.npz` | **superseded by the row above, kept** — same run at a 60 ceiling, flagged unconverged — the four-class table on 2018. BPSK/QPSK/16QAM/64QAM, 2048 frames/cell, 3 seeds: **40,654 of 40,656 correct above 10 dB**, 0.7465 overall. Peaks at 46, 48, 59 under a 60 ceiling, so the overall number is a floor; the high-SNR table is saturated and unaffected. Carries a confusion matrix at every SNR (rebuilt from the checkpoints, verified): at 0 dB the QAM pair is a coin flip and PSK is perfect; at −8 dB both QAMs drain into QPSK ~70% |
| `train_backbone_rml2018_f512_colab.npz` | **current** — ICRNNA on RadioML 2018.01A, 24 classes, 512 frames/cell, 3 seeds, 0.5699 overall / 0.8716 at SNR ≥ 10 dB. Checked: converged, best epoch 26, fourteen epochs inside the ceiling |

The embedding, discriminating-control, label-permutation and family-recovery
checks of the sink finding are **still on the old backbone**; everything
else above is current. That is the honest starting position, not an oversight. The
`README.md` "Open work" list is the queue, in order.

Results from the earlier backbone were moved out of the repository into
`archive_iqnet/` (on disk, untracked) rather than deleted. **Do not quote
anything from there, and do not treat it as a baseline.** It exists so nothing
was destroyed, not because it is still valid.

Figures in `figures/` are a mixture. The signal-level ones — spectrograms,
constellations, class spectra — are still valid. Anything showing an accuracy
curve or a confusion matrix predates the current backbone and is stale. Check
what produced a figure before reusing it.

## Environment

- Python: `C:\Users\yusuf\.venvs\amc\Scripts\python.exe` (outside OneDrive on purpose)
- Data: `C:\Users\yusuf\amc-data` — `radioml_X/y/z.npy` memmaps plus the source
  HDF5, about 40 GB, deliberately untracked
- GPU: RTX 3070 Laptop, 8 GB. **Keep the machine on mains power** — on battery
  the GPU is capped near 35 W and runs about 5× slower, measured
- `h5py` is blocked by this machine's application-control policy; `radioml.py`
  reads the `.npy` memmaps and falls back to pyfive

Run scripts from `src/`:

    cd src && "C:\Users\yusuf\.venvs\amc\Scripts\python.exe" compare_methods.py --arch ICRNNA --seeds 5 --epochs 60 --patience 20

## Conventions worth keeping

- **Long runs checkpoint and resume.** Every sweep writes partial results after
  each cell and skips finished ones on restart. This machine is a laptop and
  runs have been lost to a flat battery; do not write a sweep without it.
- **Three-way splits for anything with early stopping.** `cnn.split(...,
  val_fraction=0.15)` returns `(train, val, test)`. Early stopping, the LR
  schedule and checkpoint choice read validation only — passing the test set
  there means selecting on the number being reported.
- **New output files, not overwritten ones.** A run with different settings
  writes a differently named file, so two configurations stay comparable.
- **Negative results are kept, not deleted.** Four explanations for the domain
  gap were tested and refuted; they are part of the argument.

## Open work

**Focus moved to RML2016.10a on 2026-09-22.** The 2026-09-16 decision was
2018 only; it was reversed six days later because the people around the
project all work on 2016 and results have to be comparable with theirs. Nothing
measured on 2018 is retracted: the four-class table at 40,654 of 40,656 above
10 dB, the method table, the α sweep and the sink thread stand as the 2018
record, and the meeting deck built on them stays as it is. New work goes on
2016 first; a 2018 counterpart is run only when a 2016 result needs it.

**What Moshe asked for on 2026-09-22, all to be done on 2016** (transcript in
the session; the deck he saw was the 2018 one):

1. **Open the box.** Explain the detector end to end: what ICRNNA is, layer
   by layer with sizes and parameter counts; where it came from (El-Haryqy
   et al. 2025 via a peer's reproduction, five differences, the faithful
   build at 63.21 vs 63.24); how it was trained; and, stated plainly, what
   was written with an AI assistant, what was asked and what came back.
   "Right now it looks like magic."
2. **Define the noise exactly.** What "10 dB" is the ratio of, who made the
   noise, its properties. For 2016 that is the GNU Radio dynamic channel
   model of the 2016 dataset papers: read them and quote them.
3. **One SNR at a time.** The matrix at a single level, then lower: 10, 6,
   then ~3 dB (2 or 4 on the 2 dB grid). Error rate against SNR.
4. **Frame length.** State the input length (128 samples = 16 symbols on
   2016). Then shorten it: `--frame-len 64`, `--frame-len 32`, same four
   classes, and see the table degrade. `AMC_FRAME_LEN` in the Colab runner.
5. **Every graph says how many decisions are behind each point.** Define
   recall; show precision next to it. The QPSK "recall rises as SNR falls"
   curve is the sink: at −20 dB the model calls 84% of everything QPSK and
   QPSK precision is 25%, i.e. chance. Not a frame-length effect.
6. Do not make it more complicated than the four-class box. Break it.

Then, still on 2016:

7. The method table (`AMC_TASK=compare_methods_2016`) — not yet run
8. MSTFFNet reimplemented for the whitening prediction in `LITERATURE.md`
9. Port `dann.py` to the current backbone, or drop the comparison
10. Real SDR capture when hardware and lab access allow
11. The remaining sink corroborations on the current backbone

**The 2016 machinery (built 2026-09-22).** The 2018 protocol, run again on
RML2016.10a: `domains.RML2016Domain` (128-sample frames, SNR −20..18,
QAM16/QAM64 translated to the canonical 16QAM/64QAM), `SyntheticDomain(n_samples=128)`,
`compare_methods.py --source rml2016`, Colab task `compare_methods_2016`.
Whitening's smoothing width now scales with frame length (33 bins of 1024
→ 5 of 128), so the same α means the same bandwidth on both. Output files
carry `_rml2016`; nothing 2018 is touched. Both domains are synthetic on
this track too, and that remains the single largest weakness of the work.

Closed: the sink thread on the current backbone (16/24 same-family, all converged; the five-architecture control holds); the α sweep (full whitening ≥ partial; the WhiteNet disagreement withdrawn); the method table at a 100/150 ceiling, 20/20 converged (three redone cells bit-identical); the four-class table on 2018 (converged at 150, bit-identical seeds 0 and 1); validating the faithful build against 63.24% (reaches 63.21% trained to
convergence), and both convergence checks (best epoch 39 on 2016, 26 on 2018).

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
of 128 and 64 at 0.3. A faithful build is in `colab/icrnna_faithful_2016.py`,
unvalidated until it is run on RML2016.10a against the paper's 63.24%. Do not
describe the current model as "the published architecture".

## Which results are current

| file | status |
|---|---|
| `compare_methods_ICRNNA_es.npz` | **current**, 17/20 cells — 3 seeds of `whitening + standard` missing |
| `baseline_results.npz` | current — cumulants + SVM, no neural net involved |
| `sink_vs_geometry.npz` | current — signal geometry, no model involved |
| `overfit_2x2.json` | current — architecture comparison |
| `train_backbone_rml2016_f1000_colab.npz` | **current** — ICRNNA on RML2016.10a, 3 seeds, 0.6223 overall / 0.9148 at SNR ≥ 10 dB. Lives in Drive, not yet committed |
| `train_backbone_rml2018_f512_colab.npz` | **current** — ICRNNA on RadioML 2018.01A, 24 classes, 512 frames/cell, 3 seeds, 0.5699 overall / 0.8716 at SNR ≥ 10 dB. Lives in Drive, not yet committed |

Apart from one incomplete experiment and the two classification runs,
**nothing else is measured on the current backbone.** That is the honest starting position, not an oversight. The
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

1. Finish the last 3 cells of `compare_methods --arch ICRNNA`
2. Rerun the α sweep — α=0.75 is unverified for the current backbone
3. Rerun the sink/family thread (24-class leave-one-out, the expensive one)
4. Port `dann.py` to the current backbone, or drop the comparison
5. Validate `colab/icrnna_faithful_2016.py` against 63.24% — needs
   `RML2016.10a_dict.pkl`, which is not on this machine
6. Real SDR capture when hardware and lab access allow. Both domains are
   synthetic today, and that is the single largest weakness of the work

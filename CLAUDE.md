# Orientation

Read this before touching anything. It states what is current, because the
repository also holds superseded results and the difference is not obvious
from the filenames.

## What this project is

A study of **why** modulation classifiers lose accuracy across domains, not an
attempt to win a benchmark. Reproducing RadioML accuracy is solved; measuring,
attributing and closing the gap between the training domain and an
independently generated one is the contribution. `README.md` has the full
narrative and every caveat.

## Current backbone: ICRNNA. Do not use IQNet.

`model_zoo.ICRNNA` is the primary model for all new work.

`cnn.IQNet` was the original backbone and is **superseded**. It is still in the
tree because sixteen scripts import it and the older result files came from it,
but it should not be the default for anything new. The reason is measured, not
stylistic — `src/overfit_2x2.py`, results in `overfit_2x2.json`:

| backbone | final train | final test | gap | memorisation starts |
|---|---|---|---|---|
| IQNet | 0.9998 | 0.6750 | **+0.325** | epoch 21 |
| ICRNNA | 0.7188 | 0.7100 | **+0.009** | never |

IQNet carries one dropout, immediately before the output layer, and memorises
the training set from about epoch 20. ICRNNA regularises throughout and does
not, while scoring higher on test with fewer parameters (786k vs 894k). The
training recipe was tested on the same axis and made no difference, so this is
an architecture effect.

**Provenance caveat.** `model_zoo.ICRNNA` is transcribed from a peer's
reproduction, not from the paper. Checked against El-Haryqy et al., *Results in
Engineering* 26 (2025) 104783, it differs in five places: conv1 kernel 5 vs 3,
two max-pools vs one, one BatchNorm after the LSTM stack vs one per layer, no
attention dropout or LayerNorm, and one dense layer of 128 with dropout 0.5 vs
two of 128 and 64 at 0.3. A faithful version is in
`colab/icrnna_faithful_2016.py`, unvalidated until it is run on RML2016.10a
against the paper's 63.24%. Do not describe the current ICRNNA as "the
published architecture".

## Which results are current

| file | backbone | status |
|---|---|---|
| `compare_methods_ICRNNA_es.npz` | ICRNNA | **current**, 17/20 cells — 3 seeds of `whitening + standard` still missing |
| `overfit_2x2.json` | both | current |
| `compare_methods.npz`, `compare_dann.npz` | IQNet | superseded, keep for comparison |
| `whitening_sweep.npz`, `whitening_seeds.npz` | IQNet | **needs rerunning** — alpha=0.75 was optimal for IQNet, unverified for ICRNNA |
| `sink_*.npz/json`, `family_recovery.json` | IQNet | **needs rerunning** on ICRNNA |
| everything else | IQNet | superseded |

Most of the suite still has to be re-measured on ICRNNA. Treat any IQNet number
quoted in `README.md` as historical.

What is already known to survive the backbone change, from the 17 completed
cells: the cross-domain gap is **not** an IQNet artifact (+0.182 on IQNet,
+0.195 on ICRNNA), whitening closes it further on ICRNNA (+0.007, with 16QAM
0.060 → 0.991), and the standard AMC augmentation set — rotation, conjugate
flip, noise — does **nothing** on ICRNNA (+0.195 → +0.194) where it helped
IQNet. That last one suggests its apparent benefit was compensating for
IQNet's overfitting rather than addressing domain shift.

## Environment

- Python: `C:\Users\yusuf\.venvs\amc\Scripts\python.exe` (outside OneDrive on purpose)
- Data: `C:\Users\yusuf\amc-data` — `radioml_X/y/z.npy` memmaps plus the source
  HDF5, about 40 GB, deliberately untracked
- GPU: RTX 3070 Laptop, 8 GB. **Keep the machine on mains power** — on battery
  the GPU is capped near 35 W and runs about 5x slower
- `h5py` is blocked by this machine's application-control policy; `radioml.py`
  reads the `.npy` memmaps and falls back to pyfive

Run scripts from `src/`:

    cd src && "C:\Users\yusuf\.venvs\amc\Scripts\python.exe" compare_methods.py --arch ICRNNA --seeds 5 --epochs 60 --patience 20

## Conventions worth keeping

- **Long runs checkpoint and resume.** Every sweep writes partial results after
  each cell and skips finished ones on restart. This machine is a laptop and
  runs have been lost to a flat battery; do not write a sweep without it.
- **Three-way split for anything with early stopping.** `cnn.split(...,
  val_fraction=0.15)` returns `(train, val, test)`. Early stopping, the LR
  schedule and checkpoint choice read validation only — passing the test set
  there means selecting on the number being reported. Calling `split` without
  `val_fraction` reproduces the old two-way behaviour index for index.
- **New output files, not overwritten ones.** `compare_methods.py` writes
  `compare_methods_ICRNNA_es.npz` rather than replacing the IQNet file, so the
  two stay comparable.
- Negative results are kept, not deleted. Four explanations for the gap were
  tested and refuted; they are part of the argument.

## Open work

1. Finish the last 3 cells of `compare_methods --arch ICRNNA`
2. Rerun the alpha sweep on ICRNNA — alpha=0.75 is unverified for this backbone
3. Rerun the sink/family thread on ICRNNA (the expensive one: 24-class
   leave-one-out)
4. Validate `colab/icrnna_faithful_2016.py` against the paper's 63.24% —
   needs `RML2016.10a_dict.pkl`, which is not on this machine
5. Real SDR capture when hardware and lab access allow. Both domains are
   synthetic today, and that is the single largest weakness of the work

# Reference implementations

Not part of this project's pipeline. Kept here because `model_zoo.ICRNNA` was
transcribed from them, and a derived model without its source is a claim
nobody can check.

| file | dataset | notes |
|---|---|---|
| `train_amc_standalone_2016.py` | RML2016.10a, 11 classes, 128 samples | Colab-oriented single-file trainer. `.py` added to the original filename; contents unchanged. |
| `train_amc_standalone_2018.py` | RadioML 2018.01A, 24 classes | Same protocol, with 1024-sample frames cropped to 128 so the 2016 architecture applies unchanged. Its own docstring puts the expected accuracy near 44% and says the crop is why. |

Both are written by a colleague working on a related project, and both are
themselves reproductions of El-Haryqy et al., *Results in Engineering* 26
(2025) 104783 — which ships no code, so every implementation of ICRNNA in
circulation is somebody's reading of the paper's Figure 2.

## Where they differ from the paper

Checked line by line against the published description. The `ICRNNAFaithful`
class in these files differs in five places, all corrected in
`../colab/icrnna_faithful_2016.py`:

| | these files | paper |
|---|---|---|
| conv1 kernel | 5 | 3 |
| max-pooling | one per conv block (2 total) | one |
| BatchNorm in the RNN | one, after the whole stack | one after each LSTM layer |
| attention | plain additive | additive + dropout 0.1 + LayerNorm |
| dense head | one layer of 128, dropout 0.5 | two layers of 128 and 64, dropout 0.3 |

Their docstrings label three additions — dropout after the conv blocks,
BatchNorm after the BiLSTM, BatchNorm inside the classifier — but the paper
already specifies all three, so those are not additions. The omissions above
are the real differences.

Two hyperparameters also differ from the paper's Table 2: batch size 256 where
the paper uses 32 for RML2016.10a, and the LR scheduler keyed to validation
accuracy with patience 7 where the paper uses validation loss with patience 5.

## Two things worth knowing if you run them

- The paper's own epoch counts are 58 (RML2016.10a) and 40 (RML2016.10b). The
  `epochs: 200` in the 2016 file is a ceiling with early stopping at patience
  20, not a run length.
- `__getitem__` draws its random crop with `np.random` while the DataLoader
  runs two workers. Worker processes inherit the same NumPy seed, so both can
  emit identical crop sequences — the usual PyTorch pitfall. `torch.randint`
  avoids it. This weakens the augmentation; it does not invalidate anything.

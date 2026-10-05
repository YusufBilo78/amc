# ICRNNA against published architectures — RML2016.10a, 11 classes

Same protocol for every model (`train_backbone.py --arch`): 1,000 frames per
cell, 70/15/15 split per (class, SNR), the same three seeds, early stopping
on validation. Accuracy on the test split, 33,000 test frames per seed;
≥ 10 dB is the 8,250 of them at labels 10–18. Mean ± s.d. over seeds.

| model | paper | parameters | overall | ≥ 10 dB | best epochs (ceiling) |
|---|---|---|---|---|---|
| **ICRNNA** | El-Haryqy et al. 2025 | 786,379 | 62.23 ± 0.19 | 91.48 ± 0.13 | 39 (seed 0) (60) |
| **LSTM2** | Rajendran et al. 2018 | 201,099 | 61.44 ± 0.48 | 91.54 ± 0.52 | 46, 56, 43 (150) |
| **MCLDNN** | Xu et al. 2020 | 406,199 | 61.41 ± 0.22 | 91.06 ± 0.29 | 24, 24, 24 (150) |
| **PET-CGDNN** | Zhang et al. 2021 | 71,871 | 60.98 ± 0.08 | 90.64 ± 0.20 | 37, 41, 37 (150) |
| **VT-CNN2** | O'Shea et al. 2016 | 1,592,383 | 56.07 ± 0.02 | 81.96 ± 0.17 | 47, 63, 49 (150) |

All runs converged (best epoch + patience 20 within the ceiling). The
ICRNNA row is the committed run (ceiling 60; its best epoch was not stored,
and the convergence check reran seed 0: best epoch 39, bit-identical).
The others used a 150 ceiling.

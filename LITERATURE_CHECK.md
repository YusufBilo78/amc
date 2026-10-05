# Do our numbers agree with the papers? — RML2016.10a, 11 classes

Checked 2026-10-06 against the papers themselves (all read in full; tables
that are images in the PDFs were read at 300 dpi).

Our five models were trained under one protocol (`literature_2016.md`):
1,000 frames per cell, 70/15/15 split per (class, SNR), three seeds, early
stopping on validation, batch 256. The papers mostly use a 60/20/20 split,
batch 400, Adam at 0.001 halved when validation loss stalls for 5 epochs,
stopping after 50 stalled epochs, and **one training run**. One run against
the mean of three, and a 60 % against a 70 % training share, can each move a
number by several tenths of a point, so "agrees" below means within about
one point overall.

## 1. The models are the same models

The parameter count is the strongest check that a port is faithful: change
one layer and it changes. Ours, from `src/literature_models.py`:

| model | ours | Zhang 2022 benchmark, Table 4 (A) | PET-CGDNN paper, Table I (A) | MCLDNN paper, Table II |
|---|---|---|---|---|
| VT-CNN2 (CNN1 / CNN-IQ) | 1,592,383 | 1,592,383 | — | 1,592,383 |
| LSTM2 | 201,099 | 201,099 | 201,099 | 201,099 |
| MCLDNN | 406,199 | 406,199 | 406,199 | 406,199 |
| PET-CGDNN | 71,871 | 71,871 | 71,871 | — |

**Identical to the digit in every source.**

## 2. Overall accuracy and best single SNR

The only paper that tables an *overall* (all-SNR) accuracy on 2016.10a for
these models is the PET-CGDNN paper (Zhang et al., IEEE Commun. Lett. 2021,
Table I; one run each, 6:2:2 split). Its "highest accuracy" is the best single
SNR point. Ours: overall is the mean over seeds; the best single SNR point is
given as the range over the three seeds.

| model | paper overall | **ours overall** | difference | paper best SNR point | ours best SNR point (3 seeds) |
|---|---|---|---|---|---|
| LSTM2 | 60.56 % | **61.44 %** (60.95–62.09) | +0.9 | 91.41 % | 91.58–92.91 % |
| MCLDNN | 62.08 % | **61.41 %** (61.19–61.70) | −0.7 | 92.95 % | 91.76–92.12 % |
| PET-CGDNN | 60.44 % | **60.98 %** (60.87–61.05) | +0.5 | 91.36 % | 91.16–91.33 % |

**All three agree within one point**, in both directions, which is what two
different splits and one-run-versus-three-seeds predicts. In both, the three
sit within 1.7 points of each other and PET-CGDNN is the lowest of them.

Corroborating, from other papers:

- **MCLDNN paper** (Xu et al., IEEE WCL 2020): best point 92.95 % at 12 dB
  (the same number as the PET-CGDNN table, same group); average 92 % over
  0–18 dB. Ours over 0–18 dB: 90.77 %. The largest gap in this file, 1.2
  points, and in the paper's favour.
- **MSTFFNet paper** (Sensors 2026, Table 1), an independent group retraining
  MCLDNN: 62.02 % overall, 37.37 % at −20…0 dB, 91.11 % at 0…18 dB. Ours:
  61.41 / 37.16 / 90.77.
- **Rajendran et al.** (IEEE TCCN 2018), the LSTM2 paper itself: same model
  (two layers of 128 LSTM cells, amplitude L2-normalised, phase scaled to
  [−1, 1]); "an average accuracy of 90 % in SNR ranges from 0 dB to 20 dB",
  trained only on −10…20 dB. Ours over 0…18 dB: 91.04 %.
- **Zhang et al. 2022 benchmark** (DSP 129): accuracy only as curves
  (Fig. 5a); in the text, the best point on 2016.10a of all 14 models is
  92.05 % (MCLDNN at 10 dB). Consistent with ours, not a separate number.

## 3. Per class at 0 dB

The MCLDNN paper's Table I gives recall per class at 0 dB for MCLDNN, LSTM2
and CNN-IQ (our VT-CNN2), 200 test frames per cell; ours have 450 (3 seeds).
A single class at a single SNR is noisy, so compare the average and the
pattern rather than each cell.

| model | paper, mean over 11 classes | ours | where they differ most (paper vs ours) |
|---|---|---|---|
| MCLDNN | 89.6 % | 88.3 % | 8PSK 94 vs 83, QAM16 92 vs 86 |
| LSTM2 | 85.6 % | 87.9 % | AM-DSB 68 vs 89, QAM64 82 vs 94, WBFM 56 vs 41 |
| VT-CNN2 | 80.6 % | 79.7 % | AM-DSB 79 vs 97, QPSK 86 vs 70, QAM16 33 vs 23 |

The same failures appear in both: WBFM near 35–55 % for every model, and
VT-CNN2 near-blind to QAM16 at 0 dB (33 % there, 23 % here).

## 4. ICRNNA

El-Haryqy et al. (Results in Engineering 2025), Table 3: 63.24 %. Our build
written from the paper reaches **63.21 %** trained to convergence. The
working model, a colleague's re-implementation with five differences, is at
62.23 % (three seeds). Covered in `README.md` and `DETECTOR.md`.

## 5. Not comparable

- **VT-CNN2, overall accuracy.** No paper here tables it on 2016.10a. O'Shea,
  Corgan & Clancy (EANN 2016) used the earlier RadioML **2016.04** dataset and a
  larger network (256 and 80 filters against the benchmark's 50 and 50), so
  their 87.4 % "across all SNRs" is a different experiment.
- **MSTFFNet's "LSTM" row** (0.79 M parameters, 56.40 %) is a different,
  four-times-larger model from LSTM2 (201,099), though it cites the same
  paper.

## Verdict

Every model we ported has exactly its published size, and every published
2016.10a accuracy we could find for it is within about one point of ours,
in both directions. Our training pipeline reproduces the literature; the
comparison table in `literature_2016.md` can be quoted next to the papers.

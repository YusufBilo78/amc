# Do our numbers agree with the papers? — RML2016.10a, 11 classes

Our five models were all trained under one protocol (`literature_2016.md`):
1,000 frames per cell, 70/15/15 split per (class, SNR), three seeds, early
stopping on validation. A paper's number can differ from ours for reasons
that have nothing to do with the model: a different split (most papers use
60/20/20 or 50/50), one run instead of several, a fixed epoch count instead
of early stopping, different input normalisation. So "agrees" below means
*within the spread those differences can plausibly cause*, about a point,
and every row says which protocol the paper used where we know it.

Only numbers read from a paper we have are quoted. Where we do not have the
paper, the row says so instead of quoting from memory.

## What we can check now

| model | source we have | what it reports on 2016.10a | ours | verdict |
|---|---|---|---|---|
| **ICRNNA** | El-Haryqy et al., Results in Engineering 26 (2025), Table 3 | 63.24 % overall | faithful build: **63.21 %** (1 run, trained to convergence); our working version: 62.23 % (3 seeds) | **agrees.** The faithful build reproduces the paper to 0.03 points. Our working version is 1.0 point lower; it is a colleague's re-implementation that differs in five places, and is reported as such |
| **MCLDNN** | MSTFFNet paper (Sensors 26(16):5208, 2026), Table 1 — an *independent retraining*, not the original paper. Single run, 60/20/20 split per SNR | 0.41 M parameters; overall 62.02 %; [−20, 0] dB 37.37 %; [0, 18] dB 91.11 %; best single SNR 92.88 % | 406,199 parameters; overall **61.41 %** (seeds 61.19–61.70); [−20, 0] dB 37.16 %; [0, 18] dB 90.77 %; best single SNR 91.98 % | **agrees.** Same size to the thousand; every range within 0.6 points, their single run 0.3 above our best seed |
| LSTM | MSTFFNet paper, Table 1 | 0.79 M parameters; overall 56.40 % | our LSTM2 has 201,099 parameters; 61.44 % | **not the same model.** Theirs is four times larger and cites Rajendran 2018, the same paper; it was evidently built differently (the benchmark's version, ours, takes amplitude and phase). Not a check either way |

How our ranges were computed, for the MCLDNN row: mean accuracy over the
SNR levels in the range, then over three seeds; 0 dB counts in both ranges,
as in their table.

## What needs the paper itself

| model | paper to get | what to read off it |
|---|---|---|
| **VT-CNN2** (CNN1) | O'Shea, Corgan & Clancy, "Convolutional radio modulation recognition networks", EANN 2016 (arXiv 1602.04105) | the accuracy-vs-SNR curve on 2016.10a; the paper's network has more filters than the benchmark's version, so expect a gap |
| **LSTM2** | Rajendran et al., "Deep learning models for wireless signal classification with distributed low-cost spectrum sensors", IEEE TCCN 4(3), 2018 (arXiv 1703.09197) | accuracy at high SNR on 2016.10a, and which input (I/Q or amplitude/phase) and size |
| **MCLDNN** | Xu, Luo, Parr & Luo, "A spatiotemporal multi-channel learning framework for automatic modulation recognition", IEEE WCL 9(10), 2020 | the original's overall accuracy on 2016.10a |
| **PET-CGDNN** | Zhang, Luo, Xu & Luo, "An efficient deep learning model for automatic modulation recognition based on parameter estimation and transformation", IEEE Commun. Lett. 25(10), 2021 | overall accuracy and parameter count (ours: 71,871) |
| all four | Zhang, Luo, Xu, Luo & Zheng, "Deep learning based automatic modulation recognition: models, datasets, and challenges", Digital Signal Processing 129 (2022) 103650 | the benchmark whose code we ported: its accuracy table and parameter table (Table 1) for every model under one protocol. **The single most useful one** |

arXiv and the publishers are blocked from this environment, so these have to
be downloaded by hand and added to the session.

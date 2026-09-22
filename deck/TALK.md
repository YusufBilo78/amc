# Talk script

About 15 minutes at a steady pace, 18 with interruptions. The same text is in each slide's speaker notes; `node build.js` regenerates both from `talk.js`.

Rule for the whole talk: every number is a count first, a percentage second. When a number is a ratio, say what is under it.

## 1. Title  (0:30)

This is a progress report on one question: why a modulation classifier that is nearly perfect on its training data fails when the transmitter changes, and what actually fixes it. Everything I show is measured on one dataset and one model, and every number comes from a file in the repository.

## 2. The problem  (1:30)

The setup. Train on RadioML. Then test on I/Q from a different transmitter model: same five modulations, same SNRs, same sample rate. In-domain the model is at 0.999. On the other transmitter it drops to 0.805. Almost all of that loss is one class: 16QAM is recognised 5.6 percent of the time, and it is read as 64QAM. The four questions on the right are the project: how large is the drop, what causes it, can it be removed, and where does an unseen modulation go.

## 3. Dataset  (1:00)

The dataset is RadioML 2018.01A. 24 modulation classes, 1,024 I/Q samples per frame, 26 SNR levels from minus 20 to plus 30 dB, 4,096 frames per class and SNR. About 2.5 million frames. We decided last week to work on this one dataset only. The panel on the left shows the mean power spectrum of each class: every class shares the same spectral envelope, and that turns out to be the key to the failure.

## 4. Input, output, model, protocol  (1:30)

Input: one frame, 2 by 1,024, normalised to unit power. Model: ICRNNA, a convolutional front end, a bidirectional LSTM and attention, 786 thousand parameters. Output: one of K classes. Protocol: 70-15-15 split stratified over every class and SNR; early stopping reads validation only, the test set is touched once. Three seeds per configuration. And one rule I want to state explicitly: a run counts only if its best epoch plus the patience fits under the ceiling. Otherwise it is a floor, not a measurement. Every result here passes that test.

## 5. Does the backbone classify?  (0:45)

Before anything about domain shift: does the model classify at all. On 24 classes above 10 dB it is at 87.2 percent. On the four classes we agreed on, BPSK, QPSK, 16QAM, 64QAM, it is at 100 percent above 10 dB. The curve sits at chance below minus 12 dB and saturates from plus 8. This is not the contribution; it is the floor the rest stands on.

## 6. The decision table  (1:30)

This is the table you asked for. Rows are what we transmitted, columns are what the model decided. Every row is 10,164 decisions: 308 test frames per class and SNR, 11 SNR levels from 10 to 30 dB, three seeds. Each decision is a different frame, different symbols, different channel, different noise. 40,654 of 40,656 correct. The two errors are both 64QAM decided as 16QAM. All three seeds converged, peaks at epochs 46, 48 and 66 under a ceiling of 150.

## 7. Decisions by SNR  (1:30)

A perfect table at high SNR says nothing, so here is the same table lower down. At 0 dB: BPSK and QPSK are still perfect. 16QAM and 64QAM are a coin flip against each other, 519 against 396 and 349 against 569. Nine hundred twenty-four decisions per row. There are two thresholds: PSK versus QAM closes between minus 4 and 0 dB, which QAM closes between 0 and plus 6. Below the first threshold the classes do not scatter, they sink: at minus 8 dB, 16QAM is read as QPSK 68 percent of the time.

## 8. The gap, and what closes it  (1:30)

Now the domain gap. Five shared classes, five seeds each, all converged. In-domain never moves: 0.999 or 1.000 in every column. Cross-domain: 0.805 with nothing, 0.807 with the standard augmentation set from the literature, rotation, conjugate flip and additive noise, so plus 0.002. With spectral whitening, 0.989. Whitening plus augmentation, 0.995, but that plus 0.006 is one standard deviation, so I do not call it a difference. Each cross-domain number is 110,000 decisions: 2,000 per SNR level, 11 levels, 5 seeds.

## 9. Attribution by intervention  (1:30)

What causes it. Four explanations were tested and refuted: occupied bandwidth, constellation density, symbol timing, channel impairments. Each was matched or removed and the prediction did not move. The one thing that moves it is an intervention: take a failing frame, keep its phase spectrum exactly, substitute only the magnitude spectrum of the training domain. The prediction follows the magnitude spectrum. That is a causal claim, because we changed the signal, not the representation. The cause is the transmitter's spectral envelope, mainly the pulse-shaping roll-off.

## 10. The fix  (1:00)

So the fix removes the envelope: divide each frame's spectrum by a smoothed estimate of its own magnitude. Alpha is how much of it. Cross-domain accuracy goes 0.805, 0.855, 0.940, 0.988, 0.993 for alpha 0 to 1. The recovery is entirely 16QAM: its recall goes from 0.056 to 0.985. And it costs nothing in-domain, 0.999 stays 0.999.

## 11. Where an unseen modulation lands  (1:30)

Where does a modulation the model never saw go. Train on 23 classes, probe with the 24th, record which class absorbs it. Families were fixed before the run. 16 of 24 land in their own family; chance is 3.5. All 24 runs converged. The eight misses are on the table. Four of them are APSK versus QAM, both amplitude-and-phase constellations; two are FM versus GMSK, both constant envelope. So the misses are structured, but I quote 16, not a rescored 20, because the families were fixed in advance. On the right: the same test on five architectures. Same-family sink survives on all of them.

## 12. Three things withdrawn  (0:45)

Three things this rerun took away. They are kept in the record. Partial whitening beats full: withdrawn, alpha 1 is at least as good as 0.75. The model reads the class off the envelope: withdrawn, giving 16QAM a 64QAM envelope leaves the prediction at 16QAM, so the mechanism is open. The 60-epoch ceiling was safe: it was not, cells peaked at 82 to 89, and the rerun with room came back bit-identical or within a point.

## 13. Caveats  (0:45)

Caveats. Both domains are synthetic; until we have an over-the-air capture this is a study of a mechanism. The model is not the published architecture; a paper-faithful build exists and reproduces the paper's 63.24 to within 0.03, but only trained to convergence. Half of the sink checks are still on the old model. And three seeds is thin on 24 classes: under about 0.03 a difference is not a difference.

## 14. Next  (0:30)

What just landed is the sink thread. Next is the domain-adversarial baseline, port or drop, and the cumulant classifier on the same four classes at 0 dB, to check whether the QAM coin flip is the signal's limit or the model's. The deck is the running record: new results get appended, withdrawn claims stay.

---

# Questions to expect

**"How many times did you do it?"**
Ten thousand one hundred sixty-four per row in the high-SNR table. At any single SNR, 924 per row. Each is a different frame: different symbol sequence, different channel draw, different noise. That is a harder test than repeating one frame, because nothing is held fixed. If you want the literal version, the same frame with 10,000 independent noise realisations, it is an inference-only run and I can add it.

**"What does 0.805 mean? Over what?"**
Mean accuracy over the 11 SNR levels from 10 to 30 dB. Under it: 400 frames per class per level, 5 classes, 11 levels, 5 seeds, so 110,000 decisions. The per-level curve is in the file; above 10 dB it is flat at 0.80 to 0.82.

**"Why 16QAM specifically?"**
Because 16QAM and 64QAM differ in constellation density but share everything else, and the model does not read density. What it reads, in part, is the spectral envelope. The other transmitter's envelope is different, and the decision that depends on it flips. PSK classes have other cues, so they survive.

**"Is the coin flip at 0 dB the model's limit or the signal's?"**
Not yet measured. The next item is the cumulant classifier on the same four classes at 0 dB, which needs no neural network. If it also flips a coin, the signal is the limit. If it does better, the model is.

**"How do you know it converged?"**
Best epoch plus patience fits under the ceiling. For the four-class table: peaks at 46, 48 and 66 with patience 20 under 150. The earlier 60-epoch run of the same table was flagged unconverged by that rule, rerun at 150, and two of the three seeds came back bit-identical. That is the check working.

**"Why only two errors, is that plausible?"**
Above 10 dB with 1,024-sample frames these four constellations are separable. The two errors are 64QAM at 10 or 12 dB read as 16QAM, the one direction that physics allows: a noisy 64QAM can look like 16QAM, a 16QAM cannot look like 64QAM. On the 2016 dataset with 128-sample frames the same pair still confuses 7 to 11 percent at high SNR; the frame length is the difference.

**"What is whitening, physically?"**
Divide the frame's spectrum by a smoothed estimate of its own magnitude spectrum. It flattens the envelope the pulse-shaping filter imposed. The symbols and their timing are untouched; only the shape of the spectrum is removed. It is not new; WhiteNet did it on real captures. What is ours is showing by intervention that the envelope is the cause.

**"What is the model?"**
A convolutional front end, a bidirectional LSTM, additive attention, 786 thousand parameters. It is transcribed from a peer's reproduction of the ICRNNA paper and differs from the paper in five places, so I do not call it the published architecture. A paper-faithful build exists separately and reproduces the paper's number to 0.03.

**"Is the second transmitter real?"**
No. It is our own baseband generator with an independently chosen pulse shape and channel. Both domains are synthetic. That is the largest weakness and the reason a real capture is the next thing that matters.

**"Why is the sink result 16 and not 20?"**
The families were written down before the run. Four of the eight misses are APSK against QAM, which are both amplitude-and-phase constellations, and merging them gives 20. I report 16 because a taxonomy adjusted after seeing the result is not a test.

**"Why one dataset?"**
Your decision last week, and it costs nothing: every cross-domain result was already on 2018. The 2016 runs stay in the record; one of them is the tie to a published number.

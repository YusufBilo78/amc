# Talk script

About 19 minutes at a steady pace, 22 with interruptions. The same text is in each slide's speaker notes; `node build.js` regenerates both from `talk.js`.

Rule for the whole talk: every number is a count first, a percentage second. When a number is a ratio, say what is under it.

Each slide has a one-sentence version in italics: the thing to say if time is short or the discussion has already taken the slide.

## 1. Title  (0:30)

*One question: why a near-perfect classifier fails when the transmitter changes, and what fixes it.*

This is a progress report on one question: why a modulation classifier that is nearly perfect on its training data fails when the transmitter changes, and what actually fixes it. Everything I show is measured on one dataset and one model, and every number comes from a file in the repository.

## 2. The problem  (1:45)

*0.999 in-domain, 0.805 on our own cleaner transmitter; the loss is 16QAM read as 64QAM.*

The setup. Train on RadioML. Then test on I/Q from a second transmitter: our own baseband generator, same five modulations, same SNR grid, same 1,024-sample frames. What differs is in the small table: RadioML draws a pulse-shaping roll-off per frame between 0.1 and 0.4 and adds carrier, clock and timing offsets and multipath; our generator uses a fixed roll-off of 0.35 and nothing but noise. So the test domain is the cleaner one. In-domain the model is at 0.999. On the other transmitter it drops to 0.805. Almost all of that loss is one class: 16QAM is recognised 5.6 percent of the time, and it is read as 64QAM. The four questions on the right are the project: how large is the drop, what causes it, can it be removed, and where does an unseen modulation go.

## 3. Dataset  (1:00)

*RadioML 2018: 24 classes, 1,024 samples, 26 SNRs, 2.5 million frames; one dataset by decision.*

The dataset is RadioML 2018.01A. 24 modulation classes, 1,024 I/Q samples per frame, 26 SNR levels from minus 20 to plus 30 dB, 4,096 frames per class and SNR. About 2.5 million frames. We decided last week to work on this one dataset only. The panel on the left shows the mean power spectrum of each class: every class shares the same spectral envelope, and that turns out to be the key to the failure.

## 4. Which classes the dataset cannot separate  (1:15)

*The impairment is deliberate; the six classes it breaks are known, reproduced, and outside our four.*

One more thing about the dataset, because it decides which four classes we use. The impairment in RadioML 2018 is deliberate. Every frame draws its own random channel: a pulse-shaping roll-off between 0.1 and 0.4, carrier and clock offsets, Rayleigh fading. And for the 24-class set the authors say they apply impairments beyond what one would expect for the high-order modes, in a 1,024-sample window. So some classes stay hard even at high SNR: the high-order QAM block, 64, 128 and 256, the 16 and 32 PSK pair, and the AM pairs with and without carrier. Their explanation is that there is not enough information to separate them, by any method, and that the AM message content is a small voice corpus. Our own 24-class run reproduces exactly that: the six classes under 80 percent are those, and everything else is above 79. None of the six is in the four-class table. And the alternative dataset, 2016, is worse on exactly this table: its frames are 128 samples, so 16QAM and 64QAM confuse each other 8 to 9 percent even at high SNR, and its WBFM class is broken. So the four classes and the dataset are chosen together.

## 5. Input, output, model, protocol  (1:30)

*One frame in, one of K classes out; 70-15-15, early stopping on validation, a run counts only if it converged.*

Input: one frame, 2 by 1,024, normalised to unit power. Model: ICRNNA, a convolutional front end, a bidirectional LSTM and attention, 786 thousand parameters. Output: one of K classes. Protocol: 70-15-15 split stratified over every class and SNR; early stopping reads validation only, the test set is touched once. Three seeds per configuration. And one rule I want to state explicitly: a run counts only if its best epoch plus the patience fits under the ceiling. Otherwise it is a floor, not a measurement. Every result here passes that test.

## 6. Does the backbone classify?  (0:45)

*24 classes 87.2% above 10 dB, four classes 100%; the floor the rest stands on.*

Before anything about domain shift: does the model classify at all. On 24 classes above 10 dB it is at 87.2 percent. On the four classes we agreed on, BPSK, QPSK, 16QAM, 64QAM, it is at 100 percent above 10 dB. The curve sits at chance below minus 12 dB and saturates from plus 8. This is not the contribution; it is the floor the rest stands on.

## 7. The decision table  (1:30)

*10,164 decisions per row, 2 errors; from +14 dB, 33,264 decisions and zero errors.*

This is the table you asked for. Rows are what we transmitted, columns are what the model decided. Every row is 10,164 decisions: 308 test frames per class and SNR, 11 SNR levels from 10 to 30 dB, three seeds. Each decision is a different frame, different symbols, different channel, different noise. 40,654 of 40,656 correct. The two errors are both 64QAM decided as 16QAM, at 10 and 12 dB. From 14 dB up there are none: 33,264 decisions, zero errors, all four classes, all three seeds. That is the 100 percent you asked us to establish in the simulated environment before moving on. All three seeds converged, peaks at epochs 46, 48 and 66 under a ceiling of 150.

## 8. Decisions by SNR  (1:15)

*At 0 dB PSK is perfect and the QAM pair is a coin flip; below −4 dB the QAMs sink into QPSK.*

A perfect table at high SNR says nothing, so here is the same table lower down. At 0 dB: BPSK and QPSK are still perfect. 16QAM and 64QAM are a coin flip against each other, 519 against 396 and 349 against 569. Nine hundred twenty-four decisions per row. There are two thresholds: PSK versus QAM closes between minus 4 and 0 dB, which QAM closes between 0 and plus 6. Below the first threshold the classes do not scatter, they sink: at minus 8 dB, 16QAM is read as QPSK 68 percent of the time.

## 9. The SNR table  (1:00)

*Every SNR level, 924 decisions per cell; chance from −12 dB down; QPSK's high recall there is the sink, not recognition.*

This is the SNR table, every level from minus 20 to plus 30, recall per class and the pooled accuracy. 924 decisions in every cell. Read it bottom-up from 30 dB: everything is 100 down to plus 6; 64QAM is the first to lose, then 16QAM; by 0 dB the two QAMs are at 56 and 62 while both PSKs are still at 100; BPSK holds to minus 6. One thing to read correctly: below minus 8 the QPSK column stays around 80. That is not recognition. QPSK is where every noisy frame sinks, so its recall stays high while the pooled accuracy, in the last column, is at chance, 25, from minus 12 down.

## 10. Why accuracy falls, in that order  (1:30)

*Constellation distance over noise decides: 64QAM goes first, then 16QAM, then QPSK, BPSK last.*

And why it falls in that order. All four signals have the same power. What differs is how close their constellation points are. At unit energy BPSK's two points are 2 apart, QPSK 1.41, 16QAM 0.63, 64QAM 0.31. Noise is the same for all of them, so the ratio of distance to noise is what decides. The top table is that ratio; the bottom table is the measured recall at the same SNRs; they line up. Below 1 the neighbours overlap. 64QAM crosses 1 at about plus 7 dB, 16QAM at about plus 1, QPSK at about minus 6, BPSK at minus 9. That gives three decisions closing at three SNRs: which QAM, between plus 6 and 0; PSK versus QAM, between 0 and minus 4; and BPSK last. The accuracy curve is those three thresholds averaged. This is a rule of thumb for the order, not a prediction of exact numbers; the model integrates over about 128 symbols and does not need to decode any of them.

## 11. The gap, and what closes it  (1:30)

*0.805 with nothing, 0.807 with the literature augmentation, 0.989 with whitening; in-domain never moves.*

Now the domain gap. Five shared classes, five seeds each, all converged. In-domain never moves: 0.999 or 1.000 in every column. Cross-domain: 0.805 with nothing, 0.807 with the standard augmentation set from the literature, rotation, conjugate flip and additive noise, so plus 0.002. With spectral whitening, 0.989. Whitening plus augmentation, 0.995, but that plus 0.006 is one standard deviation, so I do not call it a difference. Each cross-domain number is 110,000 decisions: 2,000 per SNR level, 11 levels, 5 seeds.

## 12. Attribution by intervention  (1:30)

*Four explanations refuted; swapping only the magnitude spectrum moves the prediction, so the envelope is the cause.*

What causes it. Four explanations were tested and refuted: occupied bandwidth, constellation density, symbol timing, channel impairments. Each was matched or removed and the prediction did not move. The one thing that moves it is an intervention: take a failing frame, keep its phase spectrum exactly, substitute only the magnitude spectrum of the training domain. The prediction follows the magnitude spectrum. That is a causal claim, because we changed the signal, not the representation. The cause is the transmitter's spectral envelope, mainly the pulse-shaping roll-off.

## 13. The fix  (1:00)

*Divide out the envelope: 0.805 to 0.993, 16QAM from 0.056 to 0.985, nothing lost in-domain.*

So the fix removes the envelope: divide each frame's spectrum by a smoothed estimate of its own magnitude. Alpha is how much of it. Cross-domain accuracy goes 0.805, 0.855, 0.940, 0.988, 0.993 for alpha 0 to 1. The recovery is entirely 16QAM: its recall goes from 0.056 to 0.985. And it costs nothing in-domain, 0.999 stays 0.999.

## 14. Where an unseen modulation lands  (1:30)

*16 of 24 unseen modulations land in their own family against a chance of 3.5; holds on five architectures.*

Where does a modulation the model never saw go. Train on 23 classes, probe with the 24th, record which class absorbs it. Families were fixed before the run. 16 of 24 land in their own family; chance is 3.5. All 24 runs converged. The eight misses are on the table. Four of them are APSK versus QAM, both amplitude-and-phase constellations; two are FM versus GMSK, both constant envelope. So the misses are structured, but I quote 16, not a rescored 20, because the families were fixed in advance. On the right: the same test on five architectures. Same-family sink survives on all of them.

## 15. Three things withdrawn  (0:45)

*Three earlier claims withdrawn and kept in the record.*

Three things this rerun took away. They are kept in the record. Partial whitening beats full: withdrawn, alpha 1 is at least as good as 0.75. The model reads the class off the envelope: withdrawn, giving 16QAM a 64QAM envelope leaves the prediction at 16QAM, so the mechanism is open. The 60-epoch ceiling was safe: it was not, cells peaked at 82 to 89, and the rerun with room came back bit-identical or within a point.

## 16. Caveats  (0:45)

*Both domains synthetic; model not the published one; half the sink checks still old; three seeds is thin.*

Caveats. Both domains are synthetic; until we have an over-the-air capture this is a study of a mechanism. The model is not the published architecture; a paper-faithful build exists and reproduces the paper's 63.24 to within 0.03, but only trained to convergence. Half of the sink checks are still on the old model. And three seeds is thin on 24 classes: under about 0.03 a difference is not a difference.

## 17. Next  (0:30)

*Sink thread landed; next is the domain-adversarial baseline and the cumulant check at 0 dB.*

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

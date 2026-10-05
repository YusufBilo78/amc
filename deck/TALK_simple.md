# Talk script — simple deck (deck/moshe_2016_simple.pptx)

The same text is in each slide's notes. About 15 minutes. The detailed deck, deck/moshe_2016.pptx, stays as the backup for questions.

## 1. Telling four radio signals apart

Today I will show what our detector does, step by step, on the four signals you asked about: BPSK, QPSK, QAM16 and QAM64. On the right is one QAM16 frame: sixteen symbols, and you can watch them spread out as the noise grows. Everything is on the RML2016.10a dataset, the one the group uses.

## 2. What is AMC, and what am I working on?

First, in one minute, what this is about. AMC means automatic modulation classification. A receiver picks up a radio signal that nobody described to it, and has to say how it was modulated: is it BPSK, QPSK, QAM, FM? It matters for watching the spectrum, for finding interference, and for radios that adapt to what they hear. What I work on has three parts. First, a detector that does this, trained on a standard dataset that everyone in the field uses, RML2016.10a, which is simulated. Second, opening that detector up: what exactly it does, what the noise in the data is, and where it fails. That is today. Third, the bigger question of my project: why detectors like this lose accuracy when the signal comes from a different transmitter, and what fixes it. Today I stay with the first two, on four signals: BPSK, QPSK, QAM16 and QAM64.

## 3. What data do I use?

The data. I use RML2016.10a, made by DeepSig in a radio simulator in 2016; it is the dataset most papers in this field report on. It has 11 modulation types, 8 digital and 3 analog. Each type is given at 20 noise levels, from minus 20 to plus 18 dB, with 1,000 frames at each level: 220,000 frames in total. Every frame is 128 samples, which is 16 symbols. Today I use four of the eleven, in orange: BPSK and QPSK, which only change the phase, and QAM16 and QAM64, which also change the amplitude.

## 4. What does the detector do?

The detector is given 128 samples of a radio signal and has to say which of the four it is. It works in three steps. First it looks at short pieces of the frame, about one and a half symbols each, so the frame becomes 32 pieces. Second, it reads those pieces in order, forwards and backwards, so that each piece knows about the whole frame. Third, it takes a weighted average of the 32 pieces and turns it into four scores; the largest score is the answer. In total it has 785 thousand numbers that it learns, and 84 percent of them are in the second step. Once trained, nothing in it is random: the same frame always gets the same answer.

## 5. Who designed it, and who wrote the code?

Where it came from. The design is from a 2025 paper; the model is called ICRNNA. A colleague re-implemented that paper, and our code is copied from their version, which differs from the paper in five small places. To check that the paper's number is real, I rebuilt the model from the paper alone and trained it until it stopped improving: 63.21 percent against the paper's 63.24. And to be clear about the tools: the code, the plots and these slides were written with Claude Code, an AI coding assistant. Every number comes from running that code on a GPU, and what to test was my choice. [Say here who the colleague is and how the work before September was done.]

## 6. What we gave it, and what came out

This is the whole experiment on one slide. Input: the RML2016.10a dataset, made by DeepSig in a simulator: four kinds of signal, twenty noise levels, a thousand frames each. One frame is 128 samples, which is 16 symbols. The detector learns on 70 percent of the frames, uses 15 percent to decide when to stop learning, and is tested on the last 15 percent, which it never saw. I repeated this 14 times with a different split each time. Output: at 10 dB and above it is right 94.2 percent of the time, over 42,000 test decisions. BPSK and QPSK are right 99.6 percent of the time. At minus 20 dB it is at 25 percent, which with four choices is a guess.

## 7. What one frame looks like

Step one: what the detector actually sees. This is one QAM16 frame, drawn without noise. The orange line is the in-phase part, the blue line the quadrature part. Every 8 samples there is one symbol, the dots, and each dot sits on minus 3, minus 1, 1 or 3. That pattern is what makes it QAM16. On the right are the 16 symbols of one frame for each kind, drawn on all the points that kind can use. QAM64 has 64 points, but a frame only has 16 symbols, so it can show at most 16 of them. A QAM64 frame can look like QAM16 even with no noise at all. Keep that in mind.

## 8. What does “10 dB” mean in this data?

Step two: the noise. You asked what 10 dB is the ratio of. The dataset was made by DeepSig in a simulator. In their code, the label sets exactly one number: the noise amplitude, 10 to the minus label over 10. Besides noise, every frame also gets a carrier offset, a clock offset, fading and echoes, at every noise level, even the best one. I did not want to rely on their code, so I measured the noise in the file itself: the signal sits in the middle of the spectrum, and outside it there is only noise. Two findings. One dB of label is about 1.9 dB of real change, so the label moves twice as fast as its name. And at the same label, QAM64 frames have about 15 dB more signal than BPSK and QPSK frames. So 10 dB here is a setting, not a measurement, and it means a different noise level for each kind of signal.

## 9. The answer at one noise level: 10 dB

The answer at one noise level, 10 dB. 8,400 test frames. Each row is what was sent, each column what the detector said. Recall is a row: of the frames that were QPSK, how many it called QPSK. Precision is a column: of the frames it called QPSK, how many really were. BPSK and QPSK are at 99.6 percent. 448 frames out of 8,400 are wrong, and 395 of those are QAM16 and QAM64 mixed up.

## 10. Turn the noise up: when does it break?

Then, as you asked, turn the noise up. These bars are the number of wrong answers out of 8,400 at each label. From 10 dB down to 2 dB it stays at about 450 to 490: the same box every time. Below 0 dB it starts to break, and QPSK is the first to slip, into the QAMs. So the mistakes we have at 10 dB are not caused by noise.

## 11. 16, 8 or 4 symbols: what does it need?

So what does cause them? Frame length. I cut every frame to 64 samples, which is 8 symbols, and to 32 samples, 4 symbols, and trained again. The curves fall at about the same noise level, but they flatten at different heights: 5.7 percent wrong with 16 symbols, 15.5 with 8, 25.2 with 4. On the right: with fewer symbols you see fewer points of the grid, and QAM16 and QAM64 become harder to tell apart. More signal does not fix this; more symbols does.

## 12. With no signal at all, it still answers

One thing we found. At minus 20 dB there is no signal at all, but the detector still has to name one of the four. QPSK recall there is 52 percent, which looks like it recognises QPSK. It does not: of all the frames it calls QPSK, only 25 percent are QPSK, which is a guess. It sends almost everything, 98.8 percent, to BPSK or QPSK, and which of the two changes from one training run to the next. That is why every chart now shows precision next to recall, and how many decisions each point is based on.

## 13. How does ICRNNA compare with published models?

Is ICRNNA a good choice? I took four well-known published models, from 2016 to 2021, rewrote each one from its authors' benchmark code, and trained all of them exactly like ICRNNA: same data, same split, same three repeats. This time on all eleven types. ICRNNA is first overall, 62.2 percent, but the best four are within about one point of each other. Where ICRNNA gains is between minus 6 and minus 2 dB. Above 10 dB all four stop at about 91 percent, so that limit comes from the data, mostly WBFM and the QAM pair, not from the model. The oldest model, the 2016 CNN, stays about ten points lower up there.

## 14. In short

To sum up. BPSK and QPSK are essentially solved; the mistakes are QAM16 against QAM64. Those mistakes come from having only 16 symbols, not from noise: with 8 or 4 symbols they grow 2.7 and 4.3 times. And 10 dB in this dataset is a setting, not a measurement. Next, I want to switch off the carrier offset, the fading and the echoes one at a time and see which one the QAM mistakes belong to. Thank you.

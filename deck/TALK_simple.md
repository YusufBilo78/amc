# Talk script — simple deck (deck/moshe_2016_simple.pptx)

The same text is in each slide's notes. About 12 minutes. Turkish explanations and likely questions: STUDY_simple.md. The detailed deck, deck/moshe_2016.pptx, stays as the backup for questions.

## 1. Telling four radio signals apart

Today I want to open up our detector and show you what it does, step by step. I'll use the four signals you asked about: BPSK, QPSK, QAM16 and QAM64. The picture on the right is one QAM16 frame. Watch the sixteen points spread out as the noise grows. That picture is really today's whole talk: how many symbols we have, and how much noise is on them.

## 2. What is AMC, and what am I working on?

Quickly, what this is about. A receiver picks up a signal, and nobody told it what kind of signal it is. AMC, automatic modulation classification, answers one question: how was this signal modulated? You need that to watch the spectrum, to find interference, and for radios that adapt to what they hear. My work has three steps. One: a detector that does this, trained on a standard simulated dataset. Two: open it up. What does it do, what is the noise really, and where does it fail? Three, the real question of the project: why do these detectors get worse on a signal from a different transmitter? Today is steps one and two.

## 3. What data do I use?

The data is RML2016.10a. DeepSig made it in a simulator in 2016, and most papers in this field use it. Eleven signal types. Twenty noise levels, from minus twenty to plus eighteen dB. A thousand frames of each type at each level. That's two hundred and twenty thousand frames. One frame is 128 samples, and that is only sixteen symbols. Please remember that number. Today I use the four in orange. BPSK and QPSK only change the phase. QAM16 and QAM64 change the phase and the amplitude.

## 4. What does the detector do?

The detector gets 128 samples and has to say which of the four it is. It does that in three steps. First, it looks at short pieces, about one and a half symbols each. The frame becomes thirty-two pieces. Second, it reads those pieces in order, forwards and backwards, a bit like reading a sentence. After that, every piece knows about the whole frame. This part is an LSTM, and it holds most of the model, 84 percent. Third, it decides. It weighs the pieces, averages them, and gives four scores. The biggest score wins. Two things to keep in mind. Once trained, it is not random: same frame, same answer. And it can never say "I don't know". That will matter later.

## 5. Who designed it, and who wrote the code?

Where did it come from? The design is from a 2025 paper; the model is called ICRNNA. A colleague coded it up, and our code comes from theirs. It's close to the paper, with five small differences. I wanted to check the paper's number, so I built the model again from the paper alone. Trained long enough, it gets 63.21 percent. The paper says 63.24. So the number is real. And to be open about my tools: the code, the plots and these slides were written with Claude Code, an AI assistant. The numbers come from actually running that code. What to test was my decision. [Say here who the colleague is and how the work before September was done.]

## 6. What we gave it, and what came out

Here is the whole experiment on one slide. In: four signal types, twenty noise levels, a thousand frames each. The detector learns on 70 percent of the frames. It uses 15 percent to decide when to stop. And it is tested on the last 15 percent, which it has never seen. I did this fourteen times with different splits, so it isn't luck. Out: at 10 dB and above, it is right 94 percent of the time, over 42,000 decisions. BPSK and QPSK are almost perfect. At minus 20 dB it is right 25 percent of the time, and with four choices that is just a guess. Now let's open each part: the input, the noise, then the answer.

## 7. What one frame looks like

First, the input. This is one QAM16 frame, drawn without noise. Orange is the I part, blue is the Q part. Every eight samples there is a symbol, the dots. Each dot sits on one of four levels. Four levels times four levels: sixteen points. That's QAM16. On the right are the sixteen symbols of one frame, drawn on all the points each type can use. BPSK has two points, so sixteen symbols cover it many times. But QAM64 has sixty-four points, and we only have sixteen symbols. We never see the whole grid. So a QAM64 frame can look like QAM16 even with no noise at all. Keep that in mind.

## 8. What does “10 dB” mean in this data?

Second, the noise. You asked what "10 dB" actually is. The short answer: it's a setting in their code, not a measurement. In the code, the label sets one thing, the noise amplitude. Not the power. So one dB of label really moves the noise by about two dB. And noise is not the only thing added. Every frame also gets a frequency offset, a clock offset, fading and echoes. Even the cleanest frames. I didn't want to just trust the code, so I measured the noise in the file. The signal sits in the middle of the band; outside it there is only noise. Two findings. One dB of label is 1.9 dB of real change. And at the same label, QAM64 has about 15 dB more signal than BPSK and QPSK. So "10 dB" means a different noise level for each signal.

## 9. The answer at one noise level: 10 dB

Now the answer, at one noise level: 10 dB. Rows are what was sent, columns are what the detector said. The diagonal is correct. Two words I'll use. Recall is a row: of the real QPSK frames, how many did it get? Precision is a column: when it says QPSK, how often is it right? BPSK and QPSK: 99.6 percent. Solved. 448 mistakes out of 8,400. And 395 of them are just QAM16 and QAM64 confused with each other. So at 10 dB there is really one problem left: QAM16 against QAM64.

## 10. Turn the noise up: when does it break?

Then, as you asked, I turned the noise up, one level at a time. The bars are the mistakes out of 8,400. From 10 dB down to 2 dB: about 450 to 490. Eight dB more noise, and nothing changes. Same mistakes. Below 0 dB it breaks, and the first to go is QPSK: it starts to look like a QAM. So the mistakes at 10 dB are not caused by noise. Then what causes them?

## 11. 16, 8 or 4 symbols: what does it need?

My guess was the frame length. So I cut every frame in half, to eight symbols, and in half again, to four, and trained the same model again. Look at the right side of the chart, where the noise is low. Every curve goes flat. With sixteen symbols, 5.7 percent wrong. With eight, 15.5. With four, 25. Almost all of that is the QAM pair. With four symbols, QAM16 against QAM64 is a coin flip. The picture on the right shows why: fewer symbols, fewer grid points, and the two QAMs look the same. So noise decides where the curve drops. The number of symbols decides how low it can go. More signal won't fix it. More symbols will.

## 12. With no signal at all, it still answers

One thing we found along the way. At minus 20 dB there is basically no signal. But the detector can't say "I don't know", so it has to pick one. QPSK recall there is 52 percent. That looks like it still recognises QPSK. It doesn't. Precision is 25 percent: of everything it calls QPSK, only a quarter really is. That's a guess. What really happens: with nothing to go on, it says BPSK or QPSK almost every time, and which of the two changes from run to run. The lesson: recall alone can fool you. So every chart now shows precision next to recall, and how many decisions are behind each point.

## 13. How does ICRNNA compare with published models?

Is ICRNNA a good choice? I took four well-known published models and trained them exactly like ICRNNA: same data, same split, same repeats. This time all eleven types, because that's what the papers report. First, can you trust my numbers? Each model has exactly the same number of parameters as in its paper. And my accuracies are within one point of the published ones. So it's a fair comparison. The result: ICRNNA is first, at 62.2 percent, but the top four are within about one point. And above 10 dB they all stop at around 91 percent. When four different designs hit the same ceiling, the limit is in the data, not in the model. Only the oldest one, from 2016, is clearly behind.

## 14. In short

So, three things. One: BPSK and QPSK are solved. The mistakes are QAM16 against QAM64. Two: those mistakes come from having only sixteen symbols, not from noise. With eight symbols there are almost three times as many, with four more than four times. Three: "10 dB" here is a setting, not a measurement, and it's different for each signal. Next, I'll switch off the frequency offset, the fading and the echoes one at a time, and see which one the QAM mistakes follow. Thank you.

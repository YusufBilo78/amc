# Talk script — simple deck (deck/moshe_2016_simple.pptx)

The same text is in each slide's notes. About 6–7 minutes. Turkish explanations and likely questions: STUDY_simple.md. The detailed deck, deck/moshe_2016.pptx, stays as the backup for questions.

## 1. Telling four radio signals apart

Today I'll open up our detector, step by step, on the four signals you asked about: BPSK, QPSK, QAM16 and QAM64. The picture is one QAM16 frame: sixteen points, spreading out as the noise grows. That's the whole talk: how many symbols, and how much noise.

## 2. What is AMC, and what am I working on?

AMC means automatic modulation classification. A receiver gets a signal and has to say how it was modulated. You need that to watch the spectrum, to find interference, and for radios that adapt. My work has three steps: build a detector, open it up, then ask why it gets worse on a different transmitter. Today is the first two.

## 3. What data do I use?

The data is RML2016.10a, from DeepSig. It's simulated, and most papers use it. Eleven signal types, twenty noise levels, a thousand frames each. One frame is 128 samples: only sixteen symbols. Remember that. Today, the four in orange. BPSK and QPSK change only the phase; the QAMs also change the amplitude.

## 4. What does the detector do?

The detector works in three steps. It looks at short pieces of the frame. It reads them in order, forwards and backwards; that's where most of the model is. Then it gives four scores, and the biggest wins. Once trained: same frame, same answer. And it can never say "I don't know".

## 5. Who designed it, and who wrote the code?

The design is from a 2025 paper; the model is called ICRNNA. Our code comes from a colleague's version, with five small differences. I rebuilt it from the paper alone: 63.21 percent, against the paper's 63.24. So the number is real. The code and slides were written with Claude Code, an AI assistant. The numbers come from running that code, and what to test was my choice. [Say here who the colleague is and how the work before September was done.]

## 6. What we gave it, and what came out

The whole experiment: the detector learns on 70 percent of the frames, uses 15 to decide when to stop, and is tested on 15 it never saw. Fourteen times, with different splits. Result: 94 percent right at 10 dB and above. BPSK and QPSK almost perfect. At minus 20 dB, 25 percent: a guess.

## 7. What one frame looks like

This is one QAM16 frame, without noise. A symbol every eight samples, each on one of four levels. Four times four: sixteen points. Now the key point. QAM64 has sixty-four points, but a frame has only sixteen symbols. So a QAM64 frame can look like QAM16 even with no noise.

## 8. What does “10 dB” mean in this data?

You asked what "10 dB" is. It's a setting in their code, not a measurement. It sets the noise amplitude, not the power, so one dB of label is really about two. I measured it in the file: 1.9. And at the same label, QAM64 has about 15 dB more signal than BPSK and QPSK. So "10 dB" is a different noise level for each signal.

## 9. The answer at one noise level: 10 dB

At 10 dB: rows are what was sent, columns what it said. The diagonal is correct. Last time you asked what recall is. Recall is one row: of the frames that really were QAM16, how many did it call QAM16? Here 1,808 of 2,100, 86.1 percent. Precision is one column: when it says QAM64, how often is it right? 1,961 of 2,236, 87.7 percent. So recall asks "did it find them?", and precision asks "can I believe it?" BPSK and QPSK: 99.6 percent. 448 mistakes out of 8,400, and 395 of them are QAM16 and QAM64 mixed up. One problem left.

## 10. Turn the noise up: when does it break?

Then I turned the noise up. From 10 down to 2 dB the mistakes stay around 450 to 490. Nothing changes. It only breaks below 0 dB. So the mistakes at 10 dB are not noise. Then what are they?

## 11. 16, 8 or 4 symbols: what does it need?

Frame length. I cut the frames to eight symbols, then four, and trained again. At low noise every curve goes flat: 5.7 percent wrong with sixteen symbols, 15.5 with eight, 25 with four. Almost all of it is the QAM pair. Noise decides where the curve drops. Symbols decide how low it can go.

## 12. With no signal at all, it still answers

At minus 20 dB there's no signal, but it still has to pick one. QPSK recall is 52 percent, which looks good. But precision is 25 percent: a guess. It just says BPSK or QPSK almost every time. So now every chart shows precision next to recall.

## 13. How does ICRNNA compare with published models?

Is ICRNNA a good choice? I trained four published models the same way, on all eleven types. They have the same parameter counts as in their papers, and accuracy within one point of what the papers publish. ICRNNA is first, but the top four are within one point, and above 10 dB they all stop near 91 percent. That limit is in the data, not the model.

## 14. In short

Three things. BPSK and QPSK are solved; the mistakes are QAM16 against QAM64. Those come from having only sixteen symbols, not from noise. And "10 dB" is a setting, not a measurement. Next: switch off the channel effects one at a time, and see which one the QAM mistakes follow. Thank you.

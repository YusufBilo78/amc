# CALL NOTES — Prof. Ali Abdi

*Toplantı sırasında ekranda açık tut. Okumak için değil, göz atmak için.*

---

## AÇILIŞ (ilk 60 saniye)

"Thank you for making time. I saw that blind modulation recognition is one of
your research interests — that is why I wrote to you specifically.

I spent the summer measuring why a modulation classifier fails when the
transmitter changes. Let me give you the short version, and please stop me
wherever you want."

---

## ANA ANLATIM — dört adım

**1. THE MEASUREMENT**
Trained a 1-D CNN on **RadioML 2018.01A**, five modulations
(BPSK, QPSK, 8PSK, 16QAM, 64QAM).
Tested on the **same five** from an independent generator I wrote.
**At SNR ≥ 10 dB: 0.993 in-domain → 0.811 cross-domain.**
Gap **0.182 ± 0.017**, four seeds.
Almost all of it in one cell: **16QAM read as 64QAM, 75%**.

**2. THE DIAGNOSIS** — network frozen, only the test signal changed
- symbol-rate offset → drove the cyclostationary line 14.3 → 0.1 dB → **no effect**
- constellation density → amplitude histograms **overlap** → ruled out
- CFO / multipath / random phase / all combined → **best +0.045** → ruled out
- **spectral envelope** → phase-preserving magnitude substitution →
  **16QAM 0.191 → 0.972** ✓

**3. THE FIX**
Spectral whitening as **preprocessing**, both domains.
Divide each frame's spectrum by its own smoothed envelope. **Phase untouched.**
Not domain adaptation — **needs no target-domain data at fit time.**

| method | in-dom | cross | gap |
|---|---|---|---|
| none | 0.993 | 0.811 | 0.182 |
| standard AMC aug (rot/flip/noise) | 0.996 | 0.854 | 0.141 |
| **whitening** | 0.965 | 0.909 | 0.056 |
| **whitening + standard** | 0.985 | **0.948** | **0.038** |

**+3.3 s.d. over the literature baseline. They compose (+6.4 s.d.). 79% closed.**

**4. WHAT I GOT WRONG** *(say this, don't wait to be asked)*
Thought it was shortcut learning — the network reading order off the envelope.
**Envelope transplant refutes it:** gave 16QAM frames a 64QAM envelope,
prediction stayed 16QAM **99%**. Only **0.002** followed the envelope.
So it is **not** a cue. It is out-of-distribution: the input moves outside the
region where the boundaries were fitted.

---

## LIMITATIONS — say these before he asks

- **Both domains are synthetic.** RadioML is simulated, mine is simulated.
  A gap between two simulators. **This is the main limitation.**
- Five classes, one architecture, 15 epochs.
- **No comparison against DANN-style adaptation.**
- Literature: **"less than I should have."** Built the measurement first,
  read second — wrong order, correcting it now.

---

## WHAT I WANT

1. **Lab access** — real SDR captures are the missing piece.
   Plan: cabled TX → attenuator → RX. Nothing radiated, repeatable.
2. **Supervision** — senior design and beyond.
3. **Or your topic** — "I would equally welcome working on something already
   running in your lab if that is a better use of your time."

---

## MY QUESTIONS FOR HIM

- "Is the attribution method standard? Using a phase-preserving magnitude
  substitution to assign a domain gap to one signal property — I could not find
  it, but I may be searching wrong."
- "Does the field need methods, or real cross-domain captured data? My sense is
  there is a lot of the first and very little of the second."
- "Does your lab have SDR equipment an undergraduate could use?"

---

## ESCAPE PHRASES

- "Let me think about that for a second."
- "I don't know — I haven't tested that."
- "Could you say that another way?"
- "That's a good question. It's on my list and I haven't done it."

---

## IF HE ASKS "how did you do this in one month"

"I directed the work and made the research decisions. I used AI tooling heavily
for implementation. What I can do is explain every experiment and why it was
designed that way."

---

## NUMBERS — glance here

| | |
|---|---|
| gap | **0.182 ± 0.017** |
| in-dom / cross | 0.993 / 0.811 |
| after fix | **0.948**, gap 0.038 |
| closed | **79%** |
| vs literature | **+3.3 s.d.** |
| 16QAM cross | 0.123 → **0.965** |
| substitution | 0.191 → **0.972** |
| transplant | **0.002** follow envelope |
| seeds | **4** |
| dataset | RadioML 2018.01A, 24 classes, 5 used |

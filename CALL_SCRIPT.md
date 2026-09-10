# CALL SCRIPT — Prof. Ali Abdi

*Ekranda açık tut. Kısa satırlar = konuşma ritmi. Nokta = nefes.*
*Kelimesi kelimesine okuma — göz at, kendi cümlenle söyle. Tıkanırsan oku.*

---

# 1 · OPENING  (60 sn)

Thank you for making the time.

I saw that blind modulation recognition is one of your research interests.
That is why I wrote to you specifically.

I spent the summer measuring why a modulation classifier fails
when the transmitter changes.

Let me give you the short version — and please stop me wherever you want.

**→ Would it help if I shared my screen? I have the figures.**

---

# 2 · THE MEASUREMENT  (60 sn)

*(no figure yet — just talk)*

I trained a 1-D CNN on RadioML 2018.01A.
Five modulations: BPSK, QPSK, 8PSK, 16QAM, 64QAM.

Then I wrote my own signal generator,
produced the same five modulations,
and tested the same network on them.

At 10 dB SNR and above:

- in-domain **0.993**
- cross-domain **0.811**

The gap is **0.182**, plus or minus **0.017**, over four seeds.

And almost all of it sat in one cell of the confusion matrix.
**16QAM being read as 64QAM. Seventy-five percent of the time.**

The other four classes were fine.

---

# 3 · FIGURE: `summary_panel.png`  (60 sn)

*(share screen — start here)*

On the left, cross-domain accuracy against SNR.

Red is without the fix. Green is with it.
The dashed line is in-domain performance — what the model does on its own data.

The shaded area between red and green is what I recovered.

On the right, the generalization gap for four methods.
Lower is better.
Top bar is doing nothing. Bottom bar is my result.

**0.182 down to 0.038.**

---

# 4 · FIGURE: `14_narrow_the_search.png`  (90 sn)

*(the key experiment — slow down here)*

**→ Start with the middle panel.**

Three bars per class.

- Red is my signal.
- Orange is my signal with **their** magnitude spectrum substituted in —
  the phase completely untouched.
- Blue is their own data.

Look at 16QAM.
**0.19. Then 0.97. Against their 0.99.**

Everything else was already fine.
One substitution recovered essentially the whole gap.

**→ Now the right panel.** *(move the cursor there)*

This shows what actually differed.

My out-of-band floor — the red curve —
sits about **12 dB higher** than theirs.
And my main lobe is slightly wider.

After the substitution, the orange curve sits exactly on the blue one.

**→ Now the left panel.**

This corroborates it from a different direction.

If I narrow my pulse shaping —
sweeping the roll-off from 0.35 down to 0.01 —
16QAM climbs from **0.19 to 0.745**, monotonically, across eight points.

And only 16QAM moves. The others stay flat.

**But it stops at 0.745, while full spectral substitution reaches 0.97.**
That tells me their pulse shape is not a plain root-raised cosine.
No roll-off value reproduces it.

---

# 5 · WHAT I RULED OUT  (45 sn)

Before I got there, I eliminated the alternatives.
Network frozen each time — only the test signal changed.

**Symbol-rate offset.**
I drove the cyclostationary line from 14.3 dB down to 0.1 dB,
matching their own value.
16QAM moved from 0.168 to 0.202. Nothing.

**Constellation density.**
The amplitude histograms of the two domains overlap almost exactly.

**Channel impairments.**
Carrier frequency offset, multipath, random phase, and all three combined.
Best recovery was **plus 0.045**, against a shortfall of 0.83.

Only the envelope substitution reproduced the effect.

---

# 6 · FIGURE: `19_envelope_transplant.png`  (60 sn)

*(the part where I was wrong — say it, don't wait to be asked)*

I should tell you where I was wrong.

My account was shortcut learning —
the network reading modulation order off the spectral envelope.

So I tested it.
I gave 16QAM frames the magnitude spectrum of **64QAM**,
keeping their own phase.

If the envelope drives the decision, it should say 64QAM.

**It said 16QAM. Ninety-nine percent of the time.**
Only 0.002 followed the donated envelope.

The matrices are diagonal.

So the envelope is **not** a decision cue.

What is actually happening is out-of-distribution behaviour.
The input moves outside the region where the decision boundaries were fitted.
The model is a function that is only valid where it was fitted.

---

# 7 · THE FIX  (60 sn)

So the fix is not to randomise the envelope during training.

I tried that. It made things worse —
cross-domain fell from 0.825 to 0.783,
and the training loss never converged.

The fix is to **remove** the envelope. On both domains.

Spectral whitening.
Divide each frame's spectrum by a smoothed estimate of its own envelope.
Phase untouched. Nothing is added.

It is preprocessing, not domain adaptation.
**It needs no target-domain data at fit time** —
so it works on a receiver that has never met the transmitter it will see.

| method | in-dom | cross | gap |
|---|---|---|---|
| none | 0.993 | 0.811 | 0.182 |
| standard AMC augmentation | 0.996 | 0.854 | 0.141 |
| whitening | 0.965 | 0.909 | 0.056 |
| whitening + standard | 0.985 | **0.948** | **0.038** |

Against the standard augmentation baseline — rotation, flip, Gaussian noise —
whitening is ahead by **3.3 standard deviations**.

And they **compose** rather than substitute.
Combining adds another 6.4 standard deviations over whitening alone.

**Seventy-nine percent of the gap closes.**

It is a trade, not a free improvement.
In-domain drops from 0.993 to 0.965.
I lose 2.8 points in the lab to gain 10 in the field.

---

# 8 · FIGURE: `16_whitening_seeds.png`  (30 sn)

Four seeds per point. Bars are one standard deviation.
Grey dots are the individual runs.

The effect is 7.4 pooled standard deviations —
well outside run-to-run variation.

One thing worth mentioning:
my first pass used a single seed and produced a dip at alpha 0.25
that looked like a real phenomenon.
Across four seeds it disappeared.
It was one unlucky model.

---

# 9 · LIMITATIONS  (45 sn — say before he asks)

The main limitation is that **both domains are synthetic.**

RadioML is simulated. My generator is simulated.
So what I measured is a gap between two simulators.
It is real and measurable, but I cannot claim anything
about real-world generalization from it.

Also:
five classes, one architecture, fifteen epochs.
No comparison against adversarial domain adaptation.

And on the literature — **less than I should have read.**
I built the measurement first and read second.
That is the wrong order, and I am correcting it.

---

# 10 · WHAT I WANT  (30 sn)

Three things.

**First, lab access.** Real captures are the missing piece.
The plan is a cabled setup — transmitter, attenuator, receiver.
Nothing radiated, so it stays legal and repeatable.

**Second, supervision** — for senior design and beyond.

**Third — and I mean this —**
I would equally welcome working on something already running in your lab,
if that is a better use of your time than my project.

---

# 11 · MY QUESTIONS FOR HIM

**→ Ask at least two. Do not let it become a presentation.**

"What is your lab working on right now?
I would like to understand where I could actually be useful."

"Is the attribution method standard?
Using a phase-preserving magnitude substitution
to assign a domain gap to one specific signal property —
I could not find it, but I may be searching with the wrong terms."

"Does the field need more methods, or real cross-domain captured data?
My sense is there is a great deal of the first and very little of the second."

"Does your lab have SDR equipment an undergraduate could use?"

---

# 12 · IF HE ASKS

**"How did you do all this in one month?"**

I directed the work and made the research decisions —
what to test next, what to rule out,
when to stop patching and diagnose instead.
I used AI tooling heavily for implementation.
What I can do is explain every experiment and why it was designed that way.

**"How do you know it's the signals, not a fragile model?"**

Three things.
In-domain it is stable — 0.993, standard deviation 0.001, four seeds.
The failure is structured, not broad — only 16QAM collapses.
And I froze the network completely, changed only the magnitude spectrum,
and 16QAM went from 0.191 to 0.972. Same weights, same symbols.
What it does **not** rule out is architecture. I tested one network.

**"What's novel here?"**

I am not in a position to claim novelty — I have not done a proper review.
What I think is unusual is the attribution method rather than the fix.
I would value your judgement on whether that is already standard.

**"Why not compare against DANN?"**

I should have. What I can say is that DANN needs target-domain samples
at fit time and whitening does not,
so the comparison is not like-for-like — but it still needs running.

---

# ESCAPE PHRASES

- "Let me think about that for a second."
- "I don't know — I haven't tested that."
- "Could you say that another way?"
- "That's on my list and I haven't done it yet."

---

# NUMBERS

| | |
|---|---|
| gap | **0.182 ± 0.017** |
| in-dom / cross | 0.993 / 0.811 |
| after fix | **0.948**, gap **0.038** |
| closed | **79%** |
| vs literature | **+3.3 s.d.** |
| composition | +6.4 s.d. |
| effect size | 7.4 s.d. |
| 16QAM cross | 0.123 → **0.965** |
| substitution | 0.191 → **0.972** |
| transplant | **0.002** follow envelope |
| in-domain cost | 0.993 → 0.965 |
| seeds | **4** |
| dataset | RadioML 2018.01A, 24 classes, 5 used |

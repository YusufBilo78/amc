# Olası sorular ve cevaplar — ICRNNA makale sunumu

Sunum: `moshe_paper_icrnna.pptx`, 22 slayt: ana akış 1–10, yedek 11–22. Her slayt için o slaytta yazan şeylerden gelebilecek sorular var.

Her soru için:

- **Kalın satır:** Moshe'nin sorusu (İngilizce).
- **Turuncu kutu:** Sesli söyleyeceğin cevap (İngilizce, kısa).
- **TR:** Cevabın Türkçe açıklaması, ne dediğini bilmen için.

**Altın kurallar:**

- **Bilmiyorsan uydurma.** "The paper doesn't say" ya da "I haven't measured that; I can check" de. İkisi de iyi cevap.
- **Önce tek cümle cevap, sonra dur.** Moshe isterse ayrıntıya girersin.
- **Sayının kaynağını söyle:** "their Table 5", "our run", "their code".

En olası 15 soru ilk kartta. Sonra slayt slayt, en sonda genel sorular.

---

## En olası 15 soru

**1. "What do you mean by 'right'? What is accuracy here?"**
> The fraction of test frames whose label the model gets right, counted over all 20 noise levels together: 63 out of 100.

TR: Doğruluk = doğru karar / tüm karar. 20 SNR seviyesinin hepsi birlikte.

**2. "Why is 63 percent good if it's wrong a third of the time?"**
> Because the average includes the noisiest levels, where nobody can do better than guessing. At high SNR the same model is right about 93 percent of the time.

TR: −20 dB'de herkes şans seviyesinde (%9). Ortalama bu seviyeleri de içeriyor.

**3. "What is chance here?"**
> One in eleven: about 9 percent. At −20 dB the model is at 9 percent, so it is guessing there.

**4. "What is SNR here? What is 0 dB?"**
> It's the label the dataset puts on each frame, not a measured SNR. We measured it in the file: at label 0, BPSK and QPSK are at about 5 dB, QAM16 at 14 and QAM64 at 17. So I quote labels as labels.

TR: Bizim ölçümümüz (`NOISE_2016.md`). Etiket gerçek SNR değil; sınıflar arasında da fark var.

**5. "What is a frame?"**
> 128 consecutive complex samples, I and Q: 0.64 milliseconds at 200 kHz, 16 symbols at 8 samples per symbol. That's all the model sees.

**6. "What is an LSTM, in signal-processing terms?"**
> A learned nonlinear recursive filter. It reads the sequence one step at a time and keeps a state of 128 numbers. Learned gates decide how much of the state to keep, what to add and what to output.

TR: IIR filtre benzetmesi Moshe'ye uygun: durumu (state) olan, kapıları öğrenilen bir özyinelemeli filtre.

**7. "What is attention?"**
> A learned weighted average over the 64 time steps. Each step gets a score, a softmax turns the scores into weights that sum to one, and the steps are averaged with those weights.

**8. "What is 'learned'? What are the 794 thousand numbers?"**
> The filter taps and weight matrices. They start random and are adjusted, step by step, to make the right answers more probable on the 154,000 training frames.

**9. "Why is your number 62.2 on one slide and 63.21 on another?"**
> 63.21 is the rebuild from the paper, trained their way: that's the reproduction. 62.2 is our working version, trained with one common recipe for all five models so they can be compared fairly. It's a slightly different build and recipe.

TR: Özet (slayt 10) ile slayt 9 arasındaki fark. 62.2 proje sürümü (786,379 parametre, batch 256, bizim ayırma, 3 koşu ortalaması). 63.21 makaleye sadık kurulum (1 koşu, epoch 107).

**10. "Is half a point significant?"**
> Not from one run. The test set alone gives about ±0.3 points of noise, and retraining the same model with a different seed moves it by up to half a point in our runs, by over a point for LSTM2.

TR: 33,000 test çerçevesinde doğruluğun standart hatası √(0.63 × 0.37 / 33,000) ≈ 0.27 puan. Tohum farkı: ICRNNA 0.47, LSTM2 1.14 puan.

**11. "Why does WBFM fail?"**
> In this dataset the analog classes are generated from voice recordings. During silences both WBFM and AM-DSB become a bare carrier, and no model can separate them. That's the known explanation; I haven't measured it myself.

**12. "Why are QAM16 and QAM64 confused?"**
> 128 samples are only 16 symbols, too few to tell a 16-point grid from a 64-point one. When we shortened the frame, QAM16 recall at high SNR fell from 87 to 66 to 46 percent at 128, 64 and 32 samples. Raising the SNR doesn't fix it; more symbols do.

TR: Bizim dört sınıflı kırpma deneyimiz, ≥10 dB, 14 koşu.

**13. "Did they retrain the other models?"**
> The paper doesn't say. Six of them appear with confusion matrices, so they must have run those. The other five appear only in the summary table and are probably taken from other papers.

**14. "So what do we learn from this paper?"**
> That this design works and is reproducible, and that the LSTM does most of the work. It's about one point better than older models, not a big jump. For us it's a solid detector to build on.

**15. "What did you do yourself, and what did the AI do?"**
> The rebuild code and these slides were written with an AI coding assistant, Claude Code, from the paper. The numbers come from runs I did on Colab, and the rebuild matches the paper independently: 63.21 against 63.24.

TR: `DETECTOR.md` bölüm 6 ile tutarlı. Kendi yaptıklarını kendi cümlelerinle söyle.

---

## Slayt 1 — Başlık

**"Why this paper?"**
> Because our classifier is built on it. Every result I've shown you uses this architecture, so I should be able to explain it.

**"What do 'Look, Read in order, Weigh, Decide' mean?"**
> The four steps of the model, in plain words. I'll go through them on slide 4.

**"Who are the authors? Is it a good journal?"**
> A group at Ibn Tofail University in Morocco. Results in Engineering is an open-access Elsevier journal covering all of engineering, not a signal-processing journal. The paper was accepted in about three and a half months.

TR: Derginin kalitesi hakkında yorum yapma; bilmiyorsun. Sadece ne olduğunu söyle.

**"When was it published?"**
> Online in April 2025. Received December 2024.

---

## Slayt 2 — Makale tek cümlede

**"Which 11 modulations?"**
> Eight digital: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, GFSK, CPFSK. Three analog: WBFM, AM-DSB, AM-SSB.

**"Best of 12 — which 12?"**
> Their own selection of published models, from ResNet to a 2024 model called MAMCA. It's the next slide but one.

**"Is it the best model in the field?"**
> Best in their table. They cite newer models in their introduction, IQFormer for example, that don't appear in their comparison. So "best of these twelve" is the accurate claim.

**"What does 'our classifier is built on this design' mean? Is it the same?"**
> Same design, not their code. Our working version is a re-implementation that differs in five small places. I also rebuilt it faithfully from the paper, and that matches their number.

TR: Esmer'in adı slaytta yok. Doğrudan sorulursa "a colleague's re-implementation, which I then adapted" diyebilirsin. Beş fark: ilk konvolüsyon 5'lik, iki pool, tek BN, attention eki yok, tek dense katman.

**"128 samples — is that a lot or a little?"**
> Little: 16 symbols. That's why some pairs, like QAM16 and QAM64, can't be fully separated even at high SNR.

---

## Slayt 3 — Veri

**"Is the frame in the picture from the dataset?"**
> No, it's an illustration: a clean QAM16 frame drawn with the same pulse shaping, 16 symbols at 8 samples each. A real frame also has noise, fading and a frequency offset.

TR: `simple_figs.py`'deki `frame_figure`: gürültüsüz, sentetik QAM16 çerçevesi. Dürüstçe söyle.

**"What are I and Q?"**
> The real and imaginary parts of the complex baseband signal: I is the in-phase component, Q the quadrature. Together they give amplitude and phase.

**"Where does the dataset come from?"**
> RadioML 2016.10a, by O'Shea and West. Generated in GNU Radio and public since 2016; most papers in this area use it.

**"Is it real or simulated?"**
> Simulated. GNU Radio modulators, then a channel model with frequency offset, clock drift, Rician fading, three-path multipath and Gaussian noise. No over-the-air recordings.

TR: `NOISE_2016.md` bölüm 2: 200 kHz, 8 örnek/sembol, 25 kbaud, Rician K=4, 3 yol.

**"Why 220,000?"**
> 11 classes × 20 SNR labels × 1,000 frames each.

**"Why 20 SNR labels?"**
> −20 to +18 dB in steps of 2.

**"What's the sample rate? Symbol rate?"**
> 200 kHz, 8 samples per symbol, so 25 kilobaud. A frame is 0.64 ms.

**"Is the input normalized?"**
> In their code, yes: each of the 128 time positions is scaled by its mean and variance over the whole dataset. That includes the test frames, which is a small leak. Ours scales each frame to unit power.

**"Why only 2 × 128 — no spectrum, no features?"**
> That's the point of the deep-learning approach: raw samples in, and the network learns its own features. The paper uses nothing else.

**"Which are digital and which analog?"**
> Eight digital: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, GFSK and CPFSK. Three analog: WBFM, AM-DSB and AM-SSB, made from audio.

TR: Notundaki "4 analog 7 digital" yanlış. Makale (s. 3, Tablo 1) 8 dijital + 3 analog diyor; slaytta öyle yazdım.

**"Why are GFSK and CPFSK digital?"**
> They shift frequency, but the shifts are driven by bits. Analog here means a continuous audio source.

---

## Slayt 4 — Dört adım

**"What are the 'pieces'?"**
> After the first step, the 128 samples become 64 steps, each summarizing about 6 samples. Those are the pieces the LSTM reads.

**"What does 'short filters slide along the signal' mean?"**
> Convolution: a 3-tap FIR filter applied at every position. There are 64 such filters, then 128, and their taps are learned rather than designed.

**"Why read in both directions?"**
> The whole frame is available at once, so there's no need to be causal. Reading backwards too means every step's output knows what comes before and after it.

**"What is the memory?"**
> A state vector of 128 numbers in each direction, updated at every step. It's the LSTM's equivalent of a recursive filter's state.

**"What are the 11 scores?"**
> One number per modulation. A softmax turns them into probabilities that add up to one, and the largest is the answer.

**"Why 83 percent in step 2?"**
> Each LSTM layer has four gates, each a 128-by-input matrix. Two layers in two directions add up to 659 thousand numbers out of 794 thousand.

TR: Hesap: ilk katman 2 × 4 × (128 × 256 + 128) = 263,168; ikinci 2 × 4 × (128 × 384 + 128) = 394,240; artı BN.

**"If step 2 is the heart, why have the other steps?"**
> Step 1 halves the length, which makes step 2 cheaper. Step 3 turns 64 vectors into one summary. Their own test says they add only a fraction of a point.

**"Why these sizes — 64, 128, 3 taps?"**
> The paper says they tuned some settings by search and some by hand. It doesn't justify the layer sizes.

**"What do you mean by confidence?"**
> The last layer gives eleven numbers that add up to one, one per type. The largest is the answer, and its size is how sure the classifier is. Nobody has checked that 0.9 really means right 90 percent of the time, so it's a score, not a calibrated probability.

---

## Slayt 5 — Eğitim ve test

**"Why three parts? Why not just train and test?"**
> Training stops when the middle part stops improving. If the test frames were used for that decision, the reported number would be chosen on the test set and would be optimistic. So the test frames are only used once, at the end.

**"What is a 'pass'?"**
> One pass, or epoch, goes through all 154,000 training frames once, in batches of 32: about 4,800 updates.

**"How does it stop by itself?"**
> After each pass it measures the error on the 33,000 watch frames. If that error hasn't improved for 6 passes, it stops and goes back to the best version.

**"Is the split random? Balanced?"**
> Random, not balanced per class and SNR. So each class at each SNR gets about 150 test frames, not exactly 150.

**"'Done once' — why does that matter?"**
> One run gives one number. A different random split or starting point would give a slightly different one, and the paper doesn't say how much.

**"How do you know it's not memorizing?"**
> Their training accuracy, 64 percent, and validation accuracy, 63.6, are almost the same. A memorizing model would be much better on its training frames.

**"What is the error it minimizes?"**
> Cross-entropy: minus the log of the probability the model gives to the right answer, averaged over frames.

**"How long does training take?"**
> They don't say. Their code on a Colab GPU is taking about two hours per run for us.

TR: Bu sabahki gözlem: ilk koşu 2 saati geçti, hâlâ ilerliyordu.

**"70 percent of what?"**
> Of the whole dataset: 154,000 of the 220,000 frames. The split is random, so each type at each SNR gets roughly 700 of its 1,000 frames for training.

---

## Slayt 6 — Tablo 3

**"Where do the other numbers come from?"**
> The paper doesn't say whether they retrained them or copied them. Six have confusion matrices in the paper, so they ran those. The other five appear only here and are probably from other papers.

TR: Kendileri koşmuş görünen altı model: CNN-BiLSTM-DNN, MCLDNN, MCNet, CGDNet, ResNet, IC-AMCNet. Muhtemelen başka makaleden gelen beş: MAMCA, PET-CGDNN, DAE, Transformer, CVCNN.

**"Is the order reliable?"**
> The top of it isn't. Half a point from one run is within the noise. The bigger gaps, three to seven points to the CNN-only models, are real.

**"Why is the Transformer last?"**
> They suggest transformers struggle with local time features, but they don't test it. It's a guess.

**"What is CNN-BiLSTM-DNN?"**
> A 2024 model with the same ingredients, convolution plus bidirectional LSTM plus attention. It's the closest relative, and it's half a point behind.

**"What is MCLDNN?"**
> A 2020 model that processes I, Q and I+Q in separate convolution branches, then uses an LSTM. We trained it ourselves too: 61.4 percent.

**"Do published numbers for the same model agree?"**
> Not exactly. The PET-CGDNN paper reports MCLDNN at 62.08; this paper says 61.42. Same model and same data, different papers: about half a point apart.

**"What is the 10b column?"**
> A larger version of the dataset: 10 classes, without AM-SSB, and 1.2 million frames. ICRNNA gets 65.39 there.

---

## Slayt 7 — Doğruluk ve SNR (Şekil 8)

**"How many decisions behind each point?"**
> About 1,650: the test set split over 20 levels, roughly 150 per class at each level.

TR: 33,000 / 20 = 1,650. %9 için ±1.4 puanlık bant.

**"Why does it jump so fast between −10 and 0 dB?"**
> That's where the signal starts to stand out from the noise. And the label underestimates the SNR: at label 0 the PSK signals are actually around 5 dB.

**"Why does it stop at 93, not 100?"**
> Two errors that more SNR doesn't fix. WBFM is read as AM-DSB about half the time, and QAM16 and QAM64 are partly mixed. Both are limits of the data: silence in the audio, and only 16 symbols.

**"What are the other two curves?"**
> F1 score and precision, averaged over the classes. They follow the accuracy.

**"What is precision? What is recall?"**
> Recall: of the frames that really are X, how many did it call X. Precision: of the frames it called X, how many really were X. The diagonal of a confusion matrix, row by row, gives the recall.

**"Why is precision higher than accuracy overall?"**
> At low SNR the model dumps most frames into one class. The errors pile up in that one column, so only that class's precision suffers, while every class's recall suffers. It doesn't mean the model is confident.

TR: Bizim 11 sınıflı koşuda bu "çöp" sınıf AM-SSB: −20 dB'de kararların %87'si. Makaledeki "precision yüksek, model emin" yorumu yanlış.

**"Is −20 dB realistic?"**
> As a label, it means the signal is buried. Nobody classifies there; it's in the dataset to show where the methods break.

---

## Slayt 8 — 0 dB'de karışıklık (Şekil 11)

**"How do I read this matrix?"**
> Rows are the true type, columns what the classifier said. The diagonal is correct; each row adds up to 100 percent.

**"How many signals went in?"**
> The test frames labelled 0 dB: about 150 of each type, roughly 1,650 in all. The split is random, so rows are not exactly 150. For 8PSK it was 161.

TR: Makale sayı basmıyor, yüzde basıyor. 161, 8PSK satırındaki beş yüzdenin hepsini veren tek satır büyüklüğü. WBFM satırı 140, 145 ya da 152 olabilir; o yüzden slaytta "about".

**"Explain the 8PSK errors. What went in, what came out?"**
> 161 8PSK frames went in; 148 came out as 8PSK. Seven were called QPSK, three QAM16, two AM-SSB and one QAM64.
>
> QPSK is its nearest neighbour: same constant amplitude, and QPSK's four phases are every other one of 8PSK's eight. At this noise level the 45-degree steps blur and a frame can look like it has only the 90-degree ones.
>
> The QAM calls come from noise and fading making the amplitude vary, so one ring looks like several. AM-SSB is the type the classifier falls back on when it can't see a signal, probably frames caught in a fading dip.

**"Why is 8PSK the weakest digital type here?"**
> Its points are closest together. At equal power the distance between neighbouring points is 0.77 for 8PSK and 1.41 for QPSK, so 8PSK needs about 5 dB more SNR to be separated as well. At this label we measured about 4.7 dB for 8PSK.

TR: 2·sin(22.5°) = 0.765, 2·sin(45°) = 1.414; oran 1.85 → 5.3 dB. 4.7 dB: `rml2016_measured_snr.json`, 8PSK, etiket 0.

**"Why isn't QPSK called 8PSK just as often?"**
> Only 0.6 percent are. Sixteen QPSK symbols all on four of eight possible phases is strong evidence for QPSK; an 8PSK frame almost never does that by chance. The confusion goes mostly one way.

**"Why is WBFM called AM-DSB and not the other way round?"**
> AM-DSB is right 98 percent of the time. When a WBFM frame falls in a silence of the audio, it looks like a bare carrier, and the classifier has learned to call that AM-DSB. So the confusion goes one way.

TR: Bilinen açıklama; kendimiz ölçmedik.

**"What about QAM16 and QAM64?"**
> About 6 percent each way, nine or ten frames per row. Sixteen symbols per frame are too few to tell a 16-point grid from a 64-point one reliably.

**"Is 0 dB really 0 dB here?"**
> The label is 0. The measured SNR is about 5 dB for the PSK types and higher for QAM. So "signal and noise equal" isn't what this matrix shows.

**"What does '96 to 100 percent' cover?"**
> The other seven types at this level: AM-DSB 98.0, AM-SSB 96.1, BPSK 99.3, CPFSK 100, GFSK 98.6, PAM4 99.2, QPSK 99.4.

**"Do other models do better on WBFM?"**
> In their figure, yes: MCLDNN gets 63 percent and ResNet 58, against ICRNNA's 41. No model solves it.

---

## Slayt 9 — Gerçekte ne kadar iyi?

**"Why is ICRNNA 62.2 here and 63.2 on the slide before?"**
> This row is our working version, trained with one recipe shared by all five models: our split, larger batches. It's fair among these five, but it's not the reproduction; the previous slide is.

**"Same data, same split — what exactly?"**
> All 220,000 frames. A three-way split balanced per class and SNR. The same three random seeds for every model, and each trained until it stopped improving.

**"Why these five models?"**
> Published code exists for them in one benchmark collection, so we could train them exactly. They span 2016 to 2025.

TR: AMR-Benchmark (Zhang ve ark., DSP 2022) Keras kodundan port edildi (`literature_2016.md`).

**"Do your numbers match their papers?"**
> Within one point, in both directions. The parameter counts match to the digit.

**"Why is VT-CNN2 so much worse?"**
> It's the oldest, from 2016: convolution only, no recurrent part. That's the six-point gap the models with a recurrent layer closed.

**"Why are all four about 91 percent above 10 dB?"**
> Because that's the ceiling of the data: WBFM against AM-DSB, and QAM16 against QAM64. A better model can't fix either.

**"Is one point significant?"**
> ICRNNA was first in each of the three seeds, so the order is probably real. But in one seed LSTM2 was only a tenth of a point behind, so the gain is small.

TR: Tohum tohum genel doğruluk (aşağıdaki not): ICRNNA her tohumda birinci; 3. tohumda LSTM2 62.09'a karşı 62.22.

---

## Slayt 10 — Özet

**"What's next?"**
> The spread from their code, then back to the main question of my work: how much accuracy is lost when the signals come from a different source than the training data. That's something this paper never tests.

**"Would it work on real signals?"**
> Unknown. The paper never tests outside this simulated dataset, and that's exactly my research question.

**"What would you change in the model?"**
> Nothing before measuring. If anything, I'd test whether a plain bidirectional LSTM does as well, since their ablation suggests it might.

**"What's the weakest part of the paper?"**
> One run with no spread, and some claims that are stronger than their own tables: low SNR, precision and the parameter counts.

---

## Slayt 12 — Model bir bakışta

**"What's 2 × 128 → 64 × 128?"**
> 2 channels by 128 samples go in. After the convolutions it's 128 numbers at each of 128 time steps, then pooling halves the time to 64.

**"Why does the size grow from 2 to 128 numbers per step?"**
> Each filter gives one number per step. 128 filters give 128 numbers: a richer description of each short piece.

---

## Slayt 13 — Blok 1, CNN

**"What does 'larger of each pair' do?"**
> Max pooling: it keeps the larger of two neighbours, so the length halves. It keeps the strongest response and drops the exact position.

**"Why 'less than one symbol'?"**
> Each of the 64 steps sees about 6 samples, and a symbol is 8. So the CNN alone can't see a pattern across symbols; that's left to the LSTM.

**"Is there a nonlinearity?"**
> Yes, after each filter: negative outputs are set to zero (ReLU). Without it, the stacked layers would collapse into one linear filter.

**"What about the edges of the frame?"**
> The frame is zero-padded so the output has the same length.

---

## Slayt 14 — Blok 2, BiLSTM

**"What are the gates?"**
> Three learned switches between 0 and 1. Forget: how much old state to keep. Input: how much of the new step to add. Output: how much of the state to pass on.

**"Why two layers?"**
> The second reads the first one's output, so it can combine longer patterns. The paper doesn't justify two rather than one or three.

**"'Every step knows the whole frame' — really?"**
> In principle, yes: the forward state has seen everything before, the backward state everything after. How much it actually retains, we can't see directly.

**"Is there normalization or dropout here?"**
> After each LSTM layer: batch normalization, then 30 percent dropout during training.

---

## Slayt 15 — Blok 3, attention

**"What are q and k in the formula?"**
> In their code both come from the same LSTM output: q is the normalized and dropped-out version, k is the raw one. Each step is scored on its own content; steps aren't compared with each other.

**"What does the learned scale do?"**
> It multiplies all the scores, so it sets how sharp the weights are: a large scale focuses on a few steps, a small one averages evenly.

**"Where does it look? Which steps get weight?"**
> The paper never shows it, and I haven't plotted it yet. It's a good check to do.

**"Why is it called additive?"**
> Because the scores come from adding two transformed vectors inside a tanh, as opposed to a dot product. It's the 2014 Bahdanau form.

**"Is this the attention of ChatGPT?"**
> No. That's multi-head self-attention, where every step is compared with every other. This is a single learned weighting.

---

## Slayt 16 — Blok 4, karar

**"What's the pie?"**
> Where the 794 thousand numbers are: 83 percent LSTM, 8 attention, 5 dense layers, 3 convolution.

**"What's the difference between 794,188 and the 0.79 million in the paper?"**
> None: 0.79 million rounded. 1,792 of them are running averages kept by the normalization layers, so 792,396 are learned by training.

**"Are the output probabilities calibrated?"**
> Not checked, in the paper or by us. We only use the largest one.

---

## Slayt 17 — Nasıl eğitiliyor

**"Why cut the learning rate?"**
> When progress stalls, smaller steps let the weights settle into a better point. They cut it by five after 5 flat epochs.

**"What is Adam?"**
> A version of gradient descent that adapts the step size for each number separately. It's the standard choice.

**"4,813 steps per epoch?"**
> 154,000 frames in batches of 32.

**"Why batch 32?"**
> Their choice, tuned by hand. Smaller batches mean more, noisier updates per pass.

**"Is there anything in the code that isn't in the paper?"**
> Gradient clipping, which limits the size of each update; the exact normalization; and the stopping rule. None of that is in the text.

---

## Slayt 18 — Kodları ne söylüyor

**"How do you know 58 wasn't chosen?"**
> The code trains up to 100 epochs with early stopping, and there is no 58 anywhere in it. The paper's own text says the validation loss flattened after epoch 40, which fits a stop at 58.

**"Can we reproduce their exact run?"**
> Not exactly: the code's data split depends on Python's internal ordering, so it changes from one run to the next. We fixed that so our three runs share one split.

---

## Slayt 19 — "58 epoch" ne demek

**"If accuracy is still rising, why would the loss be flat?"**
> The loss also punishes confident wrong answers. The model can get more frames right while becoming over-confident on the ones it gets wrong, so accuracy rises while the loss stays flat.

**"So did they stop too early?"**
> By their rule, no: it watches the loss. By accuracy, maybe a little. Our rebuild gained 1.5 points by continuing, but that was with our recipe.

---

## Slayt 20 — Yeniden kurduk; kodları doğruluyor

**"What did you miss, and does it matter?"**
> The exact attention form, the input normalization and gradient clipping. They're small: trained to the end, our build lands at 63.21, against their 63.24.

**"How did you guess right without the code?"**
> Their Figure 2 lists the layers and sizes. The ambiguous points, like one pooling step or normalization after each LSTM, I read from the figure, and the code later confirmed them.

---

## Slayt 21 — Tablolarının desteklemediği üç iddia

**"Show me the low-SNR claim."**
> Their Table 5: at −10 dB, CNN-BiLSTM-DNN has 27.7 against 25.3. At −8, three models are higher. At −6, three are higher. ICRNNA leads from −4 to +2 dB, by three to four points.

**"How do you know MCLDNN has 406,199 parameters?"**
> We built it from its published code and counted. Every source gives the same number, to the digit. Their table writes 406 as if it were millions.

**"Does this make the paper wrong?"**
> Not the model or the headline number, which we reproduce. It means the text oversells its own tables.

**"Any other errors?"**
> Small ones: two different overall accuracies, 63.39 and 63.24; a "−5 dB" level that doesn't exist; GFSK written as BFSK. Nothing that changes the model.

---

## Slayt 22 — Makalenin söylemedikleri

**"How did you measure +9 dB?"**
> In the dataset file itself. The digital signals occupy only part of the band, so the spectrum outside it is pure noise. That gives the noise level, and the rest is signal. We used 1,000 frames per class and label, and the method was checked to within half a dB on frames of known SNR.

**"Why would QAM16 have more signal than QPSK at the same label?"**
> The generator didn't scale the constellations to equal power. A QAM grid on integer points has about 10 dB more average power than QPSK, and 16 dB for QAM64. We measured 9.5 and 15.2.

**"So is the comparison across classes unfair?"**
> Across classes at one label, yes: they don't share an SNR. Between models it's fair, since every model sees the same frames.

**"How much do your seeds vary?"**
> Overall accuracy moves by up to half a point for ICRNNA and just over a point for LSTM2, between three seeds of the same model.

---

## [↺] Çıkarılan slaytların konuları (sorulursa)

Bu üç slayt 13 Ekim'deki geri bildirimle sunumdan çıkarıldı. Moshe yine de sorarsa cevaplar burada.

**— Hangi adım önemli (makalenin Tablo 6'sı, ablation)**

**"How did they build 'step 2 only'?"**
> The paper doesn't describe the variants exactly, and they're not in their code. I can only read the numbers.

**"Is 0.4 points from attention meaningful?"**
> Not from one run each. It's within what we see between two seeds of the same model.

**"Why is 'step 2 only' better than 'all but step 3'?"**
> 63.1 against 62.8: adding the CNN without the attention lost a third of a point. That's probably noise too, which is the point: differences this size can't be read from single runs.

**"Then why use the complex model?"**
> Fair question. On their own numbers, a bidirectional LSTM alone gets almost all of it. The CNN mainly makes it cheaper by halving the sequence.

**"They say attention helps most at low SNR — is that shown?"**
> No. They only give one overall number per variant, not per SNR.

**— Onların kodu ve bizim koşularımız**

**"Why didn't they release the code with the paper?"**
> I don't know. The paper said "available on request". The code went up on GitHub in August 2026, without trained weights.

**"How do you know it's really theirs?"**
> It's on a GitHub account named after the first author. The README and the citation file name all five authors and call it the official code, and it's archived on Zenodo. The model it builds has exactly the layers of the paper's figure.

**"Why run it three times?"**
> To see how much the number moves between runs. The paper gives one number, and that's the one thing it can't tell us.

**"What did their code give?"**
> Three runs on one fixed split: 62.71, 62.45 and 62.52 percent. That's 62.56 on average, plus or minus 0.11, against their published 63.24.

TR: Onların kodu, hiç değiştirmeden.
- 1. koşu: 141.6 dk, epoch 61'de durdu, en iyi 55.
- 2. koşu: 149.9 dk, epoch 64'te durdu, en iyi 58.
- 3. koşu: 133.0 dk, epoch 57'de durdu, en iyi 51.

**"So their number doesn't reproduce?"**
> All three runs land below it: about two-thirds of a point, and within a quarter of a point of each other. Their 63.24 is a single run on a different random split, and the test set alone moves a run by about 0.3 points. So either their split was a slightly easier draw or their run a slightly lucky one; the model is the same. Training behaved like theirs: our runs stopped by themselves at epochs 57, 61 and 64, around their 58, with validation accuracy 63.4 to 63.6, like their 63.59.

**"Why would it be lower?"**
> A different random split and a different random start. Their code draws a new split every time it's run, so we can't use their exact one. We changed nothing in the model or the training.

**"And at 0 dB?"**
> 90.5 percent at label 0 over our three runs, against their 92.2. About 5,000 decisions in all. Their 92.2 is one run of about 1,650 decisions.

**"How long did the runs take?"**
> All three are done: about two and a half hours each on Colab.

**"Why no weights?"**
> The scripts save them locally; they just weren't uploaded. So anyone has to retrain.

**— Sayı gerçek mi? (makaleden yeniden kurduğumuz model)**

**"How did you rebuild it?"**
> From the paper's text and Figure 2 alone, in PyTorch, before their code existed. Same layers and sizes; the parameter count came out within 0.1 percent of theirs.

TR: 794,827'ye karşı 794,188. Fark attention'ın kurulma biçiminden.

**"How many runs gave 63.21?"**
> One, trained until it stopped improving; it peaked at epoch 107. At a fixed 58 epochs, three runs gave 61.75, plus or minus 0.07.

TR: Dürüst ol: 63.21 tek koşu.

**"Why did yours need 107 passes and theirs 58?"**
> Different stopping rules and small differences in the recipe. Theirs stops on the validation loss; ours watched validation accuracy, which was still rising. Their code also clips gradients and normalizes the input differently.

**"So is 63.24 reproducible?"**
> From the paper alone, yes: 63.21. Their own code's runs will add the spread.

---

## Genel sorular

**"Can you explain it without machine-learning words?"**
> A filter bank cuts the signal into short pieces, a recursive filter with memory reads them in order, a weighted average summarizes the frame, and a final layer picks one of eleven answers. All the filter taps are fitted to examples instead of designed.

**"How is this different from a classical classifier?"**
> Classical methods compute chosen statistics, like higher-order cumulants, and feed them to a classifier. Here the network learns its own statistics from raw samples.

**"Could you tell why it decided something?"**
> Not directly. The confusion matrices show what it gets wrong. The attention weights could show where it looks, but that hasn't been plotted, by them or by me.

**"How long to classify one frame?"**
> They report 0.23 seconds without saying for what: one frame, a batch or the test set. It isn't measured on any device.

**"Why do all papers use this dataset?"**
> It's public, labelled, and has been the standard benchmark since 2016. The weakness is that it's simulated.

**"What's the connection to your domain-gap work?"**
> This is the classifier. My question is what happens when the test signals come from a different generator than the training ones. This paper only tests on held-out frames of the same dataset.

**"What's the difference between 10a and 10b?"**
> 10b has ten classes, no AM-SSB, and six times as many frames per class. Everyone scores higher on 10b.

**"What is overfitting?"**
> When a model learns the training frames themselves rather than the pattern, so it does much better on them than on new frames. Here training and validation accuracy are within one point, so there's none to speak of.

**"What is dropout? Batch normalization?"**
> Dropout: during training, a random 30 percent of the numbers are switched off, so the network can't rely on any one of them. Batch normalization rescales each feature to a standard range, which keeps training stable.

**"What's a softmax?"**
> It exponentiates a list of scores and divides by the sum, so they become positive and add up to one.

**"I don't understand — what do you mean by X?"**
> Stop and define X in one sentence, with one number from the slide. Don't add a second idea.

TR: Moshe "bunu anlamıyorum" derse, ikinci bir kavram ekleme; tek cümle tanım ve slayttan bir sayı.

---

## Kontrol notu (söylemeden önce)

- **Slayt 9, tohum tohum genel doğruluk** (sonuç dosyalarından):

| tohum | ICRNNA | LSTM2 | MCLDNN | PET-CGDNN |
|---|---|---|---|---|
| 0 | 62.48 | 61.28 | 61.70 | 60.87 |
| 1 | 62.00 | 60.95 | 61.34 | 61.02 |
| 2 | 62.22 | 62.09 | 61.19 | 61.05 |

ICRNNA her tohumda birinci. Ama LSTM2'nin en iyi koşusu (62.09) ICRNNA'nın en kötüsünü (62.00) geçiyor; fark küçük.

- **Onların kodu:** Üç koşu da bitti; "Çıkarılan slaytların konuları" kartındaki sayılar güncel (62.56 ± 0.11).

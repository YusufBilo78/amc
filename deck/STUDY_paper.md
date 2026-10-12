# Makale sunumu çalışma metni — `moshe_paper_icrnna.pptx` (16 slayt)

Makale: El-Haryqy, Kharbouche, Ouamna, Madini, Zouine. *Improved automatic modulation recognition using deep learning with additive attention.* Results in Engineering 26 (2025) 104783. Ibn Tofail Üniversitesi, Kenitra, Fas. Aralık 2024'te gönderilmiş, Nisan 2025'te yayımlanmış. Açık erişim (CC BY-NC 4.0).

Her slayt için üç bölüm var:

- **Slaytta ne var, ne demek:** Slayttaki her sayının Türkçe açıklaması ve makalede nerede geçtiği (tablo, şekil ya da bölüm numarasıyla).
- **Okuma metni (EN):** Sunumda söyleyeceğin İngilizce metin; slayt notlarındakiyle aynı. Hepsi yaklaşık 8 dakika.
- **Olası sorular:** Moshe'nin sorabileceği sorular ve kısa cevapları.

Sunum iki parça:
- **Slayt 2–9 (makale ne diyor):** Makaleyi olduğu gibi anlatıyorsun.
- **Slayt 10–16 (bizim kontrolümüz):** Makaleden yeniden kurduğumuz modelle ve makalenin kendi tablolarıyla neyin tuttuğunu, neyin tutmadığını gösteriyorsun.

Makaleden alınan her sayı `deck/extract_paper.py` içinde tablo numarasıyla yazılı. Bizim sayılarımız sonuç dosyalarından okunuyor (`icrnna_faithful_results.json`, `icrnna_faithful_e150_results.json`, `rml2016_measured_snr.json`, `deck/deck2016_data.json`).

---

## Ezber kartı — en önemli sayılar

| sayı | ne | nereden |
|---|---|---|
| 63.24% | makalenin RML2016.10a doğruluğu (65.39% 10b'de) | Tablo 3, özet |
| 0.79 M / 48.42 MFLOPs | makalenin verdiği parametre ve işlem sayısı | Tablo 7 |
| 58 | makalenin 10a için epoch sayısı | Tablo 2 |
| 62.73% | ikinci sıradaki model, CNN-BiLSTM-DNN | Tablo 3 |
| 59.9 / 63.1 / 62.8 / 63.2 | sadece CNN / sadece BiLSTM / attention'sız / tam model | Tablo 6 |
| 57.9% | 0 dB'de WBFM'in AM-DSB denme oranı | Şekil 11 |
| 794,827 | makaleden kurduğumuz modelin parametre sayısı | bizim koşu |
| 61.75% ± 0.07 | bizim kurduğumuz model, 58 epoch, 3 koşu | `icrnna_faithful_results.json` |
| 63.21% | aynı model, en iyi epoch 107 (sınır 150) | `icrnna_faithful_e150_results.json` |
| 406,199 | MCLDNN'in gerçek parametre sayısı; makale "406 M" yazıyor | `LITERATURE_CHECK.md` |
| +9 dB | etiket 0 dB'de QAM16'nın QPSK'ya göre fazla sinyali (14.0 vs 5.1 dB) | `rml2016_measured_snr.json` |
| 61.42 / 61.41 | MCLDNN: makalenin Tablo 3'ü / bizim ölçümümüz | Tablo 3, `deck2016_data.json` |

---

## Sözlük — bu makalede yeni geçen terimler

Temel terimler (recall, precision, epoch, BiLSTM, attention...) için `STUDY_simple.html`'deki sözlüğe bak. Burada sadece bu makalede yeni çıkanlar var.

- **Ablation study** — "Remove one part of the model at a time and see how much accuracy drops. It tells you which part matters." *(Parça parça çıkarma deneyi; Tablo 6.)*
- **F1 score** — "The harmonic mean of precision and recall, averaged over classes." *(Precision ile recall'un dengeli ortalaması.)*
- **FLOPs / MFLOPs** — "Floating-point operations needed for one forward pass; MFLOPs means millions." *(Hesap yükü. Makale neye göre saydığını yazmıyor.)*
- **Bahdanau / additive attention** — "The original attention from machine translation, 2014: a small network scores each time step, a softmax turns scores into weights." *(Bizim modeldeki attention'ın kökeni.)*
- **Softmax** — "Turns a list of numbers into positive weights that add up to one." *(Ağırlıklara çevirme.)*
- **Layer normalization** — "Rescales a vector so its values have zero mean and unit spread." *(Makalenin "improved" dediği eklerden biri.)*
- **L2 regularization / weight decay** — "A penalty that keeps the weights small, to reduce overfitting." *(Tablo 2: 1e-4.)*
- **ReduceLROnPlateau** — "Cut the learning rate when the validation loss stops improving." *(Makalede: 5 epoch iyileşme yoksa ×0.2.)*
- **RML2016.10b** — "A larger sibling of 10a: 10 classes (no AM-SSB), 1.2 million frames." *(Tablo 1. Biz kullanmıyoruz.)*
- **Converged** — "Training stopped improving by itself before the epoch limit." *(Bizim kural: en iyi epoch + sabır ≤ sınır.)*
- **Baseline** — "A model you compare against." *(Tablo 3'teki diğer 11 model.)*

---

## Slayt 1 — Başlık

**Slaytta ne var, ne demek**

- **"PAPER REVIEW · ICRNNA":** Sunumun konusu; makalenin modeline verdiği isim.
- **Makale başlığı ve yazarlar:** Tam künye. Results in Engineering, Elsevier'in açık erişim dergisi.
- **Turuncu dört kutu:** Modelin dört parçası: CNN → BiLSTM → Attention → Dense. Slayt 3'te açıyorsun.

**Okuma metni (EN)**

> Today I'll walk through one paper: the one our detector comes from. ICRNNA, by El-Haryqy and colleagues, in Results in Engineering, 2025.
>
> First, what the paper says. Then, what we found when we rebuilt it.

**Olası sorular**

- *"Why this paper?"* — Our detector is a version of this model. Esmer re-implemented it, and our code comes from that version. So I wanted to know exactly what the paper claims and whether it holds.
- *"Is it a well-known venue?"* — Results in Engineering is a broad, open-access Elsevier engineering journal rather than a signal-processing one. Either way, we checked the numbers ourselves.

---

## Slayt 2 — Makale ne iddia ediyor?

**Slaytta ne var, ne demek**

- **63.24%:** RML2016.10a'da, 11 sınıfın tamamında, bütün SNR'lar birlikte doğruluk. Tablo 3 ve özet. RML2016.10b'de 65.39%.
- **0.79 M:** Parametre sayısı (Tablo 7).
- **48.42 MFLOPs:** Bir ileri geçişteki işlem sayısı (Tablo 7). Neye göre sayıldığını (bir çerçeve mi, bir batch mi) yazmıyorlar.
- **"Especially good at low SNR":** Özette ve Bölüm 4.2'de tekrar tekrar geçen iddia. Slayt 13'te kendi Tablo 5'leriyle karşılaştırıyorsun.
- Özette bir de çıkarım süresi var (0.2313 s, Tablo 7). Neyin süresi olduğu belirtilmemiş, o yüzden slayta koymadım.

**Okuma metni (EN)**

> The paper makes three claims.
>
> One: the best accuracy on RML2016.10a, 63.24 percent.
>
> Two: it is small and fast, with 0.79 million parameters and 48 million operations.
>
> Three: it is especially good at low SNR.

**Olası sorular**

- *"Best compared with what?"* — With eleven other models in their Table 3, from ResNet to a Transformer. Slide 6.
- *"Is 63% good?"* — On 2016.10a, yes: it is an average over all 20 SNR labels, including the very noisy ones where every model is at chance. Above 0 dB all good models are around 90%.

---

## Slayt 3 — Model: dört parça

**Slaytta ne var, ne demek**

Bu şema makalenin Şekil 2'sinden yeniden çizildi.

- **Input:** I ve Q, 2 × 128. Önce normalize edilip karıştırılıyor (Şekil 1). Normalizasyonun türü yazılmamış.
- **CNN (Part A):** İki 1-D konvolüsyon katmanı.
  - 64 ve 128 filtre, kernel 3, ReLU, batch normalization.
  - Sonra bir max-pool (2) ve dropout 0.3.
- **BiLSTM (Part B):** İki çift yönlü LSTM katmanı, her yönde 128 birim. Her katmandan sonra batch normalization ve dropout 0.3.
- **Attention (Part C):** "Improved additive attention". Slayt 4'te anlatılıyor.
- **Dense (Part D):** İki tam bağlı katman.
  - 128 ve 64 birim, ReLU, batch normalization, dropout 0.3.
  - Sonra softmax ile 11 sınıf.

**Esmer'in versiyonuyla 5 fark** (`DETECTOR.md`; her satırda önce makale, sonra Esmer'in versiyonu):

| | makale | Esmer'in versiyonu |
|---|---|---|
| ilk konvolüsyonun kernel'i | 3 | 5 |
| max-pool sayısı | bir | iki |
| BatchNorm | her LSTM katmanından sonra | yığının sonunda tek |
| attention'da dropout ve LayerNorm | var | yok |
| dense katmanlar | iki (128, 64), dropout 0.3 | tek (128), dropout 0.5 |

**Okuma metni (EN)**

> The model has four parts.
>
> A CNN, two convolution layers, looks at short pieces of the signal. Two bidirectional LSTM layers read those pieces in order, forwards and backwards. An attention layer weighs the pieces. And two dense layers make the decision among the eleven classes.

**Olası sorular**

- *"Is this new?"*
  - CNN + LSTM hybrids exist since at least 2018 (MCLDNN is one).
  - CNN + BiLSTM + attention also exists: their runner-up, CNN-BiLSTM-DNN (2024), uses attention too.
  - The new part is the specific combination and the attention extras (slide 4).
- *"How many pooling layers?"* — The paper is ambiguous. The text says "max pooling" once and Fig. 2 shows one pooling box. I read it as one, so the LSTM sees 64 time steps. With that reading the parameter count matches theirs.

---

## Slayt 4 — Attention tek formülde

**Slaytta ne var, ne demek**

- **Formüller** (makalenin Denklem 4–6):
  - **score = v · tanh(W₁q + W₂kₜ):** Her zaman adımı t için küçük bir ağ bir puan veriyor. q (query) ve k (key) ikisi de LSTM çıktısından geliyor.
  - **α = softmax(score):** Puanlar toplamı 1 olan ağırlıklara dönüşüyor.
  - **c = Σ αₜ vₜ:** Çıktı, zaman adımlarının ağırlıklı ortalaması (context vector).
- **"Improved" eklemeler** (Bölüm 3 ve 3.3):
  - puanların ölçeklenmesi (scaling),
  - dropout 0.1,
  - layer normalization (context vector'e).
- **Mesaj:** Büyük dil modellerindeki çok katmanlı attention değil. Tek katman, bir kez uygulanan öğrenilmiş bir ağırlıklı ortalama.

**Okuma metni (EN)**

> Attention is one formula. Each time step gets a score. A softmax turns the scores into weights that add up to one. The output is the weighted sum.
>
> What the paper calls improved is three additions: scaling the scores, dropout, and layer normalization.

**Olası sorular**

- *"Where does the dropout go?"* — The paper says the attention "includes dropout with a rate of 0.1" but not on what. I applied it to the attention weights, which is the usual choice.
- *"What is the scaling factor?"* — Not given. The text says "if scaling is enabled", so it is optional. Our rebuild leaves it out and still matches their parameter count and accuracy.
- *"Does attention help?"* — By their own ablation, less than half a point (slide 9).

---

## Slayt 5 — Nasıl eğitmişler?

**Slaytta ne var, ne demek**

- **DATA:**
  - RML2016.10a: 11 tip, 220,000 çerçeve, çerçeve başına 128 örnek (Tablo 1, Bölüm 2.2).
  - Bölme 70/15/15 (Bölüm 4.1). Bizimkiyle aynı.
- **RECIPE (Tablo 2, 10a sütunu):**
  - batch 32,
  - öğrenme oranı 0.001,
  - doğrulama kaybı 5 epoch iyileşmezse ×0.2 (en düşük 1e-6),
  - dropout 0.3, L2 1e-4.
- **EPOCHS: 58.** Bu sayı slayt 11'de önemli.
- **Belirtilmemiş olanlar:**
  - Optimizer'ın adı (Adam olduğunu varsaydık).
  - Normalizasyonun türü.
  - Bölmenin sınıf ve SNR'a göre dengeli (stratified) olup olmadığı.
  - Kaç kez koşulduğu.
- **10b için farklı ayarlar:** Öğrenme oranı her epoch ×0.95, batch 256, 40 epoch.
- **Donanım:** Google Colab GPU.

**Okuma metni (EN)**

> How they trained it. RML2016.10a: eleven types, 220,000 frames of 128 samples. Split 70, 15, 15.
>
> Batch 32, learning rate 0.001, cut by five when the validation loss stalls, and 58 epochs. Please remember that number, 58.
>
> They also test the larger 2016.10b; I stay with 10a, because that is the one we use.

**Olası sorular**

- *"How did they choose the hyperparameters?"* — Grid search over learning rate, dropout and L2, then manual tuning of batch size and epochs, chosen on validation (Section 4.1).
- *"Which optimizer?"* — The paper doesn't say. We used Adam, the usual default.

---

## Slayt 6 — Ana tablo: yarım puan farkla birinci

**Slaytta ne var, ne demek**

Grafik, makalenin Tablo 3'ünün RML2016.10a sütunu:

| model | doğruluk |
|---|---|
| ICRNNA | 63.24 |
| CNN-BiLSTM-DNN | 62.73 |
| MCLDNN | 61.42 |
| MAMCA | 60.79 |
| PET-CGDNN | 60.58 |
| DAE | 60.22 |
| ResNet | 57.10 |
| CVCNN | 56.37 |
| IC-AMCNet | 56.30 |
| CGDNet | 56.21 |
| MCNet | 56.20 |
| Transformer | 55.34 |

- **Mesaj:** Birinciyle ikinci arasında 0.51 puan var. MCLDNN 1.8, PET-CGDNN 2.7 puan geride.
- **Grafik 0'dan başlıyor:** Farkların gerçek büyüklüğü böyle görünüyor. Çubuklar neredeyse aynı boyda.
- **Diğer modellerin sayıları:** Makale bunları yeniden mi eğitti, kendi makalelerinden mi aldı, yazmıyor. Slayt 14.

**Okuma metni (EN)**

> This is their main table, on 2016.10a. ICRNNA is first, at 63.24. The runner-up, CNN-BiLSTM-DNN, is half a point behind. The well-known models, MCLDNN and PET-CGDNN, are two to three points behind.

**Olası sorular**

- *"Why is the Transformer last?"* — The paper suggests Transformers struggle with local temporal features. That is their interpretation; it is not tested in the paper.
- *"Is half a point significant?"* — We can't tell from the paper: it reports one run per model. Slide 14.

---

## Slayt 7 — SNR'a göre doğruluk

**Slaytta ne var, ne demek**

- **Grafik:** Makalenin Şekil 8'i. Doğruluk, F1 ve precision, SNR'a göre.
- **9%:** −20 dB'de doğruluk 9.0% (Bölüm 4.2.2). 11 sınıfta şans 1/11 = 9.1%; yani tahmin seviyesi.
- **~93%:** 0 dB'den itibaren. Makale: "5–15 dB'de 93–94%".
- **Ara değerler (Bölüm 4.2.2):**
  - −10 dB'de doğruluk 25.4%.
  - −5 dB'de 57.9% (grafikte −6 dB noktası).
  - 0 dB'de üç metrik de 90%'ın üstünde.
- **Not:** Burada "SNR" veri setinin etiketi. Gerçek ölçülen SNR değil (slayt 14, `NOISE_2016.md`).

**Okuma metni (EN)**

> Accuracy against SNR. Below minus 10 dB it is close to 9 percent, which is chance with eleven classes. It climbs steeply between minus 10 and 0 dB. From 0 dB up it stays around 93 percent.

**Olası sorular**

- *"Why does precision stay above accuracy at low SNR?"* — Precision is averaged over the classes. At low SNR the model names some classes rarely, and those few calls can be right more often, which lifts the average. We have not seen their code, so this is the likely reading, not a check. It is related to the sink we saw in the four-class box.
- *"Why does it stop at 93%, not 100%?"* — Mostly WBFM and the QAM pair. Next slide.

---

## Slayt 8 — 0 dB'de hâlâ nerede hata yapıyor?

**Slaytta ne var, ne demek**

- **Matris:** Makalenin Şekil 11'indeki ICRNNA paneli, 0 dB, yüzde olarak. 0 dB'de genel doğruluk 92.21%.
- **57.9%:** Gerçek WBFM çerçevelerinin 57.9%'u AM-DSB denmiş; WBFM'e doğru denen sadece 41.4%.
  - Bilinen sebep: veri setindeki WBFM'lerin bir kısmı sessiz sesten üretilmiş, sessiz bir WBFM sadece bir taşıyıcı ve AM-DSB'ye benziyor.
  - Bunu biz ölçmedik, literatürde bilinen açıklama olarak söyle.
- **~6%:** QAM16'nın 5.7%'si QAM64, QAM64'ün 6.1%'i QAM16 denmiş.
  - Bizim 4 sınıflı bulgumuzla aynı: 16 sembol QAM64'ün 64 noktasını gösteremiyor.
- **Diğerleri:** BPSK 99.3, QPSK 99.4, CPFSK 100, PAM4 99.2, 8PSK 91.9.

**Okuma metni (EN)**

> Where does it still fail? This is their matrix at 0 dB. Two problems are left.
>
> WBFM is called AM-DSB more often than WBFM: 58 percent of the time. And QAM16 and QAM64 are mixed up, about 6 percent each way.
>
> Those are the same two problems we see in our own runs.

**Olası sorular**

- *"Do other models fail the same way?"* — Yes. In the same figure, every model has WBFM between 28% and 63%, and several models confuse the QAM pair much worse (MCNet calls 63% of QAM16 QAM64).
- *"Can this be fixed?"* — The paper suggests augmentation and spectral features for WBFM. For the QAM pair, our frame-length result says more symbols, not a better model.

---

## Slayt 9 — Hangi parça önemli?

**Slaytta ne var, ne demek**

Grafik, makalenin Tablo 6'sının RML2016.10a satırları:

| model | doğruluk | F1 | precision |
|---|---|---|---|
| sadece CNN | 0.5993 | 0.6193 | 0.7486 |
| sadece BiLSTM | 0.6313 | 0.6537 | **0.7676** |
| attention'sız | 0.6280 | 0.6532 | 0.7526 |
| tam ICRNNA | 0.6324 | 0.6567 | 0.7536 |

- **+0.4:** Attention'ın katkısı. Tam model 63.24, attention'sız 62.80.
- **0.1:** Sadece BiLSTM (63.13) ile tam model (63.24) arasındaki fark. Yani CNN ve attention ikisi birlikte 0.1 puan ekliyor.
- **Mesaj:** İşin neredeyse tamamını BiLSTM yapıyor.
  - Bizim versiyonumuzda parametrelerin 84%'ü LSTM'de; bu bulguyla tutarlı.
  - Tek koşu olduğu için 0.1–0.4 puanlık farklar gürültü içinde kalabilir.

**Okuma metni (EN)**

> Which part matters? They removed parts one at a time. CNN alone: 59.9. BiLSTM alone: 63.1. Everything but the attention: 62.8. The full model: 63.2.
>
> So the BiLSTM does almost all the work, and attention adds less than half a point. That matches our version, where 84 percent of the parameters are in the LSTM.

**Olası sorular**

- *"Then why not use the BiLSTM alone?"* — By their own numbers you could: 0.1 point lower, and fewer parts. The paper doesn't discuss this.
- *"Did they say attention helps at low SNR?"* — They say so, but Table 6 gives only overall numbers; there is no per-SNR ablation to support it.

---

## Slayt 10 — Ara slayt: "Does it hold up?"

**Slaytta ne var, ne demek**

- İkinci kısma geçiş. Buradan sonra anlatılan her şey bizim ölçümümüz ya da makalenin kendi tablolarıyla karşılaştırma.

**Okuma metni (EN)**

> Part two: does it hold up? I rebuilt the model from the paper and checked the numbers against their own tables.

---

## Slayt 11 — Makaleden yeniden kurduk

**Slaytta ne var, ne demek**

- **794,827:** Sadece makalenin metninden kurduğumuz modelin parametre sayısı (`colab/icrnna_faithful_2016.py`). Makale "0.79 M" diyor; tutuyor. Esmer'in versiyonu 786,379.
- **61.75%:** Makalenin tarifiyle tam 58 epoch, 3 farklı tohum (seed 42, 43, 44).
  - Tek tek: 61.79 / 61.65 / 61.82; standart sapma 0.07.
  - En iyi epoch'lar 58, 58, 55. Yani üç koşudan ikisi son epoch'ta hâlâ iyileşiyordu.
- **63.21%:** Aynı model, 150 epoch sınırıyla.
  - En iyi epoch 107; sabır 15 epoch, yani 122'de durdu, sınıra çarpmadı.
  - Makalenin 63.24'ünden 0.03 puan farklı.
- **Sağdaki grafik:** Makalenin Şekil 7'si. Eğitim ve doğrulama doğruluğu 58. epoch'ta hâlâ yükseliyor.
  - Makale: doğrulama doğruluğu 54.08%'den 63.59%'a.
  - Öğrenme oranı 43. epoch'ta 0.0002'ye, 58'de 0.00004'e düşüyor.
- **Mesaj:** Makalenin sayısı gerçek, ama 58 epoch'ta güvenilir şekilde ulaşılmıyor.

**Okuma metni (EN)**

> I built the model from the paper's text alone. The size matches: 794,827 parameters, their 0.79 million.
>
> With their recipe and their 58 epochs, I got 61.75 percent over three runs, and two of the three were still improving at the last epoch. Their own training curve, on the right, is also still rising at epoch 58.
>
> So I let it train longer. It peaked at epoch 107, at 63.21 percent. Their 63.24 is real. It just needs more than 58 epochs.

**Olası sorular**

- *"Why did they reach 63.24 in 58 epochs and you didn't?"*
  - We can't say for sure. The paper leaves several things open: the optimizer, the normalization, whether the split was stratified, and where the attention dropout goes.
  - It also reports one run; ours is three, and all three landed at 61.7–61.8.
  - Their Fig. 7 shows the curve still rising, so at 58 they were also close to the edge.
- *"Is one 150-epoch run enough?"* — It is one seed. It matches the paper to 0.03 points, and the 58-epoch runs agree with each other to 0.07, so the picture is consistent. More seeds would tighten it.

---

## Slayt 12 — SNR'a göre: bizim model ve onların tablosu

**Slaytta ne var, ne demek**

Grafik, makalenin Tablo 5'indeki ICRNNA satırı ile bizim yeniden kurduğumuz modeli (150 epoch, seed 42) −10…+4 dB'de karşılaştırıyor:

| SNR etiketi | −10 | −8 | −6 | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|---|---|---|
| makale (Tablo 5) | 25.32 | 37.45 | 55.26 | 71.13 | 85.11 | 92.21 | 93.40 | 93.02 |
| bizim | 24.42 | 40.06 | 60.55 | 74.48 | 86.55 | 91.33 | 91.70 | 92.48 |

- **Mesaj:** Bizim model −6 ve −4 dB'de birkaç puan önde (+5.3, +3.4). Makale 0 ve 2 dB'de 1–2 puan önde. En büyük fark 5.3 puan. Genel doğruluk aynı.

**Okuma metni (EN)**

> Per SNR, my rebuild and their table agree. Mine is a few points higher at minus 6 and minus 4 dB, theirs a point or two higher at 0 and 2 dB. Same model, same curve.

**Olası sorular**

- *"Why the difference at −6 dB?"* — One run each; at −6 dB there are about 1,650 test frames per SNR, so a few points can be split and seed. I would not read anything into it.

---

## Slayt 13 — Kendi tablolarının desteklemediği üç iddia

**Slaytta ne var, ne demek**

1. **"Best at low SNR"** (özet ve Bölüm 4.2.2):
   - Makalenin kendi Tablo 5'inde (10a) CNN-BiLSTM-DNN −10 dB'de (27.73 vs 25.32) ve −8 dB'de (39.55 vs 37.45) daha yüksek.
   - MCLDNN −6 dB'de daha yüksek (56.32 vs 55.26). CNN-BiLSTM-DNN de −6'da önde (56.18).
   - Ayrıca +4 dB'de CNN-BiLSTM-DNN önde (93.18 vs 93.02).
   - ICRNNA'nın önde olduğu yer −4…+2 dB.
2. **"Full model has the highest precision"** (Bölüm 4.4):
   - Tablo 6'da sadece BiLSTM'in precision'ı daha yüksek: 10a'da 0.7676 vs 0.7536, 10b'de 0.7200 vs 0.7100.
3. **"MCLDNN: 406 million parameters"** (Tablo 7 ve Bölüm 4.5 metni):
   - MCLDNN'in gerçek sayısı 406,199, yani 0.41 milyon.
   - Bunu biz saydık; MCLDNN'in kendi makalesi, PET-CGDNN makalesi ve Zhang 2022 benchmark'ı da aynı sayıyı veriyor (`LITERATURE_CHECK.md`).
   - Yani ICRNNA (0.79 M) MCLDNN'in yaklaşık iki katı, ondan küçük değil.
   - Aynı tabloda PET-CGDNN "79.2 M" yazıyor; gerçek sayı 71,871 (0.072 M). ICRNNA ondan yaklaşık 11 kat büyük.

Slayta koymadığım küçük bir tutarsızlık daha var: genel doğruluk Tablo 3'te 63.24%, Bölüm 4.2.2'de 63.39%.

**Okuma metni (EN)**

> Reading their tables closely, three statements in the text don't match the numbers.
>
> One: best at low SNR. In their own Table 5, CNN-BiLSTM-DNN is higher at minus 10 and minus 8 dB, and MCLDNN at minus 6.
>
> Two: the full model has the highest precision. In Table 6, the BiLSTM-only model is higher.
>
> Three: MCLDNN is listed with 406 million parameters. It has 406 thousand. We counted it, and three other papers agree. So ICRNNA is about twice the size of MCLDNN, not smaller.

**Olası sorular**

- *"Is the parameter thing just a typo?"* — Probably a unit mix-up. But the text builds a claim on it: "substantially smaller than MCLDNN (406 million parameters)". With the right number the claim reverses.
- *"Are you sure about 406,199?"* — Yes: our port has exactly that count, and three independent papers print the same number to the digit.
- *"Are these errors important?"* — None changes the main result, 63.24% on 10a, which we reproduced. They change the story around it: "best at low SNR" and "smaller than the others".

---

## Slayt 14 — Makalenin söylemedikleri

**Slaytta ne var, ne demek**

1. **One run:**
   - Makale her model için tek sayı veriyor; tekrar veya standart sapma yok.
   - Bizim 3-seed ölçümlerimizde modelden modele 0.2 ile 1.1 puan arasında oynama var (LSTM2: 60.95–62.09; MCLDNN: 61.19–61.70).
   - 0.5 puanlık birincilik bu oynamanın içinde.
2. **Baselines:**
   - Diğer 11 modelin sayılarının yeniden eğitimle mi elde edildiği, yoksa kendi makalelerinden mi alındığı yazmıyor.
   - Farklı bölme ve farklı eğitimle elde edilmiş sayıları yan yana koymak adil karşılaştırma değil.
3. **SNR label:**
   - Makale etiketi gerçek SNR gibi kullanıyor (Denklem 1: H = s + n, AWGN).
   - Bizim ölçümümüzde (`NOISE_2016.md`) etiket gerçek SNR değil.
   - Etiket 0 dB'de ölçülen SNR: BPSK 4.8, QPSK 5.1, QAM16 14.0, QAM64 17.2 dB.
- **+9 dB:** QAM16 ile QPSK arasındaki fark (14.0 − 5.1), ikisi de "0 dB" etiketli.
  - Makalenin "QAM16 noise-sensitive, yet we get 94.3% at 0 dB" cümlesi bu yüzden yanıltıcı: QAM16 o etikette çok daha temiz.

**Okuma metni (EN)**

> And three things the paper doesn't tell us.
>
> It reports one training run, with no spread. Across three seeds we see up to a point of variation, so a half-point lead may not be real.
>
> It doesn't say whether the other models were retrained on the same split or copied from their papers.
>
> And it treats the SNR label as a true SNR. In this data it isn't: at label 0 dB, QAM16 frames have about 9 dB more signal than QPSK frames.

**Olası sorular**

- *"Isn't this the dataset's fault, not the paper's?"* — Yes, the label problem is in the dataset, and almost every paper treats the label the same way. The point is only that per-class comparisons at one label are not at one SNR.
- *"How did you measure SNR?"* — From the spectrum of each frame: the signal sits in the middle of the band and outside it there is only noise. Checked first on frames with known noise, within 0.5 dB. Readable between about −13 and +18 dB.

---

## Slayt 15 — Her model için tek protokol

**Slaytta ne var, ne demek**

- **Deney:** Beş modeli aynı protokolle eğittik (`literature_2016.md`): aynı çerçeveler, aynı 70/15/15 bölme, 3 seed, yakınsayana kadar.
- **"their Table 3":** Makalenin bu modeller için verdiği sayı. LSTM2 ve VT-CNN2 o tabloda yok.
- **Çarpıcı nokta:** Makalenin MCLDNN için verdiği 61.42, bizim ölçtüğümüz 61.41. PET-CGDNN için 60.58 ve 60.98. Yani baseline sayıları bizimle uyumlu.
- **ICRNNA satırı:** Esmer'in versiyonu, 62.2%. Makaleden birebir kurduğumuz model 63.21% (slayt 11).
- **Mesaj:**
  - ICRNNA birinci, ama ilk dört model 1.25 puan içinde.
  - 10 dB üstünde dördü de ~91%'de duruyor; o tavan veriden geliyor (WBFM ve QAM çifti).

**Okuma metni (EN)**

> So I ran the comparison myself, with one protocol for every model: same data, same split, three seeds.
>
> Our numbers for MCLDNN and PET-CGDNN match their Table 3 within half a point.
>
> ICRNNA is first, but LSTM2, MCLDNN and PET-CGDNN are all within about one point. Above 10 dB all four stop near 91 percent.

**Olası sorular**

- *"Why isn't CNN-BiLSTM-DNN in your table?"* — We ported the models from the public AMR-Benchmark code. CNN-BiLSTM-DNN (2024) isn't in it, and the paper doesn't give its code. It is the natural next one to add.
- *"Why compare Esmer's version and not the faithful one?"* — All our other results use Esmer's version, so the table uses it too. The faithful build is about one point higher (63.21%), which would still keep the top four within two points.

---

## Slayt 16 — Özetle

**Slaytta ne var, ne demek**

1. **Tekrarlanabilir:** Model ve 63.24% tekrarlanabiliyor (bizde 63.21%), ama 58 epoch'tan uzun eğitilince.
2. **Fark küçük:** Diğer modellere fark yaklaşık bir puan; işin çoğunu BiLSTM yapıyor (Tablo 6).
3. **Tutarsızlıklar:** Bazı iddialar makalenin kendi tablolarıyla uyuşmuyor (düşük SNR, precision, parametre sayıları).
- **"For us":** ICRNNA sağlam bir omurga (backbone), ama açıkça daha iyi bir model değil. Projede kullanmaya devam etmemiz için bu yeterli. Asıl sorumuz zaten mimari değil, alan farkı (domain gap).

**Okuma metni (EN)**

> To sum up. The model and its 63.24 percent are reproducible, if you train it long enough.
>
> Its lead over other models is about one point, and the BiLSTM does most of the work.
>
> And a few claims don't match the paper's own tables.
>
> For us, that makes ICRNNA a solid backbone, but not a clearly better one. Thank you.

**Olası sorular**

- *"Should we switch to the faithful build?"* — It is about one point better at convergence and matches the paper. Switching is one line (`model_zoo.backbone`). But all current results are on Esmer's version, so switching means rerunning them. I would decide that together with the next experiment.
- *"Would you contact the authors?"* — The parameter-count error and the missing details (optimizer, normalization, number of runs) are fair questions to send them.

---

## Genel tavsiyeler

- **Saygılı ton:** Makaleyi eleştirirken hep "their own table says..." diye bağla. Senin fikrin değil, onların sayıları konuşuyor.
- **Önce takdir, sonra eleştiri:** Sayılarını tekrarlayabildiğimiz için makale iyi bir iş. Slayt 11 bunu söylüyor, eleştiriler ondan sonra geliyor.
- **Esmer'in versiyonu ile makaleden kurduğumuz model karışmasın:**
  - Esmer'in versiyonu: 786,379 parametre, 62.2%, bütün diğer sonuçlarımız bununla.
  - Makaleden kurduğumuz (faithful) model: 794,827 parametre, 63.21%.
- **Ölçmediğin şeyi söyleme:** WBFM'in sessiz ses açıklaması literatürden; ölçmedik. CNN-BiLSTM-DNN'i biz koşmadık.

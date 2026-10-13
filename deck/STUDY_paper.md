# Makale sunumu çalışma metni — `moshe_paper_icrnna.pptx`

Makale: El-Haryqy, Kharbouche, Ouamna, Madini, Zouine. *Improved automatic modulation recognition using deep learning with additive attention.* Results in Engineering 26 (2025) 104783. Ibn Tofail Üniversitesi, Kenitra, Fas. Nisan 2025'te yayımlandı, açık erişim (CC BY-NC 4.0).

**Bu sunumun amacı:** Moshe makaleyi anlamak istiyor. O yüzden önce makalenin ne yaptığı adım adım anlatılıyor:
- veri;
- model, blok blok, her blokta boyutlarla;
- eğitim;
- sonuçlar.

Eleştiriler sonra geliyor, "What we checked" kısmında.

**Slayt sayısı:** 20 slayt, yaklaşık 9 dakika. Colab'daki `official_2016` sonuçları gelirse 16. slayttan sonra "Their own code, run by us" slaytı eklenir; sonraki slaytların numarası bir kayar.

Her slayt için üç bölüm var:
- **Slaytta ne var, ne demek:** Türkçe açıklama.
- **Okuma metni (EN):** Slayt notlarıyla aynı metin.
- **Olası sorular:** Moshe'nin sorabileceği sorular ve kısa cevapları.

**Sayıların kaynağı:**
- **Makaleden alınanlar:** `deck/extract_paper.py` içinde tablo numarasıyla yazılı.
- **Onların kodundan alınanlar:** `OFFICIAL_CODE.md`. Katman boyutları kodun kendisinden yazdırıldı.
- **Bizim sayılarımız:** Sonuç dosyalarından okunuyor.

---

## Ezber kartı

| sayı | ne | nereden |
|---|---|---|
| 63.24% | makalenin RML2016.10a doğruluğu | Tablo 3 |
| 2 × 128 | girdi: 128 I + 128 Q örneği | veri |
| 64 × 128 → 64 × 256 → 256 → 11 | CNN, BiLSTM, attention ve çıkıştan sonraki boyutlar | onların kodu |
| 794,188 | onların modelindeki sayı (BatchNorm istatistikleri dahil); 792,396'sı öğreniliyor | onların kodu |
| %83 / %8 / %5 / %3 | sayıların BiLSTM / attention / dense / CNN'deki payı | onların kodu |
| 58 | makalede "epoch sayısı"; aslında erken durdurmanın tetiklendiği epoch | Tablo 2 + kod |
| 59.9 / 63.1 / 62.8 / 63.2 | sadece CNN / sadece BiLSTM / attention'sız / tam model | Tablo 6 |
| 57.9% | 0 dB'de WBFM'in AM-DSB denme oranı | Şekil 11 |
| 61.75% / 63.21% | makaleden kurduğumuz model: 58 epoch'ta / sonuna kadar eğitilince (epoch 107) | bizim koşular |
| 406,199 | MCLDNN'in gerçek parametre sayısı; makale "406 M" yazıyor | `LITERATURE_CHECK.md` |
| 8 Ağustos 2026 | yazarların kodunu GitHub'a koyduğu gün; ağırlık yok | `OFFICIAL_CODE.md` |

---

## Sözlük — "Bu nedir?" soruları

Temel terimler (recall, precision, epoch, SNR, dB...) `STUDY_simple.html`'deki sözlükte var. Burada bu sunumda geçenler:

- **Convolution (1-D)** — "A short filter slid along the signal; at each position it multiplies and adds. Like an FIR filter whose taps are learned." *(Konvolüsyon; Blok 1.)*
- **Filter / kernel size 3** — "Each filter looks at 3 consecutive samples at a time." *(3 katsayılı filtre.)*
- **ReLU** — "Sets negative values to zero, keeps positive ones. It makes the network non-linear." *(Doğrusal olmayan adım.)*
- **Batch normalization** — "Rescales each feature to a standard range during training; makes training stable." *(Her katmandan sonra.)*
- **Max pooling** — "Keeps the larger of each two neighbouring values: the sequence gets half as long." *(128 → 64.)*
- **Dropout** — "During training, randomly switches off a fraction of the numbers (here 30%), so the network cannot rely on any single one. Off at test time." *(Aşırı ezberlemeyi önler.)*
- **LSTM** — "A layer that reads a sequence one step at a time and carries a memory vector; learned gates decide what to keep, forget and output." *(Blok 2.)*
- **Bidirectional** — "A second LSTM reads the sequence backwards; the two outputs are put side by side." *(Çift yönlü.)*
- **Attention (additive)** — "A small network gives each time step a score; softmax turns scores into weights; the output is the weighted average." *(Blok 3. 2014'teki Bahdanau attention; ChatGPT'deki çok katmanlı attention değil.)*
- **Softmax** — "Turns a list of numbers into positive weights or probabilities that add up to one." *(Hem attention'da hem çıkışta.)*
- **Layer normalization** — "Rescales one vector to zero mean and unit spread." *(Attention çıkışında.)*
- **Dense (fully connected) layer** — "Every input connected to every output: a matrix multiplication." *(Blok 4.)*
- **Cross-entropy loss** — "Minus the log of the probability the model gives the right answer." *(Eğitimde küçültülen sayı.)*
- **Adam** — "A version of gradient descent that adapts the step size for each number." *(Öğrenme oranı 0.001.)*
- **Gradient clipping** — "Caps the size of each update step so one bad batch cannot throw training off." *(Onların kodunda var, makalede yok.)*
- **Early stopping** — "Stop training when the validation result has not improved for a set number of epochs, and go back to the best weights." *(Onlarda: doğrulama kaybı, 6 epoch.)*
- **Ablation** — "Remove one block at a time to see how much it contributes." *(Tablo 6.)*
- **F1 score** — "A combined measure of precision and recall." *(Şekil 8.)*

---

## Slayt 1 — Başlık

**Slaytta ne var, ne demek**
- Makalenin adı, yazarları, üniversitesi, dergisi.
- Turuncu dört kutu modelin dört bloğu: CNN → BiLSTM → Attention → Dense. Sunum bunları sırayla açacak.

**Okuma metni (EN)**

> Today I present the paper our detector comes from: ICRNNA, by El-Haryqy and colleagues, Results in Engineering, 2025. First I'll go through what the paper does, step by step: the data, the model block by block, how it is trained, and what it reports. Then what we checked ourselves.

---

## Slayt 2 — Neden bu makale, ne iddia ediyor?

**Slaytta ne var, ne demek**
- **Üstteki zincir:** Bu makale → Esmer'in versiyonu → bizim dedektörümüz. Bizim modelimiz bu makaleden geliyor; Moshe'nin "bu model nereden çıktı" sorusunun cevabı.
- **63.24%:** RML2016.10a'da, 11 sınıfın ve 20 SNR seviyesinin tamamında doğruluk (Tablo 3).
- **0.79 M:** Öğrenilen sayı (parametre) adedi (Tablo 7). Onların kodu da bunu doğruluyor: 792,396 öğrenilen sayı.
- **"low SNR":** Makalenin düşük SNR'da özellikle iyi olduğu iddiası. 17. slaytta kendi tablolarıyla karşılaştırılıyor.

**Okuma metni (EN)**

> Why this paper? Because our detector is built on it. Esmer re-implemented it, and our code comes from that version. The paper makes three claims. The best accuracy on RML2016.10a, 63.24 percent. A small model: 0.79 million numbers to learn. And it says it is especially good at low SNR.

**Olası sorular**
- *"Is it a well-known journal?"* — Results in Engineering is a broad, open-access Elsevier engineering journal, not a signal-processing one. That is one reason we checked the numbers ourselves.

---

## Slayt 3 — Veri: model neyi görüyor?

**Slaytta ne var, ne demek**
- **Veri seti:** RML2016.10a, bizim kullandığımızla aynı. 11 modülasyon, 20 SNR etiketi (−20…18 dB), 220,000 çerçeve.
- **Sağdaki resim:** Bir çerçevenin I (turuncu) ve Q (mavi) bileşenleri. Biz çizdik, açıklama amaçlı.
- **2 × 128:** Modelin tek girdisi. 2 satır (I ve Q) × 128 örnek. Model başka hiçbir şey görmüyor: frekans bilgisi, SNR etiketi, ek bir özellik yok.
- **RML2016.10b:** Makale ayrıca daha büyük olan 10b'yi de test ediyor (65.39%). Biz 10a ile kalıyoruz.

**Okuma metni (EN)**

> The data is the same dataset we use: RML2016.10a. Eleven modulation types, twenty SNR labels, 220,000 frames. Each frame is 128 complex samples, written as two rows: I and Q. So the input to the model is a 2 by 128 array. That is all the model ever sees.

**Olası sorular**
- *"Does the model see the SNR?"* — No. Only the 2 × 128 array. The SNR label is used only afterwards, to sort the test results.

---

## Slayt 4 — Modele kuşbakışı

**Slaytta ne var, ne demek**
- **Akış:** çerçeve → CNN → BiLSTM → Attention → Dense → 11 çıktı.
- **Kutuların altındaki sayılar:** O bloktan çıkan verinin boyutu. Hepsi onların kodundan yazdırıldı:
  - giriş 2 × 128;
  - CNN'den sonra 64 × 128 (64 zaman adımı, her birinde 128 sayı);
  - BiLSTM'den sonra 64 × 256;
  - attention'dan sonra tek bir 256'lık vektör;
  - dense'ten sonra 128, sonra 64;
  - çıkışta 11 olasılık.
- **Şema:** Makalenin Şekil 2'sinden sadeleştirerek yeniden çizildi.

**Okuma metni (EN)**

> Here is the whole model. A frame of 2 by 128 goes in. Four blocks process it. Eleven probabilities come out, one per modulation, and the largest is the answer. Below each block I wrote the size of what comes out of it, so you can follow one frame through. I'll now take the blocks one at a time.

**Olası sorular**
- *"Why these sizes?"* — They are the authors' design choices (Fig. 2). The paper tuned learning rate, dropout and L2 by grid search, not the layer sizes.

---

## Slayt 5 — Blok 1: kısa öğrenilmiş filtreler (CNN)

**Slaytta ne var, ne demek**
- **Soldaki çizim (temsilî):** 3 örneklik bir pencere (turuncu) sinyal boyunca kayıyor; her konumda bir çıktı üretiyor.
- **1:** 64 filtre, her biri 3 örnek uzunluğunda. Girdi 128 adım × 2 (I, Q), çıktı 128 adım × 64. Kenarlarda "same" doldurma var, uzunluk değişmiyor.
- **2:** 128 filtre daha, yine 3 örnek: 128 × 128. Her konvolüsyondan sonra ReLU ve batch normalization.
- **3:** Max pool, komşu iki değerden büyüğünü alıyor: 64 × 128. Sonra dropout 0.3.
- **FIR benzetmesi:** Konvolüsyon, katsayıları tasarlanmak yerine veriden öğrenilen bir FIR filtre. Moshe elektrik mühendisi; bu benzetme tanıdık gelir.
- **"Yaklaşık 6 örnek":** İlk filtre 3 örnek görüyor. İkinci filtre ilkinin 3 çıkışını görüyor, yani 5 örnek. Pooling ikisini birleştiriyor, 6 örnek. Sembol 8 örnek olduğu için bu bir sembolden kısa.
- **Parametreler:** Bu blokta 25,920 sayı var, modelin %3'ü.

**Okuma metni (EN)**

> Block one: two convolution layers. A convolution here is a short filter, three samples long, slid along the frame, like an FIR filter whose taps are learned instead of designed. The first layer has 64 such filters, the second 128. So every time step is now described by 128 numbers. Then a pooling step keeps the larger of each pair of neighbours, so 128 time steps become 64. Each of the 64 steps summarizes about 6 samples, less than one symbol.

**Olası sorular**
- *"What do the filters learn?"* — We haven't looked at them. Typically such filters respond to changes between neighbouring samples: phase jumps and amplitude steps.
- *"Why pool?"* — It halves the sequence the LSTM has to read (128 → 64), which makes it cheaper.
- *"Why only 3 taps?"* — The authors' choice. Two stacked layers already see 5 samples, and the LSTM connects everything after that.

---

## Slayt 6 — Blok 2: adımları sırayla okumak (BiLSTM)

**Slaytta ne var, ne demek**
- **Çizim:** 64 adım. Üstteki oklar ileri okuyan LSTM'i, alttaki oklar geri okuyan ikinci LSTM'i gösteriyor.
- **LSTM:**
  - Adımları tek tek okuyor ve adımdan adıma bir hafıza vektörü (128 sayı) taşıyor.
  - Hafızaya neyin girip neyin silineceğine öğrenilen "kapılar" (gate) karar veriyor.
- **Bidirectional:** Her adımın çıktısı, ileri okuyandan 128 ve geri okuyandan 128 sayı; toplam 256. Böylece her adım hem öncesini hem sonrasını "biliyor".
- **2 katman:** İkinci LSTM, birincinin çıktısını okuyor. Her birinden sonra batch normalization ve dropout 0.3 var.
- **%83:** Modelin 794,188 sayısının 659,456'sı bu blokta.
  - Çünkü bir LSTM'de 4 kapı var ve her kapı girdi ile hafızanın tamamına bağlı.
  - Örneğin ikinci katmanın bir yönünde 4 × 128 × (256 + 128) ağırlık var.

**Okuma metni (EN)**

> Block two: two bidirectional LSTM layers. An LSTM reads the 64 steps in order, one at a time, and carries a memory from step to step. Bidirectional means a second LSTM reads the same steps backwards. So at every step, the output knows what came before and what comes after: 128 numbers from each direction, 256 in total. This block holds 83 percent of the model's numbers.

**Olası sorular**
- *"Why an LSTM and not just more convolutions?"* — The convolutions see less than one symbol. Telling QAM16 from QAM64 needs many symbols together; the LSTM connects all 64 steps.
- *"What is in the memory?"* — 128 numbers per direction, updated at every step. What they encode is learned; we have not interpreted them.

---

## Slayt 7 — Blok 3: öğrenilmiş ağırlıklı ortalama (attention)

**Slaytta ne var, ne demek**
- **Soldaki çubuklar (temsilî):** 64 adımın her birine bir ağırlık. Toplamları 1. Turuncu olanlar ağırlığı büyük adımlar.
- **Formüller** (makalenin Denklem 4–6):
  - **score:** Her adıma küçük bir ağın verdiği puan.
    - Onların kodunda q (query), ikinci LSTM'in batch norm ve dropout'tan geçmiş çıktısı. k, aynı LSTM'in ham çıktısı.
    - W₁ ve W₂ 128 birimlik katmanlar.
  - **weights = softmax(score):** Puanlar toplamı 1 olan ağırlıklara dönüşüyor.
  - **output:** Adımların ağırlıklı ortalaması. Ortalaması alınan, LSTM'in ham çıktısı. Sonuç 256 sayılık tek bir vektör.
- **"Improved"** (onların kodundan):
  - puanlara öğrenilen bir ölçek katsayısı (başlangıç değeri 1);
  - ağırlıklara dropout 0.1;
  - çıktıya layer normalization.
- **Parametreler:** 66,177 sayı, modelin %8'i.

**Okuma metni (EN)**

> Block three: attention. Each of the 64 steps gets a score from a small learned network. A softmax turns the scores into weights that add up to one. The output is the weighted average of the 64 steps: one vector of 256 numbers for the whole frame. The paper calls it improved because of three additions. From their code we now know exactly what they are: a learned scale on the scores, dropout on the weights, and a normalization of the result.

**Olası sorular**
- *"Is this the attention of ChatGPT?"* — No. It is the original 2014 additive attention: one layer, applied once, a weighted average over time.
- *"Which steps get large weights?"* — We have not looked. It would be a good thing to plot.
- *"Does it help?"* — By their own ablation, less than half a point (slide 13).

---

## Slayt 8 — Blok 4: karar

**Slaytta ne var, ne demek**
- **Soldaki çubuklar (temsilî):** 11 sınıfın olasılıkları. En büyüğü (burada QPSK) cevap.
- **Katmanlar:** 256 → 128 → 64 → 11. Her ara katmanda ReLU, batch normalization ve dropout 0.3. Son katmanda softmax.
- **Pasta grafiği:** 794,188 sayının bloklara dağılımı.
  - BiLSTM 659,456 (%83).
  - Attention 66,177 (%8).
  - Dense 42,635 (%5).
  - CNN 25,920 (%3).
- **Not:** Toplam sayıya batch normalization'ın 1,792 istatistiği dahil; öğrenilen sayı 792,396.

**Okuma metni (EN)**

> Block four: the decision. Two ordinary layers, 128 and 64 numbers, and then 11 outputs. A softmax turns them into probabilities, and the largest one is the answer. In total the model has 794 thousand numbers. The pie shows where they are: almost all in the LSTM.

**Olası sorular**
- *"Is the 0.72 a real probability?"* — The softmax outputs add up to one, but they are not guaranteed to be calibrated. We use only which one is largest.

---

## Slayt 9 — Nasıl eğitiliyor?

**Slaytta ne var, ne demek**
- **DATA:**
  - 220,000 çerçevenin %70'i öğrenme (154,000), %15'i doğrulama (33,000), %15'i test (33,000).
  - Onların kodunda bölme rastgele, sınıf ve SNR'a göre dengeli değil.
- **LOOP:**
  - 32 çerçevelik batch'ler, epoch başına 4,813 adım.
  - Loss: cross-entropy. Optimizer: Adam, öğrenme oranı 0.001.
  - Kodda bir de gradient clipping var (`clipnorm=1.0`), makalede yazmıyor.
- **STOPPING:**
  - Doğrulama kaybı 5 epoch iyileşmezse öğrenme oranı ×0.2.
  - 6 epoch iyileşmezse eğitim duruyor ve en iyi ağırlıklara dönülüyor.
  - En fazla 100 epoch.
- **58:** Makalenin Tablo 2'si "Number of Epochs 58" diyor. Ama koddan anlaşılıyor ki bu seçilmiş bir sayı değil, durma kuralının tetiklendiği yer.
- **"Trained once":** Kodda seed döngüsü yok; tek bölme, tek koşu.

**Okuma metni (EN)**

> How it is trained. Seventy percent of the frames to learn from, fifteen to watch, fifteen to test. Batches of 32 frames; the loss is cross-entropy; the optimizer is Adam with learning rate 0.001. When the validation loss stops improving for 5 epochs, the learning rate is cut by five. When it stops improving for 6 epochs, training stops and the best weights are restored. The paper's table says 58 epochs. From their code we now know that 58 is where this stopping rule fired, not a number they chose. They trained once.

**Olası sorular**
- *"Why stop on the loss and not the accuracy?"* — It is the common Keras default. Loss and accuracy can disagree: here the loss had flattened while accuracy was still creeping up (slide 16).
- *"How were the hyperparameters chosen?"* — Section 4.1: grid search over learning rate, dropout and L2; batch size and epochs by hand.

---

## Slayt 10 — Ana tablo

**Slaytta ne var, ne demek**

Makalenin Tablo 3'ü, RML2016.10a sütunu:

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

- Birinciyle ikinci arasında 0.51 puan var.
- Grafik 0'dan başlıyor; farkların gerçek büyüklüğü böyle görünüyor.

**Okuma metni (EN)**

> Their main result, on 2016.10a. ICRNNA is first, at 63.24 percent. The runner-up, CNN-BiLSTM-DNN, is half a point behind. MCLDNN and PET-CGDNN, the well-known ones, are two to three points behind.

**Olası sorular**
- *"Were the other models retrained?"* — The paper doesn't say (slide 18).

---

## Slayt 11 — SNR'a göre doğruluk

**Slaytta ne var, ne demek**
- **Grafik:** Makalenin Şekil 8'i.
- **9%:** −20 dB'de doğruluk. 11 sınıfta şans 1/11 = 9.1%.
- **~93%:** 0 dB'den itibaren. Makale "5–15 dB'de 93–94%" diyor.
- **Not:** Buradaki "SNR" veri setinin etiketi, ölçülmüş SNR değil (`NOISE_2016.md`).

**Okuma metni (EN)**

> Accuracy against SNR. Below minus 10 dB it is about 9 percent, which is chance with eleven classes. It climbs between minus 10 and 0 dB, and from 0 dB up it stays around 93 percent.

---

## Slayt 12 — 0 dB'de hâlâ nerede hata yapıyor?

**Slaytta ne var, ne demek**
- **Matris:** Makalenin Şekil 11'indeki ICRNNA paneli, 0 dB. Genel doğruluk 92.21%.
- **57.9%:** WBFM çerçevelerinin AM-DSB denme oranı; WBFM'e doğru denen 41.4%.
  - Literatürdeki bilinen açıklama: WBFM'lerin bir kısmı sessiz sesten üretilmiş, AM-DSB'ye benziyor.
  - Bunu biz ölçmedik.
- **~6%:** QAM16'nın 5.7%'si QAM64, QAM64'ün 6.1%'i QAM16 denmiş.

**Okuma metni (EN)**

> Where does it still fail? Their matrix at 0 dB. Two problems are left. WBFM is called AM-DSB 58 percent of the time. And QAM16 and QAM64 are mixed up, about 6 percent each way. These are the same two problems we see in our runs.

---

## Slayt 13 — Hangi blok önemli?

**Slaytta ne var, ne demek**

Makalenin Tablo 6'sı (10a):

| model | doğruluk | F1 | precision |
|---|---|---|---|
| sadece CNN | 0.5993 | 0.6193 | 0.7486 |
| sadece BiLSTM | 0.6313 | 0.6537 | 0.7676 |
| attention'sız | 0.6280 | 0.6532 | 0.7526 |
| tam model | 0.6324 | 0.6567 | 0.7536 |

- Attention 0.4 puan ekliyor.
- Sadece BiLSTM ile tam model arasında 0.1 puan var.
- Tek koşu olduğu için bu farklar gürültü içinde kalabilir.

**Okuma metni (EN)**

> Which block matters? They removed blocks one at a time. CNN alone: 59.9. BiLSTM alone: 63.1. Everything but the attention: 62.8. The full model: 63.2. So the LSTM does almost all the work, and attention adds less than half a point.

**Olası sorular**
- *"Then why keep the CNN and attention?"* — By their own numbers, they add 0.1 points together. The paper doesn't discuss it.

---

## Slayt 14 — Ara slayt

**Okuma metni (EN)**

> Part two: what we checked.

---

## Slayt 15 — Yeniden kurduk; onların kodu da doğruluyor

**Slaytta ne var, ne demek**
- **Zaman çizelgesi:**
  - **Nisan 2025:** Makale, kodsuz.
  - **8 Ağustos 2026:** Yazarlar kodu GitHub'a koydu (TensorFlow, MIT lisanslı), ağırlık yok.
  - **Eylül 2026:** Biz makaleden modeli kurduk; o sırada kodun varlığından haberimiz yoktu.
- **63.21%:** Bizim makaleden kurduğumuz model, en iyi epoch 107. Makale 63.24%.
- **Onların koduyla tutanlar:**
  - tek pooling;
  - her LSTM'den sonra normalizasyon;
  - attention ağırlıklarında dropout;
  - boyut (bizde 794,827, onlarda 794,188).
- **Kaçırdıklarımız:**
  - attention'ın iki girişli yapısı ve öğrenilen ölçeği;
  - girdi normalizasyonu;
  - gradient clipping.
  - Ayrıntılar `OFFICIAL_CODE.md`'de.

**Okuma metni (EN)**

> Before we knew any code existed, I rebuilt the model from the paper's text alone. Its size matched: 794,827 numbers. In August the authors released their code, and it confirms our reading: one pooling step, normalization after each LSTM, dropout on the attention weights. We missed a few details: the exact form of the attention, the input normalization, and gradient clipping. Trained until it stopped improving, our rebuild reached 63.21 percent, against their 63.24.

**Olası sorular**
- *"Why didn't you use their code from the start?"* — It didn't exist when the paper came out, and we didn't know it had appeared. Now we run it too (Colab, three runs).
- *"Did they publish weights?"* — No. Only the training code.

---

## Slayt 16 — "58 epoch" ne demek?

**Slaytta ne var, ne demek**
- **61.75%:** Bizim makaleden kurduğumuz model, sabit 58 epoch, 3 koşu (±0.07). İki koşu son epoch'ta hâlâ iyileşiyordu.
- **63.21%:** Aynı model, durmasına izin verilince; en iyi epoch 107.
- **Sağdaki grafik:** Makalenin Şekil 7'si. Doğrulama doğruluğu 58'de hâlâ yükseliyor.
- **Mesaj:**
  - Onların durma kuralı kaybı izliyor; kayıp 40. epoch'tan sonra düzleşmiş (makale: "around 1.05").
  - Doğruluk ise hâlâ biraz artıyordu.
  - Bu bir hata değil; yöntemin bir özelliği. Sayı gerçek, ama doğruluğun hâlâ tırmandığı bir noktada.

**Okuma metni (EN)**

> One detail about the 58 epochs. Their stopping rule watches the validation loss, and the loss had flattened. But the validation accuracy, in their own figure on the right, is still rising at 58. When I trained our rebuild for a fixed 58 epochs, I got 61.75 percent. Letting it continue, it peaked at epoch 107 at 63.21. So their number is real, but it sits where the accuracy was still climbing.

---

## (Sonuçlar gelirse) — Onların kodu, bizim koşularımız

**Slaytta ne var, ne demek**
- Colab görevi `official_2016`: onların kodu, onların bölmesi, 3 farklı başlangıçla.
- Ortalama ± sapma, her koşunun durduğu epoch, bizim eğitim eğrimiz.
- Bu slayt sadece sonuç dosyası repoya geldiyse görünür.

**Okuma metni (EN)**

> And this is their own code, run by us on Colab, three times on one split. These are the numbers it gives.

---

## Slayt 17 — Kendi tablolarının desteklemediği üç iddia

**Slaytta ne var, ne demek**
1. **"Best at low SNR":**
   - Tablo 5'te CNN-BiLSTM-DNN −10 dB'de (27.73 vs 25.32) ve −8 dB'de (39.55 vs 37.45) daha yüksek.
   - MCLDNN −6 dB'de daha yüksek (56.32 vs 55.26).
2. **"Highest precision":** Tablo 6'da sadece BiLSTM'in precision'ı daha yüksek (0.7676 vs 0.7536).
3. **"MCLDNN: 406 M parameters":**
   - Gerçek sayı 406,199, üç bağımsız kaynakta aynı.
   - ICRNNA (0.79 M) onun yaklaşık iki katı.

**Okuma metni (EN)**

> Reading their tables closely, three statements in the text don't match their own numbers. Best at low SNR: in their Table 5, other models are higher at minus 10, minus 8 and minus 6 dB. Highest precision: in Table 6, the LSTM-only model is higher. And MCLDNN is listed with 406 million parameters; it has 406 thousand. So ICRNNA is about twice the size of MCLDNN, not smaller.

---

## Slayt 18 — Makalenin söylemedikleri

**Slaytta ne var, ne demek**
1. **Tek koşu:** Kod bunu doğruluyor.
   - Bizim 3-seed ölçümlerimizde modelden modele 0.2 ile 1.1 puan arası oynama var.
   - Ayrıca kodlarındaki bölme her çalıştırmada değişiyor; sınıf listesi bir Python `set`'inden okunuyor (`OFFICIAL_CODE.md`).
2. **Baselines:** Diğer modellerin sayıları yeniden eğitimle mi elde edildi, kendi makalelerinden mi alındı, yazmıyor.
3. **SNR etiketi:** Gerçek SNR gibi kullanılıyor.
   - Bizim ölçümümüzde etiket 0 dB'de QPSK 5.1 dB, QAM16 14.0 dB.
   - Yani yaklaşık 9 dB fark var.

**Okuma metni (EN)**

> And three things the paper doesn't tell us. It reports one training run, with no spread; across seeds we see up to a point of variation, so a half-point lead may not be real. It doesn't say whether the other models were retrained or copied from their papers. And it treats the SNR label as a true SNR; in this data, at label 0 dB, QAM16 has about 9 dB more signal than QPSK.

---

## Slayt 19 — Her model için tek protokol

**Slaytta ne var, ne demek**
- **Deney:** Beş modeli aynı protokolle eğittik: aynı veri, aynı bölme, 3 seed.
- **"their Table 3":** Makalenin bu modeller için verdiği sayı.
  - MCLDNN: makalede 61.42, bizde 61.41.
  - PET-CGDNN: makalede 60.58, bizde 60.98.
- **ICRNNA satırı:** Esmer'in versiyonu (62.2%). Makaleden kurduğumuz model 63.21%.

**Okuma metni (EN)**

> So we trained five models ourselves under one protocol: same data, same split, three seeds. Our MCLDNN and PET-CGDNN numbers match their table within half a point. ICRNNA is first, but the top four are within about one point, and above 10 dB they all stop near 91 percent.

---

## Slayt 20 — Özetle

**Okuma metni (EN)**

> To sum up. ICRNNA is a CNN, then a bidirectional LSTM, then attention, then a small classifier, and the LSTM does most of the work. Its 63.24 percent is reproducible: we got 63.21 from the paper alone. Its lead over other models is about one point. For us, it is a solid backbone, not a clearly better one. Thank you.

---

## Genel tavsiyeler

- **Önce modeli anlat, sonra eleştir.** Moshe "anlamak istiyorum" dedi. 4–9. slaytlarda acele etme; her blokta "giren ne, çıkan ne, ne işe yarıyor" sırasını koru.
- **Boyutları göster:** 2 × 128 → 64 × 128 → 64 × 256 → 256 → 11. Moshe bir çerçevenin modelin içinden nasıl geçtiğini takip edebilsin.
- **Temsilî çizimleri öyle söyle.** Filtre penceresi, attention çubukları ve olasılık çubukları açıklama amaçlı; veri değil.
- **Bilmediğini söyle.** Filtrelerin ne öğrendiğine ve attention ağırlıklarına henüz bakmadık. Moshe sorarsa "we haven't looked yet; it's a good next step" de.

# Makale sunumu çalışma metni — `moshe_paper_icrnna.pptx`

Makale: El-Haryqy, Kharbouche, Ouamna, Madini, Zouine. *Improved automatic modulation recognition using deep learning with additive attention.* Results in Engineering 26 (2025) 104783. Ibn Tofail Üniversitesi, Kenitra, Fas. Nisan 2025'te yayımlandı, açık erişim (CC BY-NC 4.0).

**Bu sunumun amacı:** Moshe makaleyi anlamak istiyor ve sade anlatım istiyor. O yüzden sunum ikiye ayrıldı:
- **Ana akış (slayt 1–13, yaklaşık 6 dakika):** Her slaytta tek düz mesaj; slaytlarda makine öğrenmesi jargonu yok. Model "dört adım" olarak anlatılıyor: parçalara bak, sırayla oku, ağırlıklandır, karar ver.
- **Yedek (slayt 14–25):** Her adımın ayrıntısı, eğitim tarifi, onların kodu, "58 epoch", tablolarıyla uyuşmayan iddialar. Sadece Moshe sorarsa açılır.

**Slayt 10**, Colab'daki `official_2016` sonuçlarıyla dolu: onların kodu, bizim koşumuz, %62.71 (1 koşu; 2 koşu daha sürüyor). Yeni koşular gelince slayt ve not kendiliğinden güncellenir.

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

> Today I present the paper our detector comes from: ICRNNA, from a group in Morocco, published in 2025. I'll keep it simple: what it does, how it works, what it reports, and whether it holds up.

---

## Slayt 2 — Makale tek cümlede

**Slaytta ne var, ne demek**
- **Tek cümle:** Bir sinir ağı, bir radyo sinyalinin 128 örneğine bakıp 11 modülasyondan hangisiyle gönderildiğini söylüyor.
- **63.24%:** RML2016.10a'da doğru oranı; makalenin karşılaştırdığı 12 model içinde en iyisi (Tablo 3).
- **"our detector":** Bizim dedektörümüz bu tasarım üzerine kurulu. Moshe'nin "bu model nereden çıktı" sorusunun cevabı.

**Okuma metni (EN)**

> The paper in one sentence: a neural network looks at 128 samples of a radio signal and says which of 11 modulations sent it. They report 63.24 percent right on this dataset, the best of the twelve models they compare. Our own detector is built on this design.

**Olası sorular**
- *"Is it a well-known journal?"* — Results in Engineering is a broad, open-access Elsevier engineering journal, not a signal-processing one. That is one reason we checked the number ourselves.

---

## Slayt 3 — Veri: model neyi görüyor?

**Slaytta ne var, ne demek**
- **Veri seti:** RML2016.10a, bizim kullandığımızla aynı. 11 modülasyon, 20 SNR etiketi (−20…18 dB), 220,000 çerçeve.
- **Sağdaki resim:** Bir çerçevenin I (turuncu) ve Q (mavi) bileşenleri. Biz çizdik, açıklama amaçlı.
- **2 × 128:** Modelin tek girdisi. 2 satır (I ve Q) × 128 örnek. Model başka hiçbir şey görmüyor: frekans bilgisi, SNR etiketi, ek bir özellik yok.
- **RML2016.10b:** Makale ayrıca daha büyük olan 10b'yi de test ediyor (65.39%). Biz 10a ile kalıyoruz.

**Okuma metni (EN)**

> The data is the same dataset we use, RML2016.10a: eleven modulation types, twenty noise levels, 220,000 frames. Each frame is 128 samples of I and 128 of Q. That is the only input; the network sees nothing else.

**Olası sorular**
- *"Does the model see the SNR?"* — No. Only the 2 × 128 array. The SNR label is used only afterwards, to sort the test results.

---

## Slayt 4 — Nasıl çalışıyor: dört adım

**Slaytta ne var, ne demek**

Modelin tamamı dört adım. Slaytta düz adları büyük, teknik adları küçük yazıyor.

1. **Look at short pieces (CNN):**
   - 3 örnek uzunluğunda küçük filtreler sinyal boyunca kayıyor; katsayıları öğrenilen FIR filtreler gibi.
   - Sonunda sinyal 64 kısa parçaya dönüşüyor; her parça yaklaşık 6 örneği, bir sembolden kısa bir kısmı özetliyor.
2. **Read them in order (LSTM, iki yönlü):**
   - 64 parçayı sırayla okuyor ve parçadan parçaya bir hafıza taşıyor; bir kez ileri, bir kez geri.
   - Sonunda her parça bütün çerçeveyi "biliyor".
   - Modelin sayılarının %83'ü burada.
3. **Weigh the pieces (attention):** Her parçaya bir ağırlık veriyor (toplamları 1) ve ağırlıklı ortalamasını alıyor: çerçevenin tek bir özeti.
4. **Decide (dense katmanlar):** 11 skor, her modülasyon için bir tane; en büyüğü cevap.

- **Alttaki satır:**
  - Giriş 128 örnek (I ve Q), çıkış 11 skor.
  - 794,188 öğrenilen sayı. Bu, onların kodunun saydığı sayı, BatchNorm istatistikleri dahil; makale 0.79 M diyor.
- **Ayrıntılar** yedek slaytlarda (15–19): her adımın boyutları, formülleri, parametreleri.

**Okuma metni (EN)**

> How does it work? Four steps. One: it looks at short pieces of the signal, with small filters that slide along it. Two: it reads those pieces in order, forwards and backwards, carrying a memory from piece to piece. Three: it weighs the pieces and takes a weighted average. Four: it gives eleven scores, one per modulation, and the largest wins. Most of the network, 83 percent, is in step two. That is the heart of it.

**Olası sorular**
- *"What do the filters in step 1 learn?"* — We haven't looked. Typically they respond to changes between neighbouring samples: phase jumps and amplitude steps.
- *"What is the memory in step 2?"* — 128 numbers per direction, updated piece by piece. What they encode is learned; we haven't interpreted them.
- *"Is step 3 the attention of ChatGPT?"* — No. It is a single weighted average over the 64 pieces, the 2014 form of attention.
- *"Why is step 2 the heart of it?"* — Telling QAM16 from QAM64 needs many symbols together; step 1 sees less than one symbol, step 2 connects all of them. Their own experiment (slide 9) agrees.

---

## Slayt 5 — Nasıl eğitildi, nasıl test edildi

**Slaytta ne var, ne demek**
- **Bölme:** 220,000 çerçeve üçe bölünüyor.
  - **154,000 (%70):** Doğru cevaplarıyla birlikte modele gösteriliyor; model bunlardan öğreniyor.
  - **33,000 (%15):** Ne zaman durulacağına karar vermek için (doğrulama).
  - **33,000 (%15):** Kenara ayrılıyor, öğrenmede hiç kullanılmıyor. Bütün sonuçlar bunlar üzerinden sayılıyor (test).
- **"58 passes":** Bütün öğrenme çerçevelerinin üzerinden 58 kez geçilmiş (epoch). Bu seçilmiş bir sayı değil: model iyileşmeyi bırakınca eğitim kendiliğinden durmuş.
- **"Done once":** Tek bölme, tek eğitim, tek sayı.
- **Teknik ayrıntılar** yedek slayt 20'de: Adam, öğrenme oranı, batch ve durma kuralı.

**Okuma metni (EN)**

> How was it trained and tested? The frames are split three ways. 154,000 frames, with the right answer, to learn from. 33,000 to decide when to stop learning. And 33,000 kept aside, never used for learning: every result is counted on those. It stopped by itself after 58 passes over the data, and they did it once.

**Olası sorular**
- *"How does it learn?"* — It is shown a frame, it gives 11 scores, those are compared with the right answer, and all its numbers are nudged a little to make the right answer more likely. Repeated over all 154,000 frames, 58 times.
- *"Why keep 33,000 to decide when to stop?"* — If you decide on the test frames, you are choosing the number you report. The stopping decision uses a separate set.

---

## Slayt 6 — Ana tablo

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

> Their main result. ICRNNA is first, at 63.24 percent. The next model is half a point behind. The well-known older ones are two to three points behind.

**Olası sorular**
- *"Were the other models retrained?"* — The paper doesn't say (backup slide 25).

---

## Slayt 7 — SNR'a göre doğruluk

**Slaytta ne var, ne demek**
- **Grafik:** Makalenin Şekil 8'i.
- **9%:** −20 dB'de doğruluk. 11 sınıfta şans 1/11 = 9.1%.
- **~93%:** 0 dB'den itibaren. Makale "5–15 dB'de 93–94%" diyor.
- **Not:** Buradaki "SNR" veri setinin etiketi, ölçülmüş SNR değil (`NOISE_2016.md`).

**Okuma metni (EN)**

> Accuracy against noise. At the noisiest level it is right 9 percent of the time, which is guessing among eleven. It improves quickly between minus 10 and 0 dB, and from 0 dB up it is right about 93 percent of the time.

---

## Slayt 8 — 0 dB'de hâlâ nerede hata yapıyor?

**Slaytta ne var, ne demek**
- **Matris:** Makalenin Şekil 11'indeki ICRNNA paneli, 0 dB. Genel doğruluk 92.21%.
- **57.9%:** WBFM çerçevelerinin AM-DSB denme oranı; WBFM'e doğru denen 41.4%.
  - Literatürdeki bilinen açıklama: WBFM'lerin bir kısmı sessiz sesten üretilmiş, AM-DSB'ye benziyor.
  - Bunu biz ölçmedik.
- **~6%:** QAM16'nın 5.7%'si QAM64, QAM64'ün 6.1%'i QAM16 denmiş.

**Okuma metni (EN)**

> Where does it still make mistakes, at 0 dB? Two places. WBFM is called AM-DSB more than half the time. And QAM16 and QAM64 are mixed up, about 6 percent each way. These are the same two problems we see in our own runs.

---

## Slayt 9 — Hangi adım önemli?

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

> Which step matters? They removed steps one at a time. Step one alone: 59.9. Step two alone: 63.1. All four: 63.2. So step two, reading in order, does almost all the work; the other steps add a fraction of a point.

**Olası sorular**
- *"Then why keep the CNN and attention?"* — By their own numbers, they add 0.1 points together. The paper doesn't discuss it.

---

## Slayt 10 — Onların kodu, ve verdiği sonuç

**Slaytta ne var, ne demek**
- **Nisan 2025:** Makale, kodsuz yayımlandı.
- **Ağustos 2026:** Yazarlar kodu GitHub'a koydu. Eğitilmiş ağ (ağırlıklar) yok; sadece eğitim kodu var.
- **Sağdaki kutu:** Onların kodunu, hiçbir şeyini değiştirmeden, Colab'da biz çalıştırdık.
  - **Şimdilik 1 koşu:** %62.71. Makale %63.24 diyor; 0.53 puan altında.
  - **Eğitim onlarınkine benziyor:** Kendi kendine epoch 61'de durdu (en iyi epoch 55). En iyi epoch'ta doğrulama doğruluğu %63.60; makalenin Şekil 7'si 58. epoch'ta %63.59.
  - **0 dB'de:** %90.5 (makale %92.2). Tek koşu, yaklaşık 1,650 karar.
  - **Yarım puan ne demek:** Tek koşudan bir şey söylemez. Test setinin kendi gürültüsü ±0.3 puan; bizim modellerde tohumdan tohuma fark 0.5–1.1 puan.
  - **Diğer 2 koşu:** Colab'da sürüyor (koşu başına ~2.5 saat). Gelince kutu ortalama ± sapma gösterir.
- **Koddan öğrendiklerimiz** (makale aynı, 58 bir seçim değil, tek koşu) yedek slayt 21'de; kod satırları bu sayfanın sonundaki "Yedek" bölümünde.

**Okuma metni (EN)**

> The paper came out without code. In August this year the authors put their code on GitHub, but not the trained network itself. We ran their code ourselves, once so far. It gives 62.71 percent, against their published 63.24: 0.5 points lower. Training behaved like theirs: it stopped by itself at epoch 61, with validation accuracy at 63.6, as in their paper. Half a point is the size of the run-to-run variation we see, so the next two runs will tell us whether this is just that.

**Olası sorular**
- *"Where is it?"* — GitHub, account noureddine395; the README names the five authors, and it is archived on Zenodo.
- *"Why didn't you use their code from the start?"* — It didn't exist when the paper came out; it appeared sixteen months later.
- *"So their number doesn't reproduce?"* — Not from one run. Half a point is the size of the run-to-run variation, and training behaved like theirs: validation accuracy 63.6, as in their figure. Two more runs will settle it.
- *"Why would it be lower?"* — A different random split and a different random start; their code draws a new split each time it's run. We changed nothing.

---

## Slayt 11 — Sayıları gerçek mi?

**Slaytta ne var, ne demek**
- **63.24%:** Makalenin sayısı.
- **63.21%:** Bizim, modeli sadece makalenin metninden yeniden kurup eğittiğimizde aldığımız sayı. Bu, onların kodu yayımlanmadan önceydi.
- **Alt satır:**
  - Bizim modelin en iyi sonucu 107. geçişte (epoch) geldi.
  - Makaledeki gibi 58'de durunca 61.75% kalıyordu.
  - Yani sayı gerçek, ama bizim kurduğumuz modelde daha uzun eğitim gerekti.
- **Ayrıntı:** Yedek slayt 22 ("58 epochs") ve 23 (yeniden kurduğumuz modelle onların kodunun karşılaştırması).

**Okuma metni (EN)**

> Is their number real? Before their code existed, I rebuilt the network from the paper's description alone. It gets 63.21 percent, against their 63.24. So yes, the result can be reproduced. It needed longer training than their 58 passes; ours was best at pass 107.

**Olası sorular**
- *"Why did yours need longer?"* — Their stopping rule watches the validation loss, which had flattened by 58, while the accuracy was still creeping up (their own Fig. 7 shows it). Ours watched the accuracy and kept going. Details on backup slide 22.

---

## Slayt 12 — Gerçekte ne kadar iyi?

**Slaytta ne var, ne demek**
- **Deney:** Beş yayımlanmış modeli kendimiz, aynı şekilde eğittik: aynı veri, aynı bölme, her biri 3 tekrar.
- **Sütunlar:**
  - "right, all noise levels": bütün SNR seviyelerinde doğru oranı.
  - "right, 10 dB and above": temiz sinyallerde doğru oranı.
- **Mesaj:**
  - ICRNNA birinci, ama ilk dört model 1.25 puan içinde.
  - Temiz sinyallerde dördü de ~91%'de.
  - VT-CNN2 (2016'nın ilk modeli) belirgin şekilde geride.
- **Makalenin sayılarıyla uyum:**
  - MCLDNN: makalede 61.42, bizde 61.41.
  - PET-CGDNN: makalede 60.58, bizde 60.98.
  - Yani eğitimimiz literatürle tutarlı.
- **ICRNNA satırı:** Bizim çalışan versiyonumuz (62.2%), bütün diğer sonuçlarımız bununla. Makaleden birebir kurduğumuz model 63.21%; o da ilk dördün içinde birinci olurdu.

**Olası sorular**
- *"Why is the 2016 model so far behind?"* — It is a plain convolutional network with no step that reads in order; the others all have one.

**Okuma metni (EN)**

> How much better is it, really? We trained five published models the same way: same data, same split, three repeats each. ICRNNA is first, but only by about one point. At 10 dB and above, the best four are all about 91 percent.

---

## Slayt 13 — Özetle

**Okuma metni (EN)**

> To sum up. It is a neural network in four steps, and step two, reading in order, does most of the work. Its 63.24 percent is real: we get 63.21 from the paper alone. And it is about one point better than older models, not a big jump. For us, it is a solid detector to build on. Thank you. I have more detail on each step if you'd like.

---

## Slayt 14 — Yedek bölüm

**Slaytta ne var, ne demek**
- Buradan sonrası sadece Moshe ayrıntı sorarsa:
  - 15: Modele kuşbakışı, boyutlarla.
  - 16–19: Her adım ayrı ayrı.
  - 20: Eğitim tarifi.
  - 21: Kodun söyledikleri.
  - 22: "58 epoch".
  - 23: Yeniden kurduğumuz model ve onların kodu.
  - 24: Tablolarıyla uyuşmayan iddialar.
  - 25: Makalenin söylemedikleri.
- Sormazsa göstermeye gerek yok; sunum 13. slaytta biter.

**Okuma metni (EN)**

> Backup slides with the details, if you want to go deeper.

---

## Slayt 15 — Yedek: modele kuşbakışı

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

## Slayt 16 — Yedek: adım 1, CNN

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

## Slayt 17 — Yedek: adım 2, BiLSTM

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

## Slayt 18 — Yedek: adım 3, attention

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
- *"Does it help?"* — By their own ablation, less than half a point (slide 9).

---

## Slayt 19 — Yedek: adım 4, karar

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

## Slayt 20 — Yedek: eğitim tarifi

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
- *"Why stop on the loss and not the accuracy?"* — It is the common Keras default. Loss and accuracy can disagree: here the loss had flattened while accuracy was still creeping up (slide 19).
- *"How were the hyperparameters chosen?"* — Section 4.1: grid search over learning rate, dropout and L2; batch size and epochs by hand.

---

## Slayt 21 — Yedek: kod bize ne söylüyor?

**Slaytta ne var, ne demek**
1. **Model makaledekiyle aynı:** Aynı katmanlar. Keras'ın saydığı 794,188 sayı, makale 0.79 M diyor.
2. **"58 epoch" bir seçim değil:**
   - Kodda en fazla 100 epoch var.
   - Doğrulama kaybı 6 epoch iyileşmezse eğitim kendiliğinden duruyor ve en iyi ağırlıklara dönüyor.
   - Onların koşusunda bu 58'de olmuş.
3. **Tek koşu:** Kodda tekrar yok; bir bölme, bir eğitim, bir sayı.
- **Mesaj:** Tek koşu tek sayı verir; tekrar edilse ne kadar oynar? Bir sonraki slayt bunu soruyor.

**Okuma metni (EN)**

> What does their code tell us? Three things. First, the model in the code is the model in the paper: same layers, 794 thousand numbers. Second, the 58 epochs in the paper were not a choice: the training stops by itself when it stops improving, and for them that happened at 58. Third, they trained it once: one split of the data, one run, one number. And one run leaves a question: how much would that number move if you ran it again?

---

## Slayt 22 — Yedek: "58 epoch" ne demek?

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

## Slayt 23 — Yedek: yeniden kurduk; onların kodu da doğruluyor

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

## Slayt 24 — Yedek: kendi tablolarının desteklemediği üç iddia

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

## Slayt 25 — Yedek: makalenin söylemedikleri

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

## Yedek — Kodun ayrıntıları (slaytta yok)

Moshe ayrıntı sorarsa diye: kodun içinde ne var, satır satır. Tam karşılaştırma `OFFICIAL_CODE.md`'de.

### Ne yayımladılar?

- **Depo adresi:** Depo adı "mproved-…" diye başlıyor; "Improved"ın baş harfi eksik, onların yazım hatası.
- **İçinde olanlar:**
  - `rml2016_10a.py` ve `rml2016_10b.py`: her veri seti için bir script.
  - TensorFlow/Keras, MIT lisansı.
  - Zenodo'da arşivli (10.5281/zenodo.21854712).
- **İçinde olmayanlar:**
  - Eğitilmiş ağırlıklar. Script `best_model.keras` dosyasını kendi bilgisayarına kaydediyor, ama depoda yok.
  - Birden fazla koşu; script tek koşu yapıyor.
- **Beş adım:** Script bir kez çalıştırıldığında sırayla şunları yapıyor:
  1. pickle dosyasını okuyor;
  2. normalize ediyor;
  3. karıştırıp bölüyor;
  4. eğitiyor;
  5. test edip grafikleri çiziyor.
- **Kimin hesabı:** Hesapta isim yok. README ve atıf dosyası makalenin beş yazarını veriyor. Aynı hesapta iki AMR deposu daha var (MCSBA ve FPGA üzerinde AMR).

**Olası sorular**
- *"How do you know it is really theirs?"* — The README and citation file name the five authors and the paper's DOI, and it is archived on Zenodo under their names. The account itself shows no name, so I can't prove it, but nothing suggests otherwise.
- *"Why did they publish so late?"* — I don't know. The paper only said data are available on request.


### Model, onların kodunda

- **Kaynak:** `build_improved_model` fonksiyonunun satırları. Aşağıda batch normalization, dropout ve L2 argümanları olmadan.
- **Satır satır:**
  - `Conv1D(64, 3)` → 64 filtre, 3 örneklik. `padding='same'` uzunluğu korur.
  - `Conv1D(128, 3)` → 128 filtre.
  - `MaxPooling1D(2)` → 128 adım 64 olur.
  - `Bidirectional(LSTM(128, return_sequences=True))` → çift yönlü LSTM. `return_sequences=True` her adımın çıktısını verir, sadece sonuncusunu değil.
  - İkinci LSTM'in çıktısı `lstm_out` diye ayrıca saklanıyor; attention'a "value" olarak gidiyor.
  - `attention(x, lstm_out)` → x, batch norm ve dropout'tan geçmiş hali (query); lstm_out ham hali (value).
  - `Dense(128)`, `Dense(64)` → karar katmanları.
  - `Dense(num_classes, activation='softmax')` → 11 olasılık.
- **Ayrıca:** Girişte `Lambda(transpose)` 2 × 128'i 128 × 2'ye çeviriyor; Keras zaman eksenini önce istiyor.

**Olası sorular**
- *"Where is the L2 regularization?"* — As an argument on each Conv1D, LSTM and Dense layer: `kernel_regularizer=l2(1e-4)`. I left it out of the slide to fit.


### Attention, onların kodunda


`ImprovedAdditiveAttention.call()` satırları:

1. `score = self.V(tanh(self.W1(query) + self.W2(value)))`
   - W1 ve W2: 256 sayıyı 128'e indiren öğrenilen katmanlar (bias'sız).
   - V: 128'i tek bir puana indiriyor.
   - Her zaman adımı için bir puan.
2. `score * self.scale`: Öğrenilen tek bir sayı, başlangıçta 1. Makalenin "if scaling is enabled" dediği şey bu.
3. `softmax(score, axis=1)`: Zaman ekseni (64 adım) üzerinde; ağırlıkların toplamı 1.
4. `dropout`: Eğitimde ağırlıkların %10'u rastgele sıfırlanıyor.
5. `attention_weights * value`: Her adım kendi ağırlığıyla çarpılıyor.
6. `reduce_sum(..., axis=1)`: 64 adım toplanıyor → 256 sayılık tek vektör.
7. `layer_norm`: Sonuç vektörü normalize ediliyor.

- **Alttaki kutu:** Klasik attention'da "query" başka bir yerden gelir (örneğin çeviride, üretilen kelimeden). Burada query ve value aynı LSTM çıktısının iki hali. Yani her adım kendi içeriğine göre puanlanıyor; bu bir "attention pooling".

**Olası sorular**
- *"Is the attention looking for relations between steps?"* — No. Each step's score depends only on that step. It decides how much each step counts in the average, not how steps relate to each other.


### Eğitim, onların kodunda

- **İlk `train_test_split`:** Verinin %30'unu ayırıyor; kalan %70 öğrenme. `random_state=2016` bölmeyi sabitlemeye çalışıyor (ama bkz. aşağıdaki "Koddaki iki sorun").
- **İkinci `train_test_split`:** Ayrılan %30'u ikiye bölüyor: %15 doğrulama, %15 test.
- **`Adam(learning_rate=0.001, clipnorm=1.0)`:**
  - Adam optimizer.
  - `clipnorm=1.0`: Her güncellemede gradyanın büyüklüğü 1 ile sınırlanıyor.
  - Makalede yazmıyor.
- **`ReduceLROnPlateau`:** Doğrulama kaybı 5 epoch iyileşmezse öğrenme oranı ×0.2 (en düşük 1e-6).
- **`EarlyStopping`:** Doğrulama kaybı 6 epoch iyileşmezse durdur ve en iyi ağırlıklara dön.
- **`fit(epochs=100, batch_size=32)`:** En fazla 100 epoch, 32'lik batch'ler.
- **Mesaj:** Makaledeki "58 epoch" bu kuralın tetiklendiği yer, seçilmiş bir sayı değil.


### Kodun makalede olmayan şeyleri söyledikleri


| | makale | onların kodu |
|---|---|---|
| epoch | 58 | en fazla 100; doğrulama kaybı 6 epoch iyileşmezse dur |
| optimizer | adı yok | Adam + gradient clipping |
| girdi ölçekleme | "normalization" | zaman indeksi başına, bütün veri üzerinden |
| pooling | belirsiz | bir max-pool, 2. konvolüsyondan sonra |
| attention ölçeği | "if enabled" | öğrenilen, başlangıç 1 |
| koşu sayısı | yazmıyor | bir koşu, bir bölme |
| ağırlıklar | yok | hâlâ yok |

- **"Zaman indeksi başına" normalizasyon:**
  - Keras'ın `Normalization` katmanı 128 zaman noktasının her biri için ayrı bir ortalama ve varyans hesaplıyor.
  - Bu hesap bütün çerçeveler ve hem I hem Q üzerinden yapılıyor.
  - Bizim kodumuz ise her çerçeveyi kendi gücüne bölüyor.


### Koddaki iki sorun

1. **Bölme her çalıştırmada değişiyor:**
   - `preprocess_data` sınıf listesini `set(...)` ile kuruyor. Python'da string'lerden oluşan bir set'in sırası her süreçte değişiyor (hash randomization).
   - Veri o sırayla birleştiriliyor, sonra seed 2016 ile karıştırılıyor. Karıştırmadan önceki sıra farklı olduğu için, aynı seed farklı bir bölme veriyor.
   - Bunu sahte bir veri dosyasıyla denedik: iki farklı süreç iki farklı bölme parmak izi verdi (`54c27115e056` ve `1bf5e5673d36`).
   - README "splits are reproducible" diyor; bu, `PYTHONHASHSEED` sabitlenmedikçe doğru değil.
   - Etiket numaraları değişmiyor, çünkü `LabelEncoder` isimleri alfabetik sıralıyor.
2. **Ölçekleme test verisini de görüyor:** Normalizasyon istatistikleri bölmeden önce bütün 220,000 çerçeve üzerinden hesaplanıyor. Etkisi muhtemelen küçük, ama bir sızıntı.
- **Bizim çözümümüz:** Colab'daki sarmalayıcı `PYTHONHASHSEED`'i sabitliyor ve bölmenin parmak izini kaydediyor; bütün koşular aynı bölmeyi kullanıyor.

**Olası sorular**
- *"Does this change their 63.24%?"* — Not by itself. It means their single number came from one particular split that cannot be recreated exactly. Our three runs of their code will show how much the number moves.

---

## Genel tavsiyeler

- **Önce modeli anlat, sonra eleştir.** Moshe "anlamak istiyorum" dedi. 4–9. slaytlarda acele etme; her blokta "giren ne, çıkan ne, ne işe yarıyor" sırasını koru.
- **Boyutları göster:** 2 × 128 → 64 × 128 → 64 × 256 → 256 → 11. Moshe bir çerçevenin modelin içinden nasıl geçtiğini takip edebilsin.
- **Temsilî çizimleri öyle söyle.** Filtre penceresi, attention çubukları ve olasılık çubukları açıklama amaçlı; veri değil.
- **Bilmediğini söyle.** Filtrelerin ne öğrendiğine ve attention ağırlıklarına henüz bakmadık. Moshe sorarsa "we haven't looked yet; it's a good next step" de.

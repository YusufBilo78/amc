# Sunum çalışma metni — `moshe_2016_simple.pptx` (14 slayt)

Her slayt için üç bölüm var:

- **Slaytta ne var, ne demek:** Slayttaki her sayının, kutunun ve resmin Türkçe açıklaması. Sayının nereden geldiği de yazıyor.
- **Okuma metni (EN):** Sunumda söyleyeceğin İngilizce metin, slayt notlarındakiyle aynı. Kısa cümleler, günlük kelimeler, slayt başına 20–40 saniye; hepsi 6–7 dakika. Her slayt aynı sırayla gidiyor: slaytın cevapladığı soru → bir somut resim veya sayı → tek mesaj. Ayrıntılar slaytta ve aşağıdaki açıklamada; sorulursa oradan cevaplarsın.
- **Olası sorular:** Moshe'nin sorabileceği sorular ve kısa cevapları.

Slaytlar sadeleştirildi: büyük yazı, slayt başına tek mesaj. "Slaytta ne var" bölümleri slaytta artık yazmayan ayrıntıları da anlatıyor; onlar soru gelirse diye burada.

Buradaki her sayı repodaki bir sonuç dosyasından geliyor (`deck/deck2016_data.json`, `DETECTOR.md`, `NOISE_2016.md`, `SNR_2016.md`, `LITERATURE_CHECK.md`). Burada olmayan bir sayıyı sunumda söyleme.

---

## Ezber kartı — en önemli sayılar

| sayı | ne | nereden |
|---|---|---|
| 11 / 20 / 1,000 / 220,000 | modülasyon tipi / SNR etiketi / hücre başına çerçeve / toplam çerçeve | RML2016.10a |
| 128 örnek = 16 sembol | bir çerçeve, sembol başına 8 örnek | veri seti |
| 785,476 | 4 sınıflı modelin öğrenilen parametre sayısı (11 sınıfta 786,379) | `DETECTOR.md` |
| %84 | parametrelerin BiLSTM'deki payı (659,968) | `DETECTOR.md` |
| 63.21 vs 63.24 | makaleden birebir kurduğumuz model, yakınsayana kadar eğitildi, makaleye karşı | `icrnna_faithful_e150_results.json` |
| 70 / 15 / 15, 14 tekrar | eğitim / doğrulama / test, 14 farklı bölme | `train_backbone.py` |
| %94.2 | 10 dB ve üstünde doğru (42,000 karardan 2,421'i yanlış) | 4 sınıf, 14 seed |
| 448 / 395 | 10 dB'de 8,400 karardan yanlış / bunlardan QAM16↔QAM64 olan | `SNR_2016.md` |
| 1.9 dB | etikette 1 dB = gerçek SNR'da ~1.9 dB değişim | `rml2016_measured_snr.json` |
| +15 dB | aynı etikette QAM64'ün PSK'ya göre fazla sinyali | aynı dosya |
| %5.7 / 15.5 / 25.2 | +2 dB üstü yanlış oranı: 16 / 8 / 4 sembol | frame-length koşuları |
| ×2.7 / ×4.3 | 8 ve 4 sembolde hata artışı (6,443 ve 10,431, 2,421'e karşı) | aynı |
| %52.2 vs %24.8 | −20 dB'de QPSK recall ve precision | slayt 12 |
| %98.8 | −20 dB'de BPSK veya QPSK denen kararların payı | slayt 12 |
| 62.2 / 61.4 / 61.4 / 61.0 / 56.1 | ICRNNA / LSTM2 / MCLDNN / PET-CGDNN / VT-CNN2 genel doğruluk, 11 sınıf | slayt 13 |

---

## Sözlük — "Bu nedir?" soruları

Moshe en basit terimi de sorabiliyor: "bu nedir, bununla neyi kastettin?". Aşağıda slaytlarda geçen her terim var, geçtiği sırayla. Her birinde önce söyleyeceğin bir iki cümlelik İngilizce cevap, sonra parantez içinde Türkçe not.

**En önemlisi: recall.** Moshe bunu geçen sefer sordu, hazır ol.

> **Recall** — "Of the frames that really were QAM16, how many did it call QAM16? It is one row of the matrix. At 10 dB, 1,808 of 2,100 QAM16 frames: 86.1 percent. In detection terms it is the probability of detection for that class: P(say QAM16 | QAM16 was sent)."
>
> **Precision** — "When it says QAM64, how often is it right? It is one column. At 10 dB it said QAM64 2,236 times and was right 1,961 times: 87.7 percent. It is P(QAM64 was sent | it said QAM64)."
>
> **Why both?** — "Recall asks: did it find them? Precision asks: can I believe it when it says so? A detector that calls everything QPSK has 100 percent QPSK recall and 25 percent precision. That is exactly what happens at minus 20 dB."

- *"Is recall the same as accuracy?"*
  - For one class, yes: recall is that class's accuracy.
  - Overall accuracy is all correct over all frames. Our classes are equal in size, so overall accuracy is the average of the four recalls.
- *"Is 1 − precision the false-alarm rate?"*
  - No. The false-alarm rate is P(say QAM64 | it was not QAM64).
  - 1 − precision is the share of "QAM64" answers that were wrong. It depends on how often the other classes are sent.
  - (Moshe dedeksiyon teorisinden gelir; bu ayrımı bilmen iyi olur.)

### Sinyal

- **AMC (automatic modulation classification)** — "Looking at a received signal and saying which modulation sent it, without being told." *(Otomatik modülasyon sınıflandırma.)*
- **Modulation** — "The rule that turns bits into a radio wave: which bits change the phase, the amplitude or the frequency." *(Bitlerin dalgaya nasıl yazıldığı.)*
- **Digital / analog modulation** — "Digital sends symbols from a fixed set of points; analog, like FM or AM, follows a continuous signal such as audio." *(8 dijital, 3 analog sınıf.)*
- **Phase / amplitude** — "Amplitude is how strong the wave is; phase is where in its cycle it is, an angle." *(Genlik ve faz.)*
- **BPSK, QPSK** — "Phase-shift keying with 2 or 4 phases. All points have the same amplitude. One and two bits per symbol." *(Sadece faz değişiyor.)*
- **QAM16, QAM64** — "Quadrature amplitude modulation: points on a square grid of 16 or 64, so phase and amplitude both change. Four and six bits per symbol." *(Hem faz hem genlik.)*
- **Symbol** — "One point the transmitter sends, held for one symbol period. It is one 'letter' of the message." *(BPSK'da 2, QAM64'te 64 farklı harf var.)*
- **Constellation / grid** — "The set of all points a modulation is allowed to use, drawn on the I/Q plane." *(Slayt 7'deki gri çemberler.)*
- **Sample** — "One measurement of the signal at one moment: one I value and one Q value." *(Bir ölçüm anı.)*
- **I and Q** — "The received signal written as two numbers per sample: the in-phase part and the quadrature part. Together they give amplitude and phase." *(Turuncu ve mavi çizgiler.)*
- **Samples per symbol** — "Here 8. Each symbol lasts eight samples, so 128 samples are 16 symbols." *(128 / 8 = 16.)*
- **Frame** — "One example given to the detector: 128 samples in a row, cut from a longer signal." *(Bir çerçeve = bir karar.)*
- **Pulse shaping** — "A filter that makes the signal move smoothly from one symbol to the next, so it uses less bandwidth." *(Noktalar arasındaki yumuşak geçiş.)*
- **Different transmitter / domain gap** — "Train on signals from one source, test on signals from another. The accuracy you lose is the domain gap." *(Projenin 3. adımı.)*

### Gürültü ve kanal

- **Noise** — "Random disturbance added to the signal. Here it is white Gaussian noise added by the simulator." *(AWGN.)*
- **SNR** — "Signal-to-noise ratio: signal power divided by noise power." *(Sinyal gücü / gürültü gücü.)*
- **dB** — "Ten times the log of a power ratio. 10 dB is ten times the power, 20 dB a hundred times, 3 dB about double." *(Logaritmik ölçek.)*
- **SNR label** — "The number the dataset attaches to each frame, minus 20 to plus 18. It is a setting in their code, not a measured SNR." *(Slayt 8'in ana mesajı.)*
- **Noise amplitude vs power** — "Power is amplitude squared. Their code sets the amplitude, so one dB of label changes the noise power by about two dB." *(Ölçtüğümüz: 1.9.)*
- **Carrier frequency offset** — "The transmitter and receiver oscillators are not exactly at the same frequency, so the points slowly rotate." *(Taşıyıcı kayması.)*
- **Clock offset (sample-rate offset)** — "The receiver samples slightly faster or slower than the transmitter, so symbol timing drifts." *(Saat kayması.)*
- **Fading** — "The signal's strength and phase change over time because of movement and reflections." *(Sönümlenme.)*
- **Echoes / multipath** — "Delayed copies of the signal arrive through other paths and add to it." *(Yankılar, çok yollu yayılım.)*
- **Spectrum / band / out-of-band** — "The spectrum is the signal's power at each frequency. The band is where the signal is; out of band there is only noise, which is how I measured the noise." *(Slayt 8'deki ölçüm yöntemi.)*
- **Simulator, GNU Radio** — "GNU Radio is open-source software for building radio signal chains. DeepSig used it to generate the whole dataset; nothing was recorded over the air." *(Veri gerçek değil, simülasyon.)*
- **DeepSig, RML2016.10a** — "DeepSig is the company of Tim O'Shea, who made the RadioML datasets. 2016.10a is the version from 2016 that most papers use." *(RML = RadioML.)*

### Model

- **Detector / classifier** — "The program that takes a frame and returns one of the four names." *(Bizim ICRNNA.)*
- **Neural network** — "A function with many adjustable numbers. You show it examples with the right answer and it adjusts the numbers to make fewer mistakes." *(Sinir ağı.)*
- **Parameters ("numbers learned")** — "The adjustable numbers inside the model. Ours has 785,476." *(Ağırlıklar.)*
- **Convolution** — "A small filter slid along the signal, like an FIR filter whose taps are learned instead of designed." *(Moshe elektrik mühendisi; FIR benzetmesi işe yarar.)*
- **Pooling** — "Keeps the larger of each two neighbouring values, so the sequence gets half as long." *(128 → 64 → 32.)*
- **LSTM / BiLSTM** — "A layer that reads a sequence one step at a time and keeps a memory of what it has seen. Bi means it also reads it backwards." *(Modelin %84'ü burada.)*
- **Attention** — "It gives each of the 32 pieces a weight and takes the weighted average, so the model decides which parts to listen to." *(Ağırlıklı ortalama.)*
- **Dense layer** — "Every input connected to every output: a matrix multiplication plus a bias." *(Son iki katman.)*
- **Score / logit** — "One number per class at the end. The biggest one is the answer." *(Argmax.)*
- **Deterministic** — "Once trained, nothing is random. The same frame always gets the same answer." *(Test sırasında dropout kapalı.)*
- **ICRNNA** — "The model's name in the 2025 paper: Improved Convolutional Recurrent Neural Network with Attention." *(Açılımını ezberle.)*
- **Faithful build** — "My own version, written strictly from the paper, to check the paper's number." *(63.21 vs 63.24.)*

### Eğitim

- **Training** — "Adjusting the parameters so the model makes fewer mistakes on the training frames." *(%70.)*
- **Validation set** — "Frames used only to decide when to stop training. The model does not learn from them." *(%15.)*
- **Test set** — "Frames the model never saw. Every number I report is counted on them." *(%15.)*
- **Epoch** — "One pass through all the training frames." *(Bir tur.)*
- **Early stopping** — "Stop when validation accuracy has not improved for 20 epochs, and keep the best version." *(Patience 20.)*
- **Converged** — "It stopped improving by itself, before the epoch limit, so it was not cut short." *(Bütün koşular yakınsadı.)*
- **Seed / repeat** — "One run with its own random split and its own random starting point. We did 14 so that one lucky run cannot make the result." *(14 tekrar.)*

### Sonuçlar

- **Confusion matrix** — "A table: rows are what was sent, columns what the detector said. The diagonal is correct." *(Slayt 9.)*
- **Accuracy** — "Correct decisions divided by all decisions." *(94.2% = 1 − 2,421 / 42,000.)*
- **Decision** — "One frame, one answer. 42,000 decisions means 42,000 test frames." *(Her noktanın arkasında kaç karar var.)*
- **"At 10 dB and above"** — "Only frames with labels 10, 12, 14, 16 and 18." *(5 seviye.)*
- **Chance / a guess** — "With four classes, picking at random is right 25 percent of the time." *(1/4.)*
- **Sink** — "The class the detector falls back on when the frame carries no information." *(−20 dB'de BPSK ve QPSK.)*
- **Floor / ceiling** — "Where the curve goes flat: the error that more signal can no longer remove." *(16 sembolde %5.7.)*

### Literatür karşılaştırması

- **Overall accuracy** — "Accuracy over all twenty labels together, minus 20 to plus 18." *(Makalelerin verdiği sayı.)*
- **"Same parameter count, to the digit"** — "If I had built even one layer differently, the count would change. Matching it is the check that the model is really theirs." *(Portun doğru olduğunun kanıtı.)*
- **VT-CNN2** — "The 2016 convolutional network of O'Shea and colleagues, as rebuilt in the benchmark." *(En eski, en büyük.)*
- **LSTM2** — "Two LSTM layers that read amplitude and phase instead of I and Q. Rajendran 2018." *(Az parametre, iyi sonuç.)*
- **MCLDNN** — "A multi-channel CNN plus LSTM: it looks at I, Q and both together. Xu 2020." *(Çok kanallı.)*
- **PET-CGDNN** — "It first estimates and removes the phase offset, then classifies. Zhang 2021." *(En küçüğü, 71,871.)*

---

## Slayt 1 — Başlık

**Slaytta ne var, ne demek**

- **"AMC · DATA FUSION LAB":** Konu (Automatic Modulation Classification) ve laboratuvar.
- **"Telling four radio signals apart":** Bugünkü sunumun kapsamı. 11 sınıfın tamamı değil, Moshe'nin istediği dört sınıf: BPSK, QPSK, QAM16, QAM64.
- **Sağdaki animasyon (noise.gif):** Bir QAM16 çerçevesinin 16 sembolü. Gürültü arttıkça noktalar ızgara noktalarından uzaklaşıp dağılıyor. Bu resmi biz çizdik, açıklama amaçlı. Veri setinden alınmış bir çerçeve değil.

**Okuma metni (EN)**

> Today I'll open up our detector, step by step, on the four signals you asked about: BPSK, QPSK, QAM16 and QAM64.
>
> The picture is one QAM16 frame: sixteen points, spreading out as the noise grows. That's the whole talk: how many symbols, and how much noise.

**Olası sorular**

- *"Why only four classes?"* — Because you asked for the four-class box. It is the simplest case where something still goes wrong (the QAM pair). Slide 13 shows all eleven classes for the comparison with other models.

---

## Slayt 2 — AMC nedir, ben ne üzerinde çalışıyorum?

**Slaytta ne var, ne demek**

- **AMC:** Alıcı, tanımadığı bir radyo sinyalini yakalıyor ve "bu sinyal hangi modülasyonla gönderildi?" sorusuna cevap veriyor.
  - Slayttaki diyagram: sinyal → detector → "QPSK".
  - Kullanım alanları: spektrum izleme, girişim (interference) bulma, duyduğuna göre kendini ayarlayan radyolar (cognitive / adaptive radio).
- **Üç adım:**
  1. **Dedektör:** Standart, simülasyonla üretilmiş bir veri setinde (RML2016.10a, DeepSig) eğitilmiş bir sınıflandırıcı.
  2. **Kutuyu açmak (bugün):** Dedektör tam olarak ne yapıyor, verideki gürültü ne, nerede hata yapıyor?
  3. **Büyük soru (projenin asıl konusu):** Bu dedektörler başka bir vericiden gelen sinyalde neden doğruluk kaybediyor, bunu ne düzeltir? (Domain gap.)
- **"Today: only steps 1 and 2":** Bugün 3. adıma girmiyoruz.

**Okuma metni (EN)**

> AMC means automatic modulation classification. A receiver gets a signal and has to say how it was modulated. You need that to watch the spectrum, to find interference, and for radios that adapt.
>
> What I'm trying to do: these detectors do well on the data they were trained on, but they lose accuracy when the signal comes from a different transmitter. I want to find out why, and what fixes it.
>
> To get there, I first need a detector I understand completely. So: one, build it. Two, open it up. Three, test it on a different transmitter. Today is the first two.

**Olası sorular**

- *"What do you mean by a different transmitter?"*
  - The detector is trained on one simulated dataset. We test it on frames from an independently written generator: same modulations, different code, different impairments.
  - On 2016 the drop is small. In-domain is 0.948, cross-domain 0.920, on 5 shared classes.
  - Standard data augmentation closes it (0.969 / 0.968).
  - That is the third step, and not today.
- *"Is it real radio data?"* — No. Both datasets are simulated. Real SDR capture is on the list, when hardware and lab access allow. That is the largest weakness of the work, and I say it openly.

---

## Slayt 3 — Hangi veriyi kullanıyorum?

**Slaytta ne var, ne demek**

- **RML2016.10a:** DeepSig (O'Shea'nın grubu) 2016'da GNU Radio simülatöründe üretti. AMC makalelerinin çoğu bu veri setinde sonuç veriyor. Dosya: `RML2016.10a_dict.pkl`.
- **Sayılar:**
  - **11:** modülasyon tipi.
  - **20:** gürültü seviyesi (SNR etiketi), −20'den +18 dB'ye 2 dB adımla.
  - **1,000:** her (tip, seviye) hücresinde çerçeve.
  - **220,000:** toplam çerçeve (11 × 20 × 1,000).
  - **128:** bir çerçevedeki örnek. Sembol başına 8 örnek olduğu için 16 sembol.
- **Çipler:**
  - 8 dijital: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, GFSK, CPFSK.
  - 3 analog: WBFM, AM-DSB, AM-SSB.
  - Turuncu olanlar bugün kullanılan dört sınıf.
- **Faz ve genlik:**
  - BPSK ve QPSK sadece fazı değiştiriyor; bütün semboller aynı genlikte, bir çemberin üzerinde.
  - QAM16 ve QAM64 hem fazı hem genliği değiştiriyor; semboller kare bir ızgaranın üzerinde.
- **Bir çerçeve nedir:** 128 karmaşık sayı, yani I (in-phase) ve Q (quadrature) bileşenleri. Modele 2 × 128'lik bir dizi olarak giriyor.

**Okuma metni (EN)**

> The data is RML2016.10a, from DeepSig. It's simulated, and most papers use it.
>
> Eleven signal types, twenty noise levels, a thousand frames each. One frame is 128 samples: only sixteen symbols. Remember that.
>
> Today, the four in orange. BPSK and QPSK change only the phase; the QAMs also change the amplitude.

**Olası sorular**

- *"Why 2016 and not the newer 2018 dataset?"* — The group works on 2016, so results have to be comparable with yours. I have earlier results on 2018, but the current work is on 2016.
- *"Why is 16 symbols important?"* — QAM64 has 64 points. A frame of 16 symbols can show at most 16 of them. Slides 7 and 11.
- *"Are the frames independent?"* — Each frame is a separate slice of a simulated stream. I checked that the noise is not reused: 0 of 11,000 frames at −20 dB share a noise segment.

---

## Slayt 4 — Dedektör ne yapıyor?

**Slaytta ne var, ne demek**

- **Görev:** 128 örnek giriyor, dört etiketten biri çıkıyor. Kod: `model_zoo.backbone(4)`.
- **Ön işlem:** Her çerçeve önce birim ortalama güce ölçekleniyor. Böylece model sinyalin seviyesine bakarak sınıfı tahmin edemiyor.
- **Adım 1, "Look at short pieces":** 2 konvolüsyon katmanı, 25,792 parametre.
  - Conv1: 2→64 kanal, kernel 5.
  - Conv2: 64→128 kanal, kernel 3.
  - Her birinden sonra max-pool(2): uzunluk 128 → 64 → 32.
  - Sonuç: çerçeve 32 parçaya dönüşüyor. Her parça 12 ardışık örnekten hesaplanıyor (yaklaşık 1.5 sembol). Komşu parçalar 4 örnek (yarım sembol) aralıklı.
  - Bu adım yerel şekilleri yakalıyor: iki komşu sembol arasındaki faz ve genlik değişimi gibi.
- **Adım 2, "Read the pieces in order":** 2 katmanlı çift yönlü LSTM (BiLSTM) ve BatchNorm, 659,968 parametre.
  - Her yönde 128 birim; 32 parçayı soldan sağa ve sağdan sola okuyor.
  - Sonunda her parçanın 256 sayılık temsili bütün çerçeveden bilgi taşıyor.
- **Adım 3, "Decide":** Toplamsal (additive) attention ve iki dense katman, 99,716 parametre.
  - Attention her parçaya bir puan veriyor; puanlar softmax ile toplamı 1 olan ağırlıklara dönüşüyor; parçaların ağırlıklı ortalaması alınıyor (256 sayı).
  - Sonra Linear 256→128 ve Linear 128→4: dört skor (logit). En büyük skor cevap.
  - Bu, büyük dil modellerindeki "attention" değil. Tek katman, bir kez uygulanıyor.
- **Toplam: 785,476 parametre.** Dağılım: LSTM %84.0, attention %8.4, sınıflandırıcı %4.3, konvolüsyon %3.3. Yani model aslında küçük bir konvolüsyon ön ucu olan bir tekrarlayan (recurrent) ağ.
- **"Nothing in it is random once trained":** Test sırasında dropout kapalı, BatchNorm sabit. Aynı çerçeve her zaman aynı cevabı alıyor.
- **"Unknown" seçeneği yok:** Her çerçeveye dört etiketten biri veriliyor, sinyal olmasa bile. Slayt 12'ye bağlanıyor.

**Okuma metni (EN)**

> The detector works in three steps. It looks at short pieces of the frame. It reads them in order, forwards and backwards; that's where most of the model is. Then it gives four scores, and the biggest wins.
>
> Once trained: same frame, same answer. And it can never say "I don't know".

**Olası sorular**

- *"Why 12 samples?"*
  - Conv kernel 5 sees 5 samples. After a pool of 2 and a conv of 3, it sees 10. After the second pool, 12.
  - The steps are 4 samples apart.
- *"Why an LSTM?"* — The convolutions only see 1.5 symbols. Telling QAM16 from QAM64 needs many symbols together, and the LSTM connects them across the frame.
- *"What does the attention look at?"* — It learns which parts of the frame to weight more. We have not visualised the weights yet. I can do that if it is useful.
- *"Does the model change when you shorten the frame?"* — No. Every layer works on any length. The same 785,476 parameters are used for 128, 64 and 32 samples; only the number of LSTM steps changes (32, 16, 8).
- *"Why 785,476 and not 786,379?"* — The last layer depends on the number of classes. The total is 784,960 + 129 × classes: 785,476 for 4 classes, 786,379 for 11.

---

## Slayt 5 — Kim tasarladı, kodu kim yazdı?

**Slaytta ne var, ne demek**

- **Zincir:** makale → meslektaşın (Esmer) kodu → bizim kod.
  - **Makale:** El-Haryqy et al., *Results in Engineering* 26 (2025) 104783. Model adı ICRNNA (Improved Convolutional Recurrent Neural Network with Attention). 11 sınıfta %63.24 bildiriyor.
  - **Meslektaşın kodu:** Makalenin bir yeniden uygulaması. Bizim kodumuz bundan kopyalandı.
  - **Makaleden 5 fark:** Her satırda önce makale, sonra bizim kod.

    | | makale | bizim kod |
    |---|---|---|
    | ilk konvolüsyonun kernel'i | 3 | 5 |
    | max-pool sayısı | bir | iki |
    | BatchNorm | her LSTM katmanından sonra bir | yığının sonunda tek |
    | attention'da dropout ve LayerNorm | var | yok |
    | sınıflandırıcı | iki dense katman (128 ve 64), dropout 0.3 | tek dense katman (128), dropout 0.5 |

- **"Is the paper's number real?":** Modeli sadece makaleden yeniden kurduk (`colab/icrnna_faithful_2016.py`, 794,827 parametre).
  - Makalenin kendi 58 epoch sınırıyla eğitince %61.75 ± 0.07 çıktı.
  - Yakınsayana kadar eğitince (150 epoch sınırı, en iyi epoch 107) %63.21 çıktı. Makale %63.24 diyor.
  - Yani aradaki 1.5 puanlık farkın tamamı eğitim süresiydi, mimari değil.
- **"Written with an AI assistant":** Kod, grafikler ve slaytlar Claude Code (Anthropic) ile yazıldı. Her sayı o kodu GPU'da çalıştırarak elde edildi. Neyin test edileceği senin kararın.
- **Meslektaş = Esmer** (slaytta adı yazmıyor, sadece konuşmada söylüyorsun). Esmer de bu konu üzerinde çalışıyordu; konuyu birlikte konuştunuz. Kodumuz Esmer'in model versiyonundan geliyor, eğittiğin verilerin bir kısmı da Esmer'in.

**Okuma metni (EN)**

> The design is from a 2025 paper; the model is called ICRNNA. Esmer, a colleague here, was working on the same problem, and we talked it through together. Our code comes from Esmer's version of the model, and some of the trained data I use are Esmer's. It's close to the paper, with five small differences.
>
> I rebuilt it from the paper alone: 63.21 percent, against the paper's 63.24. So the number is real.
>
> The code and slides were written with Claude Code, an AI assistant. The numbers come from running that code, and what to test was my choice.

**Olası sorular**

- *"Why not use the faithful build then?"* — It could be the working model. All results today were made with Esmer's version, which is fully converged and tested. Its 11-class accuracy is 62.2% (3 seeds), one point below the paper. Switching is a one-line change in `model_zoo.backbone`.
- *"What exactly did the AI do?"*
  - It wrote the code, the analysis scripts, the figures and the slide layouts, from my requests.
  - I decided what to test and checked the results.
  - Every number on the slides is read from a result file, not typed by hand.
- *"How do you know the AI didn't make numbers up?"* — Every number traces to a result file in the repository, produced by a run. Converged runs repeat bit-identically on a different machine.

---

## Slayt 6 — Ne verdik, ne çıktı (tek slaytta deney)

**Slaytta ne var, ne demek**

- **INPUT:** RML2016.10a; 4 tip × 20 seviye × 1,000 çerçeve = 80,000 çerçeve. Bir çerçeve 128 örnek, 16 sembol.
- **DETECTOR:** Veri her (sınıf, SNR) hücresinde ayrı bölünüyor.
  1. **Learn, %70:** Ağırlıklar bu çerçevelerle öğreniliyor.
  2. **Tune, %15 (doğrulama):** "Ne zaman durayım?" kararı bununla veriliyor.
     - Early stopping: 20 epoch boyunca doğrulama iyileşmezse dur.
     - En iyi doğrulama epoch'u saklanıyor.
     - Öğrenme oranı düşürme de buna bakıyor.
  3. **Test, %15:** Model bunları hiç görmedi. Sadece en sonda bir kez kullanılıyor. Rapor edilen her sayı buradan.
  - **14 tekrar:** Her seed'de farklı bölme, farklı başlangıç, farklı batch sırası.
  - **Eğitim ayarları:** AdamW, lr 1e-3, batch 256, en fazla 100 epoch.
  - **Hepsi yakınsadı:** En iyi epoch'lar 16–46; hepsi early stopping ile bitti, sınıra çarpmadı.
- **OUTPUT:**
  - **%94.2:** 10 dB ve üstünde doğru oranı. 10, 12, 14, 16, 18 dB × 4 sınıf × 150 test çerçevesi × 14 seed = 42,000 karar; 2,421 yanlış.
  - **%99.6:** 10 dB'de BPSK ve QPSK recall'u.
  - **%25:** −20 dB'de doğruluk. Dört seçenekte rastgele tahmin %25, yani tahmin seviyesi.
  - **42,000:** 10 dB ve üstündeki test kararı sayısı.

**Okuma metni (EN)**

> The whole experiment: the detector learns on 70 percent of the frames, uses 15 to decide when to stop, and is tested on 15 it never saw. Fourteen times, with different splits.
>
> Result: 94 percent right at 10 dB and above. BPSK and QPSK almost perfect. At minus 20 dB, 25 percent: a guess.

**Olası sorular**

- *"Is the split per frame or per SNR?"* — Per (class, SNR) cell, stratified. Every cell has 700 / 150 / 150 frames.
- *"Why not use the validation set as test?"* — Then I would be choosing the model on the number I report. The test set is only touched once.
- *"How confident is 94.2%?"* — It is 42,000 decisions, so the pooled number is tight. Single seeds range from 91.4% to 95.8% (each seed is 3,000 decisions above 10 dB), which is why I report fourteen and pool them.

---

## Slayt 7 — Bir çerçeve neye benziyor?

**Slaytta ne var, ne demek**

- **Soldaki resim (frame.png):** Gürültüsüz çizilmiş bir QAM16 çerçevesi.
  - Turuncu çizgi I (in-phase), mavi çizgi Q (quadrature).
  - Her 8 örnekte bir nokta var: sembol anı. Her nokta −3, −1, 1 veya 3 seviyesinde. QAM16 = I için 4 seviye × Q için 4 seviye = 16 nokta.
  - Noktalar arasındaki yumuşak geçiş darbe şekillendirme filtresinden geliyor (root-raised-cosine, β 0.35).
- **Sağdaki resim (symbols_2x2.png):** Her tip için bir çerçevenin 16 sembolü (turuncu), o tipin kullanabileceği bütün noktaların (gri çemberler) üzerine çizilmiş.
  - BPSK 2 nokta, QPSK 4, QAM16 16, QAM64 64.
- **Ana mesaj:** QAM64'ün 64 noktası var ama çerçevede sadece 16 sembol var. En fazla 16 noktayı gösterebilir, genelde daha azını (bazı semboller tekrar eder). Bu yüzden gürültü hiç olmasa bile bir QAM64 çerçevesi QAM16 gibi görünebilir.
- **"Pictures drawn by us":** Bu iki resmi biz açıklama için çizdik (ölçeklenmemiş tam sayı ızgarası). Veri DeepSig'in.

**Okuma metni (EN)**

> This is one QAM16 frame, without noise. A symbol every eight samples, each on one of four levels. Four times four: sixteen points.
>
> Now the key point. QAM64 has sixty-four points, but a frame has only sixteen symbols. So a QAM64 frame can look like QAM16 even with no noise.

**Olası sorular**

- *"Is this a real frame from the dataset?"* — No. It is drawn by us with the same symbol rate and filter, to show the structure without noise and channel effects. The real frames also have carrier offset, fading and noise (slide 8).
- *"How likely is a QAM64 frame to look like QAM16?"* — We did not compute the exact probability. Slide 11 measures the effect: shorter frames make the QAM pair much worse.

---

## Slayt 8 — Bu veride "10 dB" ne demek?

**Slaytta ne var, ne demek**

- **Who made it:** DeepSig, GNU Radio simülatöründe. Gerçek bir radyodan kaydedilmedi.
- **What the label sets:** Üretici kodda etiket sadece tek bir şeyi ayarlıyor: `noise_amp = 10^(−label/10)`.
  - Bu bir **genlik**. Güç genliğin karesi olduğu için gürültü gücü 10^(−label/5) oluyor.
  - Yani etiketteki 1 dB, teoride gürültü gücünde 2 dB değişim demek. Adı "dB" ama dB gibi davranmıyor.
  - Makale bunu hiç tanımlamıyor. Bunu kodu okuyarak bulduk (`NOISE_2016.md`).
- **What else is added:** Gürültüye ek olarak her çerçeveye taşıyıcı frekans kayması (CFO), saat/örnekleme hızı kayması (SRO), sönümlenme (Rician fading, K=4) ve yankılar (3 yollu multipath) ekleniyor. Bunlar her gürültü seviyesinde var, en iyisinde (+18 dB) bile.
- **"I measured the noise in the file itself":**
  - Yöntem: Sinyal spektrumun ortasında duruyor; bandın dışında sadece gürültü var. Oradaki seviye gürültü gücünü veriyor, ortadaki toplamdan çıkarınca sinyal gücü kalıyor.
  - Önce gürültüsü bilinen çerçevelerde denendi; 0.5 dB içinde doğru.
  - Yöntem bu dosyada yaklaşık −13 ile +18 dB arasını okuyabiliyor.
- **1.9 dB:** Etikette her 1 dB, gerçek SNR'da yaklaşık 1.9 dB değişim (PSK için 1.87–1.89). Teorideki 2'yi doğruluyor.
- **+15 dB:** Aynı etikette QAM64 çerçevelerinin sinyali BPSK/QPSK'dan yaklaşık 15 dB fazla (ölçülen +15.2).
  - QAM16 +9.5, PAM4 +6.6 dB.
  - Sebep: Üretici, takımyıldızları normalize etmeden (ölçeklenmemiş tam sayı noktalarla) kullanmış görünüyor. Bunun öngördüğü değerler +7.0 / +10.0 / +16.2 dB; ölçtüklerimize çok yakın.
  - Yani yayınlanan kod, dosyayı üreten kodla birebir aynı değil.
- **Sonuç:**
  - "10 dB" bir ölçüm değil, bir ayar.
  - Her sinyal türü için farklı bir gürültü seviyesi demek.
  - Aynı etikette sınıfları "aynı SNR'da" diye karşılaştırmamak gerekir.

**Okuma metni (EN)**

> You asked what "10 dB" is. It's a setting in their code, not a measurement. It sets the noise amplitude, not the power, so one dB of label is really about two. I measured it in the file: 1.9.
>
> And at the same label, QAM64 has about 15 dB more signal than BPSK and QPSK. So "10 dB" is a different noise level for each signal.

**Olası sorular**

- *"What is the true SNR at label 10?"*
  - In the rebuilt GNU Radio chain, the per-sample SNR is about 2 × label + 2.4 dB, so label 10 is about 22.5 dB.
  - In the file, my method saturates near +18 dB, so I cannot read label 10 directly for every class.
- *"Why can't you measure below −13 dB?"* — There the signal is far below the noise. The difference between in-band and out-of-band power becomes smaller than the measurement error.
- *"Does this explain the QAM errors?"* — Not by itself. At high labels the QAM errors do not change with SNR at all (slide 10). They depend on the number of symbols (slide 11).
- *"Is the noise random?"* — Yes. In the generator version used for the dataset, noise is drawn fresh, and in the file 0 of 11,000 frames at −20 dB share a noise segment. Only the fading pattern repeats run to run. The carrier offset drifts only a few Hz, about 1° per frame.

---

## Slayt 9 — Tek gürültü seviyesinde cevap: 10 dB

**Slaytta ne var, ne demek**

- **8,400 test çerçevesi:** 14 tekrar × 4 tip × 150 test çerçevesi.
- **Matris (confusion matrix):** Satır gönderileni, sütun dedektörün dediğini gösteriyor. Köşegen doğru olanlar.
  - BPSK satırı: 2,092 doğru, 3 → QPSK, 0 → QAM16, 5 → QAM64.
  - QPSK satırı: 2,091 doğru, 6 → BPSK, 2 → QAM16, 1 → QAM64.
  - QAM16 satırı: 1,808 doğru, **269 → QAM64**, 19 → QPSK, 4 → BPSK.
  - QAM64 satırı: 1,961 doğru, **126 → QAM16**, 8 → BPSK, 5 → QPSK.
- **Recall (satır):** "Gerçekten bu tip olanların kaçına doğru isim verdi?"
  - Örnek: QAM16 için 1,808 / 2,100 = %86.1.
- **Precision (sütun):** "Bu ismi verdiklerinin kaçı doğru?"
  - Örnek: QAM64 sütunu 5 + 1 + 269 + 1,961 = 2,236 karar, bunun 1,961'i doğru = %87.7.
- **448 yanlış:** 8,400 kararın %5.3'ü. **395'i QAM16↔QAM64** (269 + 126), yani hataların %88'i.
- **Asimetri:** QAM16 → QAM64 (269), QAM64 → QAM16'dan (126) iki kat fazla. Kesin sebebini ölçmedik.
  - Olası açıklama: gürültü ve kanal QAM16 noktalarını ızgaranın arasına itiyor; bu da daha sık bir ızgaraya (QAM64) benziyor.
  - Bunu "hipotez" olarak söyle, sonuç olarak değil.

**Okuma metni (EN)**

> At 10 dB: rows are what was sent, columns what it said. The diagonal is correct.
>
> Last time you asked what recall is. Recall is one row: of the frames that really were QAM16, how many did it call QAM16? Here 1,808 of 2,100, 86.1 percent. Precision is one column: when it says QAM64, how often is it right? 1,961 of 2,236, 87.7 percent. So recall asks "did it find them?", and precision asks "can I believe it?"
>
> BPSK and QPSK: 99.6 percent. 448 mistakes out of 8,400, and 395 of them are QAM16 and QAM64 mixed up. One problem left.

**Olası sorular**

- *"What is recall, again?"* — See the **Sözlük** at the top: recall, precision, and why both.
- *"Why is QAM16 called QAM64 more often than the reverse?"* — I have not measured the reason. One likely explanation is that noise and the channel push QAM16 points off the grid, into positions that look like the denser QAM64 grid. Switching off the channel effects one at a time (the next step) should show it.
- *"Are 150 test frames enough?"* — Per seed it is small, so we pool 14 seeds: 2,100 decisions per row.
- *"Is this matrix stable across seeds?"* — Yes. The pattern is the same in every seed; the QAM pair dominates every time.

---

## Slayt 10 — Gürültüyü artır: ne zaman kırılıyor?

**Slaytta ne var, ne demek**

- **Grafik:** Her etikette 8,400 karardan kaç yanlış: +10 dB 448, +6 dB 473, +4 dB 453, +2 dB 486, 0 dB 609, −2 dB 1,042, −4 dB 2,174, −6 dB 2,842.
- **10 → 2 dB:** 450–490 civarında, neredeyse sabit. Aynı kutu, aynı hatalar. Gürültüyü 8 dB artırmak hatayı değiştirmiyor; demek ki bu hataların sebebi gürültü değil.
- **0 dB'nin altı:** Kutu kırılıyor.
  - −2 dB'de QPSK'nın 2,100 çerçevesinden 146'sı QAM16'ya, 126'sı QAM64'e gidiyor; BPSK'ya giden sadece 32.
  - −4 dB'de QPSK'nın sadece 1,132'si doğru, 375'i QAM16, 517'si QAM64.
  - −6 dB'de BPSK de bozulmaya başlıyor (1,745 doğru).
  - Yani ilk kayan QPSK, ve QAM'lere kayıyor.
- **Neden QPSK QAM'e kayıyor (yorum):**
  - Gürültü QPSK noktalarını çemberden uzaklaştırınca genlik değişiyormuş gibi görünüyor; genlik değişimi de QAM'in işareti.
  - Ayrıca aynı etikette QAM'ler ~10–15 dB daha güçlü (slayt 8). Bu yüzden düşük etikette QAM çerçeveleri hâlâ okunabilirken QPSK çoktan gürültüye gömülmüş.

**Okuma metni (EN)**

> Then I turned the noise up. From 10 down to 2 dB the mistakes stay around 450 to 490. Nothing changes. It only breaks below 0 dB.
>
> So the mistakes at 10 dB are not noise. Then what are they?

**Olası sorular**

- *"Why stop at −6 dB?"* — Below that the box is mostly gone. At −20 dB it is a guess (slide 12).
- *"Is 448 vs 486 a real difference?"* — No. The difference between 448 and 486 is within what changing seeds does. The point is that it does not trend.

---

## Slayt 11 — 16, 8 veya 4 sembol: neye ihtiyacı var?

**Slaytta ne var, ne demek**

- **Deney:** Her çerçeve ilk 64 örneğe (8 sembol) ve ilk 32 örneğe (4 sembol) kesildi; model sıfırdan, aynı protokolle tekrar eğitildi.
  - 14 seed, hepsi yakınsadı.
  - Komut: `train_backbone.py --frame-len 64 / 32`.
  - Model aynı (785,476 parametre); sadece LSTM adım sayısı değişiyor.
- **Grafik:** Etikete göre % yanlış, üç uzunluk.
  - Üç eğri yaklaşık aynı yerde düşüyor (−10 ile 0 dB arası).
  - Ama farklı yüksekliklerde düzleşiyor.
  - −20 dB'de üçü de ~%75 yanlış (dört seçenekte tahmin).
- **Taban seviyeleri (+2 dB ve üstü):** 16 sembol %5.7, 8 sembol %15.5, 4 sembol %25.2 yanlış.
- **10 dB üstü hata sayısı (42,000 karardan):** 2,421 / 6,443 / 10,431. Yani 8 sembolde ×2.7, 4 sembolde ×4.3 (slayt 14).
- **Kim kayboluyor:** Neredeyse tamamen QAM çifti.

  | 10 dB üstü recall | 16 sembol | 8 sembol | 4 sembol |
  |---|---|---|---|
  | QAM16 | %86.7 | %65.9 | %46.3 |
  | QAM64 | %91.9 | %74.9 | %62.9 |
  | BPSK | %99.3 | %99.4 | %99.4 |
  | QPSK | %98.9 | %98.5 | %92.0 |

  - 4 sembolde QAM çifti yazı tura.
  - 4 sembolde QPSK, 809 çerçeveyi BPSK'ya kaptırıyor. Sebebi: 4 sembolün hepsi aynı köşegene düşerse QPSK, BPSK gibi görünür. Bunun olasılığı 2/16 = %12.5; gerçekte %7.7 oldu.
- **Sağdaki resim (fewer.png):** QAM16 ve QAM64, 16 / 8 / 4 sembolle. Sembol azaldıkça ızgaradan daha az nokta görünüyor, ikisi birbirine benziyor.
- **Ana mesaj:**
  - Gürültü eğrinin **nerede** düştüğünü belirliyor.
  - Sembol sayısı **ne kadar aşağı** inebileceğini belirliyor.
  - Daha çok sinyal bunu düzeltmiyor; daha çok sembol düzeltiyor.

**Okuma metni (EN)**

> Frame length. I cut the frames to eight symbols, then four, and trained again.
>
> At low noise every curve goes flat: 5.7 percent wrong with sixteen symbols, 15.5 with eight, 25 with four. Almost all of it is the QAM pair.
>
> Noise decides where the curve drops. Symbols decide how low it can go.

**Olası sorular**

- *"What would 256 samples give?"* — We cannot test it on this dataset: frames are 128 samples. The trend predicts fewer QAM errors. The 2018 dataset has 1,024-sample frames, and there the four-class table above 10 dB is 40,654 of 40,656 correct; that is consistent with the trend, but it is a different dataset, so I would not lean on it.
- *"Did you take the first 64 or a random 64?"* — The first 64 samples of every frame, for training and test.
- *"Why do the curves start at 75% wrong?"* — At −20 dB there is no usable signal, and four choices means 25% right by chance.

---

## Slayt 12 — Sinyal hiç yokken bile cevap veriyor

**Slaytta ne var, ne demek**

- **Durum:** −20 dB'de pratikte sinyal yok. Ama model "bilmiyorum" diyemiyor (slayt 4), dördünden birini seçmek zorunda.
- **%52.2: QPSK recall.** Gerçek QPSK çerçevelerinin yarısından fazlasına "QPSK" demiş. Tek başına bakınca "QPSK'yı tanıyor" gibi görünüyor.
- **%24.8: QPSK precision.** "QPSK" dediği çerçevelerin sadece %24.8'i gerçekten QPSK. Dört sınıf eşit sayıda olduğu için rastgele tahmin %25 verir. Yani tahmin.
- **Formüller:**
  - Recall = doğru ÷ o tipten gönderilen.
  - Precision = doğru ÷ modelin o tip dediği.
- **Neden:** Kanıt olmayınca model hep aynı tarafa yaslanıyor.
  - Kararların %98.8'i BPSK (%46.1) veya QPSK (%52.7).
  - QAM16 (%0.7) ve QAM64 (%0.5) neredeyse hiç denmiyor.
  - İkisi arasındaki pay seed'den seed'e değişiyor: BPSK'nın payı %12 ile %86 arasında. Yani hangi sınıfa yaslanacağı rastgele; tanımayla ilgisi yok.
  - BPSK'nın kendi rakamları da aynı resmi veriyor: recall %47.2, precision %25.6.
- **Buna "sink" (lavabo) diyoruz:** Bilgisiz girdiler bir sınıfa "akıyor".
- **Sonuç:** Artık her grafikte recall'un yanında precision ve her noktanın kaç karara dayandığı yazıyor.
  - Bu, Moshe'nin 5. isteği.
  - Eski 2018 grafiğindeki "SNR düşerken QPSK recall'u artıyor" eğrisi de aynı şeydi.

**Okuma metni (EN)**

> At minus 20 dB there's no signal, but it still has to pick one. QPSK recall is 52 percent, which looks good. But precision is 25 percent: a guess. It just says BPSK or QPSK almost every time.
>
> So now every chart shows precision next to recall.

**Olası sorular**

- *"Why BPSK and QPSK and not the QAMs?"* — I do not know for certain. One reading: noise-only frames, scaled to unit power, have no consistent amplitude structure, and the model has learned that "no amplitude structure" means PSK.
- *"Could you add an 'unknown' class?"* — Yes, with a threshold on the scores or a noise-only class. It is not done; it would change the task.
- *"How many decisions are behind 52.2%?"* — 2,100 QPSK frames at −20 dB (14 seeds × 150).

---

## Slayt 13 — ICRNNA yayınlanmış modellerle nasıl kıyaslanıyor?

**Slaytta ne var, ne demek**

- **Deney:** Dört bilinen modeli, yazarlarının benchmark kodundan PyTorch'a çevirdik (`src/literature_models.py`). Kaynak: Zhang et al., *Digital Signal Processing* 2022, AMR-Benchmark Keras kodu.
  - Hepsi ICRNNA ile aynı protokolde eğitildi: aynı çerçeveler, aynı bölme, aynı 3 seed, en fazla 150 epoch. Hepsi yakınsadı.
  - Bu sefer 11 sınıfın tamamı, bütün SNR'lar.
- **Modeller:**
  - **VT-CNN2 (2016):** O'Shea'nın ilk CNN'i, benchmark versiyonu. 1,592,383 parametre; en büyüğü ama en zayıfı.
  - **LSTM2 (2018):** Rajendran et al. İki katmanlı LSTM, ama girdi I/Q değil genlik ve faz. 201,099 parametre.
  - **MCLDNN (2020):** Xu et al. Çok kanallı (I, Q ve IQ ayrı ayrı) CNN + LSTM. 406,199 parametre.
  - **PET-CGDNN (2021):** Zhang et al. Önce faz düzeltmesi tahmin ediyor, sonra CNN + GRU. 71,871 parametre; en küçüğü.
- **Tablo sütunları:**
  - "numbers learned": parametre sayısı.
  - "paper": makalede yayınlanan genel doğruluk.
  - "ours": bizim genel doğruluğumuz, 3 seed ortalaması.
  - "ours ≥ 10 dB": 10 dB ve üstü doğruluğumuz.
- **"Paper" sütunu nereden:**
  - ICRNNA: kendi makalesi (%63.24). Biz Esmer'in versiyonunu ölçüyoruz: %62.2.
  - LSTM2, MCLDNN, PET-CGDNN: PET-CGDNN makalesi, Table I (Zhang et al. 2021). Orada tek eğitim koşusu ve 60/20/20 bölme var.
  - VT-CNN2 için 2016.10a'da genel doğruluk veren bir makale bulamadık; o yüzden "—".
- **Doğrulama (`LITERATURE_CHECK.md`):**
  - Parametre sayıları her kaynakta basamağına kadar aynı. Bir katman farklı olsa sayı değişirdi; portun doğruluğunun en güçlü kanıtı bu.
  - Doğruluk farkları bizim lehimize veya aleyhimize, hepsi 1 puan içinde: LSTM2 60.56 → 61.44 (+0.9), MCLDNN 62.08 → 61.41 (−0.7), PET-CGDNN 60.44 → 60.98 (+0.5).
  - Farklı bölme ve tek koşu ile 3 seed ortalaması arasındaki fark tam da bu büyüklükte oynatır.
- **Sonuçlar:**
  - ICRNNA genel olarak birinci (%62.2), ama en iyi dördü 1.2 puan içinde.
  - ICRNNA'nın kazancı −6 ile −2 dB arasında.
  - 10 dB üstünde dördü de ~%91'de duruyor. Bu tavan modelden değil, veriden: özellikle WBFM (her modelde ~%35–55, AM-DSB ile karışıyor) ve QAM çifti.
  - VT-CNN2 orada ~10 puan aşağıda (%82.0).
- **Figür 38:** Beş modelin SNR'a göre doğruluk eğrileri.

**Okuma metni (EN)**

> Is ICRNNA a good choice? I trained four published models the same way, on all eleven types. They have the same parameter counts as in their papers, and accuracy within one point of what the papers publish.
>
> ICRNNA is first, but the top four are within one point, and above 10 dB they all stop near 91 percent. That limit is in the data, not the model.

**Olası sorular**

- *"Why is LSTM2 so good with so few parameters?"* — Its input is amplitude and phase instead of I and Q, which makes phase-based classes easy to read. That is a strong hint that the input representation matters as much as the size.
- *"Why don't you have the paper number for VT-CNN2?"* — No paper we have tables its overall accuracy on 2016.10a. O'Shea 2016 used the earlier 2016.04 dataset and a larger network, so its 87.4% is a different experiment.
- *"Is one point a real difference?"* — Between the top four, no. Three seeds each; the ranges overlap. I would not claim ICRNNA is better, only that it is not worse.
- *"Why is WBFM so bad?"* — In this dataset WBFM frames are often generated from audio with silent parts. A silent WBFM frame is just a carrier, and it looks like AM-DSB. Every model fails there; it is a known property of the dataset. (I have not measured this myself; say it as the known explanation.)

---

## Slayt 14 — Özetle

**Slaytta ne var, ne demek**

- **1. BPSK ve QPSK pratikte çözülmüş;** hatalar QAM16 ↔ QAM64. (Slayt 9: 448 yanlışın 395'i.)
- **2. Bu hatalar gürültüden değil, sadece 16 sembol olmasından geliyor.** 8 sembolde ×2.7, 4 sembolde ×4.3 artıyor. (Slayt 10: 10 → 2 dB sabit; slayt 11: 2,421 → 6,443 → 10,431.)
- **3. "10 dB" bir ayar, ölçüm değil.** Her sinyal türü için farklı gürültü demek. (Slayt 8: 1.9 dB, +15 dB.)
- **Next (sıradaki iş):** Kanal etkilerini tek tek kapatmak: taşıyıcı kayması, sönümlenme, yankılar. QAM hatalarının hangisinden geldiğini görmek.
  - Bunu yapabiliyoruz çünkü üretici zinciri GNU Radio 3.10'da yeniden kurduk (`tools/rml2016_channel_snr.py`).
  - Henüz koşulmadı; plan olarak söyle.

**Okuma metni (EN)**

> Three things. BPSK and QPSK are solved; the mistakes are QAM16 against QAM64. Those come from having only sixteen symbols, not from noise. And "10 dB" is a setting, not a measurement.
>
> Next: switch off the channel effects one at a time, and see which one the QAM mistakes follow. Thank you.

**Olası sorular**

- *"What do you expect the channel experiment to show?"* — I do not want to guess. If the QAM errors stay when everything but the noise is off, they really are the symbol count. If they drop, one of the channel effects is mixing the two grids.
- *"And the domain-gap question?"* — That is step three. On 2016 the gap is small (0.948 in-domain, 0.920 cross-domain on 5 classes) and standard augmentation closes it. I will present it separately.

---

## Genel tavsiyeler

- **Her slaytta önce mesajı, sonra sayıyı söyle.** Moshe'nin tarzı: tek cümle sonuç, sonra kanıt. Her slaytın alt satırındaki (turuncu vurgulu) cümle o slaytın mesajı.
- **Bilmediğin şeye "I have not measured that" de.** Bu sunumdaki her hipotez (QAM16 → QAM64 asimetrisi, sink'in neden PSK'ya gittiği, WBFM) bu metinde öyle işaretlendi.
- **Etiket ≠ SNR.** "At label ten" demek, "at ten dB SNR" demekten daha doğru. Slayt 8'den sonra bunu tutarlı kullanırsan Moshe fark eder.
- **Slayt 5'teki boşluğu sunumdan önce doldur.** Meslektaşın adı ve Eylül öncesi işin nasıl yapıldığı, notlarda ve `DETECTOR.md`'de.

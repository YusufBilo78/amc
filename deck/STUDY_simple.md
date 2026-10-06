# Sunum çalışma metni — `moshe_2016_simple.pptx` (14 slayt)

Her slayt için üç bölüm var:

- **Slaytta ne var, ne demek:** Slayttaki her sayının, kutunun ve resmin Türkçe açıklaması. Sayının nereden geldiği de yazıyor.
- **Okuma metni (EN):** Sunumda söyleyeceğin İngilizce metin. Slayt notlarındakinin genişletilmiş hali. Okunarak değil, anlatılarak söylenecek şekilde yazıldı.
- **Olası sorular:** Moshe'nin sorabileceği sorular ve kısa cevapları.

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

## Slayt 1 — Başlık

**Slaytta ne var, ne demek**

- **"AMC · DATA FUSION LAB":** Konu (Automatic Modulation Classification) ve laboratuvar.
- **"Telling four radio signals apart":** Bugünkü sunumun kapsamı. 11 sınıfın tamamı değil, Moshe'nin istediği dört sınıf: BPSK, QPSK, QAM16, QAM64.
- **Sağdaki animasyon (noise.gif):** Bir QAM16 çerçevesinin 16 sembolü. Gürültü arttıkça noktalar ızgara noktalarından uzaklaşıp dağılıyor. Bu resmi biz çizdik, açıklama amaçlı. Veri setinden alınmış bir çerçeve değil.

**Okuma metni (EN)**

> Good morning. Today I want to show you, step by step, what our detector does. I will stay with the four signals you asked about: BPSK, QPSK, QAM16 and QAM64.
>
> The picture on the right is one QAM16 frame. It has sixteen symbols, and you can watch them spread away from their grid points as the noise grows. Keep that picture in mind, because most of what I show today is about how many symbols a frame has and how much noise is on them.
>
> Everything today is on RML2016.10a, the dataset the group uses, so the numbers can be compared with yours.

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

> First, in one minute, what this is about.
>
> AMC stands for automatic modulation classification. A receiver picks up a radio signal that nobody described to it. Before it can do anything with that signal, it has to answer one question: how was it modulated? Is it BPSK, QPSK, a QAM, FM? That question comes up when you watch the spectrum, when you look for the source of interference, and in radios that adapt to what they hear.
>
> My work has three parts.
>
> The first is a detector that answers that question. It is a neural network trained on RML2016.10a, the standard dataset most papers in this field use. It is simulated, not recorded.
>
> The second is opening that detector up: what exactly it does, what the noise in the data really is, and where it fails. That is today.
>
> The third is the bigger question of my project: why do detectors like this lose accuracy when the signal comes from a different transmitter, and what fixes it?
>
> Today I stay with the first two, on four signals.

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

> The data. I use RML2016.10a. DeepSig made it in 2016 with GNU Radio, a radio simulator, and it is the dataset most AMC papers report on.
>
> It has eleven modulation types: eight digital and three analog. Each type is given at twenty noise levels, from minus twenty to plus eighteen dB in steps of two, with one thousand frames at every level. That is two hundred and twenty thousand frames in total.
>
> Every frame is 128 samples of the complex baseband signal, an in-phase part and a quadrature part. There are eight samples per symbol, so one frame holds sixteen symbols. That number, sixteen, will matter later.
>
> Today I use the four in orange. BPSK and QPSK carry information only in the phase: all their points sit on one circle. QAM16 and QAM64 also change the amplitude: their points sit on a square grid.

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

> What does the detector do? It is given 128 samples of a radio signal and has to say which of the four modulations sent them. Before anything else, every frame is scaled to the same average power, so the detector cannot cheat by looking at how loud the signal is.
>
> It works in three steps.
>
> Step one looks at short pieces of the frame. Two convolution layers each slide a set of learned filters along the samples, and two pooling steps halve the length twice. Each output piece is computed from twelve consecutive samples, about one and a half symbols, so the frame becomes thirty-two pieces. What this step can see is local: how the signal changes from one symbol to the next.
>
> Step two reads those thirty-two pieces in order, forwards and backwards. This is a bidirectional LSTM, two layers. After it, every piece carries information about the whole frame, not only its own neighbourhood.
>
> Step three decides. An attention layer gives each of the thirty-two pieces a weight, and takes the weighted average. Two dense layers turn that average into four scores, one per class. The largest score is the answer.
>
> In total it has 785 thousand numbers that it learns. Eighty-four percent of them are in the second step, so this is mostly a recurrent network with a small convolutional front end.
>
> Two properties matter for what follows. Once trained, nothing in it is random: the same frame always gets the same answer. And there is no "I don't know": every frame gets one of the four names.

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

- **Zincir:** makale → meslektaşın kodu → bizim kod.
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
- **⚠️ Doldurman gereken yer:** Slayt notlarında `[Say here who the colleague is and how the work before September was done.]` duruyor. `DETECTOR.md`'deki "Before 2026-09-09" bölümü de boş. Meslektaşın adını ve Eylül öncesi işin nasıl yapıldığını sen söylemelisin. Ben bilmiyorum, uydurmadım.

**Okuma metni (EN)**

> Where did the detector come from?
>
> The design is from a 2025 paper by El-Haryqy and co-authors, in Results in Engineering. The model is called ICRNNA, and the paper reports 63.24 percent on all eleven classes of this dataset.
>
> A colleague re-implemented that paper, and our code was copied from their version. Comparing it line by line with the paper, it differs in five small places: the first filter size, one extra pooling step, where the normalisation sits, no dropout in the attention, and one dense layer instead of two. So I do not call our model "the published architecture"; it is a close version of it.
>
> I wanted to know whether the paper's number is real. So I rebuilt the model from the paper alone. Trained for the paper's own 58 epochs, it reached 61.75 percent, one and a half points short. Trained until it stopped improving, which was epoch 107, it reached 63.21 percent, against the paper's 63.24. So the paper's number is reproducible, and the whole gap was training time. Since then every training script records when the model stopped improving, and warns if it was stopped too early.
>
> And to be clear about the tools: the code, the plots and these slides were written with Claude Code, an AI coding assistant from Anthropic. Every number comes from running that code on a GPU. The choices of what to test, and what to ask, were mine.
>
> *[Burayı sen doldur: meslektaşın kim olduğu ve Eylül öncesi işin nasıl yapıldığı.]*

**Olası sorular**

- *"Why not use the faithful build then?"* — It could be the working model. All results today were made with the colleague's version, which is fully converged and tested. Its 11-class accuracy is 62.2% (3 seeds), one point below the paper. Switching is a one-line change in `model_zoo.backbone`.
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

> This is the whole experiment on one slide.
>
> The input is RML2016.10a: four kinds of signal, twenty noise levels, a thousand frames of each. One frame is 128 samples, sixteen symbols.
>
> The detector learns on seventy percent of the frames. It uses fifteen percent to decide when to stop learning: when the accuracy on that part has not improved for twenty epochs, training stops and the best version is kept. The last fifteen percent is the test set. The detector never sees it while learning, and nothing is chosen on it. Every number I show is counted on that test set.
>
> I did this fourteen times, each time with a different split and a different random start, so that one lucky split cannot make the result. All fourteen runs stopped by themselves, well before the limit, so none of them was cut short.
>
> The output: at ten dB and above, it is right 94.2 percent of the time, counted over 42,000 test decisions. BPSK and QPSK at ten dB are right 99.6 percent of the time. At minus twenty dB it is right 25 percent of the time, which, with four choices, is exactly a guess.
>
> The next slides take this apart: first the input, then the noise, then the answer.

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

> Step one: what the detector actually sees.
>
> On the left is one QAM16 frame, drawn without noise. The orange line is the in-phase part, the blue line the quadrature part. Every eight samples there is one symbol, the dots. Each dot sits on minus three, minus one, one or three. Four levels on each axis gives sixteen points, and that is what makes it QAM16. Between the dots the signal moves smoothly, because of the pulse-shaping filter.
>
> On the right are the sixteen symbols of one frame for each kind, in orange, drawn on all the points that kind can use, the grey circles. For BPSK, sixteen symbols on two points: every point is seen many times. For QPSK, four points, still easy.
>
> Now look at QAM64. It has sixty-four points, but one frame only has sixteen symbols, so it can show at most sixteen of them. In practice it shows fewer, because some repeat. If those happen to fall on the points QAM16 also uses, a QAM64 frame looks like a QAM16 frame, even with no noise at all.
>
> Keep that in mind: it is the explanation for most of the mistakes I show next.

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

> Step two: the noise. You asked what "ten dB" is the ratio of. Here is the answer.
>
> The dataset was made by DeepSig in GNU Radio, a simulator. It was not recorded from a real radio. The paper does not define the label. In their code, the label sets exactly one number: the noise amplitude, ten to the minus label over ten. That is an amplitude, not a power, so one dB of label should change the noise power by two dB, not one.
>
> Besides the noise, every frame also gets a carrier frequency offset, a clock offset, fading and echoes. Those are there at every noise level, even the best one, so a frame at plus eighteen dB is still not a clean frame.
>
> I did not want to rely on their code, so I measured the noise in the file itself. The signal sits in the middle of the spectrum. Outside it there is only noise, so the level out there gives the noise power. I checked this method first on frames where I knew the noise, and it was within half a dB.
>
> Two findings. First, one dB of label is about 1.9 dB of real change, which confirms the code: the label moves twice as fast as its name. Second, at the same label, QAM64 frames have about fifteen dB more signal than BPSK and QPSK frames, and QAM16 about nine and a half. That is what you get if the constellations were never scaled to the same power.
>
> So "ten dB" in this dataset is a setting, not a measurement, and it means a different amount of noise for each kind of signal. I quote labels as labels, and I never compare two classes at one label as if they had the same SNR.

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

> Here is the answer at one noise level, ten dB, as you asked.
>
> These are 8,400 test frames: fourteen repeats, four kinds, a hundred and fifty frames each. Each row is what was sent. Each column is what the detector said. The diagonal is right; everything else is a mistake.
>
> Two words I will use from now on. Recall is a row: of the frames that really were QPSK, how many did it call QPSK. Precision is a column: of the frames it called QPSK, how many really were.
>
> BPSK and QPSK are at 99.6 percent recall, with precision near 99 percent as well. For them the problem is solved.
>
> 448 frames out of 8,400 are wrong. 395 of those, almost nine in ten, are QAM16 and QAM64 confused with each other. QAM16 frames called QAM64 are the larger part, 269 of them.
>
> So at ten dB there is really only one problem left: telling QAM16 from QAM64.

**Olası sorular**

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

> Then, as you asked, turn the noise up, one level at a time.
>
> These bars are the number of wrong answers out of 8,400 test frames at each label. From ten dB down to two dB it stays between about 450 and 490. That is eight dB more noise, and nothing changes: the same box, with the same QAM mistakes, every time. So the mistakes we have at ten dB are not caused by noise.
>
> At zero dB it starts to move, and below zero it breaks. At minus two it is a thousand, at minus four more than two thousand. And the first to slip is QPSK, not into BPSK but into the two QAMs. When noise moves QPSK points off their circle, the amplitude seems to change, and changing amplitude is what QAM looks like. Remember also from the last slide that at the same label the QAMs have ten to fifteen dB more signal than QPSK, so QPSK gets buried first.
>
> So there are two different kinds of mistake. Below zero dB, noise. Above two dB, something else. The next slide shows what.

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

> So what does cause the mistakes that noise does not explain? The frame length.
>
> I cut every frame to its first sixty-four samples, which is eight symbols, and to its first thirty-two, four symbols, and trained the same model again from scratch, fourteen times each. The model is exactly the same; only the input is shorter.
>
> This chart shows the percent wrong against the label, for sixteen, eight and four symbols. The three curves fall at about the same noise level: noise decides where the curve comes down. But they flatten at very different heights. From plus two dB up, 5.7 percent wrong with sixteen symbols, 15.5 with eight, 25.2 with four. And once flat, adding signal does not help any more.
>
> Who is lost? Almost only the QAM pair. With four symbols QAM16 is right less than half the time, a coin flip against QAM64. BPSK does not move at all. QPSK starts to lose frames to BPSK only at four symbols, and that is what four symbols predict: if all four happen to land on one diagonal, a QPSK frame looks exactly like BPSK.
>
> On the right you can see why: with fewer symbols you see fewer points of the grid, and QAM16 and QAM64 become harder to tell apart.
>
> So the ceiling at high SNR is set by the number of symbols. More signal does not fix it. More symbols does.

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

> One thing we found along the way.
>
> At minus twenty dB there is effectively no signal. But the detector has no "I don't know"; it must name one of the four. So what does it say?
>
> QPSK recall there is 52 percent. On its own, that looks as if the detector still recognises QPSK in the noise. It does not. Look at the precision: of all the frames it calls QPSK, only 24.8 percent are QPSK. With four equal classes, a guess gives 25 percent. So that is a guess.
>
> What happens is that with nothing to go on, it sends almost everything, 98.8 percent of the frames, to BPSK or QPSK. The two QAMs are almost never named. And which of the two it prefers changes from one training run to the next: in some runs it is mostly BPSK, in others mostly QPSK. That is the mark of a default, not of recognition.
>
> The lesson for me was that recall alone can look good while the detector is only guessing. That is why every chart now shows precision next to recall, and how many decisions sit behind each point.

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
  - ICRNNA: kendi makalesi (%63.24). Biz meslektaşın versiyonunu ölçüyoruz: %62.2.
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

> Is ICRNNA a good choice? To answer that, I took four well-known published models, from 2016 to 2021, and rewrote each one from its authors' public benchmark code. Then I trained all of them exactly like ICRNNA: the same frames, the same split, the same three repeats. Only the model changes. This time on all eleven types, because that is what the papers report.
>
> First, are my numbers right? Two checks. Each model has exactly the number of learned parameters its paper reports, to the digit; if one layer were different, that number would change. And the accuracies the papers publish are within one point of mine, sometimes above and sometimes below, which is what a different split and a single run against three would give. So the comparison is fair.
>
> Then the result. ICRNNA is first overall, at 62.2 percent, but the best four are within about one point of each other. Where ICRNNA gains is at low SNR, between minus six and minus two dB. Above ten dB all four stop at about 91 percent. When four different designs stop at the same height, that limit is in the data, not in the model: mostly WBFM, which every model confuses with AM-DSB, and the QAM pair we just saw. The oldest model, the 2016 CNN, stays about ten points lower up there, even though it has the most parameters.

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

> To sum up, three points.
>
> One: BPSK and QPSK are essentially solved. Almost every remaining mistake is QAM16 against QAM64.
>
> Two: those mistakes come from having only sixteen symbols, not from noise. Between ten and two dB the number of mistakes does not change, and when I give the detector eight or four symbols instead of sixteen, the mistakes grow by 2.7 and 4.3 times.
>
> Three: "ten dB" in this dataset is a setting, not a measurement. One dB of label is almost two dB of real change, and at the same label each kind of signal has a different amount of noise.
>
> Next, I want to find which part of the channel the QAM mistakes belong to. I have rebuilt the generator, so I can switch off the carrier offset, the fading and the echoes one at a time, and see which one the mistakes follow.
>
> Thank you. I am happy to take questions.

**Olası sorular**

- *"What do you expect the channel experiment to show?"* — I do not want to guess. If the QAM errors stay when everything but the noise is off, they really are the symbol count. If they drop, one of the channel effects is mixing the two grids.
- *"And the domain-gap question?"* — That is step three. On 2016 the gap is small (0.948 in-domain, 0.920 cross-domain on 5 classes) and standard augmentation closes it. I will present it separately.

---

## Genel tavsiyeler

- **Her slaytta önce mesajı, sonra sayıyı söyle.** Moshe'nin tarzı: tek cümle sonuç, sonra kanıt. Her slaytın alt satırındaki (turuncu vurgulu) cümle o slaytın mesajı.
- **Bilmediğin şeye "I have not measured that" de.** Bu sunumdaki her hipotez (QAM16 → QAM64 asimetrisi, sink'in neden PSK'ya gittiği, WBFM) bu metinde öyle işaretlendi.
- **Etiket ≠ SNR.** "At label ten" demek, "at ten dB SNR" demekten daha doğru. Slayt 8'den sonra bunu tutarlı kullanırsan Moshe fark eder.
- **Slayt 5'teki boşluğu sunumdan önce doldur.** Meslektaşın adı ve Eylül öncesi işin nasıl yapıldığı, notlarda ve `DETECTOR.md`'de.

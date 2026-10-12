# "Adım adım" sunumu çalışma metni — `moshe_steps.pptx` (15 slayt)

Bu sunum son toplantıda Moshe'nin sorduklarına cevap. Moshe şunu istedi:

> "Don't do any new work. Just try to explain to us step by step."

O yüzden yeni deney yok. Her sayı zaten var olan sonuç dosyalarından geliyor (`deck/steps_data.json`, `deck/extract_steps.py`).

Her slayt için üç bölüm var:
- **Slaytta ne var, ne demek:** Türkçe açıklama.
- **Okuma metni (EN):** Sunumda söyleyeceğin metin; slayt notlarıyla aynı, hepsi yaklaşık 9 dakika.
- **Olası sorular:** Moshe'nin sorabileceği sorular ve kısa cevapları.

---

## Toplantıda ne oldu — kısa özet

**Moshe'nin asıl sorusu:**
- 10 dB'deki 4 sinyallik tabloda bütün hatalar dört sinyalin kendi aralarında kalmıştı.
- Moshe: "Bu dedektör 11 sinyali tanıyorsa, hatalar neden hiç diğer 7 sinyale gitmiyor? Bu imkânsız."
- Haklıydı. Cevap: o tablo, **sadece bu dört sinyalle eğitilmiş ayrı bir ağdan** geliyor. O ağın 4 çıkışı var, başka bir cevap veremez.
- Kodda bu böyle: `train_backbone.py --classes BPSK,QPSK,QAM16,QAM64`, etiketleri 0..3'e çeviriyor ve modeli 4 çıkışla kuruyor (`model_zoo.backbone(4)`).

**Toplantıda yanlış verilen üç cevap:**
1. **"Aynı dedektör"** — Yanlış. 4 sinyal tablosu ayrı, 4 çıkışlı bir ağ. 11 sinyal sonuçları başka, 11 çıkışlı bir ağ. İkisi de aynı mimari (ICRNNA), ama ayrı ayrı eğitildi.
2. **"Yazarlar ağırlıkları paylaştı"** — Yanlış. Makale ne kod ne ağırlık yayımladı ("Data will be made available on request"). Bütün ağları biz, rastgele başlangıçtan eğittik. Kodda dışarıdan ağırlık yükleyen tek bir yer yok; yüklenen her checkpoint bizim kendi eğitimimizin çıktısı.
3. **"Her modülasyon için ayrı bir kod parçası var"** — Yanlış. Tek bir ağ var. Bir modülasyon ağ için sadece bir çıkış numarası (BPSK 0, QPSK 1...).

**⚠️ Senin kontrol etmen gereken bir şey:**
- Daha önce "eğittiğim verilerin bir kısmı Esmer'in" demiştin.
- Repoda gördüğüm şu: Esmer'den gelen şey modelin kodu (makalenin yeniden uygulaması). Bu sunumdaki bütün ağları bizim script'lerimiz eğitti.
- Esmer'in eğittiği bir ağırlık ya da sonuç tablosu bir yerde kullanıldıysa, slayt 4 ve 15'teki "Nobody else's weights" cümlesi değişmeli. Bana söyle, düzelteyim.

---

## Slayt 1 — Başlık

**Okuma metni (EN)**

> Last week you asked me to explain, step by step, exactly what we did. That is all this talk does. No new experiments: only what was already done, explained completely.

---

## Slayt 2 — Sorularınız

**Slaytta ne var, ne demek**
- Moshe'nin toplantıda sorduğu altı soru, sırayla. Sunumun geri kalanı bu sırayı takip ediyor.
- 3. soru (turuncu) en önemlisi: tablonun neden sadece dört sinyal arasında hata gösterdiği.

**Okuma metni (EN)**

> These are your questions from last week. Is the detector a neural network, and was it trained or fixed? Is there a separate piece for each modulation? Why were the mistakes in the four-signal table only among the four? How exactly is it trained? Which software did we use? And which paper is it, and who wrote it?

---

## Slayt 3 — Geçen hafta yanlış verdiğim üç cevap

**Slaytta ne var, ne demek**
- Sol taraf ("said"): toplantıda söylenen.
- Sağ taraf ("true"): kodda gerçekte olan.
- Bu slaytı açık ve sakin söyle. Moshe toplantıda "it's my job in life to ask this kind of question" dedi; düzeltmeyi takdir eder.

**Okuma metni (EN)**

> First, three answers I gave last week were wrong, and I want to correct them.
>
> I said the four-signal table came from the same detector as the eleven-signal results. It did not: it came from a separate network, trained only on those four.
>
> I said the authors published their weights. They did not publish code or weights; every network here, we trained ourselves.
>
> And I agreed there was a piece of code for each modulation. There isn't: there is one network, and a modulation is just an output number.

**Olası sorular**
- *"Why did you say it last week?"* — I answered too quickly without checking the code. This week I went back to the code line by line.

---

## Slayt 4 — Soru 1: Sinir ağı mı, eğitildi mi?

**Slaytta ne var, ne demek**
- **785,476:** 4 çıkışlı ağın ayarlanabilir sayıları (parametre, ağırlık). Başta rastgele.
- **56,000:** Öğrenme için kullanılan etiketli çerçeve sayısı (80,000'in %70'i).
- **frozen:** Eğitim bitince sayılar sabitleniyor, sonra hiç görmediği çerçevelerde test ediliyor.
- **Moshe'nin "fixed weights" sorusunun cevabı:** Test sırasında ağırlıklar sabit. Ama o ağırlıkları biz eğittik; hazır alınmadı.

**Okuma metni (EN)**

> Question one. Yes, it is a neural network. Think of it as a function with 785,476 adjustable numbers. At the start they are random. We show it 56,000 frames with the right answer and adjust the numbers to make fewer mistakes. Then we freeze them and test. Nobody else's weights are involved.

**Olası sorular**
- *"Where are the trained weights?"* — Each run saves its best version as a file (a checkpoint) on Google Drive. The confusion matrices at every SNR were rebuilt from those files.
- *"Fixed weights or trained?"* — Both, in order: trained by us first, then fixed for testing.

---

## Slayt 5 — Soru 2: Her modülasyon için ayrı parça var mı?

**Slaytta ne var, ne demek**
- Bir çerçeve girer, **aynı katmanlardan** geçer, sonunda 4 sayı (skor) çıkar. En büyüğü cevap.
- Sağdaki çubuklar örnek: QPSK skoru en büyük, cevap QPSK. Çubukların boyu temsilî, ölçüm değil.
- **Etiket:** Veri setinde her çerçevenin yanında bir isim var ("QPSK"). Eğitimde bu isim bir numaraya çevriliyor (QPSK → 1). Ağ sadece numarayı biliyor.
- **Toplantıdaki "image classification" benzetmesi doğru:** Resimlerde kedi 0, köpek 1 gibi; burada BPSK 0, QPSK 1. Ağ sinyalden o numarayı tahmin etmeyi öğreniyor.

**Okuma metni (EN)**

> Question two. There is no separate piece for QPSK. A frame goes into one network. The same layers handle every frame. At the end there are four numbers, one per modulation, and the largest one is the answer. For the network, BPSK is just output 0, QPSK output 1, QAM16 output 2, QAM64 output 3.

**Olası sorular**
- *"So you cannot point to where QPSK is detected?"* — Not to a piece of code. Every weight takes part in every decision. What we can point to is the last layer: one row of weights per output.

---

## Slayt 6 — Soru 3: Hatalar neden dört sinyal arasında kaldı?

**Slaytta ne var, ne demek**
- Moshe toplantıda iki yolu ayırdı; haklıydı.
  - **A:** Ağı sadece bu dört sinyalle eğit. 4 çıkış, hata sadece dördün arasında olabilir. **Tablo buydu.**
  - **B:** 11 sinyalle eğitilmiş ağa sadece bu dördünü ver. 11 çıkış, hata 11'den herhangi birine gidebilir.
- Moshe bir de **C**'yi söyledi: 11 sinyallik ağın diğer 7 çıkışını kapatmak, sadece 4 çıkış arasında en büyüğü seçmek. Bunu yapmadık. Mevcut checkpoint'lerle birkaç dakikalık bir hesap; ama "yeni iş yapma" dendiği için yapmadım.

**Okuma metni (EN)**

> Question three, the one you asked about the table. You described two different approaches, and you were right.
>
> Approach A: train a network on only the four signals. It has four outputs, so its mistakes can only be among the four.
>
> Approach B: use the eleven-signal network and feed it only these four. It has eleven outputs, so its mistakes can go anywhere.
>
> The table you saw was approach A.

**Olası sorular**
- *"Why A and not B?"* — You asked about the four-signal box, so we trained a detector whose whole job is those four. Both are valid; they answer different questions. A asks how well four can be told apart; B asks how often a general detector confuses them with anything.
- *"And with the other seven outputs switched off?"* — That is a third option. It needs no training, only a re-run of the eleven-signal network's test frames; I haven't done it, since you asked for no new work.

---

## Slayt 7 — B yaklaşımı, zaten elimizdeki sonuçlardan

**Slaytta ne var, ne demek**
- **Kaynak:** 11 sinyallik ağın test sonuçları (`train_backbone_rml2016_f1000_colab.npz`, 3 koşu). Bu dört sinyalin satırları, 10 dB. Satır başına 450 test çerçevesi (3 × 150).
- Yani B yaklaşımı, yeni eğitim yapmadan.
- **Sonuç:** 1,800 çerçevede 80 hata. Bunların **21'i dördün dışına** gidiyor: AM-SSB 13, 8PSK 5, GFSK 2, AM-DSB 1.
  - BPSK: 3 dışarı (AM-SSB 2, GFSK 1).
  - QPSK: 4 dışarı (hepsi AM-SSB).
  - QAM16: 30 → QAM64, 2 → QPSK, 4 dışarı (8PSK 3, AM-SSB 1).
  - QAM64: 27 → QAM16, 10 dışarı (AM-SSB 6, 8PSK 2, AM-DSB 1, GFSK 1).
- **Mesaj:** Moshe'nin beklediği gibi, 11 çıkış olunca hataların bir kısmı dışarı gidiyor. Ama yine de çoğu QAM16 ↔ QAM64.
- **Dikkat:** Bu tablonun karar sayısı (450/satır) 4 sinyallik tablodan (2,100/satır) az; 3 koşu, 14 değil.

**Okuma metni (EN)**

> And approach B we already have, without new work. These are the test results of the eleven-signal network on the same four signals, at 10 dB.
>
> As you expected, some mistakes leave the four: 21 of its 80 mistakes, mostly to AM-SSB. Most of the rest are still QAM16 against QAM64.

**Olası sorular**
- *"Why AM-SSB?"* — At −20 dB the eleven-signal network calls most noise AM-SSB; it is that network's "sink". A noisy-looking frame falls there. In the four-signal network the sink was BPSK and QPSK.
- *"Is 21 of 80 a lot?"* — It is 1.2% of the 1,800 frames. The QAM pair is still the main problem.

---

## Slayt 8 — Eğitim, adım 1: veri

**Slaytta ne var, ne demek**
1. **Read the file:** RML2016.10a pickle dosyasından 4 sinyal × 20 seviye × 1,000 = 80,000 çerçeve.
2. **Scale:** Her çerçeve aynı ortalama güce ölçekleniyor; ağ sinyalin seviyesinden kopya çekemesin diye.
3. **Label:** BPSK 0, QPSK 1, QAM16 2, QAM64 3.
4. **Split every cell:** Her (sinyal, seviye) hücresi ayrı bölünüyor: 700 öğrenme, 150 durma kararı (validation), 150 test.
- **Alt çubuk:** 56,000 / 12,000 / 12,000.

**Okuma metni (EN)**

> Question four: how is it trained. First the data. We read the file and keep the four signals: 20 noise levels, 1,000 frames each, 80,000 frames.
>
> Every frame is scaled to the same power. Each frame gets a label, 0 to 3.
>
> Then every signal at every noise level is split: 700 frames to learn from, 150 to decide when to stop, and 150 to test.

**Olası sorular**
- *"Why split every cell separately?"* — So every signal at every noise level has exactly 150 test frames. Otherwise some cells could end up with few test frames by chance.

---

## Slayt 9 — Eğitim, adım 2: öğrenme döngüsü

**Slaytta ne var, ne demek**
- **Take 256 frames:** Bir "batch".
- **Network gives 4 scores each:** İleri geçiş.
- **Compare with the right labels → the loss:** Cross-entropy.
  - Softmax skorları olasılığa çeviriyor.
  - Loss = −log(doğru sınıfın olasılığı).
  - Ağ doğru cevaba ne kadar emin olursa loss o kadar küçük.
- **Nudge all the numbers:**
  - Backpropagation her sayının loss'u nasıl etkilediğini hesaplıyor (türev).
  - AdamW her sayıyı loss'u azaltan yöne küçük bir adım itiyor. Öğrenme oranı 0.001.
- **219 adım = 1 epoch:** 56,000 / 256 = 218.75; son batch küçük.
- **Her epoch sonunda:** 12,000 validation çerçevesinde doğruluk; en iyisi saklanıyor.
  - Doğruluk iyileşmezse öğrenme oranı ×0.2 düşüyor.
  - 20 epoch iyileşme yoksa duruyor (en fazla 100 epoch).
- **16–46:** 14 koşunun en iyi epoch'ları. Hepsi kendiliğinden durdu (en iyi + 20 ≤ 100).

**Okuma metni (EN)**

> Then the learning loop. Take 256 frames. The network gives four scores for each. Compare the scores with the right labels; that gives one number, the loss. Nudge all 785,476 numbers a little in the direction that lowers the loss. Repeat.
>
> 219 of these steps cover all 56,000 learning frames once; that is one epoch. After every epoch we check accuracy on the 12,000 stopping frames and keep the best version. If 20 epochs pass without improvement, we stop.

**Olası sorular**
- *"What is the loss exactly?"* — Cross-entropy: minus the log of the probability the network gives to the right answer. Zero if it is certain and right; large if it is confident and wrong.
- *"How does it know which direction to nudge?"* — Backpropagation computes the derivative of the loss with respect to every number; each number takes a small step against its derivative. That is gradient descent; AdamW is the version we use.

---

## Slayt 10 — Eğitim, adım 3: bir kez test, sonra tekrar

**Slaytta ne var, ne demek**
- **12,000:** Test çerçeveleri. En iyi versiyon sadece en sonda, bir kez çalıştırılıyor.
- **14:** Bütün süreç 14 kez tekrarlandı. Her seferinde farklı bölme ve farklı rastgele başlangıç (seed).
- **8,400:** 10 dB'deki toplam test kararı: 14 koşu × 4 sinyal × 150.

**Okuma metni (EN)**

> Finally the test. The best version is run once on the 12,000 test frames, which it never learned from. That gives the tables I showed you.
>
> We did the whole thing 14 times, each time with a different split and a different random start. At 10 dB that adds up to 8,400 test decisions.

---

## Slayt 11 — Soru 5: Yazılım

**Slaytta ne var, ne demek**
- **Python + PyTorch:** Ağ ve eğitimi. PyTorch hazır bir kütüphane (Facebook/Meta), ama hazır bir dedektör değil: katmanları ve döngüyü biz yazdık.
- **NumPy, scikit-learn:** Veriyi okuma, bölme.
- **Colab GPU, laptop GPU:** Koşuların çalıştığı yer.
- **Bizim kod:**
  - `model_zoo.py`: ağ.
  - `cnn.py`: eğitim döngüsü, bölme, test.
  - `train_backbone.py`: protokol.
- **Claude Code:** Kodu benim talimatlarımla yazan yapay zekâ asistanı.
- **GNU Radio:** Toplantıda "GNU Radio kullandım" dedin. Doğrusu:
  - Veriyi 2016'da DeepSig GNU Radio ile üretti.
  - Biz GNU Radio'yu sadece gürültüyü anlamak için, onların kanal zincirini yeniden kurmakta kullandık.
  - Eğitimde GNU Radio yok.

**Okuma metni (EN)**

> Question five: software. The network and its training are written in Python with PyTorch, a standard library for neural networks. NumPy and scikit-learn handle the data and the split. It ran on Google Colab's GPUs and on my laptop's GPU.
>
> The code was written with Claude Code, an AI assistant, from my instructions.
>
> GNU Radio was not used for training: DeepSig used it to make the dataset, and I used it only to study the noise.
>
> There is no ready-made detector and no pre-trained weights anywhere.

**Olası sorular**
- *"Where did the network design come from?"* — From the 2025 paper, via Esmer's re-implementation of it; our model code was transcribed from that. Then I also rebuilt it from the paper alone, which reproduced their accuracy.

---

## Slayt 12 — Soru 6: Makale

**Slaytta ne var, ne demek**
- **Künye:** El-Haryqy, Kharbouche, Ouamna, Madini, Zouine.
  - Ibn Tofail Üniversitesi, Kenitra, Fas (ENSA, İleri Sistem Mühendisliği Laboratuvarı).
  - Results in Engineering 26 (2025) 104783. Gönderim Aralık 2024, yayın 4 Nisan 2025. Açık erişim.
- **Yayımladıkları:** Katman listesi (Şekil 2) ve eğitim tarifi (Tablo 2).
- **Yayımlamadıkları:** Kod ve ağırlıklar. "Data will be made available on request" diyorlar.
- Makalenin ayrıntılı incelemesi ayrı bir sunumda: `moshe_paper_icrnna.pptx`, yedekte dursun.

**Okuma metni (EN)**

> Question six: the paper. It is by El-Haryqy and four colleagues at Ibn Tofail University in Kenitra, Morocco, published in Results in Engineering in April 2025.
>
> They describe the layers and the training recipe. They did not publish code or weights; the paper says data are available on request. So the code had to be written from the description.

**Olası sorular**
- *"Did you get the same result as them?"* — Yes. Built from the paper alone and trained to convergence: 63.21% against their 63.24%. The details are in the paper deck if you want them.

---

## Slayt 13 — Yazarlar başka ne yapmış?

**Slaytta ne var, ne demek**

Hepsi makalenin kendi kaynakçasındaki öz-atıflar ([1], [4], [5]/[12], [8]):

- **2024:** El-Haryqy, Madini, Zouine. Bilişsel radyoda spektrum algılama ve tahmin için derin öğrenme üzerine derleme. Int. J. Comput. Appl.
- **2024:** El-Haryqy, Madini, Zouine. Mask R-CNN ile RF girişim tespiti ve modülasyon tanıma. IJCNC 16(5).
- **2023:** El-Haryqy, Madini, Zouine, Kharbouche. Derin öğrenmeyle otomatik sinyal tespiti üzerine bir derleme. ICOA konferansı.
- **2021:** Kharbouche, Madini, Zouine. Araçlar arası (V2X) iletişim için DS-OCDMA/VLC (optik) analizi. ICOA konferansı.

Makalede Madini ve Zouine "supervision", El-Haryqy "writing, methodology" olarak geçiyor. İnternette ayrıca aramadım; bu ortamdan dış siteler kapalı.

**Okuma metni (EN)**

> What else have they done? From the paper's own reference list: a review of deep learning for spectrum sensing in cognitive radio, a survey of signal detection with deep learning, and a paper on interference detection and modulation recognition with Mask R-CNN. One co-author also works on optical communication for vehicles. I did not search beyond the paper.

---

## Slayt 14 — QAM16, 10 dB, dedeksiyon terimleriyle

**Slaytta ne var, ne demek**

Toplantıda Moshe QAM16 sayılarını kendisi okudu. Onun diliyle:

- **Miss:** 2,100 QAM16'dan 292'si kaçırıldı. Miss oranı 13.9%. Moshe "almost 14%" dedi, doğru.
- **False alarm:** "QAM16" denip QAM16 olmayan 128 çerçeve (QPSK 2, QAM64 126).
  - Moshe "128 out of 2100" dedi. Doğru payda, QAM16 **olmayan** 6,300 çerçeve: false-alarm oranı 2.0%.
- **Precision:** "QAM16" denen 1,936 çerçevenin 1,808'i doğru: 93.4%.
- **Recall = 1 − miss oranı:** 86.1%.

**Okuma metni (EN)**

> Last week you read the QAM16 numbers in detection terms, so here they are that way. 2,100 QAM16 frames were sent; 292 were missed, a miss rate of 13.9 percent.
>
> 128 frames were called QAM16 that were not. Out of the 6,300 frames that were not QAM16, that is a false-alarm rate of 2 percent.
>
> Recall is one minus the miss rate.

**Olası sorular**
- *"And precision?"* — 1,808 of the 1,936 QAM16 calls were right: 93.4%. One minus precision, 6.6%, is the share of QAM16 calls that were false alarms; it is not the false-alarm rate.

---

## Slayt 15 — Özetle

**Okuma metni (EN)**

> To sum up. The four-signal table came from a network trained only on those four, which is why its mistakes stay among them.
>
> Every network was trained by us, from a random start, with no outside weights. And there is one network for all classes, not one piece per modulation.
>
> I'm happy to go deeper into any step.

---

## Genel tavsiyeler

- **Emin olmadığında söyle:** Toplantıdaki sorun, emin olmadan "evet" demekti. Bilmediğin bir şey sorulursa "I'll check in the code and come back" de.
- **Moshe'nin diliyle konuş:** O dedeksiyon teorisinden geliyor. Miss, false alarm, P(say k | k sent) gibi terimler ona tanıdık; slayt 14 bunun için var.
- **A, B ve C yaklaşımlarını karıştırma:**
  - **A:** 4 sinyalle eğitilmiş, 4 çıkışlı ağ. 4 sinyallik tablo bu.
  - **B:** 11 sinyallik ağa sadece 4'ü vermek. Slayt 7'deki tablo bu.
  - **C:** 11 sinyallik ağın 7 çıkışını kapatmak. Yapılmadı.
- **Moshe'nin bahsettiği yeni makale:** Ağustos 2026, Çinli mühendisler. Bizde yok. Bulursan bana gönder; aynı şekilde inceleriz.

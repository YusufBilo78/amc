# Makaleyi birlikte okuyalım — ICRNNA (El-Haryqy ve ark., 2025)

**Makale:** El-Haryqy, Kharbouche, Ouamna, Madini, Zouine. *Improved automatic modulation recognition using deep learning with additive attention.* Results in Engineering 26 (2025) 104783.

- 14 sayfa, 5 bölüm.
- Sayfa numaraları makalenin kendi sayfaları.

**Nasıl okunur:** Her kart makalenin bir bölümü, makaledeki sırayla. Her kartta dört kısım var:

- **Ne diyor:** Makalenin o bölümde söyledikleri, kısaca.
- **Ne demek:** Kavramların sade açıklaması.
- **Doğru mu?** Üç işaret kullanılıyor:
  - ✅ tutuyor;
  - ⚠️ dikkat, eksik ya da abartılı;
  - ❌ makalenin kendi tablolarıyla ya da bilinen bir sayıyla çelişiyor.
- **Moshe sorarsa (EN):** Kısa, sesli söylenecek cevap.

**Sayıların kaynağı:**

- **Makaleden:** Tablo ya da şekil numarasıyla, metinden kopyalandı.
- **Onların kodundan:** 8 Ağustos 2026'da yayımlanan kod, commit `ac000a0` (`OFFICIAL_CODE.md`).
- **Bizim koşulardan:** Repodaki sonuç dosyaları.

Sunumun kendisi `STUDY_paper.html`'de. Bu sayfa makalenin kendisini anlamak için.

---

## Büyük resim — beş iddia ve hükmü

**Ne diyor**

Model dört parça:

- konvolüsyon (CNN);
- iki katlı çift yönlü LSTM (BiLSTM);
- "geliştirilmiş" toplamsal attention;
- küçük bir sınıflandırıcı.

RML2016.10a'da %63.24, 10b'de %65.39 ile karşılaştırdığı 12 modelin en iyisi. Düşük SNR'de dayanıklı ve hesap açısından hafif olduğunu söylüyor.

**Doğru mu?**

1. **En yüksek doğruluk: 63.24 (10a), 65.39 (10b)** (Özet, Tablo 3)
   - ✅ Sayı gerçek: makaleden kurduğumuz model 63.21 aldı.
   - ⚠️ İkinciye fark 0.51 puan ve tek koşudan. Bizim koşularda aynı modelin tohumdan tohuma farkı 0.47 puan, LSTM2'nin 1.14 puan.
2. **Özellikle düşük SNR'de dayanıklı** (Özet, Giriş, Tablo 5)
   - ❌ 10a'da −10, −8 ve −6 dB'de başka modeller daha yüksek; bu kendi Tablo 5'lerinde yazıyor.
   - ✅ 10b'de (Tablo 4) her seviyede birinci, ama bazı seviyelerde fark puanın onda biri kadar.
3. **Hafif ve verimli, 0.79 M parametre** (Tablo 7)
   - ✅ 0.79 M doğru.
   - ❌ Tablo "MCLDNN 406 M" diyor; gerçekte 406,199, yani 0.41 M. ICRNNA, MCLDNN'in yaklaşık 2 katı, PET-CGDNN'in 11 katı.
4. **Her parça hayati (ablation)** (Tablo 6)
   - ⚠️ Sadece BiLSTM 63.13, tam model 63.24: fark 0.11 puan, tek koşu. İşi büyük ölçüde BiLSTM yapıyor.
5. **Attention yorumlanabilirlik sağlar** (Giriş)
   - ⚠️ Makalede tek bir attention ağırlığı bile gösterilmiyor.

**Tek cümleyle:** Model gerçek ve sonucu tekrar üretilebiliyor; "en iyi" ve "düşük SNR'de üstün" iddiaları ise kendi tablolarının desteklediğinden daha güçlü yazılmış.

**Moshe sorarsa (EN)**

- *"Is the paper right?"* — "The headline number is right: we rebuilt the model from the text and got 63.21 against their 63.24. The claims around it are stronger than their own tables. At low SNR on 10a other models are better, and the lead is half a point from a single run."

---

## Sözlük — makalede geçen terimler

**Sinyal tarafı**

- **AMR / AMC** — Otomatik modülasyon tanıma. Bir sinyal parçasına bakıp hangi modülasyonla gönderildiğini söylemek. Makale AMR diyor, biz AMC diyoruz; aynı şey.
- **I/Q** — Karmaşık taban bant örneğinin gerçek (I) ve sanal (Q) kısmı. Bir çerçeve 128 tane (I, Q) çifti.
- **AWGN** — Toplamsal beyaz Gauss gürültüsü: her örneğe bağımsız rastgele gürültü eklenir.
- **SNR etiketi** — Veri setinde her çerçevenin yanında yazan dB değeri. Bu veri setinde gerçek SNR değil (Bölüm 2).

**Modelin parçaları**

- **CNN / konvolüsyon** — Sinyal boyunca kaydırılan kısa bir filtre. Katsayıları öğrenilen bir FIR filtre gibi.
- **Kernel (filtre) boyutu** — Filtrenin bir anda baktığı örnek sayısı; burada 3.
- **ReLU** — Negatif sayıları sıfırlar, pozitifleri bırakır. Ağı doğrusal olmaktan çıkarır.
- **Max pooling** — Yan yana iki değerden büyüğünü tutar; dizi yarıya iner (128 → 64).
- **LSTM** — Diziyi adım adım okuyan ve bir hafıza taşıyan katman. Öğrenilen "kapılar" neyin tutulup neyin unutulacağına karar verir.
- **BiLSTM** — İki LSTM: biri baştan sona, biri sondan başa okur; çıktıları yan yana konur.
- **Attention** — Her zaman adımına bir puan verir. Puanları toplamı 1 olan ağırlıklara çevirir ve adımların ağırlıklı ortalamasını alır.
- **Query / Key / Value** — Attention'ın çeviri modellerinden gelen dili: neyi aradığın, neyle karşılaştırdığın, neyin ortalamasını aldığın (Bölüm 3.3).
- **Softmax** — Bir sayı listesini, toplamı 1 olan pozitif ağırlıklara (olasılıklara) çevirir.
- **Dense katman** — Her girişin her çıkışa bağlı olduğu katman; bir matris çarpımı.

**Eğitim**

- **Dropout** — Eğitim sırasında sayıların bir kısmını (burada %30) rastgele sıfırlar, ağ tek bir sayıya yaslanamasın diye. Testte kapalıdır.
- **Batch normalization (BN)** — Eğitim sırasında her özelliği standart bir aralığa ölçekler; eğitimi kararlı yapar.
- **Layer normalization** — Tek bir vektörü sıfır ortalama ve birim yayılıma ölçekler.
- **L2 düzenlileştirme** — Büyük ağırlıkları cezalandırır; burada katsayısı 1e-4.
- **One-hot** — Sınıf etiketini, doğru sınıfta 1, diğerlerinde 0 olan bir vektör olarak yazmak.
- **Epoch** — Eğitim setinin tamamından bir kez geçmek.
- **Batch** — Bir güncellemede birlikte işlenen çerçeve sayısı; burada 32.
- **Öğrenme oranı (learning rate)** — Her güncellemenin adım büyüklüğü.
- **Callback** — Eğitim sırasında her epoch sonunda çalışan kural. Üç tane var:
  - **EarlyStopping** — Doğrulama sonucu bir süre iyileşmezse eğitimi durdurur.
  - **ReduceLROnPlateau** — Bir süre iyileşme yoksa öğrenme oranını düşürür.
  - **ModelCheckpoint** — En iyi modeli diske kaydeder.
- **Grid search** — Ayar değerlerinin her kombinasyonunu deneyip en iyisini seçmek.
- **Eğitim / doğrulama / test** — Öğrenilen veri / ne zaman durulacağına karar verilen veri / sonucun sayıldığı, hiç dokunulmayan veri.

**Sonuçların ölçüsü**

- **Doğruluk (accuracy)** — Doğru kararlar / tüm kararlar.
- **Recall** — Gerçekten X olan çerçevelerin ne kadarına X dendi.
- **Precision** — Model X dediğinde ne sıklıkla gerçekten X'ti.
- **F1** — Recall ile precision'ın harmonik ortalaması.
- **Weighted average** — Sınıf skorlarını sınıf büyüklüğüne göre ortalamak. Burada her sınıf eşit büyüklükte, yani düz ortalama.
- **Karışıklık matrisi** — Satır gerçek sınıf, sütun modelin dediği; köşegen doğrular. Satır yüzdeleri recall'dur.
- **Ablation** — Modelin parçalarını teker teker çıkarıp neyin ne kadar katkı verdiğine bakmak.

**Hesap yükü**

- **Parametre** — Modelin öğrendiği sayılar.
- **FLOPs / MFLOPs** — Bir çerçeveyi sınıflandırmak için gereken kayan noktalı işlem sayısı (milyon cinsinden).
- **Çıkarım süresi (inference time)** — Eğitilmiş modelin karar vermesi için geçen süre.

---

## Bölüm 0 — Künye ve özet (s. 1)

**Ne diyor**

- **Başlık:** *Improved automatic modulation recognition using deep learning with additive attention.*
- **Yazarlar:** Beş yazar, Ibn Tofail Üniversitesi, Kenitra, Fas.
- **Tarihler:** Geliş 15 Aralık 2024, revizyon 22 Mart 2025, kabul 2 Nisan 2025, çevrimiçi 4 Nisan 2025.
- **Dergi:** *Results in Engineering* (Elsevier), açık erişim, CC BY-NC.
- **Özet:**
  - ICRNNA = CNN + BiLSTM + attention.
  - RML2016.10a ve 10b'de en yüksek doğruluk: 63.24 ve 65.39.
  - 48.42 MFLOPs ve 0.79 M parametre.
  - "Gerçek zamanlı uygulamalara uygun".

**Ne demek**

- **ICRNNA:** *Improved Convolutional Recurrent Neural Network with Attention.*
- **"Additive attention":** 2014'teki Bahdanau attention'ı. ChatGPT'deki transformer attention'ı değil.
- **Hakem süreci:** Geliş ile kabul arası yaklaşık 3.5 ay.
- **Dergi:** Genel bir mühendislik dergisi; sinyal işlemeye özel değil.

**Doğru mu?**

- ✅ 63.24 tekrar üretildi (Bölüm 4.1).
- ⚠️ **"State-of-the-art":** Sadece kendi seçtikleri 12 modelin en iyisi demek. Girişte kendileri IQFormer [22] ve DH-TR [21]'yi anıyor, IQFormer'ın 10a/b'de en iyi yöntemleri geçtiğini yazıyor; ama bu modeller hiçbir sonuç tablosunda yok.
- ⚠️ **"Düşük SNR'de dayanıklı":** Bölüm 4.2.2'ye bakın.
- ⚠️ **"Gerçek zamanlı":** Hiçbir cihazda gerçek zamanlı test yok. Tek bir çıkarım süresi var (0.2313 s); neyin süresi olduğu yazmıyor.

**Moshe sorarsa (EN)**

- *"Where is it published?"* — "Results in Engineering, Elsevier, open access, April 2025. A group at Ibn Tofail University in Morocco."

---

## Bölüm 1 — Giriş (s. 1–2)

**Ne diyor**

- **AMR neden önemli:** Bilişsel radyo, spektrum izleme, IoT.
- **Klasik yöntemler iki tür:**
  - **Olabilirlik tabanlı (LB):** Teoride optimal, ama hesap ağır ve kanal modeline bağlı.
  - **Öznitelik tabanlı (FB):** Kümülantlar, dalgacık, döngüsel özellikler; gürültüye hassas.
- **Derin öğrenme:** Özellikleri ham veriden kendisi öğreniyor.
- **Literatür özeti:**
  - transfer learning [15];
  - CGDNet [16];
  - CNN-BiLSTM-DNN, attention'lı [17];
  - transformer ile yarı denetimli öğrenme [18];
  - bir derleme [19];
  - MCLDNN [20], DH-TR [21], IQFormer [22].
- **Boşluk:** Çalışmalar yüksek SNR'ye odaklanmış; düşük SNR'de attention ile iyileştirme fırsatı var.
- **Üç katkı:**
  1. CNN yerel özellikleri, BiLSTM zamansal ilişkileri yakalıyor.
  2. "Geliştirilmiş" attention: ölçekleme, dropout ve layer normalization ekleniyor; yorumlanabilirlik sağlıyor.
  3. Her seviyede düzenlileştirme: dropout, BN, L2.

**Ne demek**

- **LB:** Her modülasyon için "bu sinyal bu modülasyondan gelmiş olsaydı ne kadar olası olurdu" diye hesaplayıp en olasıyı seçer.
- **FB:** Elle tasarlanmış istatistikler hesaplayıp bir sınıflandırıcıya verir. Bizim projedeki kümülant + SVM tabanı tam olarak budur (`baseline_results.npz`).
- **Derin öğrenme:** Ham I/Q'yu alıp istatistikleri de kendisi öğrenir.

**Doğru mu?**

- ⚠️ **Tekrar eden paragraf:** Bir paragraf iki kez basılmış ("Despite their theoretical advantages…", s. 1 ve s. 2). Editörlükten kaçmış.
- ⚠️ **Yanlış kaynak:** LB yöntemleri için [11] gösteriliyor; [11] bir derin öğrenme makalesi (ResNeXt-GRU).
- ⚠️ **Model yeni bir fikir değil:** CNN + BiLSTM + attention zaten vardı; kendi kaynakları [17] (CNN-BiLSTM-DNN) böyle.
- ⚠️ **Katkı 2:** Attention'a eklenen üç şey küçük ayarlar:
  - öğrenilen tek bir çarpan;
  - ağırlıklara %10 dropout;
  - çıkışa LayerNorm.
- ⚠️ **"Yorumlanabilirlik":** Hiçbir attention haritası gösterilmemiş.
- ✅ **Motivasyon makul:** Düşük SNR gerçekten zor bölge. Ama iddianın tutup tutmadığını Bölüm 4.2.2'de göreceğiz.

**Moshe sorarsa (EN)**

- *"What is new in it?"* — "Not the combination: CNN, LSTM and attention together existed before, for example their reference 17. What they add is a tuned version: three small changes to the attention and regularization everywhere. It's an engineering improvement, not a new idea."

---

## Bölüm 2 — Sistem modeli ve veri (s. 3, Tablo 1)

**Ne diyor**

- **Denklem 1:** Alınan sinyal = gönderilen sinyal + AWGN.
- **SNR:** −20 dB'den +18 dB'ye, 2 dB adımlarla.
- **Denklem 2:** Girdi, 128 tane (I, Q) çiftinden oluşan bir dizi.
- **Veri (Bölüm 2.2):**
  - **RML2016.10a:** 11 sınıf; 8 dijital, 3 analog (WBFM, AM-SSB, AM-DSB). 220,000 çerçeve, her biri 128 × 2. GNU Radio ile simüle.
  - **RML2016.10b:** 10 sınıf, AM-SSB yok. 1.2 milyon çerçeve.
  - Tablo 1 ikisini karşılaştırıyor.

**Ne demek**

- **Çerçeve sayısı:** Sınıf başına 20,000 çerçeve = 20 SNR seviyesi × 1,000 çerçeve.
- **Çerçeve uzunluğu:** 128 örnek = 16 sembol (sembol başına 8 örnek). Bu, modelin gördüğü her şey.
- **I/Q:** Sinyalin karmaşık taban bant hali. I ve Q birlikte genliği ve fazı verir.

**Doğru mu?**

- ⚠️ **Denklem 1 eksik:** Sadece AWGN diyor. Ama Bölüm 4'te kendileri verinin frekans kayması, çok yollu sönümleme ve örnekleme hızı kayması da içerdiğini yazıyor; gerçek veri ikincisi.
- ❌ **"BFSK":** Sınıf listesinde böyle yazıyor; doğrusu GFSK (Tablo 1 doğru).
- ⚠️ **Boyut sırası:** s. 3'te "[128, 2]", s. 3–4'te "(2, 128)". Aynı şey, sadece sıra farkı.
- ⚠️ **SNR etiketi gerçek SNR değil.** Makale etiketi gerçek SNR gibi kullanıyor. Bizim ölçümümüz (`NOISE_2016.md`):
  - Gerçek SNR etiketle yaklaşık 2 dB / dB eğimle artıyor.
  - Aynı etikette sınıflar farklı. Etiket 0'da ölçülen SNR: BPSK 4.8, QPSK 5.1, QAM16 14.0, QAM64 17.2 dB.
  - Yani s. 8 ve s. 11'deki "0 dB'de sinyal ve gürültü eşit" cümlesi bu veri için doğru değil.
- ⚠️ **10b'deki "ek kanal etkileri":** Kaynak gösterilmemiş. Tablo 1'deki farklar sadece şunlar: bir sınıf eksik (AM-SSB) ve sınıf başına 6 kat veri.

**Moshe sorarsa (EN)**

- *"What is 0 dB in their tables?"* — "The dataset's label, not a measured SNR. We measured it: at label 0, QPSK is about 5 dB and QAM16 about 14 dB. So 'signal and noise balanced at 0 dB' is not true for this data."

---

## Bölüm 3 — Modelin genel yapısı (s. 3–4, Şekil 1–2)

**Ne diyor**

- **Şekil 1, ön işleme:** Ham I/Q → normalizasyon → karıştırma → etiket kodlama (one-hot).
- **Şekil 2, dört parça:**
  - **A — CNN:** Conv 64 filtre (k=3) + BN, Conv 128 filtre (k=3) + BN, MaxPool 2, Dropout 0.3.
  - **B — RNN:** BiLSTM 128 + BN + Dropout 0.3, iki kez.
  - **C — Attention:** Ölçek, dropout 0.1, LayerNorm.
  - **D — Dense:** 128 + BN + Dropout 0.3, sonra 64 + BN + Dropout 0.3.
  - En sonda softmax ile 11 olasılık.

**Ne demek**

Bir çerçevenin yolculuğu ve her bloktaki sayılar (onların kodundan yazdırıldı):

| adım | çıkan boyut | parametre | pay |
|---|---|---|---|
| giriş | 2 × 128 | — | — |
| Conv 64, Conv 128, pool | 128 × 64 → 128 × 128 → 64 × 128 | 25,920 | %3 |
| BiLSTM × 2 | 64 × 256 → 64 × 256 | 659,456 | %83 |
| attention | 256 | 66,177 | %8 |
| dense 128, 64, çıkış | 128 → 64 → 11 | 42,635 | %5 |
| **toplam** | | **794,188** | |

- **Toplam parametre:** BatchNorm istatistikleri dahil. 792,396'sı öğreniliyor.
- **Dört adımda akış:** Kısa parçalara bak → sırayla oku → önemli yerleri ağırlıklandır → karar ver.

**Doğru mu?**

- ✅ **Şekil 2 kodla birebir aynı:** Kernel 3, tek pool, her LSTM'den sonra BN.
- ⚠️ **Boyutlar sadece şekilde:** Kernel ve pool boyutu metinde yazmıyor.
- ⚠️ **Normalizasyon tanımsız:** Ne olduğu yazmıyor. Kodda her zaman indeksi için ortalama ve varyans alınıyor, 220,000 çerçevenin hepsinden, yani test seti de dahil, ayırmadan önce. Küçük bir sızıntı.
- ⚠️ **Attention'ın içi tanımsız:** Boyutları ve girişleri makalede yok (Bölüm 3.3).
- **Bizim sürümler:**
  - **Makaleden kurduğumuz model** (`colab/icrnna_faithful_2016.py`): 794,827 parametre.
  - **Projede kullandığımız sürüm** (`model_zoo.ICRNNA`): 786,379 parametre. Beş yerde farklı:
    - ilk konvolüsyon 5'lik;
    - iki pool;
    - tek BN;
    - attention'da ek yok;
    - tek dense katman.

**Moshe sorarsa (EN)**

- *"Why CNN first, then LSTM?"* — "The convolutions turn each short stretch of samples into a richer description and halve the length to 64 steps. The LSTM then reads those 64 descriptions in order. Reading 64 rich steps is easier than reading 128 raw samples."

---

## Bölüm 3.1 — CNN (s. 4)

**Ne diyor**

- CNN'in tanımı.
- **Denklem 3:** İki boyutlu konvolüsyon formülü (bir görüntü üzerinde).
- Ardından BN, MaxPool, Dropout ve ReLU'nun ne işe yaradığı.

**Ne demek**

- **Konvolüsyon burada tek boyutlu.** 3 örnek uzunluğunda bir filtre zaman boyunca kayıyor; her konumda çarpıp topluyor. Katsayıları öğrenilen bir FIR filtre gibi.
- **Birinci katman:** Her filtre 3 zaman × 2 kanal (I ve Q) = 6 katsayı + 1 sabit. 64 filtre = 448 parametre.
- **İkinci katman:** 3 × 64 × 128 + 128 = 24,704 parametre.
- **Görüş alanı:** İki 3'lük konvolüsyon ve bir pool'dan sonra her çıkış adımı yaklaşık 6 örnek görüyor. Bu bir sembolden (8 örnek) az. Yani CNN tek başına semboller arası deseni göremiyor; o iş LSTM'ye kalıyor.

**Doğru mu?**

- ⚠️ **Denklem 3 bu modeli tarif etmiyor.** Görüntü için yazılmış (i, j iki boyut); model tek boyutlu konvolüsyon kullanıyor. Yanlış değil ama alakasız.
- ⚠️ **"Spatial features" diyorlar.** Burada uzay yok, zaman var. Doğru ifade "yerel zaman özellikleri".
- ⚠️ **Yanlış kaynak:** CNN tanımı için [23] gösterilmiş; o, O'Shea'nın veri seti makalesi.
- ✅ **Ablation tutarlı:** Sadece CNN %59.93; en zayıf parça bu (Bölüm 4.4).

**Moshe sorarsa (EN)**

- *"What does the convolution learn?"* — "A bank of short learned filters, three samples long: 64, then 128 of them, like learned FIR filters. Each output step sees about 6 samples, less than one symbol."

---

## Bölüm 3.2 — BiLSTM (s. 4–5)

**Ne diyor**

- LSTM uzun bağımlılıkları öğrenmek için var. Düz RNN'ler "kaybolan gradyan" sorunu yüzünden uzun dizilerde unutur.
- BiLSTM diziyi bir de tersten okur.
- Modelde 128 birimlik iki BiLSTM var; her birinden sonra BN ve dropout geliyor.

**Ne demek**

- **LSTM:** 64 adımı sırayla okur ve 128 sayılık bir hafıza taşır. Üç öğrenilen kapı var:
  - **unut:** eski hafızanın ne kadarı silinsin;
  - **yaz:** yeni adımdan ne eklensin;
  - **çıkar:** hafızanın ne kadarı dışarı verilsin.
- **Çift yön:** Biri baştan sona, biri sondan başa okur. Her adımda 128 + 128 = 256 sayı çıkar; böylece her adımın çıktısı hem öncesini hem sonrasını bilir.
- **Ağırlık:** Toplam 659,456 parametre, modelin %83'ü. Bu model büyük ölçüde bir BiLSTM.
- **Neden önemli:** Bir modülasyonu ele veren şey, sembollerin faz ve genliğinin sembolden sembole nasıl dizildiği. Bu deseni okuyan parça LSTM.

**Doğru mu?**

- ⚠️ **Yön kelimeleri ters:** "Past (backward) and future (forward)" diyorlar. İleri okuyan LSTM geçmişi, geri okuyan geleceği taşır; kelimeler yer değiştirmiş. Küçük ama dikkat çeker.
- ⚠️ **"Uzun bağımlılık" abartılı:** Dizi sadece 64 adım.
- ✅ **En güçlü kanıt kendi ablation'ları:** Sadece BiLSTM %63.13, tam model %63.24. Modelin kalbi bu.

**Moshe sorarsa (EN)**

- *"Why does the LSTM matter so much?"* — "Because a modulation is a pattern across symbols, not inside one. The LSTM is the part that reads across symbols. Their own ablation shows it: the LSTM alone gets 63.1 of the 63.2."

---

## Bölüm 3.3 — Geliştirilmiş toplamsal attention (s. 5)

**Ne diyor**

- **Bahdanau attention:** Query, Key, Value.
- **Denklem 4:** puan = v · tanh(W1·q + W2·k).
- **Denklem 5:** Puanlar softmax ile ağırlığa çevrilir.
- **Denklem 6:** Bağlam vektörü = değerlerin ağırlıklı toplamı.
- **"Improved":** Ölçekleme ("if scaling is enabled"), dropout ve LayerNorm eklenmiş.

**Ne demek**

- **Özü:** 64 adımın her birine bir puan ver; softmax ile toplamı 1 olan ağırlıklara çevir; 64 adımın ağırlıklı ortalamasını al. Sonuç, tüm çerçeveyi özetleyen 256 sayılık tek bir vektör. Yani "öğrenilen ağırlıklı ortalama".
- **Query/Key/Value çeviri modellerinden gelen dil.** Orada query, çıktı cümlesinde o an üretilen kelimedir; bu modelde öyle bir şey yok.
- **Kodda gerçekte olan:**
  - query = ikinci LSTM'nin BN ve dropout'tan geçmiş çıkışı;
  - key = value = aynı çıkışın ham hali;
  - her adım sadece kendi içeriğine göre puanlanıyor. Adımlar birbiriyle karşılaştırılmıyor.
- **"Improved"in üç eki:**
  1. **Ölçek:** Puanlar 1'den başlayan, öğrenilen tek bir sayıyla çarpılıyor. Bu, ağırlıkların ne kadar sivri olacağını ayarlıyor.
  2. **Dropout:** Eğitimde ağırlıkların %10'u rastgele sıfırlanıyor.
  3. **LayerNorm:** Çıkan vektör ölçekleniyor.
- **Parametre:** W1 32,768 + W2 32,768 + v 128 + ölçek 1 + LayerNorm 512 = 66,177.

**Doğru mu?**

- ⚠️ **Anlatım genel:** Query/Key/Value ders kitabı anlatımı; bu modelde neyin query olduğu makalede yazmıyor, kodda yazıyor.
- ⚠️ **Ölçek açık mı?** "If scaling is enabled" deniyor ama açık mı kapalı mı söylenmiyor (kodda açık).
- ⚠️ **Katkısı küçük:** Ablation'da attention'sız %62.80, attention'lı %63.24: +0.44 puan, tek koşu.
- ⚠️ **Kanıtsız iddia:** "Düşük SNR'de özellikle yardımcı" deniyor ama SNR'ye göre ablation verisi yok.
- ⚠️ **Ağırlıklar hiç çizilmemiş:** "Model nereye bakıyor?" sorusu cevapsız.

**Moshe sorarsa (EN)**

- *"What does the attention do here?"* — "It's a learned weighted average over the 64 time steps. Each step gets a score, a softmax turns the scores into weights, and the 64 vectors are averaged into one. Without it you would take the last step or a plain average. In their ablation it adds about half a point."

---

## Bölüm 4.1 — Deney düzeni (s. 5–6, Tablo 2)

**Ne diyor**

- **Veri:** Pickle dosyasından okunuyor; normalizasyon, karıştırma, one-hot.
- **Donanım:** Google Colab GPU (15 GB GPU belleği).
- **Ayırma:** %70 eğitim, %15 doğrulama, %15 test.
- **Ayar seçimi:** Öğrenme oranı, dropout ve L2 için grid search. Batch, epoch ve öğrenme oranı düşüşü elle ayarlanmış.
- **Callback'ler:**
  - EarlyStopping: doğrulama kaybı durunca eğitimi bitirir;
  - ReduceLROnPlateau;
  - ModelCheckpoint.
- **Tablo 2, 10a sütunu:**
  - öğrenme oranı 0.001; 5 epoch iyileşme yoksa ×0.2 (en az 1e-6);
  - batch 32;
  - dropout 0.3;
  - L2 1e-4;
  - epoch 58.
- **Tablo 2, 10b sütunu:** Öğrenme oranı her epoch ×0.95, batch 256, epoch 40.

**Ne demek**

- **Çerçeve sayıları:** 220,000 = 154,000 eğitim + 33,000 doğrulama + 33,000 test.
- **Doğrulama seti** ne zaman durulacağına ve öğrenme oranının ne zaman düşeceğine karar verir. **Test seti** sadece en sonda bir kez kullanılır.
- **Güncelleme sayısı:** Batch 32 ile epoch başına yaklaşık 4,813 güncelleme; 58 epoch yaklaşık 279,000 güncelleme.
- **"58 epoch" seçilmiş bir sayı değil.** Kodda kural şu:
  - en fazla 100 epoch;
  - doğrulama kaybı 6 epoch iyileşmezse dur ve en iyi ağırlıklara dön.
  - 58, durduğu yer.

**Doğru mu?**

- ✅ **Üçe bölme doğru yapılmış:** Test seti seçimde kullanılmıyor.
- ⚠️ **Tek ayırma, tek koşu:** Sonuç tek bir rastgele seçime bağlı; hata payı yok. Üstelik kodda ayırma Python'un küme sırasına bağlı; aynı script her çalıştırmada farklı bir ayırma çekiyor.
- ⚠️ **Normalizasyon sızıntısı:** İstatistikler test dahil tüm veriden hesaplanıyor.
- ⚠️ **Makalede olmayan bir ayar:** Kodda gradient clipping var (`clipnorm=1.0`).
- ⚠️ **Grid search belirsiz:** Hangi değerlerin denendiği verilmemiş.
- ⚠️ **İki farklı tarif:** 10a ve 10b farklı ayarlarla eğitilmiş. Rakiplerin hangi tarifle eğitildiği hiç yazmıyor.
- **Bizim deneyimimiz:**
  - Makaleden kurduğumuz model sabit 58 epoch'ta %61.75 ± 0.07 aldı (3 koşu).
  - 150 epoch tavanıyla epoch 107'de %63.21'e ulaştı (1 koşu).
  - Yani bizim kuralımızla 58 yetmedi. Onların kuralı farklı: kayba bakan erken durdurma ve en iyi ağırlıklara dönüş.
  - Colab'daki `official_2016` koşusu bunu onların kendi koduyla ölçüyor.

**Moshe sorarsa (EN)**

- *"Why 58 epochs?"* — "It wasn't chosen. Their code trains up to 100 epochs and stops when the validation loss hasn't improved for 6; for them that happened at 58."

---

## Bölüm 4.2 — Ana sonuç: Tablo 3 (s. 6)

**Ne diyor**

**12 modelin RML2016.10a doğruluğu:**

| model | 10a |
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

- **Yorumları:**
  - sadece CNN'den oluşan modeller zayıf;
  - CNN + RNN hibritleri daha iyi;
  - transformer kötü;
  - 10b'de herkes daha iyi.

**Ne demek**

- **"Ortalama doğruluk":** Test setinin tamamında, 20 SNR seviyesinin hepsi dahil, doğru karar oranı.
- **Şans seviyesi:** 11 sınıfta %9.1.
- **Neden 60'larda kalıyor:** En gürültülü seviyelerde herkes şansa yakın; ortalama bu seviyeleri de içerdiği için en iyi model bile %60'larda kalır.
- **Bu sayıyı yüksek SNR doğruluğuyla karıştırma:** Yüksek SNR'de doğruluk %93 civarında.

**Doğru mu?**

- ✅ 63.24 tekrar üretildi (bizim: 63.21).
- ⚠️ **Birincilik belirsiz:** İkinciyle fark 0.51 puan ve tek koşudan. Bizim koşularda aynı modelin tohumdan tohuma farkı 0.47 puan, LSTM2'ninki 1.14. Birinci ve ikinci istatistiksel olarak ayırt edilemiyor.
- ⚠️ **Rakip sayılarının nereden geldiği yazmıyor.** Makalenin yapısından iki ipucu var (tahmin, doğrulanmadı):
  - **Kendileri koşmuş görünen 6 rakip:** CNN-BiLSTM-DNN, MCLDNN, MCNet, CGDNet, ResNet ve IC-AMCNet, Tablo 5'te ve Şekil 11'de de var; karışıklık matrisleri bile var.
  - **Başka makaleden alınmış görünen 5 rakip:** MAMCA, PET-CGDNN, DAE, Transformer ve CVCNN sadece Tablo 3 ve 7'de geçiyor. Üçü [31]'e (MAMCA makalesi) atıfla geliyor; büyük ihtimalle oradan alınmış, başka koşullarda.

**Bizim karşılaştırmamız** (aynı veri, aynı ayırma, 3 koşu, `literature_2016.md`):

| model | genel | ≥10 dB |
|---|---|---|
| ICRNNA (proje sürümü) | 62.23 | 91.48 |
| LSTM2 | 61.44 | 91.54 |
| MCLDNN | 61.41 | 91.06 |
| PET-CGDNN | 60.98 | 90.64 |
| VT-CNN2 | 56.07 | 81.96 |

Sıra aynı, ama ilk dört 1.25 puan içinde ve ≥10 dB'de hepsi ~%91.

- ⚠️ **Aynı model, farklı makalede farklı sayı:** PET-CGDNN makalesinin kendi tablosunda MCLDNN 62.08 ve PET-CGDNN 60.44; bu makalede 61.42 ve 60.58.
- ⚠️ **Transformer yorumu spekülasyon:** "Yerel zaman özelliklerinde zorlanıyor olabilir" deniyor; test edilmemiş.
- ⚠️ **10b farkı tartışılmamış:** "10b'de herkes daha iyi" için olası nedenler sayılıyor ama hiçbiri test edilmiyor. Bariz iki fark var: bir sınıf eksik (AM-SSB) ve 6 kat daha çok eğitim verisi.

**Moshe sorarsa (EN)**

- *"Is 63 percent good?"* — "It's the average over all twenty noise levels, including those where nobody can do better than guessing. At high SNR it's about 93 percent. And it's half a point above the next model, from one run."

---

## Bölüm 4.2.1 — RML2016.10b sonuçları (s. 6–7, Şekil 3–5, Tablo 4)

**Ne diyor**

- **Eğitim (40 epoch):**
  - eğitim kaybı 1.2112 → 0.8489;
  - eğitim doğruluğu %53.43 → %66.44;
  - doğrulama doğruluğu %56.81 → epoch 35'te %65.48 ile tepe;
  - epoch 30'dan sonra düzleşme;
  - "35'te durmak optimal olurdu".
- **Şekil 5:**
  - genel doğruluk 0.6539, F1 0.6573, precision 0.7103;
  - −20 dB'de doğruluk 0.11;
  - 5 dB'den itibaren yaklaşık 0.94.
- **Tablo 4:** −10…+4 dB arasındaki 8 seviyenin hepsinde birinci.

**Ne demek**

- **Şans seviyesi:** 10b'de 10 sınıf var; −20 dB'deki 0.11 şans seviyesi.
- **Bizim için önemi:** Biz 10a üzerinde çalışıyoruz; bu bölüm ikincil.

**Doğru mu?**

- ✅ **Tablo 4'te her seviyede birinci.** Ama bazı farklar çok küçük ve tek koşuyla ayırt edilemez:
  - −8 dB: 45.61'e karşı 45.58 (0.03 puan);
  - 2 dB: 92.94'e karşı 92.81 (0.13 puan);
  - −10 dB: 32.62'ye karşı 32.37 (0.25 puan).
- ⚠️ **Olmayan seviye:** Metinde "−15 dB" geçiyor; böyle bir seviye yok (adımlar çift sayı).
- ⚠️ **İki yerde iki sayı:** Şekil 5 metnindeki F1 0.6573 ve precision 0.7103, Tablo 6'daki 0.6556 ve 0.7100 ile tutmuyor.
- ⚠️ **Yanlış şekil numarası:** Metin doğrulama doğruluğunun Şekil 3'te olduğunu söylüyor; Şekil 4'te.

**Moshe sorarsa (EN)**

- *"And on 10b?"* — "65.39, first at every level in their Table 4. But at several levels the lead is a tenth of a point or less, from one run."

---

## Bölüm 4.2.2 — RML2016.10a sonuçları (s. 7–9, Şekil 6–8, Tablo 5)

Bizim veri setimiz bu; en önemli bölüm.

**Ne diyor**

- **Eğitim (58 epoch):**
  - eğitim kaybı 1.5296 → 1.0243;
  - eğitim doğruluğu %47.82 → %64.24;
  - doğrulama doğruluğu %54.08 → epoch 58'de %63.59.
- **Kayıp ve doğruluk:** Epoch 40'tan sonra doğrulama kaybı ~1.05'te düzleşiyor; doğrulama doğruluğu ise yavaşça artmaya devam ediyor.
- **Öğrenme oranı:** Epoch 43'te 0.001 → 0.0002, epoch 58'de 0.00004.
- **Üç evre:**
  - 1–20: hızlı öğrenme;
  - 21–40: doygunluk başlıyor;
  - 41–58: düz.
- **Şekil 8 (doğruluk / F1 / precision):**
  - −20 dB: 9.0 / 3.1 / 9.3;
  - −10 dB: 25.4 / 22.9 / 27.9;
  - "−5 dB": 57.9 / 54.1 / 55.4;
  - 0 dB'de hepsi %90'ın üstünde;
  - 5–15 dB'de doğruluk %93–94.
- **Genel:** Doğruluk 63.39, F1 65.6, precision 75.4. "Precision yüksek → model tahminlerinden emin."
- **Tablo 5:** −10…+4 dB'de 7 model.

**Ne demek**

- **Aşırı ezber yok:** Doğrulama doğruluğu (%63.59) test doğruluğuna (%63.24) çok yakın.
- **Eğitim doğruluğu neden düşük görünüyor:** Eğitim doğruluğu (%64.24) da yakın, çünkü eğitimde dropout açık olduğundan model eğitim verisinde kendini kısıtlıyor.
- **Precision neden doğruluktan yüksek?** Bu en ilginç nokta:
  - Düşük SNR'de model kararlarının çoğunu tek bir "çöp" sınıfa yığıyor. Bizim 11 sınıflı koşumuzda bu AM-SSB: −20 dB'de bütün kararların %87'si.
  - Hatalar böylece tek bir sütunda toplanıyor. O sınıfın precision'ı düşük, ama diğer sınıflar düşük SNR'de nadiren tahmin edildiği için onların precision'ı yüksek kalıyor.
  - Recall ise her sınıfta düşüyor, çünkü her sınıfın gürültülü çerçeveleri çöpe gidiyor.
  - Ortalama precision bu yüzden şişiyor. Bu "model emin" demek değil.
- **Ölçü notu:** Onların kodu F1'i "weighted" ortalama ile hesaplıyor. Precision ise kodda hiç yok (Bölüm "Eksikler").

**Doğru mu?**

❌ **"Düşük SNR'de tutarlı şekilde üstün" (s. 8).** Tablo 5'in kendisi:

| dB | ICRNNA | ondan yüksek olanlar |
|---|---|---|
| −10 | 25.32 | CNN-BiLSTM-DNN 27.73 |
| −8 | 37.45 | CNN-BiLSTM-DNN 39.55, MCLDNN 38.09, MCNet 38.00 |
| −6 | 55.26 | MCLDNN 56.32, CNN-BiLSTM-DNN 56.18, MCNet 56.00 |
| +4 | 93.02 | CNN-BiLSTM-DNN 93.18 |

ICRNNA 8 seviyenin sadece 4'ünde birinci: −4, −2, 0 ve +2 dB.

- ❌ **−8 dB cümlesi:** Metin "MCLDNN'den (38.09) ve MCNet'ten (38.00) üstün" diyor; 37.45 ikisinden de düşük.
- ❌ **+4 dB cümlesi:** Metin "en iyi" diyor; CNN-BiLSTM-DNN 93.18 ile daha yüksek.
- ✅ **Gerçek avantaj orta bölgede:**
  - −2 dB'de 85.11'e karşı 81.05 (+4 puan);
  - 0 dB'de 92.21'e karşı 88.68 (+3.5 puan).
  - Bizim karşılaştırmamızda da ICRNNA'nın üstünlüğü −6…−2 dB'de.
- ❌ **İki farklı genel doğruluk:** Metin 63.39 diyor; Tablo 3 ve Tablo 6 63.24 diyor. Aynı model.
- ❌ **"−5 dB" yok:** Böyle bir seviye yok. 57.9 da Tablo 5'teki hiçbir değere uymuyor (−6: 55.26, −4: 71.13).
- ⚠️ **"Precision yüksek → model emin" yanlış yorum:** Precision güven ölçmez (yukarıda).
- ⚠️ **Doğrulama doğruluğu 58'de hâlâ artıyor.** Ama durdurma kuralı kayba bakıyor ve kayıp düzleşmiş; bu bir hata değil, bir seçim. Bizim sabit 58 epoch'luk koşu %61.75'te kaldı, uzun eğitim epoch 107'de %63.21'e çıktı.
- ⚠️ **"0 dB = sinyal ve gürültü eşit":** Bu veri için doğru değil (Bölüm 2).

**Moshe sorarsa (EN)**

- *"Is it better at low SNR, as they claim?"* — "Not on 10a, by their own Table 5: at −10, −8 and −6 dB other models are higher. Its real lead is between −4 and +2 dB, three to four points. Above that, everyone plateaus near 93."

---

## Bölüm 4.3 — Karışıklık matrisleri (s. 9–12, Şekil 9–12)

**Ne diyor**

- **Şekil 9 (10b) ve Şekil 11 (10a):** 0 dB'de 7 modelin matrisleri.
- **Şekil 10 (10b) ve Şekil 12 (10a):** ICRNNA'nın −4, −2, +2 ve +4 dB'deki matrisleri.

**10a, 0 dB, ICRNNA'nın sınıf doğrulukları (Şekil 11):**

| sınıf | ICRNNA | rakipler |
|---|---|---|
| genel | 92.21 | |
| 8PSK | 91.9 | |
| AM-DSB | 98.0 | MCLDNN 55, ResNet 65 |
| 16-QAM | 94.3 | CNN-BiLSTM-DNN 90, MCLDNN 84, MCNet 34, ResNet 64 |
| 64-QAM | "93.%" | |
| WBFM | 41.4 | MCLDNN 63, ResNet 58 ondan iyi |

- **10b, 0 dB:** 16-QAM 97.9, 64-QAM 98.5, BPSK 99.9, CPFSK 99.2, WBFM 38.6.
- **SNR'ye göre:**
  - −4 dB'de QPSK–BPSK ve WBFM karışıyor;
  - +2 dB'de QAM16–QAM64 karışıklığı azalıyor;
  - +4 dB'de matris neredeyse köşegen.

**Ne demek**

- **Nasıl okunur:** Satır gerçek sınıf, sütun modelin dediği; köşegen doğrular, satır yüzdeleri recall.
- **0 dB'de iki zayıf nokta:**
  - **WBFM:** Çerçevelerinin %57.9'una AM-DSB deniyor; sadece %41.4'ü doğru.
  - **QAM16 ve QAM64:** Birbirine karışıyor; QAM16'nın %5.7'si QAM64 deniyor, QAM64'ün %6.1'i QAM16.
- **WBFM neden?** Bilinen açıklama şu: bu veri setinde analog sınıflar ses kayıtlarından üretiliyor. Kayıttaki sessiz aralıklarda iki sınıf da modülesiz bir taşıyıcıya dönüşüyor ve hiçbir model ayıramıyor. Bunu kendimiz ölçmedik; "bilinen açıklama" diye söyle.
- **Bizde de aynı hatalar:**
  - **WBFM:** 11 sınıflı koşumuzda −2 dB'den yukarı her seviyede %37–46; hatalar AM-DSB'ye akıyor.
  - **QAM16:** 4 sınıflı koşumuzda +2…+18 dB boyunca recall %86–88'de düz; SNR artınca düzelmiyor. Neden 128 örnek = 16 sembol olabilir: çerçeveyi 64 örneğe kırpınca %65.9, 32'ye kırpınca %46.3.

**Doğru mu?**

- ✅ **Hata desenleri gerçek** ve bizimkilerle aynı.
- ⚠️ **Her hücrenin arkasında kaç karar var, yazmıyor.** Test seti %15 olduğundan bir SNR'de sınıf başına ortalama ~150 çerçeve düşüyor. %1 ≈ 1–2 çerçeve; "%98'e karşı %97" gibi farklar birkaç çerçeve.
- ⚠️ **Yanlış atıf:** 0 dB analizi için "Tablo 4'te görüldüğü gibi" deniyor; o bir SNR tablosu, matris değil.
- ⚠️ **Basım hatası:** "64-QAM 93.%" yazıyor; ondalık eksik.
- ⚠️ **Süslü cümleler ölçüm değil:** "Model sinyal bütünlüğünü koruyor" gibi ifadeler hiçbir şey ölçmüyor.

**Moshe sorarsa (EN)**

- *"How many decisions behind each number?"* — "About 150 per row at one SNR: 1,000 frames per class and level, 15 percent of them in the test set. So one percent is one or two frames."

---

## Bölüm 4.4 — Ablation (s. 12–13, Tablo 6)

**Ne diyor**

- **Dört varyant:** tam model, attention'sız, sadece CNN, sadece BiLSTM.

**10a:**

| varyant | doğruluk | F1 | precision |
|---|---|---|---|
| sadece CNN | 0.5993 | 0.6193 | 0.7486 |
| sadece BiLSTM | 0.6313 | 0.6537 | 0.7676 |
| attention'sız | 0.6280 | 0.6532 | 0.7526 |
| tam model | 0.6324 | 0.6567 | 0.7536 |

- **Sonuçları:**
  - her parça hayati;
  - attention özellikle düşük SNR'de yardım ediyor;
  - tam model en yüksek precision'a sahip.

**Ne demek**

- **Tablonun okunuşu:**
  - Sadece CNN %59.93.
  - Sadece BiLSTM %63.13: +3.2 puan.
  - Tam model %63.24: +0.11 puan daha.
  - Attention'ın katkısı: attention'sız %62.80'den tam modele +0.44 puan.
- **Garip bir sonuç:** Sadece BiLSTM (CNN yok, attention yok) attention'sız modelden (CNN + BiLSTM) daha iyi: 63.13'e karşı 62.80. Yani attention olmadan CNN eklemek 0.33 puan kaybettiriyor. Tek koşu olduğu için bu da gürültü olabilir.

**Doğru mu?**

- ❌ **"Tam model en yüksek precision":** Yanlış. 10a'da sadece BiLSTM 0.7676'ya karşı tam model 0.7536; 10b'de 0.7200'e karşı 0.7100. İki veri setinde de yanlış.
- ⚠️ **Farklar ölçüm değil:** 0.1–0.4 puanlık farklar tek koşudan geliyor; bizim tohumdan tohuma farkımız 0.5–1.1 puan.
- ⚠️ **Kanıtsız iddia:** "Attention düşük SNR'de yardım ediyor" deniyor ama SNR'ye göre ablation tablosu yok.
- ⚠️ **Varyantlar tarif edilmemiş.** Örneğin attention'sız modelde 64 adımdan tek vektöre nasıl geçiliyor (son adım mı, ortalama mı)? Yazmıyor. Kodda da bu varyantlar yok.
- ✅ **Tek sağlam sonuç:** CNN tek başına ~3 puan geride; işi büyük ölçüde BiLSTM yapıyor.

**Moshe sorarsa (EN)**

- *"Which part matters?"* — "The LSTM. Alone it gets 63.1. Adding the CNN and the attention takes it to 63.2. The rest is a tenth of a point, from one run."

---

## Bölüm 4.5 — Hesap yükü (s. 12–13, Tablo 7)

**Ne diyor**

**Tablo 7:**

| model | MFLOPs | parametre ("M") | süre (s) |
|---|---|---|---|
| ResNet | 1.12 × 10⁴ | 53.8 | 0.909 |
| Transformer | 408 | 0.0384 | 7.89 |
| CVCNN | 3.10 × 10³ | 68.1 | 9.22 |
| PET-CGDNN | 86.9 | 79.2 | 2.29 |
| MCLDNN | 1.17 × 10³ | 406 | 4.45 |
| DAE | 0.536 | 0.149 | 11.4 |
| MAMCA | 73.9 | 16.8 | 0.246 |
| ICRNN (Ours) | 48.42 | 0.79 | 0.2313 |

- **Yorumları:** "Kaynakları kısıtlı cihazlar için uygun"; ardından uygulama alanları: bilişsel radyo, spektrum izleme, askeri, IoT.

**Ne demek**

- **FLOPs:** Bir çerçeveyi sınıflandırmak için gereken işlem sayısı.
- **Parametre:** Öğrenilen sayı.
- **Çıkarım süresi:** Ne ölçüldüğü yazmıyor: bir çerçeve mi, bir batch mi, tüm test seti mi; hangi donanımda?

**Doğru mu?**

- ✅ 0.79 M doğru (kod: 794,188).
- ❌ **"MCLDNN 406 M":** Gerçek sayı 406,199, yani 0.41 M. ICRNNA, MCLDNN'in yaklaşık 2 katı.
- ❌ **"PET-CGDNN 79.2 M":** Gerçek sayı 71,871, yani 0.072 M. ICRNNA 11 kat büyük.
- ⚠️ **Birimler karışık görünüyor:** Sütunun bir kısmı bin, bir kısmı milyon gibi. "Daha küçük" iddiası bu karışıklığa dayanıyor.
- ⚠️ **Tutarsız süreler:** Transformer 38 bin parametreyle 7.89 s, DAE 149 bin parametreyle 11.4 s sürüyor. Farklı ölçümlerden derlenmiş izlenimi veriyor.
- ⚠️ **Muhtemelen başka makaleden (tahmin):** Kendilerinin koştuğu görünen dört model (CNN-BiLSTM-DNN, MCNet, CGDNet, IC-AMCNet) bu tabloda yok; ölçümü kendileri yapsalardı bunları eklemek kolay olurdu. Tablo büyük ihtimalle başka bir makaleden, muhtemelen [31]'den alındı. Doğrulamak için [31]'e bakmak gerek.
- ⚠️ **"Gerçek zamanlı" ölçülmemiş:** LSTM 64 adımı sırayla işlemek zorunda, kolay paralelleşmez. Hiçbir cihazda ölçüm yok.

**Moshe sorarsa (EN)**

- *"Is it lightweight?"* — "Not compared to the models it beats. It has 794 thousand parameters. MCLDNN has 406 thousand, not 406 million as their table says, and PET-CGDNN has 72 thousand."

---

## Bölüm 5 — Sonuç ve gelecek iş (s. 13–14)

**Ne diyor**

- İddiaları tekrarlıyor: en yüksek doğruluk, düşük SNR'de dayanıklılık, verimlilik, her parça gerekli.
- **Gelecek iş:**
  - WBFM için veri artırma, SNR'yi hesaba katan kayıp fonksiyonları, dalgacık ve spektral özellikler;
  - başka veri setlerine genelleme;
  - FPGA/ASIC üzerinde gerçekleme;
  - çok kafalı attention ya da transformer;
  - adversarial saldırılara dayanıklılık;
  - uçtan uca bilişsel radyo sistemi.

**Ne demek** — bizim projeyle bağlantısı

- **"Başka veri setlerine genelleme" bizim projenin asıl konusu.** Makale tek veri setinde eğitip aynı veri setinden ayrılmış bir parçada test ediyor; başka bir kaynaktan gelen sinyallerde ne olacağına hiç bakmıyor.
- **"Veri artırma" bizde ölçüldü.** 2016 yöntem tablomuzda (5 ortak sınıf, 14 koşu) standart veri artırma alan farkını kapattı:
  - artırmasız: alan içi 0.948, alan dışı (bizim ürettiğimiz sentetik sinyaller) 0.920;
  - artırmalı: alan içi 0.969, alan dışı 0.968.
- **WBFM bir mimari sorunu olmayabilir.** Sessiz aralıklar veri setinin kendi özelliği; daha iyi bir mimari bunu çözmeyebilir.

**Doğru mu?**

- ⚠️ **"Düşük SNR'de yüksek doğruluk" tekrar ediliyor;** Bölüm 4.2.2'ye bakın.
- ⚠️ **"Data will be made available on request":**
  - Veri zaten herkese açık.
  - Kod 16 ay sonra, 8 Ağustos 2026'da GitHub'a kondu.
  - Eğitilmiş ağırlık hiç yayımlanmadı.

**Moshe sorarsa (EN)**

- *"What do we take from it?"* — "A solid, reproducible classifier to build on. Not proof that attention or this exact design is needed: the LSTM does most of the work."

---

## [§] Kaynaklar ve künye ayrıntıları (s. 14)

- **[11] iki yerde yanlış gösteriliyor.** Hem LB yöntemleri hem MCNet için kullanılıyor; oysa [11], Li ve ark.'nın ResNeXt-GRU makalesi.
- **[23], O'Shea ve West 2016:** Veri setinin GNU Radio makalesi. Veri için doğru kaynak, CNN tanımı için değil.
- **[31], MAMCA makalesi:** Transformer, CVCNN ve MAMCA için tek kaynak.
- **Doğru eşleşenler:**
  - MCLDNN [20];
  - PET-CGDNN [30];
  - CGDNet [16];
  - CNN-BiLSTM-DNN [17];
  - IC-AMCNet [13];
  - ResNet [14];
  - DAE [32].
- **Kendi çalışmaları:** [1], [4], [5], [8] ve [12] yazarların kendi makaleleri. [5] ile [12] aynı makale (ICOA 2023 derlemesi), iki kez listelenmiş.
- **Yazar katkıları:**
  - El-Haryqy: yazım ve yöntem;
  - Kharbouche: analiz;
  - Ouamna: yöntem;
  - Madini ve Zouine: danışman.

---

## [✗] Hata listesi — makalenin kendi içindeki tutarsızlıklar

1. **s. 8, Tablo 5 metni** — makale: düşük SNR'de tutarlı üstün → gerçekte: −10, −8, −6 dB'de başkaları yüksek
2. **s. 8, −8 dB** — makale: MCLDNN (38.09) ve MCNet'ten (38.00) üstün → gerçekte: 37.45, ikisinden de düşük
3. **s. 8, +4 dB** — makale: en iyi → gerçekte: CNN-BiLSTM-DNN 93.18 > 93.02
4. **s. 8** — makale: genel doğruluk 63.39 → gerçekte: Tablo 3 ve 6: 63.24
5. **s. 8** — makale: −5 dB'de 57.9 → gerçekte: −5 dB yok; Tablo 5'te karşılığı yok
6. **s. 7** — makale: −15 dB → gerçekte: böyle seviye yok
7. **s. 7, Şekil 5 metni** — makale: F1 0.6573, precision 0.7103 → gerçekte: Tablo 6: 0.6556, 0.7100
8. **s. 12, ablation** — makale: tam model en yüksek precision → gerçekte: sadece BiLSTM daha yüksek, iki veri setinde
9. **s. 13, Tablo 7** — makale: MCLDNN 406 M → gerçekte: 406,199 = 0.41 M
10. **s. 13, Tablo 7** — makale: PET-CGDNN 79.2 M → gerçekte: 71,871 = 0.072 M
11. **s. 3** — makale: BFSK → gerçekte: GFSK
12. **s. 8, s. 11** — makale: 0 dB'de sinyal ve gürültü eşit → gerçekte: etiket 0'da ölçülen SNR sınıfa göre 5–17 dB
13. **s. 1–2** — aynı paragraf iki kez basılmış
14. **s. 6** — makale: doğrulama doğruluğu Şekil 3'te → gerçekte: Şekil 4'te
15. **s. 9** — makale: 0 dB matrisleri "Tablo 4'te" → gerçekte: Şekil 9'da
16. **s. 11** — makale: 64-QAM "93.%" → gerçekte: ondalık eksik
17. **s. 5** — makale: MCNet [11] → gerçekte: [11] ResNeXt-GRU makalesi
18. **s. 14** — makale: [5] ve [12] → gerçekte: aynı makale iki kez

Hiçbiri modeli geçersiz kılmıyor. Ama 1, 2, 3, 8, 9 ve 10'un hepsi aynı yöne gidiyor: makaleyi kendi tablolarından daha iyi gösteriyor.

---

## [+] Eksikler — makalede yazmayan, koddan öğrendiklerimiz

**Kod (8 Ağustos 2026, commit `ac000a0`) bunları netleştirdi:**

- **"58 epoch":** Erken durdurmanın tetiklendiği yer. Üst sınır 100 epoch; doğrulama kaybı 6 epoch iyileşmezse durup en iyi ağırlıklara dönüyor.
- **Normalizasyon:** Her zaman indeksi için ortalama ve varyans, tüm veriden, test dahil.
- **Attention'ın girişleri ve boyutları:** query = BN'li LSTM çıkışı, key = value = ham çıkış; W1 ve W2 128 birim; ölçek öğreniliyor.
- **Gradient clipping:** `clipnorm=1.0`.
- **Tek koşu, tek ayırma.** Ayırma Python'un küme sırasına bağlı; her çalıştırmada farklı.
- **Ölçü:** F1 "weighted" ortalama.

**Kodda olmayanlar** (yani tekrar üretilemeyenler):

- **Ablation varyantları:** Sadece CNN, sadece BiLSTM, attention'sız. Tablo 6'yı koddan üretemezsin.
- **Rakip modeller:** Tablo 3'ün rakip sütunu ve Şekil 9/11'in rakip panelleri koddan gelmiyor.
- **Precision:** Kod precision hesaplamıyor; Şekil 8'deki precision eğrisi ve Tablo 6'daki precision sütunu yayımlanmayan bir koddan.
- **FLOPs ve çıkarım süresi ölçümü:** Tablo 7.
- **Eğitilmiş ağırlıklar:** Hiç yayımlanmadı.

**Makale de kod da söylemiyor:**

- rakiplerin yeniden mi eğitildiği, yoksa başka makalelerden mi alındığı;
- her sayının arkasında kaç karar olduğu;
- koşudan koşuya ne kadar oynadığı.

Colab'da çalışan `official_2016` görevi bu sonuncusunu ölçüyor: onların kodunu 3 kez, aynı ayırmayla koşuyor.

---

## [EN] Moshe için beş cümle

1. "It's a CNN, then a two-layer bidirectional LSTM, then an attention average, then a small classifier: 794 thousand parameters, 83 percent of them in the LSTM."
2. "Its 63.24 percent is real: we rebuilt the model from the text alone and got 63.21."
3. "Its lead is small: half a point over the next model in their table and about one point in ours, from single runs."
4. "Some claims are stronger than their own tables: low SNR on 10a, highest precision, and the parameter counts."
5. "Its errors are the same as ours: WBFM read as AM-DSB, and QAM16 against QAM64."

**Türkçesi:**

1. CNN, sonra iki katlı çift yönlü LSTM, sonra attention ile ağırlıklı ortalama, sonra küçük bir sınıflandırıcı. 794 bin parametre, %83'ü LSTM'de.
2. 63.24 gerçek; makaleden kurduk, 63.21 aldık.
3. Önü az: kendi tablolarında yarım puan, bizim karşılaştırmamızda bir puan civarı, tek koşulardan.
4. Bazı iddialar kendi tablolarından güçlü: 10a'da düşük SNR, en yüksek precision, parametre sayıları.
5. Hataları bizimkilerle aynı: WBFM'e AM-DSB demesi ve QAM16–QAM64 karışıklığı.

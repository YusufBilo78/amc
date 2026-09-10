# Okuma listesi — Kam toplantısı öncesi

**UÇAKTAN ÖNCE İNDİR.** Hepsi arXiv veya açık erişim, PDF olarak çevrimdışı
okunur. Telefonda değil, tablette/laptopta oku — figürleri görmen lazım.

Bir uyarı: bu makaleleri **ben baştan sona okumadım.** Özetlerinden ve
alandaki yerlerinden biliyorum. O yüzden aşağıda "şunu der" demiyorum,
**"şunu ara"** diyorum. Okurken cevaplaman gereken soruları veriyorum;
cevapları senin çıkarman lazım. Bu zaten daha iyi bir okuma biçimi.

---

## Öncelik 1 — O'Shea, Roy, Clancy (2018)
### *Over-the-Air Deep Learning Based Radio Signal Classification*
IEEE JSTSP · **arXiv:1712.04578**

**Neden birinci:** Kullandığın veri setini üreten makale. Kam sana
"hangi veriyi kullandın" diye sorduğunda, veri setinin *nasıl üretildiğini*
bilmiyor olmak kötü olur.

**Okurken cevapla:**
- Sentetik veriyi tam olarak nasıl ürettiler? Hangi kanal etkileri, hangi
  parametre aralıkları?
- **Darbe şekillendirme roll-off değeri ne?** — senin bulgunun tam merkezi.
  Bir aralık mı verilmiş, sabit mi? Bu, senin `beta=0.01`'de en iyi transferi
  bulmanı açıklıyor mu?
- Havadan (OTA) kayıtlar makalede var mı, ve yayınlanan dosyada var mı?
  (DeepSig'in sayfası "synthetic simulated channel effects" diyor — makale ne
  diyor?)
- Sentetik ile OTA arasında bir performans farkı raporlamışlar mı? Varsa
  **senin ölçtüğün açığın literatürdeki karşılığı o olabilir.**

**Senin işine bağlantısı:** Eğer roll-off'u sabit bir dar değerde tuttularsa,
senin "RadioML'in etek eğimi bizimkinden dik" ölçümün doğrudan doğrulanmış
olur — ve bunu toplantıda söyleyebilirsin.

---

## Öncelik 2 — Huang ve ark. (2019)
### *Data Augmentation for Deep Learning-based Radio Modulation Classification*
**arXiv:1912.03026**

**Neden ikinci:** Senin baseline'ın. `augment.StandardAMCAugmenter` içinde
yeniden ürettiğin döndürme/çevirme/gürültü seti bu makaleden geliyor.
**Kendi karşılaştırma zeminini okumadan savunamazsın.**

**Okurken cevapla:**
- Üç dönüşümü tam olarak nasıl tanımlamışlar? Senin uygulaman uyuyor mu?
- Rotasyonu sürekli mi yoksa 90° katları olarak mı yapmışlar?
- Konjugasyon/çevirmenin etiket-güvenliği konusunda bir şey söylüyorlar mı?
  (Senin fark ettiğin şey: AM-SSB gibi spektral asimetrik sınıflarda çevirme
  etiketi bozar. Makalede bu uyarı var mı? **Yoksa bu senin katkın.**)
- Ne kadar kazanç raporlamışlar? Senin ölçtüğün +0.044 ile uyumlu mu?

**Senin işine bağlantısı:** "Literatür baseline'ını 3.3 σ geçtim" cümlesini
ancak baseline'ı doğru uyguladığından emin olursan kurabilirsin.

---

## Öncelik 3 — Kam'ın kendi makalesi
### *Optimal Data Fusion of Correlated Local Decisions in Multiple Sensor Detection Systems*
IEEE Trans. Aerospace and Electronic Systems · NJIT sayfasından erişilebilir

**Neden üçüncü ama stratejik olarak birinci:** **Karşılaşacağın kişinin
makalesi.** Bir saatlik toplantıda "sizin şu makalenizi okudum" diyebilmek,
diğer her şeyden farklı bir etki yapar.

**Okurken cevapla:**
- Karar füzyonunda "optimal" ne demek — hangi kriter altında?
- Yerel kararlar **korelasyonlu** olduğunda ne değişiyor?
- Karar bölgeleri nasıl tanımlanıyor, ve varsayımlar bozulduğunda ne oluyor?

**Senin işine bağlantısı:** Senin bulduğun şey — gözlem, sınırların
oturtulduğu bölgenin dışına düştüğünde kararın **komşu bölgeye** kayması —
onun dilinde ifade edilebilir. Ve hazırladığın soru tam buradan geliyor:
*"is there an established way to think about where a decision goes when the
observation falls outside the region the boundaries were fitted on?"*

**Not:** Tam makaleyi bulamazsan Google Scholar'dan başka bir Kam makalesi de
olur. Önemli olan onun düşünme biçimini görmen.

---

## Öncelik 4 — DANN tabanlı AMC uyarlaması (2025)
### *Deep Domain-Adversarial Adaptation for AMC under Channel Variability*
**arXiv:2508.06829**

**Neden dördüncü:** Karşılaştırmadığın asıl rakip yöntem. "Neden DANN ile
kıyaslamadın" sorusu gelirse, en azından **ne olduğunu** bilmen lazım.

**Okurken cevapla:**
- Eğitim sırasında **hedef alandan veri kullanıyorlar mı?** (Neredeyse kesin
  evet — ve bu senin en güçlü ayrımın: beyazlatma kullanmıyor.)
- Hangi alan çiftinde test etmişler? 2016.10a → 2018.01a mı?
- Ne kadar kazanç raporlamışlar?

**Senin işine bağlantısı:** Şu cümleyi kurabilmen gerek:
*"DANN needs target-domain samples at fit time. Whitening does not. That makes
the comparison not like-for-like, and it also makes whitening deployable on a
receiver that has never seen the transmitter."*

---

## Öncelik 5 — Geirhos ve ark. (2020) *(vakit kalırsa)*
### *Shortcut Learning in Deep Neural Networks*
Nature Machine Intelligence · **arXiv:2004.07780**

**Neden beşinci:** Senin test edip **çürüttüğün** çerçeve. Sinyal işleme
makalesi değil, genel bir derleme — ama kavramı doğru kullanabilmen için
kaynağını bilmen iyi olur.

**Okurken cevapla:**
- "Shortcut" tanımı tam olarak ne?
- Kestirme yolu tespit etmek için önerdikleri testler neler? Senin zarf nakli
  testin onlardan birine benziyor mu?

**Senin işine bağlantısı:** Toplantıda şunu söyleyebilirsin: *"I tested the
shortcut-learning account explicitly and it did not hold for my case — the
envelope is not a decision cue. What I have looks more like
out-of-distribution collapse."* Kavramı kaynağından bilerek kullanmak, kulaktan
dolma kullanmaktan farklıdır.

---

## Uçakta nasıl okuyacaksın

Beş makaleyi baştan sona okuma. **Süre bütçesi:**

| Makale | Süre | Nasıl |
|---|---|---|
| O'Shea | 60 dk | Tam oku. Veri üretimi bölümünü **iki kez.** |
| Augmentation | 30 dk | Yöntem bölümü + sonuç tabloları. Giriş/ilgili çalışmalar atlanabilir. |
| Kam | 45 dk | Anlamadığın matematiğe takılma. **Nasıl düşündüğünü** çıkar. |
| DANN | 20 dk | Özet + yöntem şeması + sonuç tablosu. Yeter. |
| Geirhos | 20 dk | Sadece kestirme yol tanımı ve test önerileri. |

**Toplam ~3 saat.** Uçuşta rahat sığar.

**Not tutma yöntemi:** Her makale için tek bir sayfaya şunları yaz:
1. Bir cümlede ne yaptıkları
2. Benim işimle ilişkisi (bir cümle)
3. Toplantıda söyleyebileceğim bir cümle

Üç satır × beş makale = on beş satır. Toplantıda ihtiyacın olan bu kadar.

---

## Okuduktan sonra değişecek cevap

`MEETING_PREP.md`'de şu an şöyle yazıyor:

> "Less than I should have. I built the measurement first and read second,
> which is the wrong order."

Uçuştan sonra bunu şununla değiştirebilirsin:

> "I built the measurement first and read afterwards, which is the wrong order,
> but I have now gone through the RadioML generation paper, the standard
> augmentation baseline I reproduced, and the adversarial adaptation line. What
> I still do not know is whether the attribution method I used is standard —
> that is one of the things I hoped to ask you."

**Aynı dürüstlük, çok daha güçlü pozisyon.** Ve zayıf noktan kapanmış olur.

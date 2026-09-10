# Toplantı cevapları — Türkçe karşılıkları

`MEETING_PREP.md`'deki İngilizce cevapların birebir Türkçesi.

**Nasıl kullan:** Önce Türkçesini oku, ne dediğini anla. Sonra İngilizcesini
yüksek sesle söyle. Ezberleme — anlamını bildiğin bir cümleyi kendi
kelimelerinle de kurabilirsin, ezberlediğin cümleyi kuramazsın.

---

## 0. Açılış

> "Before anything else, I owe you a correction. In my email I wrote that the
> classifier collapses onto the densest constellation. After I sent it, I
> designed a control to test that claim — a case where 'densest' and 'nearest'
> predict opposite answers. The control refuted me. The real rule is adjacency,
> not density. I would rather start there than have you find it."

**Türkçesi:** *"Her şeyden önce bir düzeltme borcum var. Mailimde
sınıflandırıcının en yoğun takımyıldıza çöktüğünü yazmıştım. Maili
gönderdikten sonra bu iddiayı test edecek bir kontrol tasarladım — 'en yoğun'
ile 'en yakın'ın zıt tahmin yaptığı bir durum. Kontrol beni çürüttü. Gerçek
kural yoğunluk değil, komşuluk. Bunu sizin bulmanızdansa buradan başlamayı
tercih ederim."*

**Kilit kelimeler:** *correction* (düzeltme), *refuted* (çürüttü),
*adjacency* (komşuluk), *density* (yoğunluk).

---

## 1. Anlama kontrolleri

### "Walk me through the problem in one minute."
> "A neural network that classifies radio modulations reaches 99% accuracy on
> the dataset it was trained on. I generated the same five modulations with an
> independent signal generator and tested the same network on them. Accuracy
> fell to 81%. The gap is 18 points, and almost all of it sat in one cell —
> 16QAM being read as 64QAM, 75% of the time. The question I worked on is why,
> and whether it can be fixed without collecting target data."

**Türkçesi:** *"Radyo modülasyonlarını sınıflandıran bir sinir ağı, eğitildiği
veri setinde %99 doğruluğa ulaşıyor. Aynı beş modülasyonu bağımsız bir sinyal
üreteciyle ürettim ve aynı ağı onlarda test ettim. Doğruluk %81'e düştü. Açık
18 puan, ve neredeyse tamamı tek bir hücrede toplanmıştı — 16QAM'in %75
oranında 64QAM olarak okunması. Üzerinde çalıştığım soru: neden, ve hedef veri
toplamadan düzeltilebilir mi."*

### "Why did the accuracy drop? What did you find?"
> "The spectral envelope. The two generators use different pulse shaping, so
> the occupied bandwidth and the roll-off skirts differ. I confirmed it by
> substitution: I took my frames, kept their phase spectrum exactly, and
> replaced only the magnitude spectrum with the average magnitude spectrum from
> the training data. 16QAM went from 0.191 to 0.972. Nothing in the time
> structure or the symbol sequence changed."

**Türkçesi:** *"Spektral zarf. İki üreteç farklı darbe şekillendirme
kullanıyor, dolayısıyla işgal edilen bant genişliği ve etek eğimleri farklı.
Bunu ikame ile doğruladım: kendi çerçevelerimi aldım, faz spektrumlarını
aynen korudum, ve sadece genlik spektrumunu eğitim verisinin ortalama genlik
spektrumuyla değiştirdim. 16QAM 0.191'den 0.972'ye çıktı. Zaman yapısında ve
sembol dizisinde hiçbir şey değişmedi."*

**Kilit kelimeler:** *spectral envelope* (spektral zarf), *pulse shaping*
(darbe şekillendirme), *roll-off skirts* (etek eğimleri), *substitution*
(ikame), *magnitude spectrum* (genlik spektrumu), *phase spectrum* (faz
spektrumu).

### "How do you know it's the envelope and not something else?"
> "I eliminated the alternatives one at a time, with the network frozen so only
> the test signal changed. Symbol-rate offset: I drove the cyclostationary line
> from 14.3 dB down to 0.1 dB, matching the training data's own value, and
> 16QAM moved from 0.168 to 0.202 — nothing. Constellation density: the
> amplitude histograms of the two domains overlap almost exactly. Channel
> impairments: carrier frequency offset, multipath, random phase, and all three
> combined — the best recovery was plus 0.045 against a shortfall of 0.83. Only
> the envelope substitution reproduced the effect."

**Türkçesi:** *"Alternatifleri teker teker eledim, ağ donduruldu ki sadece test
sinyali değişsin. Sembol hızı kayması: döngüsel durağanlık çizgisini 14.3
dB'den 0.1 dB'ye indirdim — eğitim verisinin kendi değerine eşitleyerek — ve
16QAM 0.168'den 0.202'ye gitti, yani hiçbir şey. Takımyıldız yoğunluğu: iki
alanın genlik histogramları neredeyse birebir örtüşüyor. Kanal bozulmaları:
taşıyıcı frekans kayması, çok yolluluk, rastgele faz, ve üçü birlikte — en iyi
iyileşme +0.045 iken açık 0.83'tü. Sadece zarf ikamesi etkiyi yeniden
üretebildi."*

**Kilit kelimeler:** *eliminated* (eledim), *frozen* (donduruldu),
*cyclostationary* (döngüsel durağan), *shortfall* (açık/eksiklik),
*reproduced* (yeniden üretti).

### "Plus or minus what?"
> "Plus or minus 0.017, over four seeds, varying both the train/test split and
> the weight initialisation. The improvement from whitening is 7.4 pooled
> standard deviations, so it is well outside run-to-run variation."

**Türkçesi:** *"Artı eksi 0.017, dört tohum üzerinden, hem eğitim/test bölmesi
hem ağırlık başlatması değiştirilerek. Beyazlatmanın sağladığı iyileşme 7.4
havuzlanmış standart sapma, yani koşudan koşuya değişimin çok dışında."*

### "Show me where you were wrong."
> "Four times. First I thought it was pulse shaping and swept the roll-off from
> 0.9 down to 0.15, saw only partial recovery, and concluded I was wrong — but
> my sweep stopped too early; at 0.01 the recovery was much larger. So the
> hypothesis was right and my test was inadequate. Second, I thought my
> constellations looked denser; the histograms refuted that. Third, I thought
> symbol timing regularity was the cause; removing it changed nothing. Fourth,
> I thought the network was using the envelope as a shortcut; an
> envelope-transplant test refuted that too."

**Türkçesi:** *"Dört kez. Önce darbe şekillendirme olduğunu düşündüm ve
roll-off'u 0.9'dan 0.15'e süpürdüm, sadece kısmi iyileşme gördüm ve
yanıldığıma karar verdim — ama süpürmem çok erken durmuş; 0.01'de iyileşme çok
daha büyüktü. Yani hipotez doğruydu, testim yetersizdi. İkinci olarak, kendi
takımyıldızlarımın daha yoğun göründüğünü düşündüm; histogramlar bunu
çürüttü. Üçüncü olarak, sembol zamanlaması düzenliliğinin sebep olduğunu
düşündüm; kaldırınca hiçbir şey değişmedi. Dördüncü olarak, ağın zarfı
kestirme yol olarak kullandığını düşündüm; zarf nakli testi onu da çürüttü."*

**Kilit kelimeler:** *swept* (süpürdüm), *inadequate* (yetersiz),
*refuted* (çürüttü), *shortcut* (kestirme yol).

---

## 2. Tespit teorisi çerçevesi

### "How would you describe this in decision-theoretic terms?"
> "As an M-ary hypothesis testing problem where the likelihood model is learned
> rather than specified. The training distribution defines implicit decision
> regions in a learned feature space. Domain shift moves the test samples
> outside the region where those boundaries were fitted, and the sample lands in
> whichever region happens to extend into that part of the space. That is why
> the errors are structured rather than diffuse."

**Türkçesi:** *"Olabilirlik modelinin belirtilmek yerine öğrenildiği bir M-ary
hipotez testi problemi olarak. Eğitim dağılımı, öğrenilmiş bir öznitelik
uzayında örtük karar bölgeleri tanımlıyor. Alan kayması test örneklerini o
sınırların oturtulduğu bölgenin dışına taşıyor, ve örnek uzayın o kısmına
uzanan hangi bölge varsa oraya düşüyor. Hataların dağınık değil yapılı
olmasının sebebi bu."*

**Basit hali (kendi kelimelerinle kurmak istersen):** Klasik tespitte
olasılık modelini mühendis yazar. Burada ağ onu veriden çıkarıyor. Veri
değişince model geçersizleşiyor ama karar sınırları yerinde kalıyor — ve
örnek, sınırların hiç ayarlanmadığı bir bölgeye düşüyor.

**Kilit kelimeler:** *likelihood model* (olabilirlik modeli),
*decision regions* (karar bölgeleri), *feature space* (öznitelik uzayı),
*boundaries were fitted* (sınırlar oturtuldu), *diffuse* (dağınık).

### "You say the failure is structured. What is the structure?"
> "The network maps an unseen modulation onto its nearest neighbour in
> constellation order. Train without 64QAM, show it 64QAM: 100% goes to 16QAM.
> Remove 16QAM too: everything goes to 8PSK. Remove 8PSK: everything goes to
> QPSK. Twelve of twelve cells."
>
> "But those twelve cells do not discriminate between two hypotheses, because
> in every one of them the nearest class is also the densest remaining class.
> So I built a case where they disagree: train on QPSK through 64QAM, probe
> with BPSK. Nearest is QPSK, densest is 64QAM. The answer was QPSK, at 94 and
> 99.9 percent across two seeds, with zero going to 64QAM. So it is adjacency."

**Türkçesi:** *"Ağ, görmediği bir modülasyonu takımyıldız sıralamasındaki en
yakın komşusuna eşliyor. 64QAM olmadan eğit, 64QAM göster: %100'ü 16QAM'e
gidiyor. 16QAM'i de çıkar: her şey 8PSK'ya gidiyor. 8PSK'yı çıkar: her şey
QPSK'ya. On iki hücrenin on ikisi."*

*"Ama bu on iki hücre iki hipotezi ayırt etmiyor, çünkü hepsinde en yakın
sınıf aynı zamanda kalan en yoğun sınıf. O yüzden ikisinin çeliştiği bir durum
kurdum: QPSK'dan 64QAM'e kadar eğit, BPSK ile prob et. En yakın QPSK, en yoğun
64QAM. Cevap QPSK oldu — iki tohumda %94 ve %99.9, 64QAM'e sıfır. Yani
komşuluk."*

**Kilit kelimeler:** *unseen* (görülmemiş), *nearest neighbour* (en yakın
komşu), *discriminate between* (ayırt etmek), *probe* (prob etmek/yoklamak).

### "Could that just be an artifact of your output layer?"
> "I checked. I randomly permuted which output index corresponds to which
> class and retrained. The sink stayed on the same class every time, three out
> of three. So it follows the signals, not the index."

**Türkçesi:** *"Kontrol ettim. Hangi çıkış indeksinin hangi sınıfa karşılık
geldiğini rastgele karıştırıp yeniden eğittim. Sink her seferinde aynı sınıfta
kaldı, üçte üç. Yani indeksi değil, sinyalleri takip ediyor."*

**Kilit kelimeler:** *artifact* (yapaylık), *permuted* (karıştırdım),
*output index* (çıkış indeksi).

### "What happens with pure noise?"
> "That was my third probe, and it behaves differently. Gaussian noise does not
> produce a sink — the predictions sit near chance, 20 to 30 percent across five
> classes with a mild bias toward BPSK. So 'an unfamiliar modulation' and 'no
> modulation at all' are not the same failure mode. I do not have an explanation
> for that difference yet."

**Türkçesi:** *"Üçüncü probum oydu ve farklı davranıyor. Gauss gürültüsü bir
sink üretmiyor — tahminler şans seviyesine yakın oturuyor, beş sınıfta %20-30,
BPSK'ya hafif bir eğilimle. Yani 'tanımadığı bir modülasyon' ile 'hiç
modülasyon olmaması' aynı başarısızlık modu değil. Bu farkın açıklamasına
henüz sahip değilim."*

### "Why should adjacency happen? What's the mechanism?"
> "I do not know yet, and I would rather say that than guess. The observation is
> consistent with the network holding an ordered internal representation of
> constellation order, so that an unseen class falls next to its neighbours.
> But I have measured the behaviour, not the representation."

**Türkçesi:** *"Henüz bilmiyorum, ve tahmin yürütmektense bunu söylemeyi
tercih ederim. Gözlem, ağın takımyıldız sırasını sıralı bir iç temsilde
tuttuğu fikriyle tutarlı — böylece görülmemiş bir sınıf komşularının yanına
düşüyor. Ama ben davranışı ölçtüm, temsili değil."*

---

## 3. Zayıf noktalar

### "Is any of this real data?"
> "No, and that is the main limitation. RadioML is simulated and my generator is
> simulated, so what I measured is a gap between two simulators. It is a real
> and measurable gap, but I cannot claim anything about real-world
> generalization from it. Closing that is exactly why I am looking for lab
> access — the plan is a cabled SDR setup, transmitter to attenuator to
> receiver, so nothing is radiated and the experiments stay repeatable."

**Türkçesi:** *"Hayır, ve asıl kısıt bu. RadioML simüle, benim üretecim de
simüle, dolayısıyla ölçtüğüm şey iki simülatör arasındaki bir açık. Gerçek ve
ölçülebilir bir açık, ama bundan gerçek dünyaya genelleme hakkında bir şey
iddia edemem. Bunu kapatmak, laboratuvar erişimi aramamın tam sebebi — plan
kablolu bir SDR kurulumu: verici, zayıflatıcı, alıcı. Hiçbir şey havaya
yayılmıyor ve deneyler tekrarlanabilir kalıyor."*

**Kilit kelimeler:** *limitation* (kısıt), *simulated* (simüle edilmiş),
*cabled* (kablolu), *radiated* (havaya yayılan), *repeatable* (tekrarlanabilir).

### "What have you read? What's the state of the art?"
> "Less than I should have. I know domain adaptation for modulation
> classification is active — there is DANN-style adversarial adaptation, and
> there is a standard augmentation set of rotation, flip and Gaussian noise
> that I reproduced as my baseline. I built the measurement first and read
> second, which is the wrong order. I am correcting that now."

**Türkçesi:** *"Olmam gerekenden az. Modülasyon sınıflandırmada alan
uyarlamasının aktif bir alan olduğunu biliyorum — DANN tarzı çekişmeli
uyarlama var, ve döndürme, çevirme, Gauss gürültüsünden oluşan standart bir
artırma seti var ki onu baseline olarak yeniden ürettim. Önce ölçümü kurdum,
sonra okudum — yanlış sıra. Şu an bunu düzeltiyorum."*

### "How did you do all this in one month?"
> "I directed the work and made the research decisions — what to test next,
> what to rule out, when to stop and diagnose instead of patching. I used AI
> tooling heavily for implementation, which is how I could run this many
> controlled experiments in the time I had. What I can do is explain every
> experiment and why it was designed that way."

**Türkçesi:** *"İşi ben yönlendirdim ve araştırma kararlarını ben verdim —
sırada neyi test edeceğimi, neyi eleyeceğimi, ne zaman yamalamayı bırakıp
teşhis koyacağımı. Uygulama için yoğun şekilde yapay zeka aracı kullandım;
elimdeki sürede bu kadar kontrollü deneyi koşabilmemin sebebi bu. Yapabildiğim
şey, her deneyi ve neden o şekilde tasarlandığını açıklamak."*

### "What's novel here?"
> "I am not in a position to claim novelty — I have not done a proper
> literature review. What I think is unusual is the attribution method rather
> than the fix: using a phase-preserving magnitude substitution to assign a
> domain gap to one specific signal property, and ruling out the channel
> explanations by controlled elimination. I would value your judgement on
> whether that is already standard."

**Türkçesi:** *"Yenilik iddia edecek konumda değilim — düzgün bir literatür
taraması yapmadım. Sıradışı olduğunu düşündüğüm şey çözümden çok atıf yöntemi:
bir alan açığını tek bir sinyal özelliğine atfetmek için fazı koruyan genlik
ikamesi kullanmak, ve kanal açıklamalarını kontrollü elemeyle devre dışı
bırakmak. Bunun zaten standart olup olmadığı konusunda görüşünüze değer
veririm."*

---

## 4. Ne istediğin

### "What can I do for you?"
> "Three things, in order of usefulness to me. First, a pointer to whoever at
> NJIT this is closest to — I have looked at Professor Abdi, Professor Kliewer
> and Professor Haimovich, and an introduction would carry far more weight than
> a cold email from me. Second, guidance on how an undergraduate gets access to
> RF measurement equipment here, since real captures are the missing piece.
> Third, whether this is a sensible senior design topic or whether I should
> scope it down."

**Türkçesi:** *"Üç şey, bana faydası sırasıyla. Birincisi, NJIT'te bunun en
yakın olduğu kişiye bir yönlendirme — Profesör Abdi, Profesör Kliewer ve
Profesör Haimovich'e baktım, ve bir tanıştırma benim soğuk mailimden çok daha
fazla ağırlık taşır. İkincisi, bir lisans öğrencisinin burada RF ölçüm
ekipmanına nasıl erişebileceği konusunda yönlendirme, çünkü gerçek kayıtlar
eksik parça. Üçüncüsü, bunun makul bir bitirme projesi konusu olup olmadığı,
yoksa kapsamı daraltmam mı gerektiği."*

---

## 5. Senin soracakların

> "In your work on decision fusion, is there an established way to think about
> where a decision goes when the observation falls outside the region the
> boundaries were fitted on? I could not find literature on it and I suspect I
> am using the wrong search terms."

**Türkçesi:** *"Karar füzyonu çalışmalarınızda, gözlem sınırların oturtulduğu
bölgenin dışına düştüğünde kararın nereye gideceğini düşünmenin yerleşik bir
yolu var mı? Literatürde bulamadım ve sanırım yanlış arama terimleri
kullanıyorum."*

> "If you were advising me: is the more valuable contribution the method, or a
> real captured dataset with controlled domain variation? My sense is the field
> has many methods and very little real cross-domain data."

**Türkçesi:** *"Bana tavsiye verecek olsanız: daha değerli katkı yöntem mi,
yoksa kontrollü alan değişimi içeren gerçek kaydedilmiş bir veri seti mi?
Benim izlenimim, alanın çok yöntemi ve çok az gerçek alanlar-arası verisi
olduğu."*

> "What would you want to see from a senior project before you considered it
> worth publishing?"

**Türkçesi:** *"Bir bitirme projesinde, yayınlanmaya değer bulmadan önce ne
görmek isterdiniz?"*

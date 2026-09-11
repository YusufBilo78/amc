# KAM NOTES — Prof. Moshe Kam

*Tek sayfa. Toplantıda ekranda tut, okumak için değil göz atmak için.*

*İngilizce bloklar söyleyeceğin cümleler; altlarındaki italik Türkçe sadece anlamı
tutasın diye — onu okumuyorsun, İngilizcesini söylüyorsun.*

**Bu toplantı ne değil:** Tez savunması değil, sonuç sunumu değil. Adam seninle
görüşmek istedi. Konuşmanın yarısını muhtemelen o yapacak.

**Gerçekçi en iyi sonuç:** O bir dekan, aktif bir laboratuvarda seni RA olarak
alma ihtimali düşük — zamanı yok. En muhtemel çıktı **"seni şu hocayla
tanıştırayım."** Bu kayıp değil. Dekanın tanıştırdığı öğrenci, soğuk mail atan
öğrenci değildir.

---

## 1. AÇILIŞ — üç cümle, fazlası değil

> "I've been working on modulation classification — getting a model to
> recognise a signal without knowing anything about the transmitter.
>
> It works on the standard dataset and falls apart when the transmitter
> changes. I spent the summer figuring out why, and I think it's the
> transmitter's spectral fingerprint.
>
> Everything I've done is simulated, though. What I want is to do it with real
> radios."

*Modülasyon sınıflandırma üzerine çalışıyorum — bir modele, vericisi hakkında hiçbir şey bilmeden bir sinyali tanıtmak. Standart veri setinde çalışıyor, verici değişince dağılıyor. Yazı nedenini bulmakla geçirdim, ve sanırım sebep vericinin spektral parmak izi. Ama yaptığım her şey simülasyonda. İstediğim şey bunu gerçek radyolarla yapmak.*

Dur. Nefes al. Soru sorsun.

---

## 2. ESMER — sen aç, erken

Açılışın hemen ardına. Kam ikinizi de tanıyacak; bunu sonradan çıkarmasındansa
senden duyması her senaryoda daha iyi.

> "One thing up front — you may already be talking to Esmer. We know each other,
> we work on overlapping problems, and we talk about them. I'd rather you hear
> that from me."

*Baştan bir şey — Esmer'le de görüşüyor olabilirsiniz. Birbirimizi tanıyoruz,
örtüşen problemler üzerinde çalışıyoruz, ve konuşuyoruz. Bunu benden duymanızı
tercih ederim.*

Ve hemen ardından çerçevele — örtüşme sorun değil, bölünmemiş olması sorun:

> "We're not doing the same thing, and I don't think we should. I'd rather we
> split it cleanly than both drift toward the middle — and I'd take your read on
> where the line should be."

*Aynı şeyi yapmıyoruz, yapmamız gerektiğini de düşünmüyorum. İkimizin de ortaya
kaymasındansa işi net bölmeyi tercih ederim — çizginin nerede olması gerektiği
konusunda sizin görüşünüzü alırım.*

**Toplantıdan önce onu ara.** Asıl risk Kam'ın örtüşmeyi öğrenmesi değil,
ikinizin aynı proje hakkında çelişen şeyler anlatması. Özellikle şunu netleştir:
**16QAM çöküşünü mimari değişikliği çözmedi** — ICRNNA'da hâlâ 0.060, çözen şey
beyazlatma. Mimari değişikliğinin çözdüğü şey ezberlemeydi. İki ayrı problem, ve
ikiniz de aynı şekilde anlatın.

---

## 3. DEVAMINI İSTERSE — akan hali

> "Modulation classification looks finished from the outside. Standard dataset,
> everybody trains on it, everybody reports ninety-nine percent.
>
> Then I built my own transmitter in software, generated the same modulations,
> fed them to the same trained model. Ninety-nine became eighty. And it wasn't
> spread out — it was concentrated. One class fell off a cliff while the others
> barely moved.
>
> That's the part that got me. A model doesn't fail like that by accident.
> Random degradation means noise. *Structured* degradation means it learned
> something specific, and I handed it something specific it wasn't expecting.
>
> So I stopped trying to improve the number and started trying to explain it. I
> wrote down every explanation I could think of — bandwidth, constellation
> density, symbol timing, channel effects — and designed a test for each that
> could come out either way. Every one came back negative. Four dead ends, and I
> kept all four, because dead ends are how you know the fifth one means
> something.
>
> The fifth was the transmitter's own spectral signature. Its fingerprint. I
> could hold everything about the signal identical and swap only that, and the
> model's answer would follow it. Not correlate with it — *follow* it. After
> that the fix more or less wrote itself: take the fingerprint off before the
> model ever sees the signal.
>
> And then the part I'd rather tell you myself than have you find. Halfway
> through I got suspicious of my own network and ran it against a proper one.
> Mine was memorising — training accuracy pinned at a hundred, test accuracy
> nowhere near it. So I threw it out and started the measurements over.
>
> Here's the thing, though. The failure came back. Different architecture,
> different family, same collapse, same class. And that was the best news I got
> all summer — because if it survives a change of model, it isn't my model. It's
> the problem."

*Modülasyon sınıflandırma dışarıdan bakınca bitmiş görünüyor. Standart bir veri seti, herkes onunla eğitiyor, herkes %99 raporluyor.*

*Sonra kendi vericimi yazılımda kurdum, aynı modülasyonları ürettim, aynı eğitilmiş modele verdim. %99, %80 oldu. Ve dağınık değildi — toplanmıştı. Bir sınıf uçurumdan düşerken diğerleri neredeyse hiç kıpırdamadı.*

*Beni yakalayan kısım bu. Bir model kazara böyle bozulmaz. Rastgele bozulma gürültü demektir. Yapılı bozulma, spesifik bir şey öğrendiği ve benim ona beklemediği spesifik bir şey verdiğim anlamına gelir.*

*Ben de sayıyı iyileştirmeye çalışmayı bırakıp açıklamaya çalışmaya başladım. Aklıma gelen her açıklamayı yazdım — bant genişliği, takımyıldız yoğunluğu, sembol zamanlaması, kanal etkileri — ve her biri için iki türlü de sonuçlanabilecek bir test tasarladım. Hepsi olumsuz döndü. Dört çıkmaz sokak, ve dördünü de sakladım, çünkü beşincinin bir anlam taşıdığını çıkmaz sokaklardan bilirsin.*

*Beşincisi vericinin kendi spektral imzasıydı. Parmak izi. Sinyalle ilgili her şeyi sabit tutup sadece onu değiştirebiliyordum, ve modelin cevabı onu takip ediyordu. Onunla ilişkili çıkmıyordu — takip ediyordu. Ondan sonra çözüm kendini yazdı: parmak izini, model sinyali görmeden önce sil.*

*Ve şimdi sizin bulmanızdansa benim anlatmayı tercih ettiğim kısım. Yarı yolda kendi ağımdan şüphelendim ve düzgün bir ağa karşı koşturdum. Benimki ezberliyordu — eğitim doğruluğu yüzde yüze çakılı, test doğruluğu çok uzakta. Attım ve ölçümlere baştan başladım.*

*Ama asıl mesele şu. Başarısızlık geri geldi. Farklı mimari, farklı aile, aynı çöküş, aynı sınıf. Ve bu bütün yaz aldığım en iyi haberdi — çünkü model değişikliğinden sağ çıkıyorsa, o benim modelim değil. Problemin kendisi.*

**Sözün kesilirse taşıyıcı dört cümle bunlar:**

1. "It wasn't spread out, it was concentrated." → tesadüf değil, yapı var
2. "Not correlate with it — *follow* it." → gözlem değil müdahale (onun ilgisini çekecek cümle)
3. "The failure came back." → bulgu senin ağının kusuru değil
4. "A study, not a result." → dürüstlük + isteğin, tek cümlede

**Anlatamazsan benzetme:**
> "It's like training something to recognise a language, and it quietly learns
> the microphone instead. Works perfectly until you change the microphone."

*Biraz şuna benziyor: bir şeye dil tanımayı öğretiyorsun, o da sessizce mikrofonu öğreniyor. Mikrofonu değiştirene kadar kusursuz çalışıyor.*

---

## 4. ASIL MESELE — ne istediğin

Bunu anlatının **sonuna** koy, başına değil.

> "The reason I wanted to meet is more than this project. I want to do research
> — I'd like to work as a research assistant, and I'm orienting toward a PhD
> after I graduate. This summer was mostly me finding out whether I actually
> like the work, and I do. I like the part where you don't know the answer yet."

*Görüşmek istememin sebebi bu projeden fazlası. Araştırma yapmak istiyorum — araştırma asistanı olarak çalışmak isterim, ve mezuniyet sonrası doktoraya yöneliyorum. Bu yaz büyük ölçüde bu işi gerçekten sevip sevmediğimi anlamakla geçti, ve seviyorum. Cevabı henüz bilmediğin kısmını seviyorum.*

**Ve bu cümle — kartın en değerli cümlesi:**

> "And I want to be clear that I'm not attached to my own topic. If there's
> something already running in your group that could use a pair of hands, I'd
> rather be useful there than protect my own project. I'd learn more from
> working on a real problem someone else is already invested in than from
> defending mine."

*Ve kendi konuma bağlı olmadığımı netleştirmek istiyorum. Grubunuzda halihazırda yürüyen ve bir çift ele ihtiyaç duyan bir şey varsa, kendi projemi korumaktansa orada faydalı olmayı tercih ederim. Başkasının zaten emek verdiği gerçek bir problem üzerinde çalışmaktan, kendiminkini savunmaktan öğrendiğimden fazlasını öğrenirim.*

---

## 5. SORACAKLARIN

1. > "Is this something worth pursuing here, and who should I be talking to?"

   *Bu, burada peşine düşülmeye değer bir şey mi, ve kiminle konuşmalıyım?*

   Hazır isimler: **Abdi, Kliewer, Haimovich.** "I've looked at these three —
   which is closest to this?" Somut isim vermen onun işini kolaylaştırır.

2. > "If I'm serious about a PhD — what should I be doing in the next two years
   > that would actually make a difference when I apply? I'd rather hear it now
   > than find out later."

   *Doktora konusunda ciddiysem — önümüzdeki iki yılda başvururken gerçekten fark yaratacak ne yapmalıyım? Bunu sonradan öğrenmektense şimdi duymayı tercih ederim.*

   Muhtemelen en uzun konuşacağı yer burası. Bırak konuşsun, **not al.**

3. > "How does an undergraduate here get access to RF measurement equipment?"

   *Burada bir lisans öğrencisi RF ölçüm ekipmanına nasıl erişebiliyor?*

   Somut istek. Dekanın gerçekten verebileceği türden bir şey.

---

## 6. KAPANIŞ — son 30 saniye

> "Whatever you think the right next step is — I'll take it. If that's a
> different professor, a different topic, or reading a semester's worth of
> things I haven't read, that's fine. I just want to be doing this."

*Sizce doğru bir sonraki adım neyse — yaparım. Bu başka bir hoca, başka bir konu, ya da okumadığım bir dönemlik şeyi okumak olsun, sorun değil. Sadece bu işi yapıyor olmak istiyorum.*

---

## 7. SADECE SORARSA AÇACAĞIN KUTULAR

Sen açmıyorsun. Sorarsa açıyorsun.

### "Is any of this real data?"
> "No — and that's exactly why the next thing I want is real captures.
> There's a cheaper step first, though: a third public dataset I didn't generate
> myself. Right now one side of my comparison is my own generator, and that's
> the weakest part of it."

*Hayır — ve sıradaki isteğimin gerçek kayıtlar olmasının sebebi tam olarak bu. Ama ondan önce daha ucuz bir adım var: kendimin üretmediği üçüncü bir açık veri seti. Şu an karşılaştırmamın bir tarafı kendi ürettecim, ve en zayıf kısmı da o.*

### "That's just spectral whitening — that's known."
> "It is. WhiteNet got there before me, on real captures, which is the part I
> can't match yet. What I don't think is standard is how I *found* it — by
> intervening on one property and holding everything else fixed."

*Öyle. WhiteNet benden önce ulaşmış, üstelik gerçek kayıtlarla — henüz eşleyemediğim kısım o. Standart olduğunu düşünmediğim şey onu nasıl bulduğum: tek bir özelliğe müdahale edip diğer her şeyi sabit tutarak.*

### "What have you read?"
> "Three cross-domain AMC papers properly, and a number of others by position
> rather than in full. One varies the channel, one the symbol rate, and the
> third — a TWC paper from last year — deliberately superimposes what it calls
> the typical domain difference factors: channel type, SNR, carrier frequency
> offset, sampling rate. None of them varies pulse shaping, which is the one
> that moved my failure. I don't want to call that a gap in the field, because
> three papers isn't a survey — but it's why I'd like your read on it."

*Üç cross-domain AMC makalesini düzgün okudum, birkaçını da tam okumaktan çok
alandaki yerinden biliyorum. Biri kanalı değiştiriyor, biri sembol hızını,
üçüncüsü — geçen yıldan bir TWC makalesi — "tipik alan farkı faktörleri" dediği
şeyleri bilerek üst üste bindiriyor: kanal tipi, SNR, taşıyıcı frekans kayması,
örnekleme hızı. Hiçbiri darbe şekillendirmeyi değiştirmiyor, ki benim
başarısızlığımı hareket ettiren oydu. Bunu alanda bir boşluk diye adlandırmak
istemiyorum, çünkü üç makale bir tarama değil — ama bu yüzden görüşünüzü almak
istiyorum.*

**Üçü:** SigDA (IEEE TWC 2024, kanal/SNR/CFO/örnekleme hızı bindirilmiş) ·
PMSPDMC (IEEE IoT J. 2025, sembol hızı) · Zhang ve ark. (DSA 2022, kanal).
Detay `LITERATURE.md` Q4 devamında.

### "How did you do all this so fast?"
> "I made the research decisions — what to test, what to rule out, when to stop
> patching and start diagnosing. I used AI tooling heavily for the
> implementation. I can explain every experiment and why it was designed that
> way."

*Araştırma kararlarını ben verdim — neyi test edeceğimi, neyi eleyeceğimi, ne zaman yamalamayı bırakıp teşhis koymaya başlayacağımı. Uygulama için yoğun şekilde yapay zeka aracı kullandım. Her deneyi ve neden o şekilde tasarlandığını açıklayabilirim.*

### "Would a new architecture fix it?"
> "My own results say no. I changed backbones entirely and the failure was
> unchanged — four different inductive biases, same behaviour. The architectural
> question I *do* think is real is building the invariance into the model
> instead of bolting it on: my fix is preprocessing with a fixed strength I
> tuned once, and I already know partial beats full — so one fixed number for
> every signal is almost certainly wrong."

*Kendi sonuçlarım hayır diyor. Omurgayı tamamen değiştirdim ve başarısızlık değişmedi — dört farklı tümevarım yanlılığı, aynı davranış. Gerçek olduğunu düşündüğüm mimari sorusu, değişmezliği modele cıvatalamak yerine içine inşa etmek: çözümüm, bir kere ayarladığım sabit bir güçle yapılan bir ön işleme, ve kısminin tamı yendiğini zaten biliyorum — yani her sinyal için tek bir sabit sayı neredeyse kesin yanlış.*

### "Multiple receivers?" — *köprü, sen kurabilirsin*
> "Everything I did assumes one receiver making one decision. If several
> receivers observe the same transmitter, each produces a local decision — and
> those decisions are correlated, because they share the transmitter's
> characteristics, which is precisely what I showed causes the failure. That
> looks like your problem applied to mine. I don't know how much has been done
> there, and it's one of the things I'd most like to ask you."

*Yaptığım her şey tek alıcının tek karar verdiğini varsayıyor. Birden fazla alıcı aynı vericiyi gözlerse, her biri yerel bir karar üretir — ve bu kararlar korelasyonludur, çünkü vericinin karakteristiklerini paylaşırlar ki başarısızlığa sebep olduğunu gösterdiğim şey tam olarak o. Bu, sizin probleminizin benimkine uygulanmış hali gibi görünüyor. Orada ne kadar iş yapıldığını bilmiyorum, ve size en çok sormak istediğim şeylerden biri bu.*

⚠️ İddia olarak değil, **soru olarak** kur. "Burada boşluk var" deme.

### Sayı isterse — GÜNCEL olanlar (ICRNNA, 5 seed, SNR ≥ 10 dB)

| | in-domain | cross | gap | 16QAM |
|---|---|---|---|---|
| hiçbir şey | 0.999 | 0.804 | +0.195 | 0.060 |
| literatür artırma seti | 1.000 | 0.806 | +0.194 | 0.037 |
| beyazlatma α=0.75 | 1.000 | 0.993 | +0.007 | 0.991 |

Mimari karşılaştırması: IQNet train **1.000** / test **0.675** — ezberliyor.
ICRNNA train 0.719 / test 0.710 — ayrışmıyor. İki protokolde de aynı.

Emin değilsen:
> "I'd rather not quote that from memory — half my numbers changed when I
> switched backbones and I don't want to give you a stale one. I can send you
> the current table after this."

*Bunu ezberden aktarmamayı tercih ederim — omurgayı değiştirdiğimde sayılarımın yarısı değişti ve size eski bir tane vermek istemem. Güncel tabloyu sonrasında gönderebilirim.*

---

## 8. ASLA SÖYLEME

- **`CALL_NOTES.md` ve `MEETING_PREP*.md` içindeki hiçbir sayı** — hepsi atılmış
  omurganın (IQNet). Özellikle: cross 0.811, düzeltme sonrası 0.948, "%79
  kapandı", "+3.3 s.d.", "16QAM %75 oranında 64QAM okunuyor".
- **"the published architecture"** — `model_zoo.ICRNNA` bir akranın
  reprodüksiyonundan aktarıldı, yayınlanan mimariden beş yerde ayrılıyor.
- **Beyazlatma için yenilik iddiası** — WhiteNet önce yapmış, üstelik gerçek
  kayıtlarla.
- **"16QAM sorunu mimariyi değiştirince kayboldu"** — kaybolmadı. ICRNNA'da
  16QAM hâlâ 0.060. Kaybolduğu yer beyazlatma satırı. **İki ayrı problem:**
  ezberlemeyi mimari çözdü, alanlar arası çöküşü beyazlatma çözüyor.
- **Onun makalesini okumadıysan "makalenizi okudum"** — "your work on decision
  fusion" demek yeterli.
- **"IEEE Xplore'a erişemedim"** — erişimin var, NJIT aboneliği üzerinden; okuduğun
  üç makale de oradan indi. Doğru cümle: *"my search was arXiv-weighted and I
  haven't done a proper Xplore pass yet."*
- **"Bu alanda kimse darbe şekillendirmeye bakmamış"** — üç makalede yok, bu
  "literatürde yok" demek değil. *"I haven't found the paper that isolates it."*

---

## 9. HATIRLA

**"I don't know" tam bir cümledir.** Bir hoca için "bilmiyorum, test etmedim"
diyen öğrenci, her soruya cevabı olan öğrenciden daha güvenilirdir.

Diğer kaçış cümleleri:
- "Let me think about that for a second."
- "Could you say that another way?"
- "That's on my list and I haven't done it."

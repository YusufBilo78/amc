# KAM NOTES — Prof. Moshe Kam

*Tek sayfa. Toplantıda ekranda tut, okumak için değil göz atmak için.*

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

Dur. Nefes al. Soru sorsun.

---

## 2. DEVAMINI İSTERSE — akan hali

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

**Sözün kesilirse taşıyıcı dört cümle bunlar:**

1. "It wasn't spread out, it was concentrated." → tesadüf değil, yapı var
2. "Not correlate with it — *follow* it." → gözlem değil müdahale (onun ilgisini çekecek cümle)
3. "The failure came back." → bulgu senin ağının kusuru değil
4. "A study, not a result." → dürüstlük + isteğin, tek cümlede

**Anlatamazsan benzetme:**
> "It's like training something to recognise a language, and it quietly learns
> the microphone instead. Works perfectly until you change the microphone."

---

## 3. ASIL MESELE — ne istediğin

Bunu anlatının **sonuna** koy, başına değil.

> "The reason I wanted to meet is more than this project. I want to do research
> — I'd like to work as a research assistant, and I'm orienting toward a PhD
> after I graduate. This summer was mostly me finding out whether I actually
> like the work, and I do. I like the part where you don't know the answer yet."

**Ve bu cümle — kartın en değerli cümlesi:**

> "And I want to be clear that I'm not attached to my own topic. If there's
> something already running in your group that could use a pair of hands, I'd
> rather be useful there than protect my own project. I'd learn more from
> working on a real problem someone else is already invested in than from
> defending mine."

---

## 4. SORACAKLARIN

1. > "Is this something worth pursuing here, and who should I be talking to?"

   Hazır isimler: **Abdi, Kliewer, Haimovich.** "I've looked at these three —
   which is closest to this?" Somut isim vermen onun işini kolaylaştırır.

2. > "If I'm serious about a PhD — what should I be doing in the next two years
   > that would actually make a difference when I apply? I'd rather hear it now
   > than find out later."

   Muhtemelen en uzun konuşacağı yer burası. Bırak konuşsun, **not al.**

3. > "How does an undergraduate here get access to RF measurement equipment?"

   Somut istek. Dekanın gerçekten verebileceği türden bir şey.

---

## 5. KAPANIŞ — son 30 saniye

> "Whatever you think the right next step is — I'll take it. If that's a
> different professor, a different topic, or reading a semester's worth of
> things I haven't read, that's fine. I just want to be doing this."

---

## 6. SADECE SORARSA AÇACAĞIN KUTULAR

Sen açmıyorsun. Sorarsa açıyorsun.

### "Is any of this real data?"
> "No — and that's exactly why the next thing I want is real captures.
> There's a cheaper step first, though: a third public dataset I didn't generate
> myself. Right now one side of my comparison is my own generator, and that's
> the weakest part of it."

### "That's just spectral whitening — that's known."
> "It is. WhiteNet got there before me, on real captures, which is the part I
> can't match yet. What I don't think is standard is how I *found* it — by
> intervening on one property and holding everything else fixed."

### "How did you do all this so fast?"
> "I made the research decisions — what to test, what to rule out, when to stop
> patching and start diagnosing. I used AI tooling heavily for the
> implementation. I can explain every experiment and why it was designed that
> way."

### "Would a new architecture fix it?"
> "My own results say no. I changed backbones entirely and the failure was
> unchanged — four different inductive biases, same behaviour. The architectural
> question I *do* think is real is building the invariance into the model
> instead of bolting it on: my fix is preprocessing with a fixed strength I
> tuned once, and I already know partial beats full — so one fixed number for
> every signal is almost certainly wrong."

### "Multiple receivers?" — *köprü, sen kurabilirsin*
> "Everything I did assumes one receiver making one decision. If several
> receivers observe the same transmitter, each produces a local decision — and
> those decisions are correlated, because they share the transmitter's
> characteristics, which is precisely what I showed causes the failure. That
> looks like your problem applied to mine. I don't know how much has been done
> there, and it's one of the things I'd most like to ask you."

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

---

## 7. ASLA SÖYLEME

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

---

## 8. HATIRLA

**"I don't know" tam bir cümledir.** Bir hoca için "bilmiyorum, test etmedim"
diyen öğrenci, her soruya cevabı olan öğrenciden daha güvenilirdir.

Diğer kaçış cümleleri:
- "Let me think about that for a second."
- "Could you say that another way?"
- "That's on my list and I haven't done it."

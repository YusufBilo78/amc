# Provost Kam toplantısı — soru/cevap hazırlığı

**Format:** Soru İngilizce (duyacağın hali). *Ne soruyor* ve *Dikkat* Türkçe.
**Cevap** İngilizce — yüksek sesle prova et, ezberleme, anlamını bil.

Kam kimdir, unutma: **tespit ve kestirim teorisyeni** (karar füzyonu, dağıtık
tespit). Eski IEEE Başkanı. P.E. — yani pratik mühendislik refleksi de var.
Senin sayılarını değil, **nasıl düşündüğünü** ölçecek.

---

## 0. Açılış — bunu sen başlat

Mailinde çürütülmüş bir iddia var. Sorulmasını bekleme, **ilk sen söyle.**

> "Before anything else, I owe you a correction. In my email I wrote that the
> classifier collapses onto the densest constellation. After I sent it, I
> designed a control to test that claim — a case where 'densest' and 'nearest'
> predict opposite answers. The control refuted me. The real rule is adjacency,
> not density. I would rather start there than have you find it."

**Neden böyle başlıyorsun:** bir saatlik toplantının ilk 30 saniyesinde
"kendi hipotezimi test edip yıktım" demek, geri kalan 59 dakikanın tonunu
belirler. Savunmaya geçmezsin, konuşma araştırma sohbetine döner.

---

## 1. Anlama kontrolleri

Bunlar "gerçekten sen mi yaptın" sorularıdır. Sayı değil, **mantık zinciri** ister.

### Q: "Walk me through the problem in one minute."

*Ne soruyor:* Konuyu özetleyebiliyor musun. Bu ilk 60 saniye çok önemli.

**Cevap:**
> "A neural network that classifies radio modulations reaches 99% accuracy on
> the dataset it was trained on. I generated the same five modulations with an
> independent signal generator and tested the same network on them. Accuracy
> fell to 81%. The gap is 18 points, and almost all of it sat in one cell —
> 16QAM being read as 64QAM, 75% of the time. The question I worked on is why,
> and whether it can be fixed without collecting target data."

*Dikkat:* "Deep learning" ile başlama. **Problemle** başla. Bir tespit
teorisyeni için ilginç olan kısım ağ değil, hatanın yapısı.

### Q: "Why did the accuracy drop? What did you find?"

**Cevap:**
> "The spectral envelope. The two generators use different pulse shaping, so
> the occupied bandwidth and the roll-off skirts differ. I confirmed it by
> substitution: I took my frames, kept their phase spectrum exactly, and
> replaced only the magnitude spectrum with the average magnitude spectrum from
> the training data. 16QAM went from 0.191 to 0.972. Nothing in the time
> structure or the symbol sequence changed."

*Dikkat:* Bu senin en güçlü tek deneyin. Faz/genlik ayrımını rahatça
anlatabilmelisin — çünkü sonraki soruların yarısı buradan gelir.

### Q: "How do you know it's the envelope and not something else?"

*Ne soruyor:* Kontrol grubun var mı. **En kritik soru bu.**

**Cevap:**
> "I eliminated the alternatives one at a time, with the network frozen so only
> the test signal changed. Symbol-rate offset: I drove the cyclostationary line
> from 14.3 dB down to 0.1 dB, matching the training data's own value, and
> 16QAM moved from 0.168 to 0.202 — nothing. Constellation density: the
> amplitude histograms of the two domains overlap almost exactly. Channel
> impairments: carrier frequency offset up to 1e-2, multipath with delay spread
> up to four samples, random phase, and all three combined — the best recovery
> was plus 0.045 against a shortfall of 0.83. Only the envelope substitution
> reproduced the effect."

*Dikkat:* Bu cevabı **akıcı** verebilmen lazım. Dört elemeyi sırayla
sayabiliyorsan, işi anladığın kanıtlanmış olur. Sayıları tam hatırlamazsan
"roughly" de, uydurma.

### Q: "You said the gap is 0.182. Plus or minus what?"

**Cevap:**
> "Plus or minus 0.017, over four seeds, varying both the train/test split and
> the weight initialisation. The improvement from whitening is 7.4 pooled
> standard deviations, so it is well outside run-to-run variation."

*Dikkat:* Hata çubuğu sorusu **her zaman** gelir. Hazır ol.

### Q: "Show me where you were wrong."

*Ne soruyor:* Dürüstlük testi. Bu soruyu **seviyorsun**, çünkü cevabın bol.

**Cevap:**
> "Four times. First I thought it was pulse shaping and swept the roll-off from
> 0.9 down to 0.15, saw only partial recovery, and concluded I was wrong — but
> my sweep stopped too early; at 0.01 the recovery was much larger. So the
> hypothesis was right and my test was inadequate. Second, I thought my
> constellations looked denser; the histograms refuted that. Third, I thought
> symbol timing regularity was the cause; removing it changed nothing. Fourth,
> I thought the network was using the envelope as a shortcut; an
> envelope-transplant test refuted that too."

*Dikkat:* Birinci maddeyi mutlaka anlat — **"hipotezim doğruydu, testim
yetersizdi"** ayrımı, deneysel olgunluk gösterir.

---

## 2. Tespit teorisi çerçevesi — Kam'ın sahası

Burada onun diline geçiyorsun. Bu bölüm toplantının en ilginç kısmı olabilir.

### Q: "How would you describe this in decision-theoretic terms?"

**Cevap:**
> "As an M-ary hypothesis testing problem where the likelihood model is learned
> rather than specified. The training distribution defines implicit decision
> regions in a learned feature space. Domain shift moves the test samples
> outside the region where those boundaries were fitted, and the sample lands in
> whichever region happens to extend into that part of the space. That is why
> the errors are structured rather than diffuse."

*Dikkat:* Bunu ezberden söyleme, **anla.** Eğer "learned likelihood" derse ve
sen açamıyorsan kötü olur. Basit hali: klasik tespit tekniğinde olasılık
modelini sen yazarsın; burada ağ veriden çıkarıyor, ve veri değişince model
geçersizleşiyor ama karar sınırları yerinde kalıyor.

### Q: "You say the failure is structured. What is the structure?"

**Cevap:**
> "The network maps an unseen modulation onto its nearest neighbour in
> constellation order. I tested this by removing classes and probing with the
> removed one. Train without 64QAM, show it 64QAM: 100% goes to 16QAM. Remove
> 16QAM too: everything goes to 8PSK. Remove 8PSK: everything goes to QPSK.
> Twelve of twelve cells."

**Ve hemen ardından, sormasını beklemeden:**
> "But those twelve cells do not discriminate between two hypotheses, because
> in every one of them the nearest class is also the densest remaining class.
> So I built a case where they disagree: train on QPSK through 64QAM, probe
> with BPSK. Nearest is QPSK, densest is 64QAM. The answer was QPSK, at 94 and
> 99.9 percent across two seeds, with zero going to 64QAM. So it is adjacency."

*Dikkat:* **Bu, toplantının en güçlü 90 saniyesi.** Confound'u kendin
görüp kendin çözdüğünü gösteriyorsun. Kam'ın kariyeri karar bölgeleri
üzerine — bu tam onun ilgisini çeker.

### Q: "Could that just be an artifact of your output layer?"

*Ne soruyor:* Confound avı. Cevabın hazır olduğu için bu soru sana puan yazar.

**Cevap:**
> "I checked. I randomly permuted which output index corresponds to which
> class and retrained. The sink stayed on the same class every time, three out
> of three. So it follows the signals, not the index."

### Q: "What happens with pure noise?"

**Cevap:**
> "That was my third probe, and it behaves differently. Gaussian noise does not
> produce a sink — the predictions sit near chance, 20 to 30 percent across five
> classes with a mild bias toward BPSK. So 'an unfamiliar modulation' and 'no
> modulation at all' are not the same failure mode. I do not have an explanation
> for that difference yet."

*Dikkat:* **"I do not have an explanation yet"** demek serbest. Uydurmak
serbest değil.

### Q: "Why should adjacency happen? What's the mechanism?"

**Cevap:**
> "I do not know yet, and I would rather say that than guess. The observation is
> consistent with the network holding an ordered internal representation of
> constellation order, so that an unseen class falls next to its neighbours.
> But I have measured the behaviour, not the representation. Testing it would
> mean looking at the latent space directly, which I have not done."

*Dikkat:* Bu soruyu **bilmiyorum** ile cevaplamak doğru. Bu oturumda dört kez
erken tahmin yürütüp yanıldık; bunu ona anlatmak bile bir cevap.

---

## 3. Zayıf noktalar — bunlar gelecek, hazır ol

### Q: "Is any of this real data?"

*Ne soruyor:* En büyük zafiyetin. **Baştan kabul et.**

**Cevap:**
> "No, and that is the main limitation. RadioML is simulated and my generator is
> simulated, so what I measured is a gap between two simulators. It is a real
> and measurable gap, but I cannot claim anything about real-world
> generalization from it. Closing that is exactly why I am looking for lab
> access — the plan is a cabled SDR setup, transmitter to attenuator to
> receiver, so nothing is radiated and the experiments stay repeatable."

*Dikkat:* Savunmaya geçme. **"No, and that is the main limitation"** ile
başla. Sonra planı anlat. Zafiyeti isteğine bağlamış olursun.

### Q: "What have you read? What's the state of the art?"

*Ne soruyor:* Literatür bilgin. **Şu an en zayıf yerin — 10 günde kapat.**

**Şimdilik dürüst cevap:**
> "Less than I should have. I know domain adaptation for modulation
> classification is active — there is DANN-style adversarial adaptation, and
> there is a standard augmentation set of rotation, flip and Gaussian noise
> that I reproduced as my baseline. I built the measurement first and read
> second, which is the wrong order. I am correcting that now."

*Dikkat:* Bu cevabı vermek zorunda kalmamak için 4-6. günlerde okuma yap.
Ama vermen gerekirse, **saklamaktan çok daha iyi.** "Wrong order" demek
olgunluk gösterir.

### Q: "How did you do all this in one month?"

*Ne soruyor:* Merak, şüphe değil. Ama dürüst cevapla.

**Cevap:**
> "I directed the work and made the research decisions — what to test next,
> what to rule out, when to stop and diagnose instead of patching. I used AI
> tooling heavily for implementation, which is how I could run this many
> controlled experiments in the time I had. What I can do is explain every
> experiment and why it was designed that way."

*Dikkat:* **Saklama.** 2026'da bu normal ve bir provost bunu bilir. Yakalanmak
söylemekten çok daha kötü. Ve arkasından gerçekten açıklayabilmelisin — bu
yüzden 1-3. günler önemli.

### Q: "What's novel here?"

**Cevap:**
> "I am not in a position to claim novelty — I have not done a proper
> literature review. What I think is unusual is the attribution method rather
> than the fix: using a phase-preserving magnitude substitution to assign a
> domain gap to one specific signal property, and ruling out the channel
> explanations by controlled elimination. Most of what I have seen measures a
> gap and applies a method. I would value your judgement on whether that is
> already standard."

*Dikkat:* "Yeni bir şey buldum" **deme.** Bilmiyorsun. Soruyu ona geri
vermek hem dürüst hem akıllı.

---

## 4. Ne istediğin

### Q: "What can I do for you?"

**Bu soru gelecek. Cevabın kısa ve net olmalı.**

> "Three things, in order of usefulness to me. First, a pointer to whoever at
> NJIT this is closest to — I have looked at Professor Abdi, Professor Kliewer
> and Professor Haimovich, and an introduction would carry far more weight than
> a cold email from me. Second, guidance on how an undergraduate gets access to
> RF measurement equipment here, since real captures are the missing piece.
> Third, whether this is a sensible senior design topic or whether I should
> scope it down."

*Dikkat:*
- **Maaşlı pozisyon isteme.** Provost'un vereceği şey değil.
- Üç istek de onun **tek mailiyle** çözülebilir şeyler. Kolay evet.
- Sıralama önemli: en değerlisi ilk.

---

## 5. Senin soracakların

Bir saatlik toplantıda sadece cevap verirsen sohbet olmaz. **En az üç soru hazırla.**

> "In your work on decision fusion, is there an established way to think about
> where a decision goes when the observation falls outside the region the
> boundaries were fitted on? I could not find literature on it and I suspect I
> am using the wrong search terms."

> "If you were advising me: is the more valuable contribution the method, or a
> real captured dataset with controlled domain variation? My sense is the field
> has many methods and very little real cross-domain data."

> "What would you want to see from a senior project before you considered it
> worth publishing?"

*Dikkat:* Birinci soru en iyisi — onu uzman konumuna koyuyor, gerçek bir
boşluğu soruyor, ve "yanlış terimlerle arıyor olabilirim" demek alçakgönüllü
ve muhtemelen doğru.

---

## Ezberden bilmen gereken sayılar

| Ne | Değer |
|---|---|
| Açık (SNR ≥ 10 dB) | **0.182 ± 0.017** |
| İç-alan / çapraz-alan | 0.993 / 0.811 |
| Beyazlatma sonrası çapraz | 0.909 |
| Beyazlatma + standart artırma | **0.948**, açık 0.038 |
| Kapanan oran | **%79** |
| Literatür baseline'ına fark | +0.054, **3.3 s.d.** |
| 16QAM (çapraz-alan) | 0.123 → **0.965** |
| Zarf ikamesi | 16QAM 0.191 → **0.972** |
| Zarf nakli | %0.2 zarfı takip ediyor |
| Sink kontrolü | BPSK → QPSK, **%94 ve %99.9** |
| Tohum sayısı | **4** |
| Veri seti | RadioML 2018.01A, 24 sınıf, 5'i kullanıldı |

Bu tablodaki her satırı **hatırlamalısın.** Gerisini "roughly" diyebilirsin.

---

## Son üç hatırlatma

1. **Yavaş konuş.** İkinci dilinde teknik anlatım yavaş olur, bu normaldir.
2. **"Let me think about that for a second."** — kullan, anadili olanlar da kullanıyor.
3. **Bilmediğine bilmiyorum de.** Bu oturumda dört kez yanıldık ve her seferinde
   ölçüm düzeltti. Aynı refleksi toplantıda göster: emin olmadığın şeyi
   iddia etme. Bir tespit teorisyeni için en güven verici davranış budur.

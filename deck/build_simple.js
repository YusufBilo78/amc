// The four-class box, in the format of the pycasa deck Moshe liked: one
// question per slide, a small orange kicker, the code that did it in a chip,
// INPUT -> OUTPUT, a few big numbers, one bold line to take away.
// Numbers come from deck2016_data.json (deck/extract_2016.py); pictures in
// deck/simple/ are illustrations (deck/simple_figs.py), not results.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("deck2016_data.json", "utf8"));
const TALK = require("./talk_simple.js");

const NAVY = "0F1B2D", ORANGE = "F08A24", CARD = "EEF2F6", INK = "1E293B", MUTED = "5B6B7F", WHITE = "FFFFFF", LINE = "D5DDE5";
const F = "Calibri", MONO = "Consolas";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Yusuf Bilal Cetinkaya";
pres.title = "Telling four radio signals apart";
const W = 13.33, M = 0.6;
const num = v => v.toLocaleString("en-US"), pct = v => (100 * v).toFixed(1) + "%";
const C = D.c4, L = C.lengths, cls = C.classes, SN = D.snr, FA = D.faithful;
const wrongAt = lv => L["128"].wrong_by_snr[C.snrs.indexOf(lv)];
let n = 0;

function t(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: F, fontSize: 14, color: INK, isTextBox: true, margin: 0, valign: "top", ...o });
}
function head(s, kicker, title, chip, sub) {
  t(s, kicker, M, 0.42, 8, 0.3, { fontSize: 11, bold: true, color: ORANGE, charSpacing: 3 });
  t(s, title, M, 0.72, W - 2 * M, 0.7, { fontSize: 30, bold: true, valign: "middle" });
  if (sub) t(s, sub, M, 1.45, W - 2 * M, 0.4, { fontSize: 15, color: MUTED });
  if (chip) {
    const w = 0.12 * chip.length + 0.4;
    s.addShape(pres.ShapeType.roundRect, { x: W - M - w, y: 0.38, w, h: 0.36, fill: { color: CARD }, line: { color: CARD }, rectRadius: 0.12 });
    t(s, chip, W - M - w, 0.38, w, 0.36, { fontFace: MONO, fontSize: 11, color: INK, align: "center", valign: "middle" });
  }
}
function pill(s, text, x, y, w = 1.2) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h: 0.32, fill: { color: ORANGE }, line: { color: ORANGE }, rectRadius: 0.16 });
  t(s, text, x, y, w, 0.32, { fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle", charSpacing: 2 });
}
function arrow(s, x, y) {
  s.addShape(pres.ShapeType.rightArrow, { x, y, w: 0.55, h: 0.42, fill: { color: ORANGE }, line: { color: ORANGE } });
}
function card(s, x, y, w, h, fill = CARD) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.1 });
}
function badge(s, x, y, k, fill = NAVY, d = 0.42) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  t(s, String(k), x, y, d, d, { fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle" });
}
function big(s, x, y, w, value, label, color = INK, size = 36) {
  t(s, value, x, y, w, 0.7, { fontSize: size, bold: true, color, valign: "bottom" });
  t(s, label, x, y + 0.72, w, 0.55, { fontSize: 12, color: MUTED });
}
function pic(s, path, x, y, w, pxW, pxH, frame = true) {
  const h = w * pxH / pxW;
  if (frame) s.addShape(pres.ShapeType.roundRect, { x: x - 0.04, y: y - 0.04, w: w + 0.08, h: h + 0.08, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.06 });
  s.addImage({ path, x, y, w, h });
  return h;
}
function takeaway(s, bold, rest, y = 6.55) {
  t(s, [{ text: bold, options: { bold: true } }, { text: rest ? "  " + rest : "" }], M, y, W - 2 * M, 0.6, { fontSize: 15 });
}
function done(s, dark = false) {
  n++;
  if (TALK[n]) s.addNotes(TALK[n]);
  t(s, String(n), W - M - 0.4, 7.05, 0.4, 0.25, { fontSize: 10, color: dark ? "8FA0B5" : MUTED, align: "right" });
}

// ============================ 1. title ======================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "AMC  ·  DATA FUSION LAB", M, 1.25, 7, 0.35, { fontSize: 13, bold: true, color: ORANGE, charSpacing: 4 });
  t(s, "Telling four radio signals apart", M, 1.7, 7.2, 1.7, { fontSize: 42, bold: true, color: WHITE, valign: "top" });
  t(s, "What our detector does, step by step", M, 3.55, 7, 0.5, { fontSize: 20, color: "C9D3E0" });
  t(s, "BPSK  ·  QPSK  ·  QAM16  ·  QAM64", M, 4.2, 7, 0.4, { fontSize: 16, color: WHITE, bold: true });
  s.addImage({ path: "simple/noise.gif", x: 8.35, y: 1.0, w: 4.0, h: 4.3 });
  t(s, "16 symbols of one QAM16 frame, as the noise grows", 8.35, 5.4, 4.0, 0.35, { fontSize: 11, color: "9AA8BA", align: "center" });
  t(s, "Prepared for Dr. Moshe Kam's research group  ·  NJIT  ·  October 2026", M, 6.6, 9, 0.3, { fontSize: 11, color: "9AA8BA" });
  done(s, true);
}

// ============================ 2. what it does ===============================
{ const s = pres.addSlide();
  head(s, "WHAT IT IS", "What does the detector do?", "model_zoo.backbone(4)",
    "It is given 128 samples of a radio signal and says which of four modulations sent them.");
  const steps = [["Look at short pieces", "2 convolution layers", "Each piece: 12 samples, about 1.5 symbols. The frame becomes 32 pieces.", "25,792"],
    ["Read the pieces in order", "BiLSTM, forwards and backwards", "Each piece now knows about the whole frame.", "659,968"],
    ["Decide", "attention + 2 dense layers", "Weighted average of the 32 pieces → 4 scores. The largest score wins.", "99,716"]];
  steps.forEach(([h, sub, body, p], i) => {
    const x = M + i * 4.2, y = 2.1, w = 3.5;
    card(s, x, y, w, 3.6);
    badge(s, x + 0.25, y + 0.25, i + 1);
    t(s, h, x + 0.8, y + 0.22, w - 0.95, 0.5, { fontSize: 18, bold: true, valign: "middle" });
    t(s, sub, x + 0.25, y + 0.9, w - 0.5, 0.35, { fontFace: MONO, fontSize: 11, color: MUTED });
    t(s, body, x + 0.25, y + 1.4, w - 0.5, 1.3, { fontSize: 14 });
    t(s, `${p} numbers learned`, x + 0.25, y + 3.0, w - 0.5, 0.35, { fontSize: 12, color: MUTED });
    if (i < 2) arrow(s, x + w + 0.12, y + 1.6);
  });
  takeaway(s, `${num(D.params.total4)} numbers learned in total.`, "Most of them, 84%, are in step 2. Nothing in it is random once trained: the same frame always gets the same answer.", 6.1);
  done(s);
}

// ============================ 3. where it came from =========================
{ const s = pres.addSlide();
  head(s, "WHERE IT CAME FROM", "Who designed it, and who wrote the code?", "colab/icrnna_faithful_2016.py");
  const flow = [["A paper", "El-Haryqy et al., Results in Engineering, 2025. Model name: ICRNNA. Reported 63.24% on 11 classes."],
    ["A colleague's code", "Their re-implementation of the paper. Our code is copied from it."],
    ["Our code", "Differs from the paper in 5 small places (filter size, pooling, normalisation, dropout, one layer fewer)."]];
  flow.forEach(([h, b], i) => {
    const x = M + i * 4.2, y = 2.0, w = 3.5;
    card(s, x, y, w, 2.2);
    t(s, h, x + 0.25, y + 0.2, w - 0.5, 0.45, { fontSize: 18, bold: true });
    t(s, b, x + 0.25, y + 0.75, w - 0.5, 1.4, { fontSize: 13.5 });
    if (i < 2) arrow(s, x + w + 0.12, y + 0.85);
  });
  card(s, M, 4.5, 6.0, 1.85);
  t(s, "Is the paper's number real?", M + 0.25, 4.62, 5.5, 0.4, { fontSize: 15, bold: true });
  t(s, "I rebuilt the model from the paper alone and trained it until it stopped improving:", M + 0.25, 5.02, 5.5, 0.5, { fontSize: 12.5, color: MUTED });
  t(s, [{ text: `${FA.e150.toFixed(2)}%`, options: { bold: true, color: ORANGE, fontSize: 30 } }, { text: `   against the paper's ${FA.paper.toFixed(2)}%`, options: { fontSize: 15 } }],
    M + 0.25, 5.5, 5.6, 0.7, { valign: "middle" });
  card(s, M + 6.33, 4.5, W - 2 * M - 6.33, 1.85);
  t(s, "Written with an AI assistant", M + 6.58, 4.62, 5.6, 0.4, { fontSize: 15, bold: true });
  t(s, "The code, the plots and these slides were written with Claude Code (Anthropic). Every number comes from running that code on a GPU. The choices of what to test are mine.",
    M + 6.58, 5.02, 5.3, 1.3, { fontSize: 13 });
  done(s);
}

// ============================ 4. the big picture ============================
{ const s = pres.addSlide();
  const hi = L["128"], m10 = C.mats["10"];
  head(s, "THE BIG PICTURE", "What we gave it, and what came out", "train_backbone.py --classes BPSK,QPSK,QAM16,QAM64");
  pill(s, "INPUT", M, 1.75);
  pic(s, "simple/frame.png", M, 2.2, 3.9, 1530, 544);
  t(s, [{ text: "Data: ", options: { bold: true } }, { text: "RML2016.10a (DeepSig, simulated)", options: { breakLine: true } },
    { text: "4 kinds × 20 noise levels × 1,000 frames", options: { breakLine: true } },
    { text: "One frame: ", options: { bold: true } }, { text: "128 samples = 16 symbols" }], M, 3.75, 3.95, 1.2, { fontSize: 12.5 });
  arrow(s, 4.7, 3.0);
  pill(s, "DETECTOR", 5.45, 1.75, 1.5);
  [["Learn", "on 70% of the frames"], ["Tune", "on 15%: when to stop"], ["Test", "on 15% it never saw"]].forEach(([h, b], i) => {
    card(s, 5.45, 2.2 + i * 0.85, 3.1, 0.72);
    badge(s, 5.6, 2.35 + i * 0.85, i + 1, NAVY, 0.4);
    t(s, [{ text: h + "  ", options: { bold: true } }, { text: b, options: { color: MUTED } }], 6.15, 2.2 + i * 0.85, 2.35, 0.72, { fontSize: 13, valign: "middle" });
  });
  t(s, `Repeated ${C.seeds} times, a different split each time.`, 5.45, 4.8, 3.1, 0.5, { fontSize: 12, color: MUTED });
  arrow(s, 8.75, 3.0);
  pill(s, "OUTPUT", 9.5, 1.75);
  const rightAll = 1 - hi.high_wrong / hi.high_total;
  const outs = [[pct(rightAll), "right, at 10 dB and above"], [pct(m10[1][1] / m10[1].reduce((a, b) => a + b, 0)), "BPSK and QPSK right at 10 dB"],
    ["25%", "at −20 dB: a guess"], [num(hi.high_total), "test decisions at 10 dB and above"]];
  outs.forEach(([v, l], i) => {
    const x = 9.5 + (i % 2) * 1.65, y = 2.2 + Math.floor(i / 2) * 1.35;
    card(s, x, y, 1.55, 1.22);
    t(s, v, x + 0.12, y + 0.08, 1.35, 0.55, { fontSize: 22, bold: true, color: i === 0 ? ORANGE : INK, valign: "bottom" });
    t(s, l, x + 0.12, y + 0.65, 1.35, 0.55, { fontSize: 10.5, color: MUTED });
  });
  takeaway(s, "How we check it:", "every number is counted on test frames the detector never saw while learning.", 5.6);
  done(s);
}

// ============================ 5. one frame ==================================
{ const s = pres.addSlide();
  head(s, "STEP 1  ·  THE INPUT", "What one frame looks like", "128 samples = 16 symbols");
  pic(s, "simple/frame.png", M, 1.8, 7.6, 1530, 544);
  t(s, "One frame of QAM16, drawn without noise. Orange: the in-phase part (I); blue: the quadrature part (Q). A symbol every 8 samples (the dots); each dot sits on −3, −1, 1 or 3. That is what makes it QAM16.",
    M, 4.6, 7.6, 1.0, { fontSize: 13 });
  pic(s, "simple/symbols_2x2.png", 9.1, 1.8, 3.6, 1020, 1071);
  t(s, "16 symbols of one frame (orange) on all the points each kind can use (circles).", 9.1, 5.65, 3.6, 0.6, { fontSize: 11.5, color: MUTED });
  t(s, [{ text: "16 symbols cannot show all 64 points of QAM64.", options: { bold: true } }, { text: "  So a QAM64 frame can look like QAM16 even with no noise at all." }], M, 5.75, 8.2, 0.7, { fontSize: 15 });
  t(s, "Pictures drawn by us to explain; the data are DeepSig's.", M, 6.75, 8, 0.3, { fontSize: 10.5, color: MUTED, italic: true });
  done(s);
}

// ============================ 6. the noise ==================================
{ const s = pres.addSlide();
  head(s, "STEP 2  ·  THE NOISE", "What does “10 dB” mean in this data?", "tools/measure_rml2016_snr.py");
  const facts = [["Who made it", "DeepSig, in a simulator (GNU Radio). Not recorded from a real radio."],
    ["What the label sets", "One number in their code: the noise amplitude = 10^(−label/10)."],
    ["What else is added", "Carrier offset, clock offset, fading and echoes (multipath) — at every noise level, even the best."]];
  facts.forEach(([h, b], i) => {
    card(s, M, 1.85 + i * 1.32, 5.6, 1.18);
    t(s, h, M + 0.25, 1.95 + i * 1.32, 5.1, 0.4, { fontSize: 15, bold: true });
    t(s, b, M + 0.25, 2.35 + i * 1.32, 5.1, 0.65, { fontSize: 13 });
  });
  t(s, "I measured the noise in the file itself:", 6.75, 1.85, 6, 0.4, { fontSize: 15, bold: true });
  const sl = Math.max(...Object.values(SN.slopes)).toFixed(1);
  card(s, 6.75, 2.35, 2.85, 2.3); card(s, 9.88, 2.35, 2.85, 2.3);
  t(s, `${sl} dB`, 6.95, 2.5, 2.5, 0.8, { fontSize: 40, bold: true, color: INK, align: "center", valign: "bottom" });
  t(s, "of real change for every 1 dB of label", 6.95, 3.35, 2.5, 0.9, { fontSize: 12.5, color: MUTED, align: "center" });
  t(s, `+${SN.offsets.QAM64.toFixed(0)} dB`, 10.08, 2.5, 2.5, 0.8, { fontSize: 40, bold: true, color: ORANGE, align: "center", valign: "bottom" });
  t(s, "more signal in QAM64 than in BPSK or QPSK, at the same label", 10.08, 3.35, 2.5, 0.9, { fontSize: 12.5, color: MUTED, align: "center" });
  t(s, "How: the signal sits in the middle of the spectrum; outside it there is only noise, so its level there gives the noise power. Checked first on frames of known noise (within 0.5 dB).",
    6.75, 4.85, 5.98, 1.1, { fontSize: 12, color: MUTED });
  takeaway(s, "“10 dB” is a setting, not a measurement,", "and it means a different amount of noise for each kind of signal.", 6.2);
  done(s);
}

// ============================ 7. result at 10 dB ============================
{ const s = pres.addSlide();
  const m = C.mats["10"], tot = m.flat().reduce((a, b) => a + b, 0), ok = m.reduce((a, r, i) => a + r[i], 0);
  head(s, "RESULT", "The answer at one noise level: 10 dB", "tools/single_snr.py");
  t(s, `${num(tot)} test frames: ${C.seeds} repeats × 4 kinds × 150. Rows: what was sent. Columns: what the detector said.`, M, 1.5, W - 2 * M, 0.4, { fontSize: 14, color: MUTED });
  const colW = [1.6, 1.25, 1.25, 1.25, 1.25, 1.2];
  const hc = (x, o = {}) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, align: "center", fontSize: 13, ...o } });
  const rows = [[hc("sent \\ said", { fontSize: 11 }), ...cls.map(c => hc(c)), hc("recall")]];
  m.forEach((r, i) => {
    const tt = r.reduce((a, b) => a + b, 0);
    rows.push([{ text: cls[i], options: { bold: true, fontSize: 13 } },
      ...r.map((v, j) => ({ text: num(v), options: { align: "center", fontSize: 14, bold: i === j, color: i === j ? WHITE : (v >= 20 ? INK : "8A99A8"),
        fill: { color: i === j ? NAVY : (v >= 20 ? "FCE3C8" : WHITE) } } })),
      { text: pct(r[i] / tt), options: { align: "center", fontSize: 13, bold: true } }]);
  });
  const dec = cls.map((_, j) => m.reduce((a, r) => a + r[j], 0));
  rows.push([{ text: "precision", options: { bold: true, fontSize: 12, fill: { color: CARD } } },
    ...dec.map((d, j) => ({ text: pct(m[j][j] / d), options: { align: "center", fontSize: 12, fill: { color: CARD } } })), { text: "", options: { fill: { color: CARD } } }]);
  s.addTable(rows, { x: M, y: 2.05, w: 7.8, colW, fontFace: F, color: INK, border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.5, autoPage: false, valign: "middle" });
  t(s, [{ text: "Recall", options: { bold: true } }, { text: " = of the frames that were this kind, how many it named right (a row).", options: { breakLine: true } },
    { text: "Precision", options: { bold: true } }, { text: " = of the frames it gave this name, how many were right (a column)." }], M, 5.25, 7.8, 0.8, { fontSize: 12.5, color: MUTED });
  card(s, 8.9, 2.05, 3.83, 3.0);
  t(s, num(tot - ok), 9.1, 2.2, 3.4, 0.9, { fontSize: 46, bold: true, color: ORANGE, valign: "bottom" });
  t(s, `wrong out of ${num(tot)}`, 9.1, 3.1, 3.4, 0.4, { fontSize: 14, color: MUTED });
  t(s, num(m[2][3] + m[3][2]), 9.1, 3.55, 3.4, 0.8, { fontSize: 34, bold: true, color: INK, valign: "bottom" });
  t(s, "of them are QAM16 and QAM64 mixed up", 9.1, 4.35, 3.4, 0.5, { fontSize: 14, color: MUTED });
  takeaway(s, "BPSK and QPSK are essentially solved.", "Almost every mistake is QAM16 against QAM64.", 6.2);
  done(s);
}

// ============================ 8. lower ======================================
{ const s = pres.addSlide();
  const lv = [10, 6, 4, 2, 0, -2, -4, -6];
  head(s, "STEP BY STEP  ·  LESS SIGNAL", "Turn the noise up: when does it break?", "tools/single_snr.py");
  t(s, "Wrong answers out of 8,400 test frames at each label.", M, 1.5, W - 2 * M, 0.4, { fontSize: 14, color: MUTED });
  s.addChart(pres.ChartType.bar, [{ name: "wrong of 8,400", labels: lv.map(v => (v > 0 ? "+" : "") + v + " dB"), values: lv.map(wrongAt) }],
    { x: M, y: 1.95, w: 8.6, h: 4.2, barDir: "col", barGapWidthPct: 45, chartColors: [NAVY, NAVY, NAVY, NAVY, ORANGE, ORANGE, ORANGE, ORANGE],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: INK, dataLabelFontFace: F, dataLabelFormatCode: "#,##0",
      showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 13, catAxisLabelFontFace: F, valAxisMinVal: 0, valAxisMaxVal: 3300 });
  card(s, 9.55, 2.1, 3.18, 1.7);
  t(s, "10 → 2 dB", 9.75, 2.2, 2.8, 0.6, { fontSize: 24, bold: true, color: NAVY });
  t(s, "about 450–490 wrong every time: the same box", 9.75, 2.8, 2.8, 0.9, { fontSize: 13, color: MUTED });
  card(s, 9.55, 4.0, 3.18, 1.7);
  t(s, "below 0 dB", 9.75, 4.1, 2.8, 0.6, { fontSize: 24, bold: true, color: ORANGE });
  t(s, "it breaks: QPSK starts going to the QAMs first", 9.75, 4.7, 2.8, 0.9, { fontSize: 13, color: MUTED });
  takeaway(s, "From 10 dB down to 2 dB nothing changes.", "The mistakes there are not caused by noise.", 6.35);
  done(s);
}

// ============================ 9. frame length ===============================
{ const s = pres.addSlide();
  head(s, "COMPARING FRAME LENGTHS", "16, 8 or 4 symbols: what does it need?", "train_backbone.py --frame-len 64 / 32");
  const lens = ["128", "64", "32"], snrs = C.snrs;
  s.addChart(pres.ChartType.line, lens.map(k => ({ name: `${k / 8} symbols`, labels: snrs.map(v => String(v)), values: L[k].wrong_by_snr.map(w => +(100 * w / 8400).toFixed(1)) })),
    { x: M, y: 1.65, w: 6.2, h: 4.4, chartColors: [NAVY, ORANGE, "8FA0B5"], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5,
      showLegend: true, legendPos: "t", legendFontSize: 12, legendFontFace: F, legendColor: INK,
      valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, showValAxisTitle: true, valAxisTitle: "% wrong", showCatAxisTitle: true, catAxisTitle: "label (dB)",
      catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" },
      valAxisTitleColor: MUTED, catAxisTitleColor: MUTED, valAxisTitleFontSize: 11, catAxisTitleFontSize: 11 });
  pic(s, "simple/fewer.png", 7.0, 1.8, 3.75, 1428, 986);
  const fl = lens.map(k => L[k].floor_pct + "%");
  [[fl[0], "16 symbols"], [fl[1], "8 symbols"], [fl[2], "4 symbols"]].forEach(([v, l], i) => {
    t(s, v, 11.05, 1.75 + i * 0.85, 1.7, 0.5, { fontSize: 24, bold: true, color: [NAVY, ORANGE, "8FA0B5"][i] });
    t(s, l, 11.05, 2.2 + i * 0.85, 1.7, 0.3, { fontSize: 11, color: MUTED });
  });
  t(s, "wrong, from +2 dB up", 11.05, 4.3, 1.7, 0.35, { fontSize: 11, color: MUTED, italic: true });
  t(s, "Fewer symbols, fewer points of the grid shown: QAM64 and QAM16 start to look alike.", 7.0, 4.5, 3.75, 0.9, { fontSize: 12, color: MUTED });
  takeaway(s, "More signal does not fix it. More symbols does.", "Noise decides where the curve falls; the number of symbols decides how low it can go.", 6.25);
  done(s);
}

// ============================ 10. something we found ========================
{ const s = pres.addSlide();
  const m = C.m20;
  head(s, "SOMETHING WE FOUND", "With no signal at all, it still answers", "tools/single_snr.py");
  card(s, M + 0.4, 1.75, 3.6, 2.4); card(s, M + 4.9, 1.75, 3.6, 2.4);
  t(s, pct(m.recall[1]).replace(".0", ""), M + 0.4, 1.9, 3.6, 1.0, { fontSize: 50, bold: true, color: INK, align: "center", valign: "bottom" });
  t(s, "QPSK recall at −20 dB", M + 0.4, 2.95, 3.6, 0.4, { fontSize: 14, color: MUTED, align: "center" });
  t(s, "looks like it recognises QPSK", M + 0.4, 3.4, 3.6, 0.5, { fontSize: 15, bold: true, align: "center" });
  t(s, "vs", M + 4.05, 2.6, 0.8, 0.5, { fontSize: 20, color: MUTED, align: "center" });
  t(s, pct(m.precision[1]).replace(".0", ""), M + 4.9, 1.9, 3.6, 1.0, { fontSize: 50, bold: true, color: ORANGE, align: "center", valign: "bottom" });
  t(s, "QPSK precision at −20 dB", M + 4.9, 2.95, 3.6, 0.4, { fontSize: 14, color: MUTED, align: "center" });
  t(s, "it is guessing: 1 in 4", M + 4.9, 3.4, 3.6, 0.5, { fontSize: 15, bold: true, align: "center" });
  t(s, [{ text: "Why? ", options: { bold: true, color: ORANGE } }, { text: `It must name one of the four. With nothing to go on it says BPSK or QPSK ${pct(m.decided_share[0] + m.decided_share[1])} of the time, and which of the two changes from one training run to the next.` }],
    M + 0.4, 4.45, 8.1, 1.0, { fontSize: 14 });
  card(s, 9.55, 1.75, 3.18, 3.7);
  t(s, "How they are counted", 9.75, 1.85, 2.8, 0.4, { fontSize: 13, bold: true, color: MUTED });
  t(s, [{ text: "Recall", options: { bold: true, breakLine: true } }, { text: "= right ÷ frames sent of that kind", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Precision", options: { bold: true, breakLine: true } }, { text: "= right ÷ frames it called that kind", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Recall alone can look good while the detector is only guessing.", options: { color: MUTED } }], 9.75, 2.3, 2.85, 3.0, { fontSize: 13 });
  takeaway(s, "Every chart now shows precision next to recall,", "and how many decisions sit behind each point.", 6.2);
  done(s);
}

// ============================ 11. in short ==================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "In short", M, 0.6, 8, 0.8, { fontSize: 36, bold: true, color: WHITE });
  const pts = ["BPSK and QPSK are essentially solved. The mistakes are QAM16 against QAM64.",
    "Those mistakes come from having only 16 symbols, not from noise: with 8 or 4 symbols they grow 2.7 and 4.3 times.",
    "“10 dB” in this data is a setting, not a measurement, and it means a different noise level for each kind of signal."];
  pts.forEach((p, i) => {
    badge(s, M, 1.75 + i * 1.3, i + 1, ORANGE, 0.55);
    t(s, p, M + 0.85, 1.7 + i * 1.3, 6.9, 1.1, { fontSize: 17, color: WHITE, valign: "middle" });
  });
  t(s, [{ text: "Next: ", options: { bold: true, color: ORANGE } }, { text: "switch off the carrier offset, fading and echoes one at a time, and see which one the QAM mistakes belong to.", options: { color: "C9D3E0" } }],
    M, 5.75, 7.8, 0.8, { fontSize: 14 });
  pic(s, "simple/fewer.png", 8.4, 1.75, 4.33, 1428, 986, false);
  t(s, "QAM16 and QAM64 with 16, 8 and 4 symbols", 8.4, 4.8, 4.33, 0.3, { fontSize: 11, color: "9AA8BA", align: "center" });
  done(s, true);
}

pres.writeFile({ fileName: "moshe_2016_simple.pptx" }).then(f => console.log("wrote", f, "slides:", n));

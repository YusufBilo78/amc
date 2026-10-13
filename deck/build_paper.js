// The ICRNNA paper (El-Haryqy et al., Results in Engineering 26 (2025) 104783),
// presented so that Moshe can follow it: the data, the model block by block,
// the training, the results; then what we checked. Same style as the other
// decks: large type, one message per slide.
// Numbers come from paper_data.json (deck/extract_paper.py): paper values are
// transcribed with their table numbers, the authors' code (OFFICIAL_CODE.md) is
// read from the code itself, ours from the result files. Figures in deck/paper/
// cropped from the paper (CC BY-NC 4.0) say so; drawings marked "illustration"
// are ours and carry no data. The "official" slide appears only once the Colab
// results of the authors' code are in the repository.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("paper_data.json", "utf8"));
const TALK = require("./talk_paper.js");
const P = D.paper, O = D.ours, OF = D.official, OR = D.official_runs;

const NAVY = "0F1B2D", ORANGE = "F08A24", CARD = "EEF2F6", INK = "1E293B", MUTED = "5B6B7F", WHITE = "FFFFFF", LINE = "D5DDE5", SOFT = "C9D3E0", GREY = "8FA0B5";
const F = "Calibri";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Yusuf Bilal Cetinkaya";
pres.title = "ICRNNA: the paper, step by step";
const W = 13.33, M = 0.6;
const num = v => v.toLocaleString("en-US");
const FROM = "Figure from El-Haryqy et al. 2025 (CC BY-NC 4.0)";
let n = 0;

function t(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: F, fontSize: 20, color: INK, isTextBox: true, margin: 0, valign: "top", ...o });
}
function title(s, text, kicker) {
  if (kicker) t(s, kicker, M, 0.3, 9, 0.35, { fontSize: 14, bold: true, color: ORANGE, charSpacing: 2 });
  t(s, text, M, 0.6, W - 2 * M, 0.8, { fontSize: 34, bold: true, valign: "middle" });
}
function takeaway(s, text, y = 6.3, dark = false) {
  t(s, text, M, y, W - 2 * M, 0.7, { fontSize: 24, bold: true, color: dark ? WHITE : NAVY, valign: "middle" });
}
function card(s, x, y, w, h, fill = CARD) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
}
function box(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: fill }, line: { color: fill } });
}
function badge(s, x, y, k, fill = NAVY, d = 0.6) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  t(s, String(k), x, y, d, d, { fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
}
function arrow(s, x, y, w = 0.5, kind = "rightArrow", color = ORANGE) {
  s.addShape(pres.ShapeType[kind], { x, y, w, h: kind === "downArrow" ? 0.5 : 0.42, fill: { color }, line: { color } });
}
function big(s, x, y, w, value, label, color = INK, size = 56, align = "left") {
  t(s, value, x, y, w, 1.0, { fontSize: size, bold: true, color, align, valign: "bottom" });
  t(s, label, x, y + 1.05, w, 0.8, { fontSize: 19, color: MUTED, align });
}
function pic(s, path, x, y, w, pxW, pxH) {
  const h = w * pxH / pxW;
  s.addImage({ path, x, y, w, h });
  return h;
}
function credit(s, text, x, y, w) {
  t(s, text, x, y, w, 0.3, { fontSize: 12, color: MUTED, italic: true });
}
function done(s, id, dark = false) {
  n++;
  if (!TALK[id]) throw new Error(`no notes for slide ${id}`);
  s.addNotes(TALK[id]);
  t(s, String(n), W - M - 0.5, 7.0, 0.5, 0.3, { fontSize: 12, color: dark ? "8FA0B5" : MUTED, align: "right" });
}
// One row per code line: the line in a monospaced box on the left, what it
// does on the right. Lines are quoted from their rml2016_10a.py (commit
// ac000a0); a slide says when arguments were left out to fit.
function codeRows(s, rows, y0, rowH = 0.62, codeW = 8.55, size = 14) {
  const tags = rows.some(r => r[2] !== undefined), off = tags ? 1.55 : 0.15;
  rows.forEach(([code, note, tag], i) => {
    const y = y0 + i * rowH;
    box(s, M, y, codeW, rowH - 0.08, i % 2 ? "F6F8FA" : "EEF2F6");
    t(s, code, M + 0.15, y, codeW - 0.2, rowH - 0.08, { fontFace: "Courier New", fontSize: size, valign: "middle", color: INK });
    if (tag) t(s, tag, M + codeW + 0.15, y, 1.35, rowH - 0.08, { fontSize: 16, bold: true, color: ORANGE, valign: "middle" });
    t(s, note, M + codeW + off, y, W - 2 * M - codeW - off, rowH - 0.08, { fontSize: 16, color: MUTED, valign: "middle" });
  });
}
const PARTS = [["CNN", "short filters"], ["BiLSTM", "reads in order"], ["Attention", "weighs the steps"], ["Dense", "decides"]];
const pct = (a, b) => Math.round(100 * a / b);

// title =========================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "PAPER PRESENTATION", M, 1.0, 6, 0.4, { fontSize: 16, bold: true, color: ORANGE, charSpacing: 4 });
  t(s, "ICRNNA", M, 1.45, 12, 1.2, { fontSize: 66, bold: true, color: WHITE });
  t(s, "Improved automatic modulation recognition using deep learning with additive attention", M, 2.7, 11.5, 1.0, { fontSize: 26, color: SOFT });
  t(s, "El-Haryqy, Kharbouche, Ouamna, Madini, Zouine  ·  Ibn Tofail University, Morocco  ·  Results in Engineering 26 (2025) 104783", M, 3.85, 12.2, 0.5, { fontSize: 17, color: "9AA8BA" });
  PARTS.forEach(([h], i) => {
    const x = M + i * 2.6;
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.85, w: 2.1, h: 0.65, fill: { color: ORANGE }, line: { color: ORANGE }, rectRadius: 0.12 });
    t(s, h, x, 4.85, 2.1, 0.65, { fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle" });
    if (i < 3) arrow(s, x + 2.15, 4.97, 0.4);
  });
  t(s, "Yusuf Bilal Cetinkaya  ·  Moshe Kam's research group  ·  NJIT  ·  October 2026", M, 6.5, 10, 0.4, { fontSize: 16, color: "9AA8BA" });
  done(s, "title", true);
}

// why this paper, what it claims ==================================================
{ const s = pres.addSlide();
  title(s, "Why this paper, and what it claims", "THE PAPER");
  ["This paper (2025)", "Our detector"].forEach((h, i) => {
    const x = M + i * 4.15;
    card(s, x, 1.6, 3.55, 0.9, i === 1 ? NAVY : CARD);
    t(s, h, x, 1.6, 3.55, 0.9, { fontSize: 22, bold: true, align: "center", valign: "middle", color: i === 1 ? WHITE : INK });
    if (i < 1) arrow(s, x + 3.6, 1.85, 0.5);
  });
  const H = P.headline;
  [[`${H.acc_10a}%`, "best accuracy on RML2016.10a", ORANGE], [`${H.params_m} M`, "numbers to learn: a small model", INK], ["low SNR", "claimed to be its strength", INK]].forEach(([v, l, c], i) => {
    const x = M + i * 4.15;
    card(s, x, 2.85, 3.85, 2.45);
    big(s, x + 0.3, 3.0, 3.3, v, l, c, 50);
  });
  takeaway(s, "Today: what it does, step by step. Then what we checked.", 5.75);
  done(s, "claims");
}

// the data =======================================================================
{ const s = pres.addSlide();
  title(s, "The data: what the model sees", "THE PAPER  ·  DATA");
  [["11", "modulation types"], ["20", "SNR labels, −20 to 18 dB"], ["220,000", "frames"]].forEach(([v, l], i) => {
    big(s, M + i * 3.0, 1.45, 2.9, v, l, INK, 50);
  });
  pic(s, "simple/frame.png", 9.3, 1.55, 3.45, 1530, 544);
  t(s, "one frame: I (orange), Q (blue)", 9.3, 2.85, 3.45, 0.4, { fontSize: 15, color: MUTED });
  card(s, M, 3.65, W - 2 * M, 1.7, NAVY);
  t(s, "2 × 128", M + 0.4, 3.75, 4, 1.5, { fontSize: 66, bold: true, color: ORANGE, valign: "middle" });
  t(s, "Each frame: 128 samples of I and 128 of Q.\nThat array is the only input.", M + 4.6, 3.75, 7.2, 1.5, { fontSize: 24, color: WHITE, valign: "middle" });
  takeaway(s, "RML2016.10a, the same dataset we use.", 5.8);
  done(s, "data");
}

// the model at a glance ===========================================================
{ const s = pres.addSlide();
  title(s, "The model at a glance", "THE PAPER  ·  MODEL");
  const flow = [["frame", "2 × 128", CARD, INK], ["CNN", "64 × 128", NAVY, WHITE], ["BiLSTM", "64 × 256", ORANGE, WHITE],
    ["Attention", "256", NAVY, WHITE], ["Dense", "128 → 64", NAVY, WHITE], ["output", "11", CARD, INK]];
  const bw = 1.62, gap = 0.44;
  flow.forEach(([h, sz, fill, ink], i) => {
    const x = M + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: 2.0, w: bw, h: 1.6, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
    t(s, h, x, 2.0, bw, 1.6, { fontSize: 21, bold: true, color: ink, align: "center", valign: "middle" });
    t(s, sz, x, 3.8, bw, 0.5, { fontSize: 20, bold: true, color: MUTED, align: "center" });
    if (i < 5) arrow(s, x + bw + 0.03, 2.6, 0.38);
  });
  t(s, "size of what comes out", M, 4.35, 6, 0.4, { fontSize: 16, italic: true, color: MUTED });
  takeaway(s, "One frame in, eleven probabilities out. The largest is the answer.", 5.4);
  credit(s, "Redrawn from their Fig. 2; sizes from their code.", M, 6.25, 8);
  done(s, "model");
}

// block 1: CNN =====================================================================
{ const s = pres.addSlide();
  title(s, "Block 1: short learned filters", "THE PAPER  ·  CNN");
  // a strip of samples with a 3-sample window sliding along it (illustration)
  const cells = 22, cw = 0.27, x0 = M, y0 = 1.9;
  for (let i = 0; i < cells; i++) box(s, x0 + i * cw, y0, cw - 0.04, 0.45, i >= 6 && i < 9 ? ORANGE : CARD);
  t(s, "samples →", x0, y0 + 0.5, 3, 0.35, { fontSize: 15, color: MUTED });
  arrow(s, x0 + 7 * cw - 0.05, y0 + 0.55, 0.5, "downArrow");
  for (let i = 0; i < cells; i++) box(s, x0 + i * cw, y0 + 1.15, cw - 0.04, 0.45, i === 7 ? NAVY : "DDE3EA");
  t(s, "one output per sample, per filter", x0, y0 + 1.65, 6, 0.35, { fontSize: 15, color: MUTED });
  credit(s, "illustration", x0, y0 + 2.0, 3);
  const steps = [["64 filters, 3 taps", "2 × 128  →  128 × 64"], ["128 filters, 3 taps", "→  128 × 128"], ["pool: larger of each pair", "→  64 × 128"]];
  steps.forEach(([a, b], i) => {
    const y = 1.6 + i * 1.05;
    badge(s, 7.0, y + 0.1, i + 1, i === 2 ? ORANGE : NAVY, 0.55);
    t(s, a, 7.75, y, 5.0, 0.45, { fontSize: 21, bold: true });
    t(s, b, 7.75, y + 0.45, 5.0, 0.45, { fontSize: 19, color: MUTED });
  });
  card(s, M, 4.95, W - 2 * M, 0.95);
  t(s, "Like an FIR filter whose taps are learned instead of designed.", M + 0.3, 4.95, W - 2 * M - 0.6, 0.95, { fontSize: 22, valign: "middle" });
  takeaway(s, "Each of the 64 steps sums up about 6 samples: less than one symbol.", 6.15);
  done(s, "cnn");
}

// block 2: BiLSTM ==================================================================
{ const s = pres.addSlide();
  title(s, "Block 2: reading the steps in order", "THE PAPER  ·  BiLSTM");
  const k = 6, bw = 1.25, gap = 0.45, y = 2.55;
  ["1", "2", "3", "…", "63", "64"].forEach((lab, i) => {
    const x = M + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: bw, h: 0.9, fill: { color: CARD }, line: { color: CARD }, rectRadius: 0.1 });
    t(s, `step ${lab}`, x, y, bw, 0.9, { fontSize: 17, bold: true, align: "center", valign: "middle" });
    if (i < k - 1) {
      arrow(s, x + bw + 0.02, y - 0.5, gap - 0.04, "rightArrow", NAVY);
      arrow(s, x + bw + 0.02, y + 1.0, gap - 0.04, "leftArrow", ORANGE);
    }
  });
  t(s, "forwards, with a memory", M, y - 0.95, 6, 0.4, { fontSize: 17, bold: true, color: NAVY });
  t(s, "backwards, a second LSTM", M, y + 1.45, 6, 0.4, { fontSize: 17, bold: true, color: ORANGE });
  card(s, M, 4.55, 7.2, 1.4);
  t(s, [{ text: "2 layers, 128 numbers each way", options: { breakLine: true } }, { text: "→  64 × 256" , options: { color: MUTED } }],
    M + 0.3, 4.55, 6.7, 1.4, { fontSize: 22, bold: true, valign: "middle" });
  card(s, 8.1, 4.55, W - M - 8.1, 1.4, NAVY);
  t(s, `${pct(OF.blocks.BiLSTM, OF.params)}%`, 8.35, 4.55, 2.0, 1.4, { fontSize: 48, bold: true, color: ORANGE, valign: "middle" });
  t(s, "of the model's numbers are here", 10.3, 4.55, 2.3, 1.4, { fontSize: 18, color: WHITE, valign: "middle" });
  takeaway(s, "After this block, every step knows the whole frame.", 6.2);
  done(s, "lstm");
}

// block 3: attention ===============================================================
{ const s = pres.addSlide();
  title(s, "Block 3: a learned weighted average", "THE PAPER  ·  ATTENTION");
  const hs = [0.25, 0.4, 0.3, 0.9, 1.6, 1.2, 0.5, 0.35, 0.6, 1.4, 1.8, 0.9, 0.4, 0.3, 0.5, 0.7];
  hs.forEach((h, i) => box(s, M + 0.15 + i * 0.36, 3.9 - h, 0.26, h, h > 1.0 ? ORANGE : GREY));
  t(s, "a weight for each of the 64 steps; they add up to 1", M, 4.05, 6.2, 0.4, { fontSize: 15, color: MUTED });
  credit(s, "illustration", M, 4.4, 3);
  const eq = [["score", " = v · tanh(W₁q + W₂k)"], ["weights", " = softmax(score)"], ["output", " = Σ weightₜ · stepₜ"]];
  eq.forEach(([a, b], i) => t(s, [{ text: a, options: { bold: true, color: ORANGE } }, { text: b }], 6.75, 1.65 + i * 0.75, 6.0, 0.65,
    { fontFace: "Cambria", fontSize: 24, valign: "middle" }));
  t(s, "“Improved” (from their code):", 6.75, 4.0, 6, 0.45, { fontSize: 19, bold: true });
  t(s, "a learned scale on the scores · dropout 0.1 on the weights · normalization of the output", 6.75, 4.45, 6.0, 0.9, { fontSize: 18, color: MUTED });
  takeaway(s, "64 steps in, one vector of 256 out: the frame's summary.", 5.75);
  done(s, "attention");
}

// block 4: decision + where the numbers are ==========================================
{ const s = pres.addSlide();
  title(s, "Block 4: the decision", "THE PAPER  ·  DENSE");
  const probs = [["8PSK", 0.03], ["AM-DSB", 0.01], ["AM-SSB", 0.02], ["BPSK", 0.04], ["CPFSK", 0.01], ["GFSK", 0.01], ["PAM4", 0.02], ["QAM16", 0.08], ["QAM64", 0.03], ["QPSK", 0.72], ["WBFM", 0.03]];
  probs.forEach(([c, p], i) => {
    const y = 1.55 + i * 0.37;
    t(s, c, M, y, 1.3, 0.33, { fontSize: 15, valign: "middle", bold: c === "QPSK" });
    box(s, M + 1.35, y + 0.06, 4.0 * p + 0.04, 0.22, c === "QPSK" ? ORANGE : GREY);
  });
  t(s, "256 → 128 → 64 → 11 probabilities; the largest wins (illustration)", M, 5.65, 6.3, 0.4, { fontSize: 15, color: MUTED });
  const b = OF.blocks;
  s.addChart(pres.ChartType.doughnut, [{ name: "numbers", labels: ["BiLSTM", "Attention", "Dense", "CNN"], values: [b.BiLSTM, b.Attention, b.Dense, b.CNN] }],
    { x: 6.9, y: 1.4, w: 3.4, h: 3.6, holeSize: 55, chartColors: [ORANGE, NAVY, GREY, "C9D3E0"], showLegend: false, showPercent: false, showValue: false, dataLabelColor: WHITE });
  [["BiLSTM", b.BiLSTM, ORANGE], ["Attention", b.Attention, NAVY], ["Dense", b.Dense, GREY], ["CNN", b.CNN, "B7C3D0"]].forEach(([k, v, c], i) => {
    const y = 1.9 + i * 0.62;
    box(s, 10.45, y + 0.12, 0.3, 0.3, c);
    t(s, `${k}  ${pct(v, OF.params)}%`, 10.9, y, 2.0, 0.55, { fontSize: 19, bold: k === "BiLSTM", valign: "middle" });
  });
  t(s, `${num(OF.params)} numbers in total`, 6.9, 5.1, 6, 0.5, { fontSize: 20, bold: true });
  takeaway(s, "Most of the model is the LSTM.", 6.15);
  done(s, "decision");
}

// training =========================================================================
{ const s = pres.addSlide();
  title(s, "How it is trained", "THE PAPER  ·  TABLE 2 AND THEIR CODE");
  const steps = Math.ceil(OF.train_frames / OF.batch);
  const cols = [["DATA", ["70% learn: 154,000 frames", "15% watch: 33,000", "15% test: 33,000"]],
    ["LOOP", [`batches of ${OF.batch} frames`, `${num(steps)} steps per epoch`, "loss: cross-entropy", "Adam, learning rate 0.001"]]];
  cols.forEach(([h, lines], i) => {
    const x = M + i * 4.15;
    card(s, x, 1.6, 3.85, 3.9);
    t(s, h, x + 0.3, 1.75, 3.3, 0.45, { fontSize: 16, bold: true, color: ORANGE, charSpacing: 2 });
    t(s, lines.map((l, k) => ({ text: l, options: { breakLine: k < lines.length - 1 } })), x + 0.3, 2.35, 3.35, 3.0, { fontSize: 20, paraSpaceAfter: 10 });
  });
  const x = M + 2 * 4.15;
  card(s, x, 1.6, W - M - x, 3.9, NAVY);
  t(s, "STOPPING", x + 0.3, 1.75, 3, 0.45, { fontSize: 16, bold: true, color: ORANGE, charSpacing: 2 });
  t(s, [{ text: `loss flat ${OF.lr_patience} epochs: rate × 0.2`, options: { breakLine: true } },
    { text: `loss flat ${OF.es_patience} epochs: stop`, options: { breakLine: true } },
    { text: `at most ${OF.max_epochs} epochs` }], x + 0.3, 2.3, 3.6, 1.6, { fontSize: 18, color: WHITE, paraSpaceAfter: 6 });
  t(s, String(P.training.epochs), x + 0.3, 3.85, 1.6, 1.0, { fontSize: 54, bold: true, color: ORANGE, valign: "middle" });
  t(s, "where it stopped", x + 1.9, 3.85, 2.0, 1.0, { fontSize: 18, color: SOFT, valign: "middle" });
  takeaway(s, "Trained once: one split, one run.", 5.85);
  done(s, "training");
}

// main table ===========================================================================
{ const s = pres.addSlide();
  title(s, "Their main table: first by half a point", "THE PAPER  ·  TABLE 3");
  const rows = [...P.table3_10a].reverse();
  s.addChart(pres.ChartType.bar, [{ name: "accuracy %", labels: rows.map(r => r[0]), values: rows.map(r => r[1]) }],
    { x: M, y: 1.45, w: W - 2 * M, h: 4.75, barDir: "bar", barGapWidthPct: 35,
      chartColors: rows.map(r => (r[0] === "ICRNNA" ? ORANGE : (r[0] === "CNN-BiLSTM-DNN" ? GREY : NAVY))),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 15, dataLabelColor: INK, dataLabelFontFace: F, dataLabelFormatCode: "0.00",
      showLegend: false, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 70, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 15, catAxisLabelFontFace: F });
  takeaway(s, "63.24 vs 62.73: the runner-up is half a point behind.", 6.3);
  done(s, "table3");
}

// accuracy vs SNR ======================================================================
{ const s = pres.addSlide();
  title(s, "Accuracy against SNR", "THE PAPER  ·  FIG. 8");
  pic(s, "paper/fig8_accuracy_snr.jpg", M, 1.45, 6.9, 1561, 1023);
  credit(s, FROM, M, 6.0, 7);
  big(s, 8.1, 1.7, 4.6, `${P.acc_m20}%`, "at −20 dB: chance\n(1 in 11)", INK, 60);
  big(s, 8.1, 3.75, 4.6, "~93%", "from 0 dB up", ORANGE, 60);
  done(s, "snr");
}

// confusion at 0 dB ====================================================================
{ const s = pres.addSlide();
  const F11 = P.fig11;
  title(s, "Where it still fails, at 0 dB", "THE PAPER  ·  FIG. 11");
  pic(s, "paper/fig11_icrnna_0db.jpg", M, 1.45, 4.9, 525, 448);
  credit(s, FROM, M, 5.7, 7);
  big(s, 7.3, 1.6, 5.4, `${F11.wbfm_as_amdsb}%`, "of WBFM called AM-DSB", ORANGE, 60);
  big(s, 7.3, 3.6, 5.4, `~${Math.round(F11.q16_as_q64)}%`, "QAM16 ↔ QAM64, each way", INK, 60);
  takeaway(s, "The same two problems we see in our runs.", 6.2);
  done(s, "confusion");
}

// ablation =============================================================================
{ const s = pres.addSlide();
  title(s, "Which block matters?", "THE PAPER  ·  TABLE 6");
  const r = P.table6_10a;
  s.addChart(pres.ChartType.bar, [{ name: "accuracy %", labels: r.map(x => x[0]), values: r.map(x => +(100 * x[1]).toFixed(1)) }],
    { x: M, y: 1.45, w: 8.0, h: 4.6, barDir: "col", barGapWidthPct: 40, chartColors: [NAVY, ORANGE, NAVY, NAVY],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 20, dataLabelColor: INK, dataLabelFontFace: F, dataLabelFormatCode: "0.0",
      showLegend: false, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 72, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 18, catAxisLabelFontFace: F });
  big(s, 9.1, 1.7, 3.6, `+${(100 * (r[3][1] - r[2][1])).toFixed(1)}`, "points from attention", INK, 54);
  big(s, 9.1, 3.75, 3.6, `${(100 * (r[3][1] - r[1][1])).toFixed(1)}`, "points between BiLSTM alone and the full model", ORANGE, 54);
  takeaway(s, "The LSTM does almost all the work.", 6.25);
  done(s, "ablation");
}

// part 2: their code ==========================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "PART 2", M, 2.4, 6, 0.5, { fontSize: 20, bold: true, color: ORANGE, charSpacing: 4 });
  t(s, "Their code", M, 2.95, 12, 1.2, { fontSize: 60, bold: true, color: WHITE });
  t(s, "Published on GitHub on 8 August 2026, sixteen months after the paper.", M, 4.25, 12, 0.6, { fontSize: 24, color: SOFT });
  done(s, "code_divider", true);
}

{ const s = pres.addSlide();
  title(s, "What they published", "THEIR CODE");
  card(s, M, 1.55, W - 2 * M, 1.0);
  t(s, "github.com/noureddine395/mproved-automatic-modulation-recognition-using-deep-learning-with-additive-attention",
    M + 0.3, 1.55, W - 2 * M - 0.6, 1.0, { fontFace: "Courier New", fontSize: 14, valign: "middle" });
  [["IN IT", ["two scripts: 2016.10a and 2016.10b", "TensorFlow / Keras, MIT licence"]], ["NOT IN IT", ["trained weights", "more than one run"]]].forEach(([h, lines], i) => {
    const x = M + i * 6.15;
    card(s, x, 2.8, 5.9, 1.55, i ? NAVY : CARD);
    t(s, h, x + 0.3, 2.9, 5, 0.4, { fontSize: 15, bold: true, color: ORANGE, charSpacing: 2 });
    t(s, lines.join("\n"), x + 0.3, 3.3, 5.4, 1.0, { fontSize: 20, color: i ? WHITE : INK });
  });
  const flow = ["load the file", "normalize", "shuffle, split", "train", "test, plot"];
  flow.forEach((h, i) => {
    const x = M + i * 2.45;
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.65, w: 2.0, h: 0.8, fill: { color: i === 3 ? ORANGE : CARD }, line: { color: i === 3 ? ORANGE : CARD }, rectRadius: 0.1 });
    t(s, h, x, 4.65, 2.0, 0.8, { fontSize: 18, bold: true, align: "center", valign: "middle", color: i === 3 ? WHITE : INK });
    if (i < 4) arrow(s, x + 2.03, 4.85, 0.38);
  });
  takeaway(s, "One run of their script: these five steps, once.", 6.0);
  done(s, "code_intro");
}

{ const s = pres.addSlide();
  title(s, "The model, in their code", "THEIR CODE  ·  THE MODEL");
  codeRows(s, [
    ["x = Conv1D(64, 3, activation='relu', padding='same')(x)", "64 filters, 3 taps", "CNN"],
    ["x = Conv1D(128, 3, activation='relu', padding='same')(x)", "128 filters, 3 taps", ""],
    ["x = MaxPooling1D(pool_size=2)(x)", "128 steps → 64", ""],
    ["x = Bidirectional(LSTM(128, return_sequences=True))(x)", "both directions", "BiLSTM"],
    ["lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)", "kept for attention", ""],
    ["x = attention(x, lstm_out)", "weighted average", "Attention"],
    ["x = Dense(128, activation='relu')(x)", "", "Dense"],
    ["x = Dense(64, activation='relu')(x)", "", ""],
    ["outputs = Dense(num_classes, activation='softmax')(x)", "11 probabilities", ""],
  ], 1.5, 0.52, 8.55, 16);
  credit(s, "Their lines, with the BatchNormalization, Dropout and L2 arguments left out to fit.", M, 6.25, 10);
  takeaway(s, "Nine lines are the whole model.", 5.65);
  done(s, "code_model");
}

{ const s = pres.addSlide();
  title(s, "The attention, in their code", "THEIR CODE  ·  THE ATTENTION");
  codeRows(s, [
    ["score = self.V(tf.nn.tanh(self.W1(query) + self.W2(value)))", "a score per step"],
    ["score = score * self.scale", "learned scale, starts at 1"],
    ["attention_weights = tf.nn.softmax(score, axis=1)", "weights over 64 steps, sum to 1"],
    ["attention_weights = self.dropout(attention_weights)", "10% dropped while training"],
    ["context_vector = attention_weights * value", "weight each step"],
    ["context_vector = tf.reduce_sum(context_vector, axis=1)", "add them up: 256 numbers"],
    ["context_vector = self.layer_norm(context_vector)", "normalize the result"],
  ], 1.55, 0.6, 8.3, 16);
  credit(s, "Their lines from call(), slightly shortened (the if around the scale, the training flag).", M, 5.75, 11);
  card(s, M, 6.1, W - 2 * M, 0.75);
  t(s, "query = the LSTM output after normalization;  value = the same output before it. Each step is scored on its own.", M + 0.3, 6.1, W - 2 * M - 0.6, 0.75, { fontSize: 18, valign: "middle" });
  done(s, "code_attention");
}

{ const s = pres.addSlide();
  title(s, "The training, in their code", "THEIR CODE  ·  THE TRAINING");
  codeRows(s, [
    ["train_test_split(X, Y, snrs, test_size=0.3, random_state=2016)", "70% learn, 30% held out"],
    ["train_test_split(X_temp, ..., test_size=0.5, random_state=2016)", "15% watch, 15% test"],
    ["Adam(learning_rate=0.001, clipnorm=1.0)", "steps capped: not in the paper"],
    ["ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)", "loss flat 5 epochs: rate × 0.2"],
    ["EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)", "loss flat 6 epochs: stop"],
    ["model.fit(X_train, Y_train, epochs=100, batch_size=32, ...)", "at most 100 epochs"],
  ], 1.55, 0.66, 8.75);
  credit(s, "Their lines, slightly shortened (mode='min', min_lr=1e-6 and the checkpoint callback left out).", M, 5.6, 11);
  takeaway(s, "The paper's “58 epochs” is where this stopping rule fired.", 6.0);
  done(s, "code_training");
}

{ const s = pres.addSlide();
  title(s, "What the code tells us that the paper didn't", "THEIR CODE");
  const hc = (x) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, fontSize: 18 } });
  const rows = [[hc(""), hc("the paper"), hc("their code")],
    ["epochs", "58", "up to 100; stop when val loss is flat 6 epochs"],
    ["optimizer", "not named", "Adam, with gradient clipping"],
    ["input scaling", "“normalization”", "per time index, over the whole dataset"],
    ["pooling", "unclear", "one max-pool, after the 2nd conv"],
    ["attention scale", "“if enabled”", "learned, starts at 1"],
    ["runs", "not stated", "one run, one split"],
    ["weights", "not published", "still not published"]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 18, bold: j === 0, color: j === 1 ? MUTED : INK } })));
  s.addTable(rows, { x: M, y: 1.5, w: W - 2 * M, colW: [2.6, 3.0, W - 2 * M - 5.6], fontFace: F, color: INK,
    border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.56, autoPage: false, valign: "middle" });
  takeaway(s, "Most of what the paper left open, the code settles.", 6.25);
  done(s, "code_vs_paper");
}

{ const s = pres.addSlide();
  title(s, "Two problems in their code", "THEIR CODE");
  const items = [["The split changes every run",
    "The class list comes from a Python set, whose order changes between runs. The data are put together in that order, then shuffled with seed 2016: a different train/test split each time, despite the fixed seed."],
    ["The scaling sees the test set",
    "The normalization statistics are computed on all 220,000 frames before the split. Probably a small effect, but the test frames are not untouched."]];
  items.forEach(([h, b], i) => {
    const y = 1.6 + i * 2.05;
    card(s, M, y, W - 2 * M, 1.8);
    badge(s, M + 0.3, y + 0.3, i + 1, ORANGE);
    t(s, h, M + 1.15, y + 0.15, W - 2 * M - 1.5, 0.55, { fontSize: 23, bold: true });
    t(s, b, M + 1.15, y + 0.7, W - 2 * M - 1.5, 1.05, { fontSize: 18, color: MUTED });
  });
  takeaway(s, "Neither changes the model; both make one run harder to repeat.", 5.85);
  done(s, "code_issues");
}

// part 3: what we checked ======================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "PART 3", M, 2.4, 6, 0.5, { fontSize: 20, bold: true, color: ORANGE, charSpacing: 4 });
  t(s, "What we checked", M, 2.95, 12, 1.2, { fontSize: 60, bold: true, color: WHITE });
  t(s, "Our rebuild from the paper, their own code, and their own tables.", M, 4.25, 12, 0.6, { fontSize: 24, color: SOFT });
  done(s, "divider", true);
}

// rebuild + their code ======================================================================
{ const s = pres.addSlide();
  title(s, "We rebuilt it; their code agrees", "OUR CHECK");
  [["Apr 2025", "paper, no code"], ["Aug 2026", "their code on GitHub, no weights"], ["Sep 2026", "our rebuild from the paper alone"]].forEach(([d, l], i) => {
    const x = M + i * 4.15;
    card(s, x, 1.55, 3.85, 1.05, i === 2 ? NAVY : CARD);
    t(s, d, x + 0.25, 1.6, 3.4, 0.5, { fontSize: 20, bold: true, color: ORANGE });
    t(s, l, x + 0.25, 2.05, 3.5, 0.5, { fontSize: 17, color: i === 2 ? WHITE : MUTED });
  });
  card(s, M, 2.9, 3.85, 2.75);
  big(s, M + 0.3, 3.0, 3.4, `${O.e150_acc}%`, `our rebuild, trained to the end (paper: ${P.headline.acc_10a}%)`, ORANGE, 48);
  card(s, M + 4.15, 2.9, 3.85, 2.75);
  t(s, "MATCHED THEIR CODE", M + 4.45, 3.05, 3.4, 0.4, { fontSize: 15, bold: true, color: ORANGE, charSpacing: 1 });
  t(s, "one pooling step\nnormalization after each LSTM\ndropout on attention weights\nsize: " + num(O.params) + " vs " + num(OF.params), M + 4.45, 3.5, 3.4, 2.0, { fontSize: 17 });
  card(s, M + 8.3, 2.9, W - 2 * M - 8.3, 2.75);
  t(s, "MISSED", M + 8.6, 3.05, 3.4, 0.4, { fontSize: 15, bold: true, color: ORANGE, charSpacing: 1 });
  t(s, "exact attention form and its learned scale\ninput normalization\ngradient clipping", M + 8.6, 3.5, 3.4, 2.0, { fontSize: 17 });
  takeaway(s, "Their 63.24% is reproducible from the paper alone.", 5.95);
  done(s, "rebuild");
}

// what 58 epochs means =======================================================================
{ const s = pres.addSlide();
  title(s, "What “58 epochs” means", "OUR CHECK");
  big(s, M, 1.5, 5.6, `${O.e58_mean}%`, `our rebuild at a fixed ${P.training.epochs} epochs (${O.e58_seeds} runs)`, INK, 48);
  big(s, M, 3.4, 5.6, `${O.e150_acc}%`, `the same, left to run: best at epoch ${O.e150_best_epoch}`, ORANGE, 48);
  pic(s, "paper/fig7_accuracy_epochs.jpg", 6.55, 1.5, 6.15, 1041, 659);
  t(s, "Their Fig. 7: accuracy still rising at 58", 6.55, 5.45, 6.15, 0.4, { fontSize: 18, bold: true });
  credit(s, FROM, 6.55, 5.85, 6.15);
  takeaway(s, "Their rule stopped on the loss; the accuracy was still climbing.", 6.3);
  done(s, "epochs");
}

// their code, our runs (only when the results are in) ===========================================
if (OR) {
  const s = pres.addSlide();
  title(s, "Their own code, run by us", "OUR CHECK");
  big(s, M, 1.5, 5.2, `${OR.mean.toFixed(2)}%`, `mean of ${OR.n} runs, ± ${OR.std.toFixed(2)} (paper: ${P.headline.acc_10a}%)`, ORANGE, 56);
  t(s, `stopped after ${OR.stopped.join(", ")} epochs`, M, 3.55, 5.4, 0.5, { fontSize: 20 });
  t(s, `runs: ${OR.accs.map(a => a.toFixed(2) + "%").join(", ")}`, M, 4.05, 5.4, 0.5, { fontSize: 20, color: MUTED });
  const img = "paper/ours_official_curve.png";
  if (fs.existsSync(img)) pic(s, img, 6.4, 1.5, 6.3, 1275, 782);
  takeaway(s, "Their code, their split, three runs.", 6.2);
  done(s, "official");
}

// claims vs their tables ======================================================================
{ const s = pres.addSlide();
  title(s, "Three claims their own tables don't support", "OUR CHECK");
  const t6 = P.table6_10a, mc = D.verified_params.MCLDNN;
  const rows = [["“Best at low SNR”", "Table 5: beaten at −10, −8 and −6 dB"],
    ["“Full model has the highest precision”", `Table 6: LSTM alone is higher, ${t6[1][3].toFixed(3)} vs ${t6[3][3].toFixed(3)}`],
    [`“MCLDNN: ${P.table7_params_m.MCLDNN} million parameters”`, `It has ${num(mc)}: 0.41 million`]];
  rows.forEach(([a, b], i) => {
    const y = 1.6 + i * 1.4;
    card(s, M, y, W - 2 * M, 1.2);
    badge(s, M + 0.25, y + 0.3, i + 1, ORANGE);
    t(s, a, M + 1.1, y, 5.0, 1.2, { fontSize: 21, color: MUTED, italic: true, valign: "middle" });
    t(s, b, M + 6.2, y, W - 2 * M - 6.4, 1.2, { fontSize: 22, bold: true, valign: "middle" });
  });
  takeaway(s, "ICRNNA is about twice MCLDNN's size, not smaller.", 5.95);
  done(s, "tables");
}

// what the paper doesn't tell us ===============================================================
{ const s = pres.addSlide();
  title(s, "What the paper doesn't tell us", "OUR CHECK");
  const q0 = O.snr_at_label0;
  const items = [["One run", "no spread; our seeds vary by up to a point"],
    ["Baselines", "retrained on the same split, or copied from their papers?"],
    ["SNR label", "taken as the true SNR"]];
  items.forEach(([h, b], i) => {
    const y = 1.65 + i * 1.3;
    badge(s, M, y + 0.05, i + 1, NAVY);
    t(s, h, M + 0.85, y, 2.4, 0.7, { fontSize: 24, bold: true, valign: "middle" });
    t(s, b, M + 3.25, y, 4.6, 0.75, { fontSize: 20, color: MUTED, valign: "middle" });
  });
  card(s, 8.7, 1.6, W - M - 8.7, 3.85);
  t(s, `+${(q0.QAM16 - q0.QPSK).toFixed(0)} dB`, 8.95, 1.8, 3.6, 1.1, { fontSize: 60, bold: true, color: ORANGE, valign: "bottom" });
  t(s, "more signal in QAM16 than in QPSK, both labelled 0 dB (measured in the file)", 8.95, 2.95, 3.5, 1.6, { fontSize: 19, color: MUTED });
  takeaway(s, "A half-point lead from one run may not be real.", 5.95);
  done(s, "unknown");
}

// one protocol for every model ===================================================================
{ const s = pres.addSlide();
  title(s, "One protocol for every model", "OUR CHECK");
  t(s, "All 11 types. Same data, split and 3 seeds for every model.", M, 1.35, W - 2 * M, 0.5, { fontSize: 20, color: MUTED });
  const theirs = Object.fromEntries(P.table3_10a);
  const hc = (x, o = {}) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, fontSize: 18, align: "center", ...o } });
  const rows = [[hc("model", { align: "left" }), hc("their Table 3"), hc("ours"), hc("ours, ≥ 10 dB")]];
  O.lit.forEach(r => {
    const me = r.name === "ICRNNA", o = { fontSize: 22, align: "center", bold: me, color: me ? ORANGE : INK };
    rows.push([{ text: me ? "ICRNNA (our version)" : r.name, options: { ...o, align: "left" } },
      { text: theirs[r.name] ? theirs[r.name].toFixed(2) + "%" : "—", options: { ...o, color: MUTED, bold: false } },
      { text: (100 * r.overall).toFixed(1) + "%", options: o }, { text: (100 * r.high).toFixed(1) + "%", options: o }]);
  });
  s.addTable(rows, { x: M, y: 2.0, w: W - 2 * M, colW: [4.3, 2.6, 2.3, 2.93], fontFace: F, border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.62, autoPage: false, valign: "middle" });
  takeaway(s, "First, but the top four are within about one point.", 6.0);
  done(s, "protocol");
}

// summary =========================================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "In short", M, 0.55, 8, 0.9, { fontSize: 40, bold: true, color: WHITE });
  const pts = ["CNN → BiLSTM → attention → classifier; the LSTM does most of the work.",
    "63.24% is reproducible: 63.21% from the paper alone.",
    "Its lead over other models is about one point."];
  pts.forEach((p, i) => {
    badge(s, M, 1.85 + i * 1.25, i + 1, ORANGE, 0.7);
    t(s, p, M + 1.0, 1.75 + i * 1.25, W - 2 * M - 1.0, 0.9, { fontSize: 27, color: WHITE, valign: "middle" });
  });
  t(s, [{ text: "For us: ", options: { bold: true, color: ORANGE } }, { text: "a solid backbone, not a clearly better one.", options: { color: SOFT } }],
    M, 5.75, W - 2 * M, 0.7, { fontSize: 26 });
  done(s, "summary", true);
}

pres.writeFile({ fileName: "moshe_paper_icrnna.pptx" }).then(f => console.log("wrote", f, "slides:", n, OR ? "(with official runs)" : "(official runs not in yet)"));

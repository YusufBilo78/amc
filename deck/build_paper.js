// The ICRNNA paper (El-Haryqy et al., Results in Engineering 26 (2025) 104783),
// presented for Moshe's group: what the paper says, then what we found when we
// rebuilt it. Same style as the simple deck: large type, one message per slide.
// Numbers come from paper_data.json (deck/extract_paper.py): paper values are
// transcribed with their table numbers, ours are read from the result files.
// Figures in deck/paper/ are cropped from the paper (CC BY-NC 4.0) and say so.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("paper_data.json", "utf8"));
const TALK = require("./talk_paper.js");
const P = D.paper, O = D.ours;

const NAVY = "0F1B2D", ORANGE = "F08A24", CARD = "EEF2F6", INK = "1E293B", MUTED = "5B6B7F", WHITE = "FFFFFF", LINE = "D5DDE5", SOFT = "C9D3E0";
const F = "Calibri";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Yusuf Bilal Cetinkaya";
pres.title = "ICRNNA: the paper, and what we found";
const W = 13.33, M = 0.6;
const num = v => v.toLocaleString("en-US");
const FROM = "Figure from El-Haryqy et al. 2025 (CC BY-NC 4.0)";
let n = 0;

function t(s, text, x, y, w, h, o = {}) {
  s.addText(text, { x, y, w, h, fontFace: F, fontSize: 20, color: INK, isTextBox: true, margin: 0, valign: "top", ...o });
}
function title(s, text, kicker) {
  if (kicker) t(s, kicker, M, 0.3, 8, 0.35, { fontSize: 14, bold: true, color: ORANGE, charSpacing: 2 });
  t(s, text, M, 0.6, W - 2 * M, 0.8, { fontSize: 34, bold: true, valign: "middle" });
}
function takeaway(s, text, y = 6.3, dark = false) {
  t(s, text, M, y, W - 2 * M, 0.7, { fontSize: 24, bold: true, color: dark ? WHITE : NAVY, valign: "middle" });
}
function card(s, x, y, w, h, fill = CARD) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
}
function badge(s, x, y, k, fill = NAVY, d = 0.6) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  t(s, String(k), x, y, d, d, { fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
}
function arrow(s, x, y, w = 0.5) {
  s.addShape(pres.ShapeType.rightArrow, { x, y, w, h: 0.45, fill: { color: ORANGE }, line: { color: ORANGE } });
}
function big(s, x, y, w, value, label, color = INK, size = 60, align = "left") {
  t(s, value, x, y, w, 1.05, { fontSize: size, bold: true, color, align, valign: "bottom" });
  t(s, label, x, y + 1.1, w, 0.8, { fontSize: 20, color: MUTED, align });
}
function pic(s, path, x, y, w, pxW, pxH) {
  const h = w * pxH / pxW;
  s.addImage({ path, x, y, w, h });
  return h;
}
function credit(s, text, x, y, w) {
  t(s, text, x, y, w, 0.3, { fontSize: 12, color: MUTED, italic: true });
}
function done(s, dark = false) {
  n++;
  if (TALK[n]) s.addNotes(TALK[n]);
  t(s, String(n), W - M - 0.5, 7.0, 0.5, 0.3, { fontSize: 12, color: dark ? "8FA0B5" : MUTED, align: "right" });
}
const PARTS = [["CNN", "2 conv layers"], ["BiLSTM", "2 layers"], ["Attention", "weights"], ["Dense", "→ 11 classes"]];

// 1. title ======================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "PAPER REVIEW", M, 1.0, 6, 0.4, { fontSize: 16, bold: true, color: ORANGE, charSpacing: 4 });
  t(s, "ICRNNA", M, 1.45, 12, 1.2, { fontSize: 66, bold: true, color: WHITE });
  t(s, "Improved automatic modulation recognition using deep learning with additive attention", M, 2.7, 11.5, 1.0, { fontSize: 26, color: SOFT });
  t(s, "El-Haryqy, Kharbouche, Ouamna, Madini, Zouine  ·  Results in Engineering 26 (2025) 104783", M, 3.85, 12, 0.5, { fontSize: 18, color: "9AA8BA" });
  PARTS.forEach(([h], i) => {
    const x = M + i * 2.6;
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.85, w: 2.1, h: 0.65, fill: { color: ORANGE }, line: { color: ORANGE }, rectRadius: 0.12 });
    t(s, h, x, 4.85, 2.1, 0.65, { fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle" });
    if (i < 3) arrow(s, x + 2.15, 4.95, 0.4);
  });
  t(s, "Yusuf Bilal Cetinkaya  ·  Moshe Kam's research group  ·  NJIT  ·  October 2026", M, 6.5, 10, 0.4, { fontSize: 16, color: "9AA8BA" });
  done(s, true);
}

// 2. the claims ===================================================================
{ const s = pres.addSlide();
  title(s, "What the paper claims", "THE PAPER");
  const H = P.headline;
  [[`${H.acc_10a}%`, "highest accuracy on RML2016.10a", ORANGE], [`${H.params_m} M`, "parameters", INK], [`${H.mflops}`, "MFLOPs", INK]].forEach(([v, l, c], i) => {
    const x = M + i * 4.15;
    card(s, x, 1.75, 3.75, 2.75);
    big(s, x + 0.35, 1.95, 3.1, v, l, c, 60);
  });
  takeaway(s, "And: especially good at low SNR.", 5.0);
  done(s);
}

// 3. the model ====================================================================
{ const s = pres.addSlide();
  title(s, "The model: four parts", "THE PAPER");
  const boxes = [["Input", "I and Q\n2 × 128"], ["CNN", "2 conv layers\n64, 128 filters"], ["BiLSTM", "2 layers\n128 each way"], ["Attention", "weighs the\n time steps"], ["Dense", "128, 64\n→ 11 classes"]];
  boxes.forEach(([h, b], i) => {
    const x = M + i * 2.48, w = 2.05, fill = i === 0 ? CARD : (i === 2 ? ORANGE : NAVY), ink = i === 0 ? INK : WHITE;
    s.addShape(pres.ShapeType.roundRect, { x, y: 2.0, w, h: 2.6, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
    t(s, h, x, 2.2, w, 0.6, { fontSize: 26, bold: true, color: ink, align: "center" });
    t(s, b, x + 0.1, 3.0, w - 0.2, 1.4, { fontSize: 18, color: ink, align: "center" });
    if (i < 4) arrow(s, x + w + 0.0, 3.05, 0.43);
  });
  credit(s, "Redrawn from their Fig. 2. Each block also has batch normalization and dropout.", M, 4.85, 11);
  takeaway(s, "CNN for short pieces, BiLSTM for order, attention to focus.", 5.6);
  done(s);
}

// 4. attention =====================================================================
{ const s = pres.addSlide();
  title(s, "Attention in one formula", "THE PAPER");
  card(s, M, 1.7, 7.4, 3.6);
  const eq = [["score", " = v · tanh(W₁q + W₂kₜ)"], ["α", " = softmax(score)"], ["c", " = Σ αₜ vₜ"]];
  eq.forEach(([a, b], i) => t(s, [{ text: a, options: { bold: true, color: ORANGE } }, { text: b }], M + 0.4, 1.95 + i * 1.05, 6.8, 0.9,
    { fontFace: "Cambria", fontSize: 30, valign: "middle" }));
  t(s, "Their “improved” version adds:", 8.4, 1.75, 4.4, 0.5, { fontSize: 20, bold: true });
  ["scaling of the scores", "dropout 0.1", "layer normalization"].forEach((x, i) => {
    badge(s, 8.4, 2.45 + i * 0.95, "+", ORANGE, 0.55);
    t(s, x, 9.15, 2.45 + i * 0.95, 3.6, 0.55, { fontSize: 22, valign: "middle" });
  });
  takeaway(s, "A learned weighted average over the time steps.", 5.75);
  done(s);
}

// 5. training =======================================================================
{ const s = pres.addSlide();
  const T = P.training;
  title(s, "How they trained it", "THE PAPER");
  const cols = [["DATA", ["RML2016.10a", "11 types, 220,000 frames", "128 samples per frame", `split ${T.split.join(" / ")}`]],
    ["RECIPE", [`batch ${T.batch}`, `learning rate ${T.lr}`, `× ${T.lr_drop} after ${T.lr_patience} flat epochs`, `dropout ${T.dropout}, L2 ${T.l2}`]]];
  cols.forEach(([h, lines], i) => {
    const x = M + i * 4.15;
    card(s, x, 1.7, 3.85, 3.9);
    t(s, h, x + 0.3, 1.85, 3.3, 0.45, { fontSize: 16, bold: true, color: ORANGE, charSpacing: 2 });
    t(s, lines.map((l, k) => ({ text: l, options: { breakLine: k < lines.length - 1 } })), x + 0.3, 2.45, 3.3, 3.0, { fontSize: 21, paraSpaceAfter: 10 });
  });
  const x = M + 2 * 4.15;
  card(s, x, 1.7, W - M - x, 3.9, NAVY);
  t(s, "EPOCHS", x + 0.3, 1.85, 3, 0.45, { fontSize: 16, bold: true, color: ORANGE, charSpacing: 2 });
  t(s, String(T.epochs), x + 0.3, 2.3, 3.5, 1.9, { fontSize: 110, bold: true, color: WHITE, valign: "middle" });
  t(s, "remember this number", x + 0.3, 4.4, 3.5, 0.6, { fontSize: 20, color: SOFT });
  takeaway(s, "They also test RML2016.10b; we stay with 10a.", 6.0);
  done(s);
}

// 6. headline table ==================================================================
{ const s = pres.addSlide();
  title(s, "Their main table: first by half a point", "THE PAPER  ·  TABLE 3");
  const rows = [...P.table3_10a].reverse();
  s.addChart(pres.ChartType.bar, [{ name: "accuracy %", labels: rows.map(r => r[0]), values: rows.map(r => r[1]) }],
    { x: M, y: 1.45, w: W - 2 * M, h: 4.75, barDir: "bar", barGapWidthPct: 35,
      chartColors: rows.map(r => (r[0] === "ICRNNA" ? ORANGE : (r[0] === "CNN-BiLSTM-DNN" ? "8FA0B5" : NAVY))),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 15, dataLabelColor: INK, dataLabelFontFace: F, dataLabelFormatCode: "0.00",
      showLegend: false, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 70, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 15, catAxisLabelFontFace: F });
  takeaway(s, "63.24 vs 62.73: the runner-up is half a point behind.", 6.3);
  done(s);
}

// 7. accuracy vs SNR =================================================================
{ const s = pres.addSlide();
  title(s, "Accuracy against SNR", "THE PAPER  ·  FIG. 8");
  pic(s, "paper/fig8_accuracy_snr.jpg", M, 1.45, 6.9, 1561, 1023);
  credit(s, FROM, M, 6.0, 7);
  big(s, 8.1, 1.7, 4.6, `${P.acc_m20}%`, "at −20 dB: chance\n(1 in 11)", INK, 60);
  big(s, 8.1, 3.75, 4.6, "~93%", "from 0 dB up", ORANGE, 60);
  done(s);
}

// 8. confusion at 0 dB ===============================================================
{ const s = pres.addSlide();
  const F11 = P.fig11;
  title(s, "Where it still fails, at 0 dB", "THE PAPER  ·  FIG. 11");
  pic(s, "paper/fig11_icrnna_0db.jpg", M, 1.45, 4.9, 525, 448);
  credit(s, FROM, M, 5.7, 7);
  big(s, 7.3, 1.6, 5.4, `${F11.wbfm_as_amdsb}%`, "of WBFM called AM-DSB", ORANGE, 60);
  big(s, 7.3, 3.6, 5.4, `~${Math.round(F11.q16_as_q64)}%`, "QAM16 ↔ QAM64, each way", INK, 60);
  takeaway(s, "The same two problems we see in our runs.", 6.2);
  done(s);
}

// 9. ablation ========================================================================
{ const s = pres.addSlide();
  title(s, "Which part matters?", "THE PAPER  ·  TABLE 6");
  const r = P.table6_10a;
  s.addChart(pres.ChartType.bar, [{ name: "accuracy %", labels: r.map(x => x[0]), values: r.map(x => +(100 * x[1]).toFixed(1)) }],
    { x: M, y: 1.45, w: 8.0, h: 4.6, barDir: "col", barGapWidthPct: 40, chartColors: [NAVY, ORANGE, NAVY, NAVY],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 20, dataLabelColor: INK, dataLabelFontFace: F, dataLabelFormatCode: "0.0",
      showLegend: false, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 72, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 18, catAxisLabelFontFace: F });
  big(s, 9.1, 1.7, 3.6, `+${(100 * (r[3][1] - r[2][1])).toFixed(1)}`, "points from attention", INK, 54);
  big(s, 9.1, 3.75, 3.6, `${(100 * (r[3][1] - r[1][1])).toFixed(1)}`, "points between BiLSTM alone and the full model", ORANGE, 54);
  takeaway(s, "The BiLSTM does almost all the work.", 6.25);
  done(s);
}

// 10. divider =========================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "PART 2", M, 2.4, 6, 0.5, { fontSize: 20, bold: true, color: ORANGE, charSpacing: 4 });
  t(s, "Does it hold up?", M, 2.95, 12, 1.2, { fontSize: 60, bold: true, color: WHITE });
  t(s, "We rebuilt the model from the paper and checked it against their own tables.", M, 4.25, 12, 0.6, { fontSize: 24, color: SOFT });
  done(s, true);
}

// 11. rebuild ==========================================================================
{ const s = pres.addSlide();
  title(s, "We rebuilt it from the paper", "OUR CHECK");
  const items = [[num(O.params), `parameters (paper: ${P.headline.params_m} M)`, INK],
    [`${O.e58_mean}%`, `at their ${P.training.epochs} epochs, ${O.e58_seeds} runs`, INK],
    [`${O.e150_acc}%`, `trained to epoch ${O.e150_best_epoch} (paper: ${P.headline.acc_10a}%)`, ORANGE]];
  items.forEach(([v, l, c], i) => {
    t(s, v, M, 1.5 + i * 1.5, 4.3, 0.9, { fontSize: 44, bold: true, color: c, valign: "bottom" });
    t(s, l, M, 2.4 + i * 1.5, 5.6, 0.5, { fontSize: 19, color: MUTED });
  });
  pic(s, "paper/fig7_accuracy_epochs.jpg", 6.55, 1.5, 6.15, 1041, 659);
  t(s, "Their Fig. 7: still rising at epoch 58", 6.55, 5.45, 6.15, 0.4, { fontSize: 18, bold: true, color: INK });
  credit(s, FROM, 6.55, 5.85, 6.15);
  takeaway(s, "Their 63.24% is real. It needs more than 58 epochs.", 6.3);
  done(s);
}

// 12. per SNR ===========================================================================
{ const s = pres.addSlide();
  title(s, "Per SNR: our rebuild against their table", "OUR CHECK");
  const snr = P.table5_snr;
  s.addChart(pres.ChartType.line, [
    { name: "their Table 5", labels: snr.map(v => String(v).replace("-", "−") + " dB"), values: P.table5.ICRNNA },
    { name: "our rebuild", labels: snr.map(v => String(v).replace("-", "−") + " dB"), values: snr.map(v => O.per_snr[v]) }],
    { x: M, y: 1.45, w: W - 2 * M, h: 4.6, chartColors: [NAVY, ORANGE], lineSize: 4, lineDataSymbol: "circle", lineDataSymbolSize: 9,
      showLegend: true, legendPos: "t", legendFontSize: 18, legendFontFace: F, legendColor: INK,
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, showValAxisTitle: true, valAxisTitle: "accuracy %", valAxisTitleFontSize: 16, valAxisTitleColor: MUTED,
      catAxisLabelColor: INK, valAxisLabelColor: MUTED, catAxisLabelFontSize: 16, valAxisLabelFontSize: 15,
      valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" } });
  takeaway(s, "Same model, same curve: within five points at every SNR.", 6.25);
  done(s);
}

// 13. claims vs tables ==================================================================
{ const s = pres.addSlide();
  title(s, "Three claims their own tables don't support", "OUR CHECK");
  const t5 = P.table5, t6 = P.table6_10a, mc = D.verified_params.MCLDNN;
  const rows = [["“Best at low SNR”", "Table 5: beaten at −10, −8 and −6 dB"],
    ["“Full model has the highest precision”", `Table 6: BiLSTM alone is higher, ${t6[1][3].toFixed(3)} vs ${t6[3][3].toFixed(3)}`],
    [`“MCLDNN: ${P.table7_params_m.MCLDNN} million parameters”`, `It has ${num(mc)}: 0.41 million`]];
  rows.forEach(([a, b], i) => {
    const y = 1.6 + i * 1.4;
    card(s, M, y, W - 2 * M, 1.2);
    badge(s, M + 0.25, y + 0.3, i + 1, ORANGE);
    t(s, a, M + 1.1, y, 5.0, 1.2, { fontSize: 21, color: MUTED, italic: true, valign: "middle" });
    t(s, b, M + 6.2, y, W - 2 * M - 6.4, 1.2, { fontSize: 22, bold: true, valign: "middle" });
  });
  takeaway(s, "ICRNNA is about twice MCLDNN's size, not smaller.", 5.95);
  done(s);
}

// 14. not said ==========================================================================
{ const s = pres.addSlide();
  title(s, "What the paper doesn't tell us", "OUR CHECK");
  const q0 = O.snr_at_label0;
  const items = [["One run", "no spread reported; our seeds vary by up to a point"],
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
  done(s);
}

// 15. same protocol ======================================================================
{ const s = pres.addSlide();
  title(s, "One protocol for every model", "OUR CHECK");
  t(s, "All 11 types. Same data, split and 3 seeds for every model.", M, 1.35, W - 2 * M, 0.5, { fontSize: 20, color: MUTED });
  const theirs = Object.fromEntries(P.table3_10a);
  const hc = (x, o = {}) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, fontSize: 18, align: "center", ...o } });
  const rows = [[hc("model", { align: "left" }), hc("their Table 3"), hc("ours"), hc("ours, ≥ 10 dB")]];
  O.lit.forEach(r => {
    const me = r.name === "ICRNNA", o = { fontSize: 22, align: "center", bold: me, color: me ? ORANGE : INK };
    rows.push([{ text: me ? "ICRNNA (Esmer's version)" : r.name, options: { ...o, align: "left" } },
      { text: theirs[r.name] ? theirs[r.name].toFixed(2) + "%" : "—", options: { ...o, color: MUTED, bold: false } },
      { text: (100 * r.overall).toFixed(1) + "%", options: o }, { text: (100 * r.high).toFixed(1) + "%", options: o }]);
  });
  s.addTable(rows, { x: M, y: 2.0, w: W - 2 * M, colW: [4.3, 2.6, 2.3, 2.93], fontFace: F, border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.62, autoPage: false, valign: "middle" });
  takeaway(s, "First, but the top four are within about one point.", 6.0);
  done(s);
}

// 16. summary =============================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "In short", M, 0.55, 8, 0.9, { fontSize: 40, bold: true, color: WHITE });
  const pts = ["The model and its 63.24% are reproducible, if trained long enough.",
    "Its lead is about one point; the BiLSTM does most of the work.",
    "Some claims don't match the paper's own tables."];
  pts.forEach((p, i) => {
    badge(s, M, 1.85 + i * 1.25, i + 1, ORANGE, 0.7);
    t(s, p, M + 1.0, 1.75 + i * 1.25, W - 2 * M - 1.0, 0.9, { fontSize: 28, color: WHITE, valign: "middle" });
  });
  t(s, [{ text: "For us: ", options: { bold: true, color: ORANGE } }, { text: "a solid backbone, not a clearly better one.", options: { color: SOFT } }],
    M, 5.75, W - 2 * M, 0.7, { fontSize: 26 });
  done(s, true);
}

pres.writeFile({ fileName: "moshe_paper_icrnna.pptx" }).then(f => console.log("wrote", f, "slides:", n));

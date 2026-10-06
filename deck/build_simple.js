// The four-class box, for Moshe: one message per slide, large type, little
// text. What is said aloud is in the notes (talk_simple.js); the details behind
// every slide are in STUDY_simple.md. Pictures only where they carry the point.
// Numbers come from deck2016_data.json (deck/extract_2016.py); pictures in
// deck/simple/ are illustrations (deck/simple_figs.py), not results.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("deck2016_data.json", "utf8"));
const TALK = require("./talk_simple.js");

const NAVY = "0F1B2D", ORANGE = "F08A24", CARD = "EEF2F6", INK = "1E293B", MUTED = "5B6B7F", WHITE = "FFFFFF", LINE = "D5DDE5", SOFT = "C9D3E0";
const F = "Calibri";
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
  s.addText(text, { x, y, w, h, fontFace: F, fontSize: 20, color: INK, isTextBox: true, margin: 0, valign: "top", ...o });
}
function title(s, text) {
  t(s, text, M, 0.45, W - 2 * M, 0.9, { fontSize: 36, bold: true, valign: "middle" });
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
function arrow(s, x, y) {
  s.addShape(pres.ShapeType.rightArrow, { x, y, w: 0.6, h: 0.5, fill: { color: ORANGE }, line: { color: ORANGE } });
}
function big(s, x, y, w, value, label, color = INK, size = 66, align = "center") {
  t(s, value, x, y, w, 1.15, { fontSize: size, bold: true, color, align, valign: "bottom" });
  t(s, label, x, y + 1.2, w, 0.9, { fontSize: 20, color: MUTED, align });
}
function pic(s, path, x, y, w, pxW, pxH) {
  const h = w * pxH / pxW;
  s.addImage({ path, x, y, w, h });
  return h;
}
function done(s, dark = false) {
  n++;
  if (TALK[n]) s.addNotes(TALK[n]);
  t(s, String(n), W - M - 0.5, 7.0, 0.5, 0.3, { fontSize: 12, color: dark ? "8FA0B5" : MUTED, align: "right" });
}

// 1. title =====================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "Telling four radio signals apart", M, 1.5, 7.4, 2.0, { fontSize: 50, bold: true, color: WHITE });
  t(s, "BPSK  ·  QPSK  ·  QAM16  ·  QAM64", M, 3.7, 7.4, 0.6, { fontSize: 26, bold: true, color: ORANGE });
  t(s, "What our detector does, step by step", M, 4.4, 7.4, 0.6, { fontSize: 22, color: SOFT });
  s.addImage({ path: "simple/noise.gif", x: 8.3, y: 1.1, w: 4.3, h: 4.62 });
  t(s, "Moshe Kam's research group  ·  NJIT  ·  October 2026", M, 6.5, 8, 0.4, { fontSize: 16, color: "9AA8BA" });
  done(s, true);
}

// 2. what AMC is ==================================================================
{ const s = pres.addSlide();
  title(s, "What is AMC?");
  t(s, "Automatic Modulation Classification: a receiver gets a signal and says how it was modulated.", M, 1.4, W - 2 * M, 0.9, { fontSize: 22, color: MUTED });
  // the idea, as one picture
  const dy = 0.35;
  pic(s, "simple/frame.png", M, 2.65 + dy, 3.4, 1530, 544);
  arrow(s, M + 3.6, 2.98 + dy);
  s.addShape(pres.ShapeType.roundRect, { x: M + 4.4, y: 2.73 + dy, w: 1.9, h: 1.05, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.1 });
  t(s, "detector", M + 4.4, 2.73 + dy, 1.9, 1.05, { fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
  arrow(s, M + 6.5, 2.98 + dy);
  t(s, "QPSK", M + 7.3, 2.73 + dy, 1.8, 1.05, { fontSize: 36, bold: true, color: ORANGE, valign: "middle" });
  // my work, in three steps
  const steps = [["Build a detector", NAVY], ["Open it up: today", ORANGE], ["Why it fails on a new transmitter", NAVY]];
  steps.forEach(([h, c], i) => {
    const x = M + i * 4.1;
    badge(s, x, 5.4, i + 1, c);
    t(s, h, x + 0.8, 5.3, 3.2, 0.85, { fontSize: 22, bold: i === 1, color: i === 1 ? ORANGE : INK, valign: "middle" });
  });
  done(s);
}

// 3. the data =====================================================================
{ const s = pres.addSlide();
  title(s, "The data: RML2016.10a");
  t(s, "Made by DeepSig in a simulator. Most AMC papers use it.", M, 1.4, W - 2 * M, 0.6, { fontSize: 22, color: MUTED });
  const xs = [0, 2.5, 5.0, 8.6];
  [["11", "signal types"], ["20", "noise levels"], ["1,000", "frames each"], ["16", "symbols per frame"]].forEach(([v, l], i) => {
    big(s, M + xs[i], 2.0, i === 2 ? 3.4 : 2.9, v, l, i === 3 ? ORANGE : INK, 66, "left");
  });
  const all = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "PAM4", "GFSK", "CPFSK", "WBFM", "AM-DSB", "AM-SSB"];
  const used = new Set(["BPSK", "QPSK", "QAM16", "QAM64"]);
  all.forEach((name, k) => {
    const x = M + k * 1.1, on = used.has(name);
    s.addShape(pres.ShapeType.roundRect, { x, y: 4.6, w: 1.02, h: 0.55, fill: { color: on ? ORANGE : CARD }, line: { color: on ? ORANGE : CARD }, rectRadius: 0.12 });
    t(s, name, x, 4.6, 1.02, 0.55, { fontSize: 14, bold: on, color: on ? WHITE : INK, align: "center", valign: "middle" });
  });
  takeaway(s, "Today: the four in orange.", 5.75);
  done(s);
}

// 4. what it does =================================================================
{ const s = pres.addSlide();
  title(s, "What does the detector do?");
  const steps = [["Look at short pieces", "2 convolutions"], ["Read them in order", "BiLSTM: 84% of the model"], ["Decide", "4 scores, the largest wins"]];
  steps.forEach(([h, sub], i) => {
    const x = M + i * 4.15, y = 1.8, w = 3.55;
    card(s, x, y, w, 3.2);
    badge(s, x + 0.3, y + 0.3, i + 1, i === 1 ? ORANGE : NAVY);
    t(s, h, x + 0.3, y + 1.1, w - 0.6, 1.1, { fontSize: 28, bold: true });
    t(s, sub, x + 0.3, y + 2.3, w - 0.6, 0.7, { fontSize: 20, color: MUTED });
    if (i < 2) arrow(s, x + w + 0.02, y + 1.35);
  });
  takeaway(s, `${num(D.params.total4)} numbers learned.  Same frame, same answer.`, 5.6);
  done(s);
}

// 5. where it came from ===========================================================
{ const s = pres.addSlide();
  title(s, "Where did it come from?");
  const flow = ["2025 paper: ICRNNA", "A colleague's code", "Our code: 5 small differences"];
  flow.forEach((h, i) => {
    const x = M + i * 4.15, w = 3.55;
    card(s, x, 1.7, w, 1.3);
    t(s, h, x + 0.3, 1.7, w - 0.6, 1.3, { fontSize: 22, bold: true, valign: "middle" });
    if (i < 2) arrow(s, x + w + 0.02, 2.1);
  });
  t(s, `${FA.e150.toFixed(2)}%`, M, 3.35, 4.6, 1.3, { fontSize: 72, bold: true, color: ORANGE, valign: "bottom" });
  t(s, `rebuilt from the paper alone,\nagainst the paper's ${FA.paper.toFixed(2)}%`, M + 4.7, 3.55, 7.3, 1.1, { fontSize: 24, color: INK, valign: "bottom" });
  t(s, "Code, plots and slides written with Claude Code, an AI assistant. Every number comes from running that code.",
    M, 5.3, W - 2 * M, 0.9, { fontSize: 20, color: MUTED });
  done(s);
}

// 6. the whole experiment =========================================================
{ const s = pres.addSlide();
  const hi = L["128"];
  title(s, "The whole experiment");
  const cols = [["IN", ["4 kinds", "20 noise levels", "1,000 frames each"]],
    ["DETECTOR", ["learns on 70%", "stops on 15%", "tested on 15%", `${C.seeds} repeats`]]];
  cols.forEach(([h, lines], i) => {
    const x = M + i * 4.15, w = 3.55;
    card(s, x, 1.7, w, 3.9);
    t(s, h, x + 0.3, 1.85, w - 0.6, 0.5, { fontSize: 18, bold: true, color: ORANGE, charSpacing: 2 });
    t(s, lines.map((l, k) => ({ text: l, options: { breakLine: k < lines.length - 1 } })), x + 0.3, 2.5, w - 0.6, 3.0, { fontSize: 24, paraSpaceAfter: 8 });
    arrow(s, x + w + 0.02, 3.4);
  });
  const x = M + 2 * 4.15;
  card(s, x, 1.7, W - M - x, 3.9);
  t(s, "OUT", x + 0.3, 1.85, 3, 0.5, { fontSize: 18, bold: true, color: ORANGE, charSpacing: 2 });
  t(s, pct(1 - hi.high_wrong / hi.high_total), x + 0.3, 2.3, 3.6, 1.1, { fontSize: 60, bold: true, color: ORANGE, valign: "bottom" });
  t(s, "right at 10 dB and above", x + 0.3, 3.4, 3.6, 0.5, { fontSize: 20, color: MUTED });
  t(s, "25%", x + 0.3, 4.0, 3.6, 0.8, { fontSize: 40, bold: true, valign: "bottom" });
  t(s, "at −20 dB: a guess", x + 0.3, 4.85, 3.6, 0.5, { fontSize: 20, color: MUTED });
  takeaway(s, "Every number is counted on frames it never saw.", 6.0);
  done(s);
}

// 7. one frame ====================================================================
{ const s = pres.addSlide();
  title(s, "What one frame looks like");
  pic(s, "simple/frame.png", M, 1.75, 7.5, 1530, 544);
  t(s, "One QAM16 frame: 128 samples, 16 symbols (dots)", M, 4.55, 7.5, 0.5, { fontSize: 18, color: MUTED });
  pic(s, "simple/symbols_2x2.png", 8.7, 1.45, 4.0, 1020, 1071);
  takeaway(s, "16 symbols cannot show all 64 points of QAM64.", 5.85);
  t(s, "Drawn by us, without noise.", M, 6.55, 8, 0.4, { fontSize: 14, color: MUTED, italic: true });
  done(s);
}

// 8. the noise ====================================================================
{ const s = pres.addSlide();
  title(s, "What does “10 dB” mean here?");
  const sl = Math.max(...Object.values(SN.slopes)).toFixed(1);
  card(s, M, 1.7, 5.85, 3.0); card(s, M + 6.28, 1.7, 5.85, 3.0);
  big(s, M + 0.3, 1.95, 5.25, `${sl} dB`, "real change per 1 dB of label");
  big(s, M + 6.58, 1.95, 5.25, `+${SN.offsets.QAM64.toFixed(0)} dB`, "more signal in QAM64 than in BPSK or QPSK, at the same label", ORANGE);
  t(s, "Measured in the file itself. Every frame also gets frequency offset, fading and echoes.", M, 4.95, W - 2 * M, 0.8, { fontSize: 20, color: MUTED });
  takeaway(s, "“10 dB” is a setting, not a measurement.", 6.0);
  done(s);
}

// 9. result at 10 dB ==============================================================
{ const s = pres.addSlide();
  const m = C.mats["10"], tot = m.flat().reduce((a, b) => a + b, 0), ok = m.reduce((a, r, i) => a + r[i], 0);
  title(s, "The answer at 10 dB");
  const hc = (x, o = {}) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, align: "center", fontSize: 18, ...o } });
  const rows = [[hc("sent \\ said", { fontSize: 14 }), ...cls.map(c => hc(c)), hc("recall")]];
  m.forEach((r, i) => {
    const tt = r.reduce((a, b) => a + b, 0);
    rows.push([{ text: cls[i], options: { bold: true, fontSize: 18 } },
      ...r.map((v, j) => ({ text: num(v), options: { align: "center", fontSize: 20, bold: i === j, color: i === j ? WHITE : (v >= 20 ? INK : "8A99A8"),
        fill: { color: i === j ? NAVY : (v >= 20 ? "FCE3C8" : WHITE) } } })),
      { text: pct(r[i] / tt), options: { align: "center", fontSize: 18, bold: true } }]);
  });
  const dec = cls.map((_, j) => m.reduce((a, r) => a + r[j], 0));
  rows.push([{ text: "precision", options: { bold: true, fontSize: 16, fill: { color: CARD } } },
    ...dec.map((d, j) => ({ text: pct(m[j][j] / d), options: { align: "center", fontSize: 16, fill: { color: CARD } } })), { text: "", options: { fill: { color: CARD } } }]);
  s.addTable(rows, { x: M, y: 1.6, w: 8.3, colW: [1.6, 1.35, 1.35, 1.35, 1.35, 1.3], fontFace: F, color: INK,
    border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.62, autoPage: false, valign: "middle" });
  t(s, [{ text: "Recall", options: { bold: true } }, { text: " = a row     " }, { text: "Precision", options: { bold: true } }, { text: " = a column" }],
    M, 5.5, 8.3, 0.5, { fontSize: 20, color: MUTED });
  big(s, 9.35, 1.55, 3.4, num(tot - ok), `wrong of ${num(tot)}`, ORANGE, 60, "left");
  big(s, 9.35, 3.55, 3.4, num(m[2][3] + m[3][2]), "are QAM16 ↔ QAM64", INK, 60, "left");
  takeaway(s, "One problem left: QAM16 against QAM64.", 6.2);
  done(s);
}

// 10. lower =======================================================================
{ const s = pres.addSlide();
  const lv = [10, 6, 4, 2, 0, -2, -4, -6];
  title(s, "Turn the noise up");
  s.addChart(pres.ChartType.bar, [{ name: "wrong of 8,400", labels: lv.map(v => (v > 0 ? "+" : "") + v + " dB"), values: lv.map(wrongAt) }],
    { x: M, y: 1.4, w: W - 2 * M, h: 4.6, barDir: "col", barGapWidthPct: 45, chartColors: [NAVY, NAVY, NAVY, NAVY, ORANGE, ORANGE, ORANGE, ORANGE],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 18, dataLabelColor: INK, dataLabelFontFace: F, dataLabelFormatCode: "#,##0",
      showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      catAxisLabelColor: INK, catAxisLabelFontSize: 18, catAxisLabelFontFace: F, valAxisMinVal: 0, valAxisMaxVal: 3300,
      showTitle: true, title: "wrong answers out of 8,400 at each label", titleFontSize: 18, titleColor: MUTED, titleFontFace: F });
  takeaway(s, "10 → 2 dB: nothing changes. These mistakes are not noise.", 6.2);
  done(s);
}

// 11. frame length ================================================================
{ const s = pres.addSlide();
  title(s, "16, 8 or 4 symbols");
  const lens = ["128", "64", "32"], snrs = C.snrs, col = [NAVY, ORANGE, "8FA0B5"];
  s.addChart(pres.ChartType.line, lens.map(k => ({ name: `${k / 8} symbols`, labels: snrs.map((v, i) => (i % 2 ? "" : String(v).replace("-", "\u2212"))), values: L[k].wrong_by_snr.map(w => +(100 * w / 8400).toFixed(1)) })),
    { x: M - 0.1, y: 1.35, w: 6.6, h: 4.7, chartColors: col, lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      showLegend: true, legendPos: "t", legendFontSize: 16, legendFontFace: F, legendColor: INK,
      valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, showValAxisTitle: true, valAxisTitle: "% wrong", showCatAxisTitle: true, catAxisTitle: "label (dB)",
      catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 14, valAxisLabelFontSize: 14, valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" },
      valAxisTitleColor: MUTED, catAxisTitleColor: MUTED, valAxisTitleFontSize: 15, catAxisTitleFontSize: 15 });
  pic(s, "simple/fewer.png", 7.2, 1.45, 5.5, 1428, 986);
  lens.forEach((k, i) => {
    const x = 7.2 + i * 1.85;
    t(s, L[k].floor_pct + "%", x, 5.3, 1.8, 0.65, { fontSize: 32, bold: true, color: col[i], valign: "bottom" });
  });
  t(s, "wrong at low noise", 7.2, 5.95, 5.5, 0.4, { fontSize: 16, color: MUTED });
  takeaway(s, "More signal does not fix it. More symbols does.", 6.35);
  done(s);
}

// 12. something we found ==========================================================
{ const s = pres.addSlide();
  const m = C.m20;
  title(s, "No signal at all, and it still answers");
  card(s, M, 1.6, 5.6, 3.15); card(s, W - M - 5.6, 1.6, 5.6, 3.15);
  big(s, M + 0.3, 1.8, 5.0, pct(m.recall[1]), "QPSK recall at −20 dB:\nlooks like recognition", INK, 80);
  big(s, W - M - 5.3, 1.8, 5.0, pct(m.precision[1]), "QPSK precision at −20 dB:\na guess, 1 in 4", ORANGE, 80);
  t(s, "vs", M + 5.6, 2.6, W - 2 * M - 11.2, 0.6, { fontSize: 24, color: MUTED, align: "center" });
  t(s, `It says BPSK or QPSK ${pct(m.decided_share[0] + m.decided_share[1])} of the time.`, M, 5.25, W - 2 * M, 0.6, { fontSize: 22, color: MUTED });
  takeaway(s, "Recall alone can fool you. Precision now sits next to it.", 6.1);
  done(s);
}

// 13. literature ==================================================================
{ const s = pres.addSlide();
  title(s, "ICRNNA against published models");
  t(s, "All 11 types. Same data, split and repeats for every model.", M, 1.35, W - 2 * M, 0.5, { fontSize: 20, color: MUTED });
  const hc = (x, o = {}) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, fontSize: 18, align: "center", ...o } });
  const rows = [[hc("model", { align: "left" }), hc("paper"), hc("ours"), hc("ours, ≥ 10 dB")]];
  D.lit.forEach(r => {
    const me = r.name === "ICRNNA", o = { fontSize: 22, align: "center", bold: me, color: me ? ORANGE : INK };
    rows.push([{ text: `${r.name} (${r.year})`, options: { ...o, align: "left" } },
      { text: r.paper ? pct(r.paper) : "—", options: { ...o, color: MUTED, bold: false } },
      { text: pct(r.overall), options: o }, { text: pct(r.high), options: o }]);
  });
  s.addTable(rows, { x: M, y: 2.0, w: W - 2 * M, colW: [4.3, 2.4, 2.4, 3.03], fontFace: F, border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.62, autoPage: false, valign: "middle" });
  takeaway(s, "Within 1 point of the papers. Above 10 dB, all stop near 91%.", 6.0);
  done(s);
}

// 14. in short ====================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "In short", M, 0.55, 8, 0.9, { fontSize: 40, bold: true, color: WHITE });
  const pts = ["BPSK and QPSK: solved. The mistakes are QAM16 against QAM64.",
    "The cause is 16 symbols, not noise.",
    "“10 dB” is a setting, not a measurement."];
  pts.forEach((p, i) => {
    badge(s, M, 1.85 + i * 1.25, i + 1, ORANGE, 0.7);
    t(s, p, M + 1.0, 1.75 + i * 1.25, W - 2 * M - 1.0, 0.9, { fontSize: 28, color: WHITE, valign: "middle" });
  });
  t(s, [{ text: "Next: ", options: { bold: true, color: ORANGE } }, { text: "switch off offset, fading and echoes one at a time.", options: { color: SOFT } }],
    M, 5.75, W - 2 * M, 0.7, { fontSize: 24 });
  done(s, true);
}

pres.writeFile({ fileName: "moshe_2016_simple.pptx" }).then(f => console.log("wrote", f, "slides:", n));

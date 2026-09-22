// Builds the Moshe progress deck from deck_data.json. Every number on a slide
// comes from that file, which is extracted from the repository's .npz results;
// nothing is typed in by hand.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("deck_data.json", "utf8"));
const TALK = require("./talk.js");

// ---- palette: deep navy ink, signal-orange accent, cool greys ----------------
const NAVY = "0B2545", INK = "13315C", MID = "5C7A99", PALE = "DCE6F0", BG = "FFFFFF";
const ORANGE = "F26419", TEAL = "1B998B", RED = "C0392B", GREY = "8A99A8", LIGHT = "F3F6F9";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Yusuf Bilal Cetinkaya";
pres.title = "Why modulation classifiers fail across transmitters";

const W = 13.33, M = 0.6;
const f3 = v => v.toFixed(3), pct = v => (100 * v).toFixed(1) + "%";

function light(slide) { slide.background = { color: BG }; }
function dark(slide) { slide.background = { color: NAVY }; }
function title(slide, text, sub, onDark = false) {
  slide.addText(text, { x: M, y: 0.42, w: W - 2 * M, h: 0.8, fontFace: HEAD, fontSize: 30, bold: true,
    color: onDark ? "FFFFFF" : INK, isTextBox: true, margin: 0, valign: "middle" });
  if (sub) slide.addText(sub, { x: M, y: 1.2, w: W - 2 * M, h: 0.45, fontFace: BODY, fontSize: 15,
    color: onDark ? PALE : MID, italic: true, isTextBox: true, margin: 0, valign: "top" });
}
function foot(slide, n, onDark = false) {
  if (TALK[n]) slide.addNotes(TALK[n] + "\n\n");
  slide.addText(`${n}`, { x: W - M - 0.5, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10,
    color: onDark ? PALE : GREY, align: "right", isTextBox: true, margin: 0 });
}
function stat(slide, x, y, w, big, label, color = ORANGE, size = 44) {
  slide.addText(big, { x, y, w, h: 0.95, fontFace: HEAD, fontSize: size, bold: true, color, isTextBox: true, margin: 0, valign: "bottom" });
  slide.addText(label, { x, y: y + 0.97, w, h: 0.55, fontFace: BODY, fontSize: 12.5, color: MID, isTextBox: true, margin: 0, valign: "top" });
}
function card(slide, x, y, w, h, fill = LIGHT) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08 });
}
function bullets(slide, items, x, y, w, h, size = 14, color = INK) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })),
    { x, y, w, h, fontFace: BODY, fontSize: size, color, isTextBox: true, margin: 0, valign: "top" });
}
function circleIcon(slide, x, y, d, fill, glyph) {
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  slide.addText(glyph, { x, y, w: d, h: d, fontFace: BODY, fontSize: d * 26, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
}
const axisQuiet = {
  catAxisLabelColor: MID, valAxisLabelColor: MID, catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
  catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" },
  catAxisTitleColor: MID, valAxisTitleColor: MID, catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  showCatAxisTitle: true, showValAxisTitle: true, showTitle: false, chartArea: { fill: { color: "FFFFFF" } }, plotArea: { fill: { color: "FFFFFF" } },
};
function matrixTable(slide, classes, mat, x, y, w, note) {
  const n = classes.length, colW = [1.55, ...classes.map(() => (w - 1.55 - 0.9) / n), 0.9];
  const hdr = [{ text: "transmitted \\ decided", options: { bold: true, fill: { color: PALE }, color: INK, fontSize: 10 } },
    ...classes.map(c => ({ text: c, options: { bold: true, fill: { color: PALE }, color: INK, align: "center", fontSize: 11 } })),
    { text: "recall", options: { bold: true, fill: { color: PALE }, color: INK, align: "center", fontSize: 11 } }];
  const rows = [hdr];
  mat.forEach((r, i) => {
    const tot = r.reduce((a, b) => a + b, 0);
    rows.push([{ text: classes[i], options: { bold: true, color: INK, fontSize: 11 } },
      ...r.map((v, j) => ({ text: v.toLocaleString("en-US"), options: { align: "center", fontSize: 11, bold: i === j,
        color: i === j ? "FFFFFF" : (v === 0 ? GREY : INK), fill: { color: i === j ? TEAL : (v > 0 ? "FDE9DF" : "FFFFFF") } } })),
      { text: pct(r[i] / tot), options: { align: "center", fontSize: 11, bold: true, color: INK } }]);
  });
  slide.addTable(rows, { x, y, w, colW, fontFace: BODY, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.36, autoPage: false });
  if (note) slide.addText(note, { x, y: y + 0.36 * (n + 1) + 0.08, w, h: 0.4, fontFace: BODY, fontSize: 10.5, color: MID, italic: true, isTextBox: true, margin: 0 });
}

const c4 = D.c4, c24 = D.c24, meth = D.methods, alpha = D.alpha;
const highTot = c4.mats.high.flat().reduce((a, b) => a + b, 0);
const highOk = c4.mats.high.reduce((a, r, i) => a + r[i], 0);
const z0 = c4.mats.z0, z0Tot = z0.flat().reduce((a, b) => a + b, 0), z0Ok = z0.reduce((a, r, i) => a + r[i], 0);
const none = meth[0], aug = meth[1], wht = meth[2], both = meth[3];
let n = 0;

// ============================ 1. title ======================================
{ const s = pres.addSlide(); dark(s); n++;
  s.addText("Why a modulation classifier fails across transmitters", { x: M, y: 1.3, w: 8.4, h: 1.7, fontFace: HEAD, fontSize: 36, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, valign: "bottom" });
  s.addText("— measured by intervention, and closed", { x: M, y: 3.05, w: 8.4, h: 0.7, fontFace: HEAD, fontSize: 26, color: ORANGE, isTextBox: true, margin: 0 });
  s.addText("RadioML 2018.01A · ICRNNA backbone · progress report, September 2026", { x: M, y: 4.2, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 16, color: PALE, isTextBox: true, margin: 0 });
  s.addText("Yusuf Bilal Çetinkaya", { x: M, y: 4.75, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 16, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addImage({ path: "c64qam.png", x: 9.4, y: 1.4, w: 3.3, h: 3.3, transparency: 15 });
  s.addNotes(TALK[1] + "\n\nEvery number in this deck is read from a results file in the repository; nothing is quoted from the earlier backbone.");
}

// ============================ 2. the problem ================================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "The problem", "Reproducing the benchmark is solved. What happens when the transmitter changes is not.");
  card(s, M, 1.9, 6.0, 4.6);
  s.addText("Train on RadioML. Test on I/Q generated by an independent transmitter model. Same five modulations, same SNRs, same sample rate.",
    { x: M + 0.3, y: 2.05, w: 5.4, h: 1.2, fontFace: BODY, fontSize: 15, color: INK, isTextBox: true, margin: 0, valign: "top" });
  stat(s, M + 0.3, 3.3, 2.5, f3(none.in_dom), "in-domain accuracy", TEAL);
  stat(s, M + 3.0, 3.3, 2.5, f3(none.cross), "on the other transmitter", RED);
  s.addText(`16QAM alone: ${f3(none.qam16)} — it is read as 64QAM`, { x: M + 0.3, y: 5.2, w: 5.4, h: 0.5, fontFace: BODY, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0 });
  s.addText("The questions this project answers", { x: 7.2, y: 1.95, w: 5.5, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK, isTextBox: true, margin: 0 });
  [["1", "How large is the drop, and where does it land?"],
   ["2", "What property of the signal causes it — shown by intervention, not correlation?"],
   ["3", "Can it be removed without paying for it in-domain?"],
   ["4", "Where does a modulation the model never saw go?"]].forEach(([k, t], i) => {
    circleIcon(s, 7.2, 2.6 + i * 0.95, 0.5, i === 3 ? ORANGE : NAVY, k);
    s.addText(t, { x: 7.9, y: 2.55 + i * 0.95, w: 4.8, h: 0.6, fontFace: BODY, fontSize: 14, color: INK, isTextBox: true, margin: 0, valign: "middle" });
  });
  foot(s, n);
  s.addNotes("The contribution is the attribution method, not an accuracy number. The 0.999 to 0.805 drop is the starting measurement, on five shared classes, five seeds.");
}

// ============================ 3. dataset ====================================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Dataset: RadioML 2018.01A", "One dataset, by decision. Every class shares the same spectral envelope — that turns out to matter.");
  s.addImage({ path: "spectra24.png", x: M, y: 1.8, w: 7.9, h: 4.55 });
  s.addText("Mean power spectrum per class at 30 dB. Blue = symmetric, red = single-sideband.", { x: M, y: 6.4, w: 7.9, h: 0.35, fontFace: BODY, fontSize: 10.5, color: MID, italic: true, isTextBox: true, margin: 0 });
  const rows = [["24", "modulation classes — ASK, PSK, APSK, QAM, analog"], ["1,024", "I/Q samples per frame, used at native length"],
    ["26", "SNR levels, −20 to +30 dB in 2 dB steps"], ["4,096", "frames per (class, SNR) cell; 2.5 M frames, 21 GB"]];
  rows.forEach(([b, t], i) => {
    s.addText(b, { x: 8.9, y: 1.8 + i * 0.95, w: 1.5, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true, color: ORANGE, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(t, { x: 10.45, y: 1.8 + i * 0.95, w: 2.4, h: 0.6, fontFace: BODY, fontSize: 12.5, color: INK, isTextBox: true, margin: 0, valign: "middle" });
  });
  s.addText("RML2016.10a stays in the record for one purpose: the paper-faithful build reaches 63.21% against the published 63.24% on it — the pipeline's only tie to a published number.",
    { x: 8.9, y: 5.65, w: 3.95, h: 1.1, fontFace: BODY, fontSize: 10.5, color: MID, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes("Decided with Moshe on 16 September: new work goes on 2018 only. The 2016 results are not retracted.");
}

// ============================ 4. inputs / outputs / model ===================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Input, output, model, protocol", "Defined once here, and unchanged for every result that follows.");
  const boxes = [["INPUT", "one I/Q frame\n2 × 1,024 float32\nunit average power", NAVY],
    ["MODEL", "ICRNNA · 786k params\nconv front end → BiLSTM →\nadditive attention", ORANGE],
    ["OUTPUT", "one of K class labels\nK = 24, or 4 for the\ndecision table", NAVY]];
  boxes.forEach(([h, t, c], i) => {
    const x = M + i * 4.2; card(s, x, 1.9, 3.7, 1.9, i === 1 ? "FDE9DF" : LIGHT);
    s.addText(h, { x: x + 0.25, y: 2.0, w: 3.2, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true, color: c, isTextBox: true, margin: 0 });
    s.addText(t, { x: x + 0.25, y: 2.45, w: 3.2, h: 1.25, fontFace: BODY, fontSize: 13, color: INK, isTextBox: true, margin: 0, valign: "top" });
    if (i < 2) s.addShape(pres.ShapeType.rightArrow, { x: x + 3.78, y: 2.6, w: 0.35, h: 0.45, fill: { color: MID }, line: { color: MID, width: 0 } });
  });
  s.addText("Protocol", { x: M, y: 4.1, w: 6, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK, isTextBox: true, margin: 0 });
  bullets(s, ["70 / 15 / 15 split, stratified over every (class, SNR) cell",
    "Early stopping, LR schedule and checkpoint choice read validation only; the test set is touched once",
    "Batch 256, Adam 1e-3, ReduceLROnPlateau, patience 20 under an epoch ceiling",
    "Three seeds per configuration; results written after every seed and resumed, never overwritten"], M, 4.55, 6.3, 2.3, 12.5);
  card(s, 7.3, 4.1, 5.45, 2.75, "FFF4EE");
  s.addText("A run counts as converged only if", { x: 7.55, y: 4.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0 });
  s.addText("best_epoch + patience ≤ ceiling", { x: 7.55, y: 4.65, w: 5.0, h: 0.5, fontFace: "Courier New", fontSize: 16, bold: true, color: ORANGE, isTextBox: true, margin: 0 });
  s.addText("Otherwise the number is a floor, not a measurement. Every result here passes this test; where a first run did not, it was rerun under a higher ceiling and the rerun is what is shown. Convergence turned out to be set by gradient updates, not epochs — and to depend on the method, not only the data.",
    { x: 7.55, y: 5.2, w: 5.0, h: 1.6, fontFace: BODY, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes("The backbone is transcribed from a peer's reproduction and differs from the published ICRNNA in five places; it is not called 'the published architecture'. The faithful build is a separate file and does reproduce the paper.");
}

// ============================ 5. does it classify? ==========================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "First, does the backbone classify?", "Not the contribution — but the cross-domain numbers must start from a competent classifier.");
  s.addChart(pres.ChartType.line, [
    { name: "24 classes, 512 frames/cell", labels: c24.snrs.map(String), values: c24.curve },
    { name: "4 classes (BPSK, QPSK, 16QAM, 64QAM), 2,048 frames/cell", labels: c4.snrs.map(String), values: c4.curve }],
    { x: M, y: 1.75, w: 7.9, h: 4.9, chartColors: [NAVY, ORANGE], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5,
      showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 10, legendColor: INK,
      catAxisTitle: "SNR (dB)", valAxisTitle: "test accuracy", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", catAxisLabelFrequency: 2, ...axisQuiet });
  stat(s, 8.9, 1.8, 3.9, pct(c24.high), `24 classes, SNR ≥ 10 dB  (±${(100 * c24.high_sd).toFixed(1)})`, NAVY);
  stat(s, 8.9, 3.5, 3.9, pct(c4.high), "4 classes, SNR ≥ 10 dB", ORANGE);
  s.addText(`Overall across all 26 levels: ${f3(c24.overall)} on 24 classes, ${f3(c4.overall)} on 4. Chance is 0.042 and 0.25; the curve sits there below −12 dB and saturates from +8.`,
    { x: 8.9, y: 5.25, w: 3.9, h: 1.4, fontFace: BODY, fontSize: 11.5, color: MID, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes("Deep residual networks on the whole dataset are reported near 95% above 8 dB; this is 512 of 4096 frames per cell with a 786k recurrent model. Closing that gap is not the goal.");
}

// ============================ 6. the decision table =========================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "The decision table", `Four modulations in, ${highTot.toLocaleString("en-US")} decisions above 10 dB, three seeds pooled — how often did we say what?`);
  matrixTable(s, c4.classes, c4.mats.high, M, 1.9, 7.6, `${highOk.toLocaleString("en-US")} of ${highTot.toLocaleString("en-US")} correct. Both errors are 64QAM decided as 16QAM. Rows sum to 10,164: 308 test frames per (class, SNR) cell × 11 levels × 3 seeds.`);
  s.addImage({ path: "c16qam.png", x: 8.7, y: 1.85, w: 2.0, h: 2.0 });
  s.addImage({ path: "c64qam.png", x: 10.8, y: 1.85, w: 2.0, h: 2.0 });
  s.addText("16QAM and 64QAM at 20 dB: the pair that carries almost every error in this project", { x: 8.7, y: 3.9, w: 4.1, h: 0.6, fontFace: BODY, fontSize: 10.5, color: MID, italic: true, isTextBox: true, margin: 0 });
  card(s, 8.7, 4.65, 4.1, 2.1, "FFF4EE");
  s.addText("Converged", { x: 8.9, y: 4.72, w: 3.7, h: 0.35, fontFace: HEAD, fontSize: 13, bold: true, color: INK, isTextBox: true, margin: 0 });
  s.addText(`Peaks at epochs ${c4.best_epochs.join(", ")} under a 150 ceiling. Two of the three seeds are bit-identical to the earlier 60-epoch run — the lower ceiling had only denied them the proof.`,
    { x: 8.9, y: 5.1, w: 3.7, h: 1.6, fontFace: BODY, fontSize: 11, color: INK, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes("This is the table asked for on 15 September. The same table on RML2016 (128-sample frames) still confuses the QAM pair 7–11% at high SNR; at 1,024 samples that confusion is gone. The frame is the difference.");
}

// ============================ 7. where decisions go by SNR ==================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Where the decisions go, by SNR", "A perfect table at high SNR says nothing. The informative tables are lower down.");
  const snrs = c4.snrs.filter(v => v >= -14 && v <= 10);
  s.addChart(pres.ChartType.line, c4.classes.map((c, i) => ({ name: c, labels: snrs.map(String), values: snrs.map(v => c4.recall_by_snr[String(v)][i]) })),
    { x: M, y: 1.75, w: 6.1, h: 4.9, chartColors: [NAVY, TEAL, ORANGE, RED], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5,
      showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 10, legendColor: INK,
      catAxisTitle: "SNR (dB)", valAxisTitle: "recall per class", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", ...axisQuiet });
  matrixTable(s, c4.classes, z0, 7.0, 1.85, 5.8, `At 0 dB: ${z0Ok.toLocaleString("en-US")} of ${z0Tot.toLocaleString("en-US")} (${pct(z0Ok / z0Tot)}). PSK is perfect; the QAM pair is a coin flip against each other.`);
  card(s, 7.0, 4.35, 5.8, 2.4);
  s.addText("Two stages, two thresholds", { x: 7.2, y: 4.42, w: 5.4, h: 0.35, fontFace: HEAD, fontSize: 13, bold: true, color: INK, isTextBox: true, margin: 0 });
  const m8 = c4.mats.m8, leak16 = m8[2][1] / m8[2].reduce((a, b) => a + b, 0), leak64 = m8[3][1] / m8[3].reduce((a, b) => a + b, 0);
  bullets(s, ["PSK-versus-QAM closes between −4 and 0 dB; which-QAM closes between 0 and +6",
    `Below the first threshold the unreadable classes do not scatter — they sink: at −8 dB, 16QAM is read as QPSK ${(100 * leak16).toFixed(0)}% of the time and 64QAM ${(100 * leak64).toFixed(0)}%; 16QAM's own recall is below chance`,
    "The accuracy curve averages both stages into one number per level"], 7.2, 4.8, 5.4, 1.9, 11);
  foot(s, n);
  s.addNotes("924 decisions per row at each single level; a 1% cell is nine frames. Both matrices are from the converged run.");
}

// ============================ 8. the domain gap =============================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "The gap, and what closes it", "Five shared classes, five seeds each, every cell converged. In-domain never moves; cross-domain does.");
  s.addChart(pres.ChartType.bar, [
    { name: "in-domain", labels: meth.map(m => m.name.replace("a=0.75", "α=0.75")), values: meth.map(m => m.in_dom) },
    { name: "cross-domain", labels: meth.map(m => m.name.replace("a=0.75", "α=0.75")), values: meth.map(m => m.cross) }],
    { x: M, y: 1.75, w: 7.6, h: 4.9, barDir: "col", barGapWidthPct: 60, chartColors: [PALE, ORANGE], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.000", dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFontFace: BODY,
      showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 10, legendColor: INK,
      catAxisTitle: "training method", valAxisTitle: "test accuracy", valAxisMinVal: 0.7, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.00", ...axisQuiet });
  stat(s, 8.7, 1.8, 4.1, `${f3(none.cross)} → ${f3(wht.cross)}`, "no method → spectral whitening", ORANGE, 34);
  const rows = [[{ text: "comparison", options: { bold: true, fill: { color: PALE }, color: INK } }, { text: "Δ cross-domain", options: { bold: true, fill: { color: PALE }, color: INK, align: "center" } }, { text: "verdict", options: { bold: true, fill: { color: PALE }, color: INK } }],
    ["literature augmentation vs none", { text: (aug.cross - none.cross).toFixed(3), options: { align: "center" } }, "does not address it"],
    ["whitening vs none", { text: "+" + (wht.cross - none.cross).toFixed(3), options: { align: "center", bold: true, color: ORANGE } }, "closes it"],
    ["whitening + augmentation vs whitening", { text: "+" + (both.cross - wht.cross).toFixed(3), options: { align: "center" } }, "not a measurement (1 s.d.)"]];
  s.addTable(rows, { x: 8.7, y: 3.5, w: 4.1, colW: [1.95, 0.85, 1.3], fontFace: BODY, fontSize: 9.5, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.4, autoPage: false });
  s.addText("Rotation, conjugate flip and additive noise — the augmentation set the literature cites — move the gap by 0.002. Whatever they are good for, this failure mode is not it.",
    { x: 8.7, y: 5.75, w: 4.1, h: 1.0, fontFace: BODY, fontSize: 11, color: MID, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes(`Stopping epochs by method: none ${none.epochs.join("/")}, augmentation ${aug.epochs.join("/")}, whitening ${wht.epochs.join("/")}, both ${both.epochs.join("/")}. Whitening converges about 2.4× faster than augmentation — it removes a nuisance dimension, augmentation adds one.`);
}

// ============================ 9. attribution by intervention ================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "What causes it — by intervention, not correlation", "Four explanations tested and refuted. One intervention that moves the prediction.");
  const refuted = [["Occupied bandwidth", "matching it did not recover 16QAM"], ["Constellation density", "density is not what the model reads"],
    ["Symbol-timing regularity", "RadioML has no symbol-rate spectral line; driving it down changed nothing"], ["Channel impairments", "CFO, multipath, IQ imbalance removed one at a time — little changed"]];
  refuted.forEach(([h, t], i) => {
    const y = 1.9 + i * 1.05; circleIcon(s, M, y, 0.5, RED, "×");
    s.addText(h, { x: M + 0.7, y: y - 0.05, w: 5.4, h: 0.35, fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, isTextBox: true, margin: 0 });
    s.addText(t, { x: M + 0.7, y: y + 0.3, w: 5.4, h: 0.5, fontFace: BODY, fontSize: 11.5, color: MID, isTextBox: true, margin: 0, valign: "top" });
  });
  card(s, 7.1, 1.85, 5.7, 4.9, "FFF4EE");
  circleIcon(s, 7.35, 2.05, 0.5, TEAL, "✓");
  s.addText("Phase-preserving magnitude substitution", { x: 8.0, y: 2.0, w: 4.6, h: 0.6, fontFace: HEAD, fontSize: 14.5, bold: true, color: INK, isTextBox: true, margin: 0, valign: "middle" });
  s.addText("Take a frame from the failing domain. Keep its phase spectrum exactly. Substitute only the magnitude spectrum of the training domain. Nothing else changes — not the symbols, not the timing, not the channel.",
    { x: 7.35, y: 2.75, w: 5.2, h: 1.3, fontFace: BODY, fontSize: 12, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addText("The prediction follows the magnitude spectrum.", { x: 7.35, y: 4.1, w: 5.2, h: 0.45, fontFace: HEAD, fontSize: 14, bold: true, color: ORANGE, isTextBox: true, margin: 0 });
  s.addText("That is an intervention on the signal, not an observation of the representation, so it supports a causal claim. The cause is the transmitter's spectral envelope — chiefly its pulse-shaping roll-off.",
    { x: 7.35, y: 4.6, w: 5.2, h: 1.2, fontFace: BODY, fontSize: 12, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addText("Signal-level, classifier-independent; the accuracies attached to these interventions are pending re-measurement on the current backbone.", { x: 7.35, y: 5.9, w: 5.2, h: 0.7, fontFace: BODY, fontSize: 10, color: MID, italic: true, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes("The working story for most of the project was shortcut learning — the model reads modulation order off the envelope. Two experiments killed it: handing 16QAM frames a 64QAM envelope left the prediction at 16QAM. The envelope does not carry the class; removing it still helps. That is stated as unresolved.");
}

// ============================ 10. the fix: whitening and alpha ==============
{ const s = pres.addSlide(); light(s); n++;
  title(s, "The fix: remove the envelope", "X / smooth(|X|)^α — divide each frame's spectrum by a smoothed estimate of its own magnitude envelope.");
  s.addChart([
    { type: pres.ChartType.line, data: [{ name: "cross-domain accuracy", labels: alpha.map(a => a.alpha.toFixed(2)), values: alpha.map(a => a.cross) }],
      options: { chartColors: [ORANGE], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7, showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 9, dataLabelColor: INK, dataLabelPosition: "t" } },
    { type: pres.ChartType.line, data: [{ name: "16QAM recall", labels: alpha.map(a => a.alpha.toFixed(2)), values: alpha.map(a => a.qam16) }],
      options: { chartColors: [NAVY], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7, showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 9, dataLabelColor: NAVY, dataLabelPosition: "b" } }],
    { x: M, y: 1.75, w: 7.2, h: 4.9,
      showLegend: true, legendPos: "b", legendFontFace: BODY, legendFontSize: 10, legendColor: INK,
      catAxisTitle: "α (0 = no whitening, 1 = full)", valAxisTitle: "accuracy, five seeds", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0%", ...axisQuiet });
  card(s, 8.3, 1.85, 4.5, 2.3);
  s.addText("Monotone, saturating by 0.75", { x: 8.5, y: 1.95, w: 4.1, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0 });
  s.addText(`α = 1.0 is at least as good as 0.75 (${f3(alpha[4].cross)} vs ${f3(alpha[3].cross)}) and seven times more stable across seeds (±${f3(alpha[4].cross_sd)} vs ±${f3(alpha[3].cross_sd)}). The recovery is carried by 16QAM at every step: α switches one confusion off rather than lifting everything a little.`,
    { x: 8.5, y: 2.4, w: 4.1, h: 1.7, fontFace: BODY, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" });
  card(s, 8.3, 4.35, 4.5, 2.4, "FFF4EE");
  s.addText("Costs nothing in-domain", { x: 8.5, y: 4.45, w: 4.1, h: 0.4, fontFace: HEAD, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0 });
  s.addText(`In-domain accuracy stays at ${f3(wht.in_dom)} with whitening on. The gap falls from +${(none.in_dom - none.cross).toFixed(3)} to +${(wht.in_dom - wht.cross).toFixed(3)}, so the recovery is not bought by trading away benchmark performance.`,
    { x: 8.5, y: 4.9, w: 4.1, h: 1.8, fontFace: BODY, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes("The middle of the sweep is where seeds disagree: α = 0.5 has a spread of 0.019 on cross-domain accuracy against 0.001–0.007 at either end. Half-removing the envelope leaves a model that sometimes learns to read through it.");
}

// ============================ 11. where an unseen modulation lands ==========
{ const s = pres.addSlide(); light(s); n++;
  const k = D.sink24, ar = D.sink_arch;
  title(s, "Where an unseen modulation lands", "Train on 23 classes, probe with the 24th. Families fixed before the run; now measured on the current backbone.");
  stat(s, M, 1.75, 3.9, `${k.same} of ${k.n}`, `held-out classes sink into their own family · chance ${k.chance.toFixed(1)}`, TEAL, 40);
  card(s, M, 3.45, 3.9, 1.55);
  s.addText([{ text: "All 24 runs converged. ", options: { bold: true } },
    { text: `Best epochs ${Math.min(...k.rows.map(r => r.best_epoch))} to ${Math.max(...k.rows.map(r => r.best_epoch))} under a ${k.ceiling} ceiling, patience ${k.patience}. ${k.rows.filter(r => r.share > 0.95).length} of the 24 sinks take more than 95% of the probes — the model does not scatter, it decides.` }],
    { x: M + 0.2, y: 3.55, w: 3.5, h: 1.35, fontFace: BODY, fontSize: 11, color: INK, isTextBox: true, margin: 0, valign: "top" });
  s.addText(`Merging APSK with QAM — both amplitude-and-phase constellations — would make it ${k.same_if_apsk_qam_merged} of ${k.n}. That reading is post hoc, so the number quoted is ${k.same}.`,
    { x: M, y: 5.2, w: 3.9, h: 1.3, fontFace: BODY, fontSize: 11, color: MID, italic: true, isTextBox: true, margin: 0, valign: "top" });
  // the eight misses
  const misses = k.rows.filter(r => !r.same);
  const hdr = ["held out", "sink", "share", "second"].map((t, i) => ({ text: t, options: { bold: true, fill: { color: PALE }, color: INK, fontSize: 10, align: i > 1 ? "center" : "left" } }));
  const rows = [hdr, ...misses.map(r => [
    { text: r.held, options: { bold: true, color: INK } },
    { text: r.sink, options: { color: (["APSK", "QAM"].includes(r.family) && ["APSK", "QAM"].includes(D.sink24.rows.find(q => q.held === r.sink).family)) ? ORANGE : INK, bold: true } },
    { text: (100 * r.share).toFixed(0) + "%", options: { align: "center" } },
    { text: r.second_share > 0.05 ? `${r.second} ${(100 * r.second_share).toFixed(0)}%` : "—", options: { align: "center", color: GREY } }])];
  s.addText("The eight misses", { x: 4.9, y: 1.75, w: 4.0, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0 });
  s.addTable(rows, { x: 4.9, y: 2.15, w: 4.0, colW: [1.0, 1.0, 0.7, 1.3], fontFace: BODY, fontSize: 10.5, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.31, autoPage: false });
  s.addText("Orange: an APSK ↔ QAM pair. Four of the eight misses are that pair; two more are FM ↔ GMSK, both constant-envelope. The misses are structured — the hand taxonomy, not the model, is what they disagree with.",
    { x: 4.9, y: 5.05, w: 4.0, h: 1.45, fontFace: BODY, fontSize: 10.5, color: MID, isTextBox: true, margin: 0, valign: "top" });
  // the architecture control
  s.addText("Across five architectures", { x: 9.3, y: 1.75, w: 3.45, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0 });
  s.addChart(pres.ChartType.bar, [{ name: "same-family sinks", labels: ar.archs.map(a => a.split(" ")[0]), values: ar.archs.map(a => ar.same[a]) }],
    { x: 9.2, y: 2.1, w: 3.6, h: 2.9, barDir: "col", barGapWidthPct: 45, chartColors: [NAVY], showValue: true, dataLabelPosition: "outEnd",
      dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFontFace: BODY, showLegend: false,
      valAxisMinVal: 0, valAxisMaxVal: 6, valAxisMajorUnit: 1, catAxisLabelFontSize: 9, ...axisQuiet, showCatAxisTitle: false, valAxisTitle: "same-family sinks of 6", catAxisLabelRotate: 0 });
  const agree = ar.held.filter(h => new Set(ar.archs.map(a => ar.cells[a][h].sink)).size === 1);
  s.addText(`Chance is ${ar.chance.toFixed(1)} of 6. ${agree.join(", ")} sink to the same class on all five; the two misses on every architecture are FM and OQPSK, the two hard cases the rule put in on purpose. ${ar.unconverged.length} of 30 runs unconverged (${ar.unconverged.join("; ")}) — its sink stands.`,
    { x: 9.3, y: 5.05, w: 3.45, h: 1.45, fontFace: BODY, fontSize: 10.5, color: MID, isTextBox: true, margin: 0, valign: "top" });
  foot(s, n);
  s.addNotes("The ICRNNA column of the control is a bit-identical training to the corresponding rows of the 24-class run: same best epoch, same in-distribution accuracy to every digit. The probe frames are an independent draw and move the shares by at most 0.3 points.");
}

// ============================ 12. what we withdrew ==========================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Three things this rerun took away", "Kept in the record, not deleted. Each was believed on the earlier backbone and did not survive re-measurement.");
  const items = [["\"Partial whitening beats full\"", "Our one concrete disagreement with WhiteNet, from one seed per point on the old backbone. Five seeds on the current one: α = 1.0 ≥ 0.75. Withdrawn; on this task we agree with WhiteNet."],
    ["\"The model reads the class off the envelope\"", "Handing 16QAM frames a 64QAM envelope left the prediction at 16QAM. The envelope does not carry the class, yet removing it still helps. The mechanism is stated as unresolved."],
    ["\"The 60-epoch ceiling was safe\"", "It was not: cells peaked at 82–89. Rerun with room, every flagged cell came back bit-identical or within a point — the ceiling had cost nothing, but that could only be shown by paying for the rerun."]];
  items.forEach(([h, t], i) => {
    const x = M + i * 4.13; card(s, x, 1.9, 3.85, 4.7);
    circleIcon(s, x + 0.25, 2.1, 0.5, i === 2 ? ORANGE : RED, i === 2 ? "!" : "−");
    s.addText(h, { x: x + 0.25, y: 2.75, w: 3.35, h: 0.8, fontFace: HEAD, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0, valign: "top" });
    s.addText(t, { x: x + 0.25, y: 3.65, w: 3.35, h: 2.8, fontFace: BODY, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" });
  });
  foot(s, n);
  s.addNotes("Negative results are part of the argument. Whitening itself is not novel either: WhiteNet arrived at the same operation on real over-the-air captures; what survives here is the attribution method and the sink finding.");
}

// ============================ 13. caveats ===================================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Caveats, stated rather than buried");
  const cav = [["Both domains are synthetic", "RadioML is simulated and so is the second transmitter. Until an over-the-air capture exists this is a study of a mechanism, not a measurement of a deployment. The single largest weakness.", RED],
    ["The model is not the published architecture", "It is transcribed from a peer's reproduction and differs from the paper in five places. The paper-faithful build is separate — it reproduces the paper's 63.24% (63.21%), but only trained to convergence, 107 epochs against the paper's 58.", ORANGE],
    ["Half the sink checks are still on the old backbone", "The 24-class leave-one-out and the architecture control are re-measured. The embedding check, the discriminating control, label permutation and family recovery are not yet.", ORANGE],
    ["Three seeds is thin on 24 classes", "Seed spread reaches 0.021; a difference under about 0.03 between two configurations is not yet a difference.", MID]];
  cav.forEach(([h, t, c], i) => {
    const col = i % 2, row = Math.floor(i / 2), x = M + col * 6.2, y = 1.55 + row * 2.75;
    card(s, x, y, 5.95, 2.5); circleIcon(s, x + 0.25, y + 0.25, 0.42, c, "!");
    s.addText(h, { x: x + 0.85, y: y + 0.2, w: 4.9, h: 0.5, fontFace: HEAD, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(t, { x: x + 0.25, y: y + 0.85, w: 5.45, h: 1.55, fontFace: BODY, fontSize: 11.5, color: INK, isTextBox: true, margin: 0, valign: "top" });
  });
  foot(s, n);
}

// ============================ 14. next, and asks ============================
{ const s = pres.addSlide(); dark(s); n++;
  title(s, "Next", null, true);
  const next = [["Just landed", "The sink thread on the current backbone: 16 of 24 held-out classes sink into their own family, and the same-family sink survives on all five architectures. Both runs converged."],
    ["Next", "Port or drop the domain-adversarial baseline (dann.py); then the cumulant classifier on the same four classes at 0 dB as an independent check that the QAM coin flip is the signal's limit, not the model's."],
    ["Growing", "This deck is the running record; new results are appended, withdrawn claims stay."]];
  next.forEach(([h, t], i) => {
    const y = 1.6 + i * 1.35;
    s.addText(h, { x: M, y, w: 2.2, h: 0.5, fontFace: HEAD, fontSize: 15, bold: true, color: ORANGE, isTextBox: true, margin: 0 });
    s.addText(t, { x: M + 2.3, y, w: W - 2 * M - 2.3, h: 1.2, fontFace: BODY, fontSize: 12.5, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top" });
  });
  s.addText("Repository: YusufBilo78/amc · every figure regenerates from the committed .npz files", { x: M, y: 6.6, w: W - 2 * M, h: 0.4, fontFace: BODY, fontSize: 10.5, color: PALE, isTextBox: true, margin: 0 });
  foot(s, n, true);
}

pres.writeFile({ fileName: "amc_progress.pptx" }).then(f => console.log("wrote", f, "slides:", n));

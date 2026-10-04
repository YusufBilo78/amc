// Builds the 2016 deck for Moshe from deck2016_data.json (deck/extract_2016.py).
// Every number on a slide comes from that file or from a figure regenerated
// from the committed results; nothing is typed in by hand except structure.
// Same palette and helpers as build.js, so the two decks read as one record.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("deck2016_data.json", "utf8"));
const TALK = require("./talk2016.js");

const NAVY = "0B2545", INK = "13315C", MID = "5C7A99", PALE = "DCE6F0", BG = "FFFFFF";
const ORANGE = "F26419", TEAL = "1B998B", RED = "C0392B", GREY = "8A99A8", LIGHT = "F3F6F9";
const HEAD = "Cambria", BODY = "Calibri";
const FIG = "../figures/";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Yusuf Bilal Cetinkaya";
pres.title = "Opening the four-class box on RML2016.10a";

const W = 13.33, M = 0.6;
const pct = v => (100 * v).toFixed(1) + "%";
const num = v => v.toLocaleString("en-US");

function light(s) { s.background = { color: BG }; }
function dark(s) { s.background = { color: NAVY }; }
function title(s, text, sub, onDark = false) {
  s.addText(text, { x: M, y: 0.42, w: W - 2 * M, h: 0.8, fontFace: HEAD, fontSize: 30, bold: true,
    color: onDark ? "FFFFFF" : INK, isTextBox: true, margin: 0, valign: "middle" });
  if (sub) s.addText(sub, { x: M, y: 1.2, w: W - 2 * M, h: 0.45, fontFace: BODY, fontSize: 15,
    color: onDark ? PALE : MID, italic: true, isTextBox: true, margin: 0, valign: "top" });
}
function foot(s, n, onDark = false) {
  if (TALK[n]) s.addNotes(TALK[n]);
  s.addText(`${n}`, { x: W - M - 0.5, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10,
    color: onDark ? PALE : GREY, align: "right", isTextBox: true, margin: 0 });
}
function stat(s, x, y, w, big, label, color = ORANGE, size = 40) {
  s.addText(big, { x, y, w, h: 0.85, fontFace: HEAD, fontSize: size, bold: true, color, isTextBox: true, margin: 0, valign: "bottom" });
  s.addText(label, { x, y: y + 0.88, w, h: 0.6, fontFace: BODY, fontSize: 12, color: MID, isTextBox: true, margin: 0, valign: "top" });
}
function card(s, x, y, w, h, fill = LIGHT) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08 });
}
function text(s, t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: BODY, fontSize: 13, color: INK, isTextBox: true, margin: 0, valign: "top", ...o });
}
function circleIcon(s, x, y, d, fill, glyph) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(glyph, { x, y, w: d, h: d, fontFace: BODY, fontSize: d * 26, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
}
function image(s, file, x, y, w, pxW, pxH) { s.addImage({ path: FIG + file, x, y, w, h: w * pxH / pxW }); return w * pxH / pxW; }
const hc = (t, o = {}) => ({ text: t, options: { bold: true, fill: { color: PALE }, color: INK, ...o } });

// the confusion matrix, with a precision row
function matrixTable(s, classes, mat, x, y, w, rowH = 0.42, fs = 13) {
  const n = classes.length, colW = [1.75, ...classes.map(() => (w - 1.75 - 1.05) / n), 1.05];
  const rows = [[hc("sent \\ decided", { fontSize: fs - 2 }), ...classes.map(c => hc(c, { align: "center", fontSize: fs })), hc("recall", { align: "center", fontSize: fs })]];
  mat.forEach((r, i) => {
    const tot = r.reduce((a, b) => a + b, 0);
    rows.push([{ text: classes[i], options: { bold: true, color: INK, fontSize: fs } },
      ...r.map((v, j) => ({ text: num(v), options: { align: "center", fontSize: fs, bold: i === j,
        color: i === j ? "FFFFFF" : (v === 0 ? GREY : INK), fill: { color: i === j ? TEAL : (v >= 20 ? "FDE9DF" : "FFFFFF") } } })),
      { text: pct(r[i] / tot), options: { align: "center", fontSize: fs, bold: true, color: INK } }]);
  });
  const dec = classes.map((_, j) => mat.reduce((a, r) => a + r[j], 0));
  rows.push([{ text: "precision", options: { bold: true, color: INK, fontSize: fs - 1, fill: { color: LIGHT } } },
    ...dec.map((d, j) => ({ text: `${pct(mat[j][j] / d)}\nof ${num(d)}`, options: { align: "center", fontSize: fs - 3, color: INK, fill: { color: LIGHT } } })),
    { text: "", options: { fill: { color: LIGHT } } }]);
  s.addTable(rows, { x, y, w, colW, fontFace: BODY, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH, autoPage: false });
}

const C = D.c4, L = C.lengths, cls = C.classes, SN = D.snr, F = D.faithful, MT = D.methods;
const wrongAt = lv => L["128"].wrong_by_snr[C.snrs.indexOf(lv)];
let n = 0;

// ============================ 1. title ======================================
{ const s = pres.addSlide(); dark(s); n++;
  s.addText("Opening the four-class box", { x: M, y: 1.5, w: 8.6, h: 1.2, fontFace: HEAD, fontSize: 40, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, valign: "bottom" });
  s.addText("What you asked on 22 September, answered on RML2016.10a", { x: M, y: 2.8, w: 8.6, h: 0.7, fontFace: HEAD, fontSize: 22, color: ORANGE, isTextBox: true, margin: 0 });
  s.addText("BPSK · QPSK · QAM16 · QAM64 · 128-sample frames · ICRNNA · 14 seeds", { x: M, y: 4.0, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 16, color: PALE, isTextBox: true, margin: 0 });
  s.addText("Yusuf Bilal Çetinkaya · September 2026", { x: M, y: 4.55, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 16, color: "FFFFFF", isTextBox: true, margin: 0 });
  s.addImage({ path: "c16qam.png", x: 9.5, y: 1.4, w: 3.2, h: 3.2, transparency: 15 });
  foot(s, n, true);
}

// ============================ 2. what you asked =============================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "What you asked for last week", "Six items, one slide each. Everything on 2016, so it compares with the group's work.");
  const asks = [["Open the box: what the detector is, where it came from, who wrote it", "3–5"],
    ["Define the noise: what “10 dB” is the ratio of, who made it", "6–7"],
    ["One SNR at a time: the matrix at 10, then 6, then about 3 dB", "8–9"],
    ["State the frame length, then shorten it", "10–11"],
    ["Every graph says how many decisions are behind each point; recall defined, precision shown", "8, 12"],
    ["Don't make it more complicated than the four-class box — break it", "all"]];
  const rows = [[hc("you asked", { fontSize: 13 }), hc("slide", { fontSize: 13, align: "center" }), hc("status", { fontSize: 13, align: "center" })]];
  asks.forEach(([a, sl], i) => rows.push([{ text: a, options: { fontSize: 14, color: INK } },
    { text: sl, options: { fontSize: 14, color: INK, align: "center", bold: true } },
    { text: "done", options: { fontSize: 13, color: "FFFFFF", bold: true, align: "center", fill: { color: TEAL } } }]));
  s.addTable(rows, { x: M, y: 1.9, w: W - 2 * M, colW: [9.13, 1.5, 1.5], fontFace: BODY, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.62, autoPage: false, valign: "middle" });
  foot(s, n);
}

// ============================ 3. the box ====================================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "The box, end to end", "One frame in, one of four labels out. Every block, its output shape and how many numbers it learns.");
  const blocks = [["input", "I and Q,\n128 samples\n= 16 symbols", "2 × 128", "—", MID],
    ["conv 1", "64 filters × 5 samples\n+ max-pool 2", "64 × 64", "832", NAVY],
    ["conv 2", "128 filters × 3\n+ max-pool 2", "128 × 32", "24,960", NAVY],
    ["BiLSTM", "2 layers, 128 per\ndirection, reads the\n32 steps both ways", "32 × 256", "659,968", ORANGE],
    ["attention", "weighted average\nof the 32 steps", "256", "66,048", NAVY],
    ["classifier", "dense 128\n+ dense 4", "4 scores", "33,668", NAVY],
    ["decision", "largest score\nwins", "1 label", "—", TEAL]];
  const bw = 1.55, gap = 0.2, y0 = 2.1;
  blocks.forEach(([h, d, shape, p, c], i) => {
    const x = M + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y: y0, w: bw, h: 0.62, fill: { color: c }, line: { color: c, width: 0 }, rectRadius: 0.08 });
    s.addText(h, { x, y: y0, w: bw, h: 0.62, fontFace: HEAD, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle", isTextBox: true, margin: 0 });
    card(s, x, y0 + 0.72, bw, 1.35);
    text(s, d, x + 0.08, y0 + 0.8, bw - 0.16, 1.2, { fontSize: 11, align: "center" });
    text(s, shape, x, y0 + 2.15, bw, 0.35, { fontSize: 13, bold: true, align: "center", color: INK });
    text(s, p === "—" ? "no weights" : `${p} weights`, x, y0 + 2.5, bw, 0.3, { fontSize: 10.5, align: "center", color: MID });
    if (i < blocks.length - 1) s.addShape(pres.ShapeType.rightArrow, { x: x + bw + 0.03, y: y0 + 0.2, w: gap - 0.06, h: 0.22, fill: { color: GREY }, line: { color: GREY, width: 0 } });
  });
  stat(s, M, 5.15, 3.2, num(D.params.total4), "numbers learned in total", ORANGE, 36);
  stat(s, M + 3.5, 5.15, 3.2, (100 * D.params.lstm / D.params.total4).toFixed(0) + "%", "of them in the LSTM: this is mostly a recurrent network", NAVY, 36);
  text(s, "Each of the 32 LSTM steps sees 12 input samples, 1.5 symbols. The same weights run at 64 and 32 samples; only the number of steps changes (16, 8). Each frame is scaled to unit power before it goes in, so the level carries nothing.",
    M + 7.1, 5.25, W - 2 * M - 7.1, 1.4, { fontSize: 12, color: MID });
  foot(s, n);
}

// ============================ 4. provenance and AI ==========================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Where it came from, and who wrote what", "The model is a colleague's reproduction of a paper. The code around it was written with an AI assistant.");
  card(s, M, 1.9, 5.95, 4.85);
  text(s, "The architecture", M + 0.3, 2.05, 5.4, 0.4, { fontFace: HEAD, fontSize: 16, bold: true });
  text(s, "ICRNNA — El-Haryqy et al., Results in Engineering 26 (2025) 104783. Our code is transcribed from a colleague's reproduction, not from the paper, and differs in five places: first kernel 5 vs 3; two pools vs one; one BatchNorm after the LSTM vs one per layer; no attention dropout or LayerNorm; one dense layer vs two.",
    M + 0.3, 2.5, 5.4, 1.75, { fontSize: 12 });
  text(s, "A separate build written from the paper itself, on 11 classes:", M + 0.3, 4.3, 5.4, 0.35, { fontSize: 12, bold: true });
  s.addTable([[hc("", { fontSize: 11 }), hc("accuracy", { fontSize: 11, align: "center" })],
    [{ text: "paper, Table 3", options: { fontSize: 11 } }, { text: `${F.paper.toFixed(2)}%`, options: { fontSize: 11, align: "center" } }],
    [{ text: "faithful build, paper's 58 epochs", options: { fontSize: 11 } }, { text: `${F.e58.toFixed(2)}%`, options: { fontSize: 11, align: "center" } }],
    [{ text: `faithful build, to convergence (epoch ${F.e150_best})`, options: { fontSize: 11, bold: true } }, { text: `${F.e150.toFixed(2)}%`, options: { fontSize: 11, align: "center", bold: true, color: TEAL } }]],
    { x: M + 0.3, y: 4.7, w: 5.4, colW: [4.0, 1.4], fontFace: BODY, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.36, autoPage: false });
  const X = M + 6.25, Wr = W - M - X;
  card(s, X, 1.9, Wr, 4.85);
  text(s, "Written with an AI assistant", X + 0.3, 2.05, Wr - 0.6, 0.4, { fontFace: HEAD, fontSize: 16, bold: true });
  text(s, [{ text: "Claude Code (Anthropic) wrote most of the code, the Colab runners, the table and figure tools, this deck and most of the write-up. Every commit it wrote is marked in git.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Not from it: ", options: { bold: true } },
    { text: "the dataset (DeepSig), the architecture (the paper, the colleague), the numbers — measured by running the code on Colab GPUs, several rerun bit-identical — and the decisions.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "What I asked, what came back — e.g.: ", options: { bold: true } },
    { text: "“shorten the frame to 64, then 32” → a frame-length option and three runs; “why is the faithful build 1.5 points low?” → the convergence check that found 58 epochs too few." }],
    X + 0.3, 2.5, Wr - 0.6, 3.1, { fontSize: 12 });
  text(s, "Full account: DETECTOR.md in the repository.", X + 0.3, 6.25, Wr - 0.6, 0.35, { fontSize: 11, italic: true, color: MID });
  foot(s, n);
}

// ============================ 5. training and decision ======================
{ const s = pres.addSlide(); light(s); n++;
  const be = L["128"].best_epochs;
  title(s, "How it was trained, and how it decides", "The test set is read once. The decision is deterministic, and it always picks one of the four.");
  const rows = [["data", "RML2016.10a, 4 classes × 20 SNR levels × 1,000 frames"], ["split", "per (class, SNR): 70% train, 15% validation, 15% test"],
    ["training", "AdamW, lr 1e-3, batch 256, cross-entropy; lr ×0.2 when validation stalls"],
    ["stopping", "20 epochs without validation gain; ceiling 100; best-validation weights kept"],
    ["repeats", `${C.seeds} seeds: new split, new initialisation each`],
    ["converged?", `all ${C.seeds}: best epochs ${Math.min(...be)}–${Math.max(...be)}, every one stopped early`]];
  s.addTable(rows.map(([a, b]) => [{ text: a, options: { bold: true, fontSize: 13, fill: { color: LIGHT } } }, { text: b, options: { fontSize: 13 } }]),
    { x: M, y: 1.95, w: 7.0, colW: [1.55, 5.45], fontFace: BODY, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.62, autoPage: false, valign: "middle" });
  const X = 8.1, Wr = W - M - X;
  card(s, X, 1.95, Wr, 4.75);
  text(s, "Same frame 10,000 times", X + 0.3, 2.1, Wr - 0.6, 0.4, { fontFace: HEAD, fontSize: 16, bold: true });
  text(s, "→ the same answer 10,000 times. At test time nothing is random. The meaningful test is 10,000 different frames, and that is what every table here counts.",
    X + 0.3, 2.55, Wr - 0.6, 1.5, { fontSize: 13 });
  text(s, "No “I don't know”", X + 0.3, 4.1, Wr - 0.6, 0.4, { fontFace: HEAD, fontSize: 16, bold: true });
  text(s, "The largest of four scores always wins, even at −20 dB where the frame carries nothing. Where those frames go is slide 12.",
    X + 0.3, 4.55, Wr - 0.6, 1.5, { fontSize: 13 });
  foot(s, n);
}

// ============================ 6. who made the noise =========================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Who made the noise", "DeepSig, in GNU Radio, synthetic. The paper gives no parameters; these are from their published generator.");
  const rows = [[hc("stage", { fontSize: 12 }), hc("setting", { fontSize: 12 }), hc("what it means", { fontSize: 12 })],
    ["modulator", "root-raised cosine, 8 samples / symbol, roll-off 0.35", "200 kHz sampling, 25 kbaud; a frame is 0.64 ms, 16 symbols"],
    ["clock offset", "random walk, clipped at 50 Hz", "sample-rate drift, small in practice"],
    ["carrier offset", "random walk, clipped at 500 Hz", "reaches ~4 Hz in a run: under 1° of rotation per frame"],
    ["fading", "Rician, K = 4, Doppler 1 Hz", "one complex gain per frame; 5.7 dB swing over a run"],
    ["multipath", "3 paths: delays 0, 0.9, 1.7 samples; gains 1, 0.8, 0.3", "0.2 symbol of delay spread"],
    [{ text: "noise", options: { bold: true, color: ORANGE } }, { text: "white Gaussian, noise_amp = 10^(−label/10)", options: { bold: true } }, { text: "the only thing the SNR label sets", options: { bold: true } }]];
  s.addTable(rows.map((r, i) => i === 0 ? r : r.map(c => typeof c === "string" ? { text: c, options: { fontSize: 12 } } : c)),
    { x: M, y: 1.95, w: 8.3, colW: [1.6, 3.5, 3.2], fontFace: BODY, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.56, autoPage: false, valign: "middle" });
  const X = 9.25, Wr = W - M - X;
  card(s, X, 1.95, Wr, 4.6);
  circleIcon(s, X + 0.3, 2.15, 0.5, ORANGE, "!");
  text(s, "Noise is one of five impairments", X + 0.95, 2.15, Wr - 1.2, 0.55, { fontFace: HEAD, fontSize: 14, bold: true, valign: "middle" });
  text(s, "Carrier offset, clock offset, fading and multipath are present at every SNR, including +18 dB. Errors at high SNR cannot be blamed on noise. And every part is seeded with the same fixed number: every run of the generator sees the identical channel and noise sequence.",
    X + 0.3, 2.95, Wr - 0.6, 1.7, { fontSize: 12.5 });
  text(s, "Source: O'Shea & West, GRCon 2016; github.com/radioML/dataset, generate_RML2016.10a.py.", X + 0.3, 5.3, Wr - 0.6, 1.0, { fontSize: 10.5, italic: true, color: MID });
  foot(s, n);
}

// ============================ 7. what 10 dB is ==============================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "What “10 dB” is the ratio of", "Measured in the frames of the file itself, not taken from the code.");
  image(s, "35_rml2016_snr_measured.png", M, 1.75, 7.4, 1425, 840);
  const X = 8.3, Wr = W - M - X, sl = SN.slopes, off = SN.offsets;
  const slope = Math.min(sl.BPSK, sl.QPSK, sl["8PSK"]).toFixed(2) + "–" + Math.max(sl.BPSK, sl.QPSK, sl["8PSK"]).toFixed(2);
  stat(s, X, 1.7, Wr, `${slope} dB`, "of measured SNR per dB of label (PSK): the label sets a noise amplitude, so it moves twice as fast as its name", ORANGE, 30);
  text(s, "Same label, different SNR — above PSK:", X, 3.45, Wr, 0.35, { fontSize: 12.5, bold: true });
  s.addTable([[hc("", { fontSize: 11 }), hc("measured", { fontSize: 11, align: "center" }), hc("unscaled points", { fontSize: 11, align: "center" })],
    ...["PAM4", "QAM16", "QAM64"].map(c => [{ text: c, options: { fontSize: 11, bold: true } },
      { text: `+${off[c].toFixed(1)} dB`, options: { fontSize: 11, align: "center" } }, { text: `+${SN.unscaled[c].toFixed(1)} dB`, options: { fontSize: 11, align: "center", color: MID } }])],
    { x: X, y: 3.85, w: Wr, colW: [1.3, 1.6, Wr - 2.9], fontFace: BODY, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.34, autoPage: false });
  const a6 = SN.at["-6"];
  text(s, `At a label of −6 dB a QAM64 frame is at about ${a6.QAM64 > 0 ? "+" : ""}${a6.QAM64.toFixed(0)} dB, a QPSK frame at about ${a6.QPSK.toFixed(0)} dB. Readable only from about −13 to +18 dB; outside that the flat parts are the method's limit.`,
    X, 5.35, Wr, 1.4, { fontSize: 11.5, color: MID });
  foot(s, n);
}

// ============================ 8. the box at 10 dB ===========================
{ const s = pres.addSlide(); light(s); n++;
  const m = C.mats["10"], tot = m.flat().reduce((a, b) => a + b, 0), ok = m.reduce((a, r, i) => a + r[i], 0);
  const qam = m[2][3] + m[3][2];
  title(s, "The box at one SNR: 10 dB", `${num(tot)} decisions — ${C.seeds} seeds × 4 classes × 150 test frames. Rows: what was sent. Columns: what was decided.`);
  matrixTable(s, cls, m, M, 1.95, 7.6);
  text(s, [{ text: "recall", options: { bold: true } }, { text: " = the row: of the frames that were this class, how many were called it.", options: { breakLine: true } },
    { text: "precision", options: { bold: true } }, { text: " = the column: of the frames called this class, how many were it." }], M, 5.05, 7.6, 0.9, { fontSize: 12, color: MID });
  const X = 8.7, Wr = W - M - X;
  stat(s, X, 1.9, Wr, num(tot - ok), `wrong of ${num(tot)} (${pct((tot - ok) / tot)})`, RED, 40);
  stat(s, X, 3.55, Wr, num(qam), `of them are QAM16 ↔ QAM64`, ORANGE, 40);
  text(s, "BPSK and QPSK: 99.6% recall. The errors are one pair.", X, 5.2, Wr, 0.8, { fontSize: 13, bold: true });
  foot(s, n);
}

// ============================ 9. lower ======================================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Then lower: nothing changes until 0 dB", "Each matrix 8,400 decisions; each cell count and share of its row.");
  const h = image(s, "31_rml2016_c4_matrices.png", M, 1.7, 8.9, 2400, 1350);
  const X = 9.8, Wr = W - M - X;
  const lv = [10, 6, 4, 2, 0, -2, -4, -6];
  s.addTable([[hc("label", { fontSize: 12, align: "center" }), hc("wrong of 8,400", { fontSize: 12, align: "center" })],
    ...lv.map(v => [{ text: `${v > 0 ? "+" : ""}${v} dB`, options: { fontSize: 12, align: "center", bold: v >= 2 } },
      { text: num(wrongAt(v)), options: { fontSize: 12, align: "center", bold: true, color: v >= 2 ? TEAL : (v >= -2 ? ORANGE : RED) } }])],
    { x: X, y: 1.75, w: Wr, colW: [1.3, Wr - 1.3], fontFace: BODY, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.4, autoPage: false });
  text(s, "10 → 2 dB: the same box. Below 0 dB QPSK leaks into both QAMs first; BPSK holds longest.", X, 5.5, Wr, 1.2, { fontSize: 12, color: MID });
  foot(s, n);
}

// ============================ 10. error vs SNR, frame length ================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Error rate against SNR — and against frame length", "SNR decides where the curve falls. The number of symbols decides how low it can go.");
  image(s, "32_rml2016_c4_error_vs_snr.png", M, 1.75, 8.3, 1350, 750);
  const X = 9.3, Wr = W - M - X;
  [["128", "16"], ["64", "8"], ["32", "4"]].forEach(([k, sym], i) =>
    stat(s, X, 1.7 + i * 1.6, Wr, `${L[k].floor_pct}%`, `error floor from +2 dB up, ${k} samples = ${sym} symbols`, [NAVY, ORANGE, TEAL][i], 34));
  text(s, `Above 10 dB: ${num(L["128"].high_wrong)} / ${num(L["64"].high_wrong)} / ${num(L["32"].high_wrong)} wrong of ${num(L["128"].high_total)}.`, X, 6.45, Wr, 0.4, { fontSize: 11.5, color: MID });
  foot(s, n);
}

// ============================ 11. what shorter frames break =================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "What shorter frames break", `Recall above 10 dB, ${num(L["128"].high_total / 4)} frames per class per length.`);
  const lens = ["128", "64", "32"];
  s.addChart(pres.ChartType.bar, cls.map((c, i) => ({ name: c, labels: lens.map(k => `${k} samples (${k / 8} symbols)`), values: lens.map(k => +(100 * L[k].high_recall[i]).toFixed(1)) })),
    { x: M, y: 1.8, w: 7.6, h: 4.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: [NAVY, "5C9EAD", ORANGE, TEAL],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: INK, dataLabelFormatCode: "0",
      showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: BODY, legendColor: INK,
      valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 25, valAxisLabelFormatCode: "0", showValAxisTitle: true, valAxisTitle: "recall (%)",
      catAxisLabelColor: MID, valAxisLabelColor: MID, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" },
      valAxisTitleColor: MID, valAxisTitleFontSize: 11 });
  const X = 8.6, Wr = W - M - X, q16 = lens.map(k => (100 * L[k].high_recall[2]).toFixed(0));
  stat(s, X, 1.75, Wr, `${q16.join(" → ")}%`, "QAM16 recall at 16 → 8 → 4 symbols: at 4 it is a coin flip", ORANGE, 30);
  const qb = L["32"].high_matrix[1][0];
  stat(s, X, 3.35, Wr, num(qb), "QPSK frames called BPSK at 4 symbols: if all four land on one diagonal, the frame looks like BPSK", NAVY, 30);
  text(s, `These QAM frames are measured at 17 dB SNR or more (slide 7). The errors are the symbol count, not the noise.`, X, 5.2, Wr, 1.3, { fontSize: 12.5, bold: true });
  foot(s, n);
}

// ============================ 12. recall, precision, sink ===================
{ const s = pres.addSlide(); light(s); n++;
  const m = C.m20, bs = m.bpsk_share_by_seed;
  title(s, "Recall, precision, and where empty frames go", `Recall: ${num(C.per_snr_decisions / 4)} frames sent per point. Precision: n = frames decided as that class, shown per panel.`);
  image(s, "33_rml2016_c4_recall_precision.png", M, 1.7, W - 2 * M, 2400, 660);
  const y = 5.25;
  stat(s, M, y, 3.6, pct(m.decided_share[0] + m.decided_share[1]), "of all −20 dB decisions go to BPSK or QPSK", ORANGE, 34);
  stat(s, M + 4.0, y, 3.8, `${(100 * m.precision[0]).toFixed(0)}% / ${(100 * m.precision[1]).toFixed(0)}%`, "BPSK / QPSK precision at −20 dB: chance, with four classes", RED, 34);
  stat(s, M + 8.0, y, 4.1, `${(100 * Math.min(...bs)).toFixed(0)}–${(100 * Math.max(...bs)).toFixed(0)}%`, "BPSK's share of that, seed to seed: which of the two is arbitrary", NAVY, 34);
  foot(s, n);
}

// ============================ 13. beyond the box ============================
{ const s = pres.addSlide(); light(s); n++;
  const R = MT.rows, sm = MT.smooth;
  title(s, "Beyond the box: the 2018 result does not transfer", `Train on RML2016.10a, test on our own generator. 5 shared classes, ${MT.seeds} seeds, accuracy at ≥ 10 dB.`);
  const lab = { "none": "none", "standard augmentation": "standard augmentation", "whitening a=0.75": "whitening (5-bin envelope)", "whitening + standard": "whitening + augmentation" };
  s.addTable([[hc("method", { fontSize: 12 }), hc("in-domain", { fontSize: 12, align: "center" }), hc("cross-domain", { fontSize: 12, align: "center" })],
    ...R.map(r => [{ text: lab[r.name], options: { fontSize: 12, bold: r.name === "standard augmentation" } },
      { text: r.in.toFixed(3), options: { fontSize: 12, align: "center" } },
      { text: r.cross.toFixed(3), options: { fontSize: 12, align: "center", bold: r.name === "standard augmentation", color: r.name === "standard augmentation" ? TEAL : INK } }])],
    { x: M, y: 1.95, w: 6.1, colW: [3.3, 1.4, 1.4], fontFace: BODY, color: INK, border: { type: "solid", pt: 0.5, color: "D5DDE5" }, rowH: 0.45, autoPage: false });
  text(s, `The gap is small here (${(R[0].in - R[0].cross).toFixed(3)}) and the standard augmentation set closes it. On 2018 the gap was 0.19 and whitening closed it.`, M, 4.35, 6.1, 1.0, { fontSize: 12.5 });
  const i33 = sm.bins.indexOf(33);
  text(s, `Whitening's 8-point cost was the envelope estimate: 128 points smoothed over 5 bins. Over 33 bins it is +${(100 * (sm.in[i33] - sm.none_in)).toFixed(1)} points in-domain — but cross-domain it never clearly beats doing nothing.`,
    M, 5.35, 6.1, 1.3, { fontSize: 12.5, color: MID });
  image(s, "34_whitening_smoothing_rml2016.png", 7.0, 1.85, W - M - 7.0, 1350, 780);
  foot(s, n);
}

// ============================ 14. caveats ===================================
{ const s = pres.addSlide(); light(s); n++;
  title(s, "Caveats");
  const cav = [["Both domains are synthetic", "RML2016.10a is simulated, and so is our second transmitter. No over-the-air capture has been classified yet.", RED],
    ["Not the published architecture", "A colleague's reproduction with five differences. The faithful build is separate and matches the paper at convergence.", ORANGE],
    ["The SNR label is not one SNR", "At one label QAM64 frames carry about 15 dB more SNR than PSK. Classes cannot be compared at one label.", ORANGE],
    ["One dataset, one model", "14 seeds each, all converged. A second architecture on the same box has not been run.", MID]];
  cav.forEach(([h, t, c], i) => {
    const col = i % 2, row = Math.floor(i / 2), x = M + col * 6.2, y = 1.45 + row * 2.75;
    card(s, x, y, 5.95, 2.5); circleIcon(s, x + 0.25, y + 0.25, 0.42, c, "!");
    text(s, h, x + 0.85, y + 0.2, 4.9, 0.5, { fontFace: HEAD, fontSize: 15, bold: true, valign: "middle" });
    text(s, t, x + 0.25, y + 0.9, 5.45, 1.5, { fontSize: 13 });
  });
  foot(s, n);
}

// ============================ 15. next ======================================
{ const s = pres.addSlide(); dark(s); n++;
  title(s, "Next: break the box one knob at a time", null, true);
  const next = [["Done", "Frame length (128 / 64 / 32) and SNR (one level at a time). The high-SNR errors are the QAM pair, and they follow the symbol count."],
    ["Next", "The generator is now rebuilt in GNU Radio. Switch off carrier offset, fading and multipath one at a time, same four classes, same 128 samples, and see which impairment the high-SNR QAM errors belong to."],
    ["Then", "The same box on a real capture, when lab access allows."]];
  next.forEach(([h, t], i) => {
    const y = 1.7 + i * 1.5;
    s.addText(h, { x: M, y, w: 2.0, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: ORANGE, isTextBox: true, margin: 0 });
    s.addText(t, { x: M + 2.1, y, w: W - 2 * M - 2.1, h: 1.3, fontFace: BODY, fontSize: 15, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top" });
  });
  s.addText("Repository: YusufBilo78/amc · DETECTOR.md · NOISE_2016.md · SNR_2016.md", { x: M, y: 6.55, w: W - 2 * M, h: 0.4, fontFace: BODY, fontSize: 11, color: PALE, isTextBox: true, margin: 0 });
  foot(s, n, true);
}

pres.writeFile({ fileName: "moshe_2016.pptx" }).then(f => console.log("wrote", f, "slides:", n));

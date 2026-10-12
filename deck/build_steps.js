// What we did, step by step: answers to Moshe's questions from the last
// meeting. No new experiments; every number is read from steps_data.json
// (deck/extract_steps.py), which reads the committed result files.
// Same style as the other decks: large type, one message per slide.
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("steps_data.json", "utf8"));
const TALK = require("./talk_steps.js");
const C4 = D.c4, C11 = D.c11, Q = D.qam16_10db;

const NAVY = "0F1B2D", ORANGE = "F08A24", CARD = "EEF2F6", INK = "1E293B", MUTED = "5B6B7F", WHITE = "FFFFFF", LINE = "D5DDE5", SOFT = "C9D3E0", PALE = "FCE3C8";
const F = "Calibri";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Yusuf Bilal Cetinkaya";
pres.title = "What we did, step by step";
const W = 13.33, M = 0.6;
const num = v => v.toLocaleString("en-US"), pct = (a, b) => (100 * a / b).toFixed(1) + "%";
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
function badge(s, x, y, k, fill = NAVY, d = 0.6) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  t(s, String(k), x, y, d, d, { fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
}
function arrow(s, x, y, w = 0.5, dir = "right") {
  s.addShape(dir === "down" ? pres.ShapeType.downArrow : pres.ShapeType.rightArrow,
    { x, y, w: dir === "down" ? 0.45 : w, h: dir === "down" ? w : 0.45, fill: { color: ORANGE }, line: { color: ORANGE } });
}
function big(s, x, y, w, value, label, color = INK, size = 56, align = "left") {
  t(s, value, x, y, w, 1.0, { fontSize: size, bold: true, color, align, valign: "bottom" });
  t(s, label, x, y + 1.05, w, 0.8, { fontSize: 19, color: MUTED, align });
}
function done(s, dark = false) {
  n++;
  if (TALK[n]) s.addNotes(TALK[n]);
  t(s, String(n), W - M - 0.5, 7.0, 0.5, 0.3, { fontSize: 12, color: dark ? "8FA0B5" : MUTED, align: "right" });
}
const QS = ["Is it a neural network? Trained, or fixed weights?", "Is there a separate piece for each modulation?",
  "Why are the four-signal mistakes only among the four?", "How exactly is it trained?", "Which software did we use?", "Which paper, and who wrote it?"];

// 1. title =========================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "ANSWERS TO LAST WEEK'S QUESTIONS", M, 1.6, 10, 0.4, { fontSize: 16, bold: true, color: ORANGE, charSpacing: 4 });
  t(s, "What we did, step by step", M, 2.05, 12, 1.2, { fontSize: 56, bold: true, color: WHITE });
  t(s, "No new experiments: only what was done, explained completely.", M, 3.35, 12, 0.6, { fontSize: 24, color: SOFT });
  t(s, "Yusuf Bilal Cetinkaya  ·  Moshe Kam's research group  ·  NJIT  ·  October 2026", M, 6.5, 10, 0.4, { fontSize: 16, color: "9AA8BA" });
  done(s, true);
}

// 2. the questions ====================================================================
{ const s = pres.addSlide();
  title(s, "Your questions");
  QS.forEach((q, i) => {
    const x = M + (i % 2) * 6.15, y = 1.75 + Math.floor(i / 2) * 1.45;
    card(s, x, y, 5.9, 1.2);
    badge(s, x + 0.25, y + 0.3, i + 1, i === 2 ? ORANGE : NAVY);
    t(s, q, x + 1.05, y, 4.7, 1.2, { fontSize: 21, valign: "middle", bold: i === 2 });
  });
  done(s);
}

// 3. corrections ======================================================================
{ const s = pres.addSlide();
  title(s, "Three answers I got wrong last week");
  const rows = [["“The four-signal table used the same 11-signal detector.”", "It used a separate network, trained only on those four."],
    ["“They published their weights.”", "No code, no weights. We trained every network ourselves."],
    ["“There is a piece of code for each modulation.”", "One network. A modulation is just an output number."]];
  rows.forEach(([a, b], i) => {
    const y = 1.65 + i * 1.45;
    card(s, M, y, W - 2 * M, 1.25);
    t(s, "said", M + 0.3, y + 0.12, 1.0, 0.4, { fontSize: 14, bold: true, color: MUTED, charSpacing: 2 });
    t(s, a, M + 0.3, y + 0.4, 5.6, 0.8, { fontSize: 19, italic: true, color: MUTED });
    arrow(s, M + 6.05, y + 0.42, 0.5);
    t(s, "true", M + 6.8, y + 0.12, 1.0, 0.4, { fontSize: 14, bold: true, color: ORANGE, charSpacing: 2 });
    t(s, b, M + 6.8, y + 0.4, W - 2 * M - 7.0, 0.8, { fontSize: 20, bold: true });
  });
  takeaway(s, "The rest of the talk shows each step.", 6.2);
  done(s);
}

// 4. Q1 trained ========================================================================
{ const s = pres.addSlide();
  title(s, "It is a neural network, and we trained it", "QUESTION 1");
  const steps = [[num(C4.params), "adjustable numbers,\nrandom at the start"], [num(C4.train), "labelled frames to\nlearn from"], ["frozen", "then tested on frames\nit never saw"]];
  steps.forEach(([v, l], i) => {
    const x = M + i * 4.15, w = 3.55;
    card(s, x, 1.75, w, 3.1, i === 2 ? NAVY : CARD);
    t(s, v, x + 0.3, 2.0, w - 0.6, 1.0, { fontSize: 46, bold: true, color: i === 2 ? WHITE : (i === 0 ? ORANGE : INK), valign: "bottom" });
    t(s, l, x + 0.3, 3.15, w - 0.6, 1.4, { fontSize: 20, color: i === 2 ? SOFT : MUTED });
    if (i < 2) arrow(s, x + w + 0.03, 3.05, 0.5);
  });
  takeaway(s, "Trained by us, from a random start. Nobody else's weights.", 5.4);
  done(s);
}

// 5. Q2 one network ======================================================================
{ const s = pres.addSlide();
  title(s, "One network for all four", "QUESTION 2");
  card(s, M, 2.3, 2.3, 1.6);
  t(s, "one frame\n2 × 128", M, 2.3, 2.3, 1.6, { fontSize: 22, bold: true, align: "center", valign: "middle" });
  arrow(s, M + 2.45, 2.85, 0.55);
  s.addShape(pres.ShapeType.roundRect, { x: M + 3.2, y: 1.8, w: 3.6, h: 2.6, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.12 });
  t(s, "the same layers\nfor every frame", M + 3.2, 1.8, 3.6, 2.6, { fontSize: 24, bold: true, color: WHITE, align: "center", valign: "middle" });
  arrow(s, M + 6.95, 2.85, 0.55);
  const outs = [["0", "BPSK", 0.15], ["1", "QPSK", 0.92], ["2", "QAM16", 0.3], ["3", "QAM64", 0.1]];
  outs.forEach(([k, name, v], i) => {
    const y = 1.75 + i * 0.68, x = M + 7.7;
    t(s, `output ${k}`, x, y, 1.45, 0.55, { fontSize: 17, color: MUTED, valign: "middle" });
    t(s, name, x + 1.5, y, 1.3, 0.55, { fontSize: 19, bold: i === 1, valign: "middle" });
    s.addShape(pres.ShapeType.rect, { x: x + 2.85, y: y + 0.12, w: 1.6 * v + 0.05, h: 0.32, fill: { color: i === 1 ? ORANGE : "B8C4D2" }, line: { color: i === 1 ? ORANGE : "B8C4D2" } });
  });
  t(s, "largest score wins → QPSK", M + 7.7, 4.55, 4.5, 0.5, { fontSize: 19, bold: true, color: ORANGE });
  takeaway(s, "No separate piece per modulation: a modulation is only an output number.", 5.6);
  done(s);
}

// 6. Q3 two approaches ======================================================================
{ const s = pres.addSlide();
  title(s, "Why the mistakes stayed among the four", "QUESTION 3");
  const cols = [["A", "Train a network on the four only", "4 outputs", "mistakes can only fall among the four", "what the table showed", ORANGE],
    ["B", "Use the 11-signal network, feed it the four", "11 outputs", "mistakes can go to any of the eleven", "next slide", NAVY]];
  cols.forEach(([k, h, o, b, tag, c], i) => {
    const x = M + i * 6.15, w = 5.9;
    card(s, x, 1.65, w, 3.85);
    badge(s, x + 0.3, 1.9, k, c, 0.7);
    t(s, h, x + 1.2, 1.85, w - 1.5, 0.85, { fontSize: 22, bold: true, valign: "middle" });
    t(s, o, x + 0.3, 2.95, w - 0.6, 0.9, { fontSize: 44, bold: true, color: c });
    t(s, b, x + 0.3, 3.9, w - 0.6, 0.8, { fontSize: 20, color: MUTED });
    t(s, tag, x + 0.3, 4.75, w - 0.6, 0.5, { fontSize: 18, bold: true, color: c, charSpacing: 1 });
  });
  takeaway(s, "The table was A: with 4 outputs it cannot answer anything else.", 5.85);
  done(s);
}

// 7. approach B from existing results ===========================================================
{ const s = pres.addSlide();
  title(s, "Approach B, from results we already had", "QUESTION 3");
  t(s, `The 11-signal network at 10 dB: ${C11.per_row} test frames per signal (${C11.seeds} runs × 150). No new training.`, M, 1.35, W - 2 * M, 0.5, { fontSize: 19, color: MUTED });
  const hc = (x, o = {}) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, align: "center", fontSize: 18, ...o } });
  const rows = [[hc("sent \\ said", { fontSize: 15 }), ...C4.classes.map(c => hc(c)), hc("other 7", { color: ORANGE })]];
  C11.rows.forEach((r, i) => {
    const out = Object.values(r.outside).reduce((a, b) => a + b, 0);
    const where = Object.entries(r.outside).map(([k, v]) => `${k} ${v}`).join(", ");
    rows.push([{ text: r.name, options: { bold: true, fontSize: 18 } },
      ...r.inside.map((v, j) => ({ text: num(v), options: { align: "center", fontSize: 20, bold: i === j, color: i === j ? WHITE : (v >= 10 ? INK : "8A99A8"),
        fill: { color: i === j ? NAVY : (v >= 10 ? PALE : WHITE) } } })),
      { text: `${out}  (${where})`, options: { align: "left", fontSize: 15, bold: true, color: ORANGE } }]);
  });
  s.addTable(rows, { x: M, y: 1.95, w: W - 2 * M, colW: [1.7, 1.35, 1.35, 1.35, 1.35, W - 2 * M - 7.1], fontFace: F, color: INK,
    border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.62, autoPage: false, valign: "middle" });
  takeaway(s, `${C11.left} of its ${C11.errors} mistakes leave the four, mostly to AM-SSB.`, 5.6);
  done(s);
}

// 8. Q4 data ==========================================================================
{ const s = pres.addSlide();
  title(s, "Training, step 1: the data", "QUESTION 4");
  const steps = [["Read the file", `4 signals × 20 levels × 1,000 = ${num(C4.frames)} frames`], ["Scale", "every frame to the same power"],
    ["Label", "BPSK 0, QPSK 1, QAM16 2, QAM64 3"], ["Split every cell", "700 learn · 150 stop · 150 test"]];
  steps.forEach(([h, b], i) => {
    const y = 1.6 + i * 0.95;
    badge(s, M, y + 0.05, i + 1, NAVY);
    t(s, h, M + 0.85, y, 3.1, 0.7, { fontSize: 23, bold: true, valign: "middle" });
    t(s, b, M + 4.0, y, 8.2, 0.7, { fontSize: 22, color: MUTED, valign: "middle" });
  });
  const segs = [[0.70, "learn  " + num(C4.train), NAVY], [0.15, "stop  " + num(C4.val), "8FA0B5"], [0.15, "test  " + num(C4.test), ORANGE]];
  let x = M; const total = W - 2 * M;
  segs.forEach(([f, l, c]) => {
    s.addShape(pres.ShapeType.rect, { x, y: 5.5, w: total * f - 0.04, h: 0.6, fill: { color: c }, line: { color: c } });
    t(s, l, x, 5.5, total * f - 0.04, 0.6, { fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle" });
    x += total * f;
  });
  takeaway(s, `The ${num(C4.test)} test frames are never used for learning.`, 6.3);
  done(s);
}

// 9. Q4 the loop ========================================================================
{ const s = pres.addSlide();
  title(s, "Training, step 2: the learning loop", "QUESTION 4");
  const loop = [["Take 256 frames", ""], ["Network gives\n4 scores each", ""], ["Compare with the\nright labels", "→ one number: the loss"], ["Nudge all\nthe numbers", "to lower the loss"]];
  loop.forEach(([h, sub], i) => {
    const x = M + i * 3.1, w = 2.7;
    card(s, x, 1.65, w, 2.0, i === 3 ? NAVY : CARD);
    t(s, h, x + 0.2, 1.75, w - 0.4, 1.1, { fontSize: 21, bold: true, align: "center", valign: "middle", color: i === 3 ? WHITE : INK });
    t(s, sub, x + 0.2, 2.9, w - 0.4, 0.6, { fontSize: 16, align: "center", color: i === 3 ? SOFT : MUTED });
    if (i < 3) arrow(s, x + w + 0.02, 2.45, 0.36);
  });
  t(s, "↻ repeat", M + 9.3, 3.75, 2.7, 0.5, { fontSize: 20, bold: true, color: ORANGE, align: "center" });
  const facts = [["219 steps", `= one pass over the ${num(C4.train)} learning frames: one epoch`],
    ["after each epoch", `check accuracy on the ${num(C4.val)} stopping frames; keep the best version`],
    ["20 epochs", "with no improvement: stop"]];
  facts.forEach(([a, b], i) => {
    const y = 4.3 + i * 0.6;
    t(s, a, M, y, 2.9, 0.55, { fontSize: 21, bold: true, color: ORANGE, valign: "middle" });
    t(s, b, M + 3.0, y, 9.1, 0.55, { fontSize: 20, valign: "middle" });
  });
  const be = C4.best_epochs, lo = Math.min(...be), hi = Math.max(...be);
  takeaway(s, `Best version found at epoch ${lo}–${hi}; every run stopped by itself.`, 6.25);
  done(s);
}

// 10. Q4 test and repeat ===================================================================
{ const s = pres.addSlide();
  title(s, "Training, step 3: test once, then repeat", "QUESTION 4");
  [[num(C4.test), "test frames, used once,\nat the very end", NAVY], [String(C4.seeds), "repeats, each with a new\nsplit and a new random start", INK], ["8,400", "test decisions at 10 dB\n(14 × 4 × 150)", ORANGE]].forEach(([v, l, c], i) => {
    const x = M + i * 4.15;
    card(s, x, 1.7, 3.85, 3.0);
    big(s, x + 0.3, 1.95, 3.3, v, l, c, 60);
  });
  takeaway(s, "Every number you saw is counted on frames the network never learned from.", 5.3);
  done(s);
}

// 11. Q5 software ============================================================================
{ const s = pres.addSlide();
  title(s, "The software", "QUESTION 5");
  const hc = (x) => ({ text: x, options: { bold: true, fill: { color: CARD }, color: INK, fontSize: 18 } });
  const rows = [[hc("what"), hc("used for")],
    ["Python + PyTorch", "the network and its training"], ["NumPy, scikit-learn", "reading the data, splitting it"],
    ["Google Colab GPU, laptop GPU", "where it ran"], ["Our code: model_zoo.py, cnn.py, train_backbone.py", "the detector, the loop, the protocol"],
    ["Claude Code (AI assistant)", "wrote the code from my instructions"],
    ["GNU Radio", "not for training: DeepSig made the data with it; I used it only to study the noise"]]
    .map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { fontSize: 19, bold: j === 0 } })));
  s.addTable(rows, { x: M, y: 1.55, w: W - 2 * M, colW: [5.3, W - 2 * M - 5.3], fontFace: F, color: INK,
    border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.6, autoPage: false, valign: "middle" });
  takeaway(s, "No ready-made detector and no pre-trained weights.", 6.1);
  done(s);
}

// 12. Q6 the paper =============================================================================
{ const s = pres.addSlide();
  title(s, "The paper", "QUESTION 6");
  t(s, "Improved automatic modulation recognition using deep learning with additive attention", M, 1.5, W - 2 * M, 0.9, { fontSize: 24, bold: true });
  t(s, "El-Haryqy, Kharbouche, Ouamna, Madini, Zouine  ·  Ibn Tofail University, Kenitra, Morocco\nResults in Engineering 26 (2025) 104783  ·  online April 2025", M, 2.45, W - 2 * M, 0.9, { fontSize: 20, color: MUTED });
  card(s, M, 3.7, 5.9, 1.9); card(s, M + 6.15, 3.7, 5.9, 1.9, NAVY);
  t(s, "PUBLISHED", M + 0.3, 3.85, 5, 0.4, { fontSize: 15, bold: true, color: ORANGE, charSpacing: 2 });
  t(s, "the layers (Fig. 2) and the training recipe (Table 2)", M + 0.3, 4.3, 5.3, 1.1, { fontSize: 21 });
  t(s, "NOT PUBLISHED", M + 6.45, 3.85, 5, 0.4, { fontSize: 15, bold: true, color: ORANGE, charSpacing: 2 });
  t(s, "code and weights (“available on request”)", M + 6.45, 4.3, 5.3, 1.1, { fontSize: 21, color: WHITE });
  takeaway(s, "So the code had to be written from the description.", 6.0);
  done(s);
}

// 13. Q6 the authors ==============================================================================
{ const s = pres.addSlide();
  title(s, "What else the authors have done", "QUESTION 6");
  const works = [["2024", "Review: deep learning for spectrum sensing and prediction in cognitive radio", "Int. J. Comput. Appl."],
    ["2024", "Radio-frequency interference detection and modulation recognition with Mask R-CNN", "IJCNC"],
    ["2023", "Survey: automatic signal detection using deep learning", "ICOA conference"],
    ["2021", "Kharbouche: optical (DS-OCDMA/VLC) communication for vehicles", "ICOA conference"]];
  works.forEach(([y, w, v], i) => {
    const yy = 1.6 + i * 1.08;
    t(s, y, M, yy, 1.3, 0.8, { fontSize: 26, bold: true, color: ORANGE, valign: "middle" });
    t(s, [{ text: w, options: { breakLine: true } }, { text: v, options: { fontSize: 16, color: MUTED } }], M + 1.5, yy, W - 2 * M - 1.5, 0.95, { fontSize: 20, valign: "middle" });
  });
  t(s, "From the paper's own reference list; we did not search further.", M, 6.0, W - 2 * M, 0.4, { fontSize: 16, italic: true, color: MUTED });
  done(s);
}

// 14. detection terms ===============================================================================
{ const s = pres.addSlide();
  title(s, "QAM16 at 10 dB, in detection terms", "YOUR NUMBERS");
  [[`${Q.miss} / ${num(Q.sent)}`, `missed: miss rate ${pct(Q.miss, Q.sent)}`, ORANGE],
    [`${Q.fa} / ${num(Q.others)}`, `false alarms: rate ${pct(Q.fa, Q.others)}`, INK],
    [`${num(Q.hit)} / ${num(Q.called)}`, `of the “QAM16” calls were right: precision ${pct(Q.hit, Q.called)}`, INK]].forEach(([v, l, c], i) => {
    const x = M + i * 4.15;
    card(s, x, 1.7, 3.85, 2.3);
    big(s, x + 0.3, 1.85, 3.3, v, l, c, 38);
  });
  t(s, `False alarms are counted against the ${num(Q.others)} frames that were not QAM16.`, M, 4.25, W - 2 * M, 0.5, { fontSize: 19, color: MUTED });
  takeaway(s, "Recall = 1 − miss rate.", 5.0);
  done(s);
}

// 15. summary ======================================================================================
{ const s = pres.addSlide(); s.background = { color: NAVY };
  t(s, "In short", M, 0.55, 8, 0.9, { fontSize: 40, bold: true, color: WHITE });
  const pts = ["The four-signal table came from a network trained only on those four.",
    "Every network was trained by us, from a random start. No outside weights.",
    "One network for all classes: no piece per modulation."];
  pts.forEach((p, i) => {
    badge(s, M, 1.85 + i * 1.25, i + 1, ORANGE, 0.7);
    t(s, p, M + 1.0, 1.75 + i * 1.25, W - 2 * M - 1.0, 0.9, { fontSize: 27, color: WHITE, valign: "middle" });
  });
  t(s, [{ text: "Next: ", options: { bold: true, color: ORANGE } }, { text: "any step you'd like explained in more depth.", options: { color: SOFT } }],
    M, 5.75, W - 2 * M, 0.7, { fontSize: 24 });
  done(s, true);
}

pres.writeFile({ fileName: "moshe_steps.pptx" }).then(f => console.log("wrote", f, "slides:", n));

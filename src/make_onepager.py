"""
make_onepager.py -- the one-page PDF attached to faculty emails.

A README is the wrong thing to attach to a cold email: nobody reads 350 lines
from a stranger. This is the version that has to survive thirty seconds of
attention -- claim, evidence, honest limits, and two figures.

    python make_onepager.py
"""

import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "Cetinkaya_modulation_classification_summary.pdf"

INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#1f4e79")

body = ParagraphStyle("body", fontName="Helvetica", fontSize=8.2, leading=10.2,
                      alignment=TA_JUSTIFY, textColor=INK, spaceAfter=4)
head = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=8.8, leading=11, 
                      textColor=ACCENT, spaceBefore=4, spaceAfter=2)
title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=13.5,
                       leading=16, textColor=INK, spaceAfter=3)
byline = ParagraphStyle("byline", fontName="Helvetica", fontSize=8.2, leading=10.5,
                        textColor=colors.HexColor("#555555"), spaceAfter=8)
cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.4, leading=9,
                     textColor=colors.HexColor("#555555"), spaceBefore=2,
                     spaceAfter=5)


def table(data, widths):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f5f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0d8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


S = []
S.append(Paragraph("Why modulation classifiers fail across transmitters "
                   "&mdash; and a preprocessing fix", title))
S.append(Paragraph(
    "Yusuf Bilal &Ccedil;etinkaya &nbsp;|&nbsp; &#304;T&Uuml;&ndash;NJIT dual-degree "
    "programme, Electronics &amp; Communications Engineering, entering senior year "
    "&nbsp;|&nbsp; ybc6@njit.edu", byline))

S.append(Paragraph("Problem", head))
S.append(Paragraph(
    "Deep classifiers for automatic modulation classification reach ~99% on the "
    "dataset they were trained on, then lose much of it the moment the transmitter "
    "changes. Labelled real RF data being scarce, models are trained on synthetic "
    "data and deployed elsewhere &mdash; so this gap, not benchmark accuracy, is "
    "the practical bottleneck for spectrum monitoring and cognitive radio.", body))

S.append(Paragraph("Measurement", head))
S.append(Paragraph(
    "A 1-D CNN was trained on RadioML 2018.01A (five modulations shared across "
    "domains, 26 SNR levels, ~100k frames of 1024 IQ samples) and evaluated on an "
    "independently written signal generator. Gap at SNR &ge; 10 dB: "
    "<b>0.182 &plusmn; 0.017</b>, and almost all of it sat in a single cell "
    "&mdash; 16QAM read as 64QAM 75% of the time.", body))

S.append(Paragraph("Diagnosis by controlled elimination &mdash; four hypotheses, "
                   "network frozen so only the test signal varied. Three refuted:",
                   head))
S.append(table([
    ["Hypothesis", "Test", "Outcome"],
    ["Symbol-rate offset", "cyclostationary line driven 14.3 → 0.1 dB",
     "no effect (16QAM 0.168 → 0.202)"],
    ["Constellation density", "amplitude histograms, both domains",
     "refuted — distributions overlap"],
    ["Channel impairments", "CFO, multipath, random phase, combined",
     "refuted — best recovery +0.045"],
    ["Spectral envelope", "phase-preserving magnitude substitution",
     "confirmed — 16QAM 0.191 → 0.972"],
], [1.05 * inch, 2.5 * inch, 2.55 * inch]))

S.append(Paragraph("Fix, and comparison against the literature", head))
S.append(Paragraph(
    "Spectral whitening applied as preprocessing to both domains: divide each "
    "frame's spectrum by a smoothed estimate of its own magnitude envelope, leaving "
    "phase untouched. Unlike adversarial domain adaptation, this needs no "
    "target-domain data at fit time, so it is deployable on a receiver that has "
    "never met the transmitter it will see. Four seeds, accuracy at SNR &ge; 10 dB:",
    body))
S.append(table([
    ["Method", "in-domain", "cross-domain", "gap"],
    ["none", "0.993", "0.811", "0.182"],
    ["standard AMC augmentation (rotation / flip / noise)", "0.996", "0.854",
     "0.141"],
    ["spectral whitening", "0.965", "0.909", "0.056"],
    ["whitening + standard augmentation", "0.985", "0.948", "0.038"],
], [3.3 * inch, 0.95 * inch, 1.0 * inch, 0.85 * inch]))
S.append(Spacer(1, 4))
S.append(Paragraph(
    "Whitening beats the standard augmentation baseline by <b>3.3 standard "
    "deviations</b> and composes with it rather than substituting for it "
    "(+6.4 s.d. over whitening alone). <b>79% of the gap closes.</b>", body))

S.append(Image(str(ROOT / "figures" / "summary_panel.png"),
               width=6.35 * inch, height=2.35 * inch))
S.append(Paragraph(
    "<b>A.</b> Cross-domain accuracy with and without the fix (dashed = in-domain). "
    "<b>B.</b> Gap by method, four seeds, bars are one s.d.", cap))

S.append(Paragraph("Mechanism &mdash; and a hypothesis I had to discard", head))
S.append(Paragraph(
    "The obvious account was shortcut learning: the network reads modulation order "
    "off the spectral envelope. An envelope-transplant test refutes it &mdash; "
    "giving frames another class's magnitude spectrum while keeping their own phase "
    "leaves the prediction unchanged, with only 0.002 of cases following the "
    "donated envelope. What happens instead is out-of-distribution collapse: "
    "mismatched frames fall off the training manifold and the classifier defaults "
    "to one class, consistently the densest constellation. The errors are "
    "systematically biased rather than diffuse &mdash; which matters for any "
    "monitoring system expected to be trusted in unfamiliar conditions.", body))

S.append(Paragraph("Limitations and next step", head))
S.append(Paragraph(
    "Both domains are synthetic; there is no real capture yet, and closing that is "
    "the main reason I am seeking lab access. Five classes, one architecture, "
    "15 epochs, no comparison against adversarial domain adaptation. The planned "
    "next step is hardware: a small programmable RF front end (digital step "
    "attenuator, gain block driven into compression, switchable filter bank) that "
    "reproduces the same domain variation in measured physical units.", body))

doc = SimpleDocTemplate(
    str(OUT), pagesize=LETTER,
    leftMargin=0.62 * inch, rightMargin=0.62 * inch,
    topMargin=0.55 * inch, bottomMargin=0.5 * inch,
    title="Modulation classification across transmitters",
    author="Yusuf Bilal Cetinkaya",
)
doc.build(S)
print(f"wrote {OUT.name}")

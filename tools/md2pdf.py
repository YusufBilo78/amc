"""Render a markdown file in this repository to a printable PDF.

    python tools/md2pdf.py MEETING_MOSHE.md            # -> MEETING_MOSHE.pdf
    python tools/md2pdf.py MEETING_MOSHE.md out.pdf

The meeting documents are written to be read on screen, but they also get
printed and carried into the room, so the layout is fixed here rather than
left to whatever the browser defaults to: A4, tables that do not split across
a page break, and the English lines set apart from the Turkish scaffolding.

Needs `markdown` (pip) and a Chromium binary. The PDF embeds DejaVu, which
covers the Turkish glyphs -- check that before swapping the font family.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
:root {
  --ink: #16181d; --mute: #5d6470; --rule: #d8dce3;
  --accent: #1f5fa8; --quotebg: #f4f7fb;
}
* { box-sizing: border-box; }
body {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 9.4pt; line-height: 1.5; color: var(--ink);
  margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 {
  font-size: 19pt; line-height: 1.2; margin: 0 0 4pt;
  letter-spacing: -0.2pt; border-bottom: 2.5pt solid var(--ink);
  padding-bottom: 6pt;
}
h2 {
  font-size: 12.5pt; margin: 20pt 0 6pt; padding-top: 5pt;
  border-top: 0.8pt solid var(--rule); color: var(--accent);
  break-after: avoid; break-inside: avoid;
}
h3 { font-size: 10.3pt; margin: 13pt 0 4pt; break-after: avoid; }
p { margin: 5pt 0; }
em { color: var(--mute); font-style: italic; }

table {
  border-collapse: collapse; width: 100%; margin: 8pt 0 10pt;
  font-size: 8.6pt; break-inside: avoid;
}
th, td {
  border: 0.6pt solid var(--rule); padding: 3.4pt 5pt;
  text-align: left; vertical-align: top;
}
th { background: #eef1f6; font-weight: bold; }
tbody tr:nth-child(even) td { background: #fafbfd; }

/* The English lines meant to be said out loud are block quotes. */
blockquote {
  margin: 8pt 0; padding: 7pt 10pt; background: var(--quotebg);
  border-left: 2.5pt solid var(--accent); break-inside: avoid;
}
blockquote p { margin: 2pt 0; }

ul, ol { margin: 5pt 0; padding-left: 15pt; }
li { margin: 2.5pt 0; }

code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 8.1pt;
  background: #eef1f6; padding: 0.6pt 2.6pt; border-radius: 2pt;
}
th code, td code { font-size: 7.8pt; }

hr {
  border: none; border-top: 0.8pt solid var(--rule);
  margin: 14pt 0; break-after: avoid;
}
/* a horizontal rule already separates the sections; don't draw a second one */
hr + h2 { border-top: none; margin-top: 4pt; padding-top: 0; }

.box {
  display: inline-block; width: 8pt; height: 8pt;
  border: 0.9pt solid var(--mute); border-radius: 1.5pt;
  margin-right: 3pt; vertical-align: -0.5pt;
}
.box.done { background: var(--mute); }
"""


def to_html(md_text, title):
    # python-markdown has no task-list extension here; render the boxes ourselves.
    md_text = re.sub(r'^(\s*)- \[ \] ', r'\1- <span class="box"></span> ',
                     md_text, flags=re.M)
    md_text = re.sub(r'^(\s*)- \[x\] ', r'\1- <span class="box done"></span> ',
                     md_text, flags=re.M)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list",
                    "md_in_html"],
    )
    return (f'<!DOCTYPE html>\n<html lang="tr"><head><meta charset="utf-8">\n'
            f"<title>{title}</title>\n<style>{CSS}</style></head><body>\n"
            f"{body}\n</body></html>\n")


def find_chromium():
    for name in ["chromium", "chromium-browser", "google-chrome", "chrome"]:
        found = shutil.which(name)
        if found:
            return found
    # The remote execution environment ships Chromium under the Playwright dir.
    for cand in sorted(Path("/opt/pw-browsers").glob("chromium*/chrome-linux/chrome")):
        if cand.is_file():
            return str(cand)
    raise SystemExit(
        "no Chromium binary found. Install one, or point PATH at it -- this "
        "script shells out to headless Chrome to do the printing.")


def main():
    if not 2 <= len(sys.argv) <= 3:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else src.with_suffix(".pdf")

    html = to_html(src.read_text(encoding="utf-8"), src.stem)

    # Chromium reads the page over file://, so the HTML has to live on disk.
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [find_chromium(), "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", f"--print-to-pdf={out.resolve()}",
             page.resolve().as_uri()],
            check=True, capture_output=True)

    print(f"{out} ({out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()

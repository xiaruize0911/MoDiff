"""Render one of this project's report markdowns to PDF, tables and figures intact.

WHY A SCRIPT AND NOT A ONE-LINER. These documents are mostly wide numeric tables (up to 11 columns) and
full-width figures, and the two things that go wrong in a default conversion are exactly those: portrait
pages clip the tables, and a table split across a page boundary loses its header so the columns become
unreadable. So: landscape, `thead` repeated on every page fragment, and figures allowed to shrink to the
text width but never to grow past it.

Relative markdown links (`[KERNEL_SPEEDUP.md](KERNEL_SPEEDUP.md)`) cannot resolve in a PDF. They are kept
as visible text rather than dropped, because knowing which file a number came from is the point of them.

Run: python docs/bench_report_2026-08-13_postzp/scripts/md_to_pdf.py [in.md] [out.pdf]
"""
import os
import sys

import markdown
from weasyprint import HTML

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
SRC = sys.argv[1] if len(sys.argv) > 1 else "docs/bench_report_2026-08-13_postzp/SUMMARY.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(SRC)[0] + ".pdf"

CSS = """
@page { size: A4 landscape; margin: 13mm 14mm 15mm 14mm;
        @bottom-right { content: counter(page) " / " counter(pages);
                        font: 8pt "DejaVu Sans"; color: #999; } }
body { font-family: "DejaVu Sans", sans-serif; font-size: 8.6pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 2mm 0; }
h2 { font-size: 12pt; margin: 7mm 0 2mm 0; padding-bottom: 1mm;
     border-bottom: 1px solid #ccc; break-after: avoid; }
h3 { font-size: 10pt; margin: 5mm 0 1.5mm 0; break-after: avoid; }
p { margin: 1.6mm 0; }
strong { color: #000; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 7.6pt;
       background: #f4f4f4; padding: 0 1px; }
pre { background: #f6f6f6; border-left: 2.5px solid #bbb; padding: 2mm 3mm; margin: 2mm 0;
      font-size: 7.6pt; break-inside: avoid; }
pre code { background: none; }
table { border-collapse: collapse; margin: 2.5mm 0 3.5mm 0; font-size: 7.9pt; width: auto; }
thead { display: table-header-group; }                 /* repeat the header if a table does split */
tr { break-inside: avoid; }
th { background: #eee; border-bottom: 1.2px solid #888; padding: 1mm 2.2mm;
     text-align: left; font-weight: bold; }
td { border-bottom: 1px solid #e2e2e2; padding: 0.9mm 2.2mm; }
/* A4 landscape is 210mm tall and the margins take 28mm, so anything over ~170mm gets CLIPPED at the
   page bottom rather than scaled -- which silently ate the last row of the 5-row sample grid. Capping
   the height also lets the two suite plots sit side by side instead of one per page. */
img { max-width: 100%; max-height: 148mm; height: auto; width: auto;
      display: inline-block; vertical-align: top; margin: 2mm 1mm; }
p:has(> img:only-child) { text-align: center; }
/* two figures in one markdown paragraph mean "show these side by side" -- without a width cap they
   are each wide enough to force a wrap and end up one per page. */
p img:not(:only-child) { max-width: 48%; max-height: 110mm; }
a { color: #1a4f8a; text-decoration: none; }
blockquote { margin: 2mm 0; padding: 1.5mm 3mm; background: #fbf7ec;
             border-left: 2.5px solid #d9b25f; }
em { color: #444; }
"""

html = markdown.markdown(open(SRC).read(),
                         extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
       f"<style>{CSS}</style></head><body>{html}</body></html>")
#: base_url is the markdown's own directory, so `plots/06_samples.png` resolves the same way it does
#: when the markdown is viewed in place. Getting this wrong silently produces a PDF with no figures.
HTML(string=doc, base_url=os.path.dirname(os.path.abspath(SRC)) + os.sep).write_pdf(OUT)
size = os.path.getsize(OUT)
print(f"wrote {OUT}  ({size / 1e6:.2f} MB)")
assert size > 200_000, "suspiciously small -- the figures probably did not resolve"

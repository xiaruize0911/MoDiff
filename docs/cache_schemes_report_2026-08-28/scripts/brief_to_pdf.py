"""Portrait A4 PDF for cache_schemes BRIEF.md (Noto CJK)."""
import os
import markdown
from weasyprint import HTML

ROOT = "/workspace/MoDiff"
SRC = os.path.join(ROOT, "docs/cache_schemes_report_2026-08-28/BRIEF.md")
OUT = os.path.join(ROOT, "docs/cache_schemes_report_2026-08-28/BRIEF.pdf")

CSS = r"""
@page { size: A4 portrait; margin: 14mm 15mm 16mm 15mm;
        @bottom-left { content: "缓存方案简报  ·  层先行，e2e 在后";
                       font: 8pt "Noto Sans CJK SC"; color: #999; }
        @bottom-right { content: counter(page) " / " counter(pages);
                        font: 8pt "Noto Sans CJK SC"; color: #999; } }
body { font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", sans-serif;
       font-size: 9.2pt; line-height: 1.48; color: #1a1a1a; }
h1 { font-size: 16pt; margin: 0 0 2.5mm 0; }
h2 { font-size: 12pt; margin: 6mm 0 2mm 0; padding-bottom: 1mm;
     border-bottom: 1px solid #ccc; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 4mm 0 1.5mm 0; break-after: avoid; }
p { margin: 1.6mm 0; }
strong { color: #000; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt;
       background: #f4f4f4; padding: 0 1px; }
table { border-collapse: collapse; margin: 2mm 0 3mm 0; font-size: 8.2pt; width: 100%; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #eee; border-bottom: 1.2px solid #888; padding: 1mm 1.8mm;
     text-align: left; font-weight: bold; }
td { border-bottom: 1px solid #e2e2e2; padding: 0.9mm 1.8mm; }
img { max-width: 100%; max-height: 95mm; height: auto; display: block;
      margin: 2mm auto; }
a { color: #1a4f8a; text-decoration: none; }
ul { margin: 1.5mm 0 2mm 5mm; }
li { margin: 0.6mm 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 4mm 0; }
"""

os.chdir(ROOT)
html = markdown.markdown(open(SRC, encoding="utf-8").read(),
                         extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
       f"<style>{CSS}</style></head><body>{html}</body></html>")
HTML(string=doc, base_url=os.path.dirname(SRC) + os.sep).write_pdf(OUT)
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.2f} MB)")

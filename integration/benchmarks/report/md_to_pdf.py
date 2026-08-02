"""Render the measurement report's markdown to PDF with its tables and figures intact.

Written because this container has no pandoc, weasyprint, wkhtmltopdf or LaTeX -- only reportlab.
Scope is deliberately the subset of markdown MEASUREMENT_REPORT actually uses: ATX headings,
paragraphs, `**bold**` / `*italic*` / `` `code` ``, bullet lists, fenced code, horizontal rules,
GFM pipe tables (2-10 columns, up to 289 chars wide) and `![alt](path)` images.

Landscape A4: the widest tables run 10 columns and the figures are up to 17 inches wide, both of
which get squeezed to illegibility in portrait.

Usage: md_to_pdf.py REPORT.md --out REPORT.pdf
"""
import argparse
import html
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

PAGE = landscape(A4)
MARGIN = 14 * mm
AVAIL = PAGE[0] - 2 * MARGIN

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#57606a")
RULE = colors.HexColor("#d0d7de")
HEADBG = colors.HexColor("#eef2f6")
ZEBRA = colors.HexColor("#f7f9fb")
CODEBG = colors.HexColor("#f2f4f7")


def styles():
    ss = getSampleStyleSheet()
    mk = lambda **kw: ParagraphStyle(**kw)
    return {
        "h1": mk(name="h1", fontName="Helvetica-Bold", fontSize=19, leading=23,
                 spaceBefore=2, spaceAfter=9, textColor=INK),
        "h2": mk(name="h2", fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                 spaceBefore=15, spaceAfter=6, textColor=INK),
        "h3": mk(name="h3", fontName="Helvetica-Bold", fontSize=11, leading=14,
                 spaceBefore=11, spaceAfter=4, textColor=INK),
        "h4": mk(name="h4", fontName="Helvetica-Bold", fontSize=9.5, leading=12,
                 spaceBefore=9, spaceAfter=3, textColor=MUTED),
        "body": mk(name="body", fontName="Helvetica", fontSize=8.6, leading=12.2,
                   spaceAfter=5, textColor=INK, alignment=TA_LEFT),
        "li": mk(name="li", fontName="Helvetica", fontSize=8.6, leading=12.2,
                 leftIndent=11, bulletIndent=3, spaceAfter=2, textColor=INK),
        "cell": mk(name="cell", fontName="Helvetica", fontSize=6.6, leading=8.2,
                   textColor=INK),
        "cellh": mk(name="cellh", fontName="Helvetica-Bold", fontSize=6.6, leading=8.2,
                    textColor=INK),
        "code": mk(name="code", fontName="Courier", fontSize=7.2, leading=9.4,
                   textColor=INK, backColor=CODEBG, borderPadding=5, spaceAfter=7),
        "cap": mk(name="cap", fontName="Helvetica-Oblique", fontSize=7.4, leading=9.6,
                  textColor=MUTED, spaceBefore=2, spaceAfter=9),
    }


def inline(t):
    """Markdown inline spans -> reportlab mini-HTML. Escapes first so `<` in kernel names
    (e.g. `Kernel<half>`) is not eaten as a tag.

    Code spans are pulled out to placeholders before emphasis runs and put back after. The
    report contains spans like **`*_qout` entries.** whose code content holds a literal `*`;
    substituting in place leaves that `*` in the stream, the bold pattern then fails to match
    across it, and the italic pattern pairs it with a `*` outside the <font> element -- which
    reportlab rejects as crossed tags rather than rendering wrong.
    """
    t = html.escape(t, quote=False)
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return "\x00%d\x00" % (len(spans) - 1)

    t = re.sub(r"`([^`]+)`", stash, t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<u>\1</u>", t)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: '<font face="Courier" size="-0.6">%s</font>'
                            % spans[int(m.group(1))], t)


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def make_table(rows, st):
    head, body = rows[0], rows[1:]
    ncol = len(head)
    data = [[Paragraph(inline(c), st["cellh"]) for c in head]]
    for r in body:
        r = (r + [""] * ncol)[:ncol]
        data.append([Paragraph(inline(c), st["cell"]) for c in r])
    # Width by longest raw cell, so a column of kernel names gets the room and a column of
    # numbers does not. Clamped so one long cell cannot starve the rest.
    raw = [head] + [(r + [""] * ncol)[:ncol] for r in body]
    w = [max(3, min(46, max(len(rr[i]) for rr in raw))) for i in range(ncol)]
    tot = float(sum(w))
    widths = [AVAIL * x / tot for x in w]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def make_image(path, alt, st):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = AVAIL
    h = w * ih / float(iw)
    maxh = PAGE[1] - 2 * MARGIN - 26 * mm     # leave room for the caption
    if h > maxh:
        h = maxh
        w = h * iw / float(ih)
    img = Image(path, width=w, height=h)
    img.hAlign = "LEFT"
    return KeepTogether([img, Paragraph(inline(alt), st["cap"])]) if alt else img


def build(md_path, out_path):
    st = styles()
    base = os.path.dirname(os.path.abspath(md_path))
    lines = open(md_path).read().splitlines()
    flow = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            flow.append(Paragraph("<br/>".join(buf) or " ", st["code"]))
            continue

        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", s)
        if m:
            alt, rel = m.group(1), m.group(2)
            p = rel if os.path.isabs(rel) else os.path.join(base, rel)
            flow.append(make_image(p, alt, st) if os.path.exists(p)
                        else Paragraph("[missing figure: %s]" % html.escape(rel), st["cap"]))
            i += 1
            continue

        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = split_row(lines[i])
                # the |---|---| separator row carries no data
                if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r):
                    rows.append(r)
                i += 1
            if rows:
                flow.append(make_table(rows, st))
                flow.append(Spacer(1, 8))
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s):
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                   spaceBefore=7, spaceAfter=9))
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = min(len(m.group(1)), 4)
            flow.append(Paragraph(inline(m.group(2)), st["h%d" % lvl]))
            i += 1
            continue

        m = re.match(r"^[-*+]\s+(.*)$", s)
        if m:
            flow.append(Paragraph(inline(m.group(1)), st["li"], bulletText="•"))
            i += 1
            continue

        m = re.match(r"^(\d+)[.)]\s+(.*)$", s)
        if m:
            flow.append(Paragraph(inline(m.group(2)), st["li"],
                                  bulletText="%s." % m.group(1)))
            i += 1
            continue

        # Consecutive plain lines are one paragraph, as markdown means them.
        buf = []
        while i < len(lines):
            t = lines[i].strip()
            if (not t or t.startswith(("|", "#", "```", "!["))
                    or re.match(r"^[-*+]\s", t) or re.match(r"^\d+[.)]\s", t)
                    or re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", t)):
                break
            buf.append(t)
            i += 1
        flow.append(Paragraph(inline(" ".join(buf)), st["body"]))

    title = os.path.basename(md_path)

    def furniture(canv, doc):
        canv.saveState()
        canv.setFont("Helvetica", 7)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN, 9 * mm, title)
        canv.drawRightString(PAGE[0] - MARGIN, 9 * mm, "page %d" % canv.getPageNumber())
        canv.setStrokeColor(RULE)
        canv.line(MARGIN, 12.5 * mm, PAGE[0] - MARGIN, 12.5 * mm)
        canv.restoreState()

    doc = BaseDocTemplate(out_path, pagesize=PAGE,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=17 * mm, title=title)
    frame = Frame(MARGIN, 17 * mm, AVAIL, PAGE[1] - MARGIN - 17 * mm, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=furniture)])
    doc.build(flow)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("wrote %s" % build(a.md, a.out))

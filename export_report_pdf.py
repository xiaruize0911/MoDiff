#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
REPORT_MD = ROOT / "experiment_report" / "BENCHMARK_REPORT.md"
REPORT_PDF = ROOT / "experiment_report" / "BENCHMARK_REPORT.pdf"


def md_inline_to_html(text: str) -> str:
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    img = Image(str(path))
    width = float(img.imageWidth)
    height = float(img.imageHeight)
    scale = min(max_width / width, max_height / height, 1.0)
    img.drawWidth = width * scale
    img.drawHeight = height * scale
    return img


def parse_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for line in lines:
        if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append([Paragraph(md_inline_to_html(cell), styles["table"]) for cell in cells])

    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0B0B0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F7F7F7")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_story(md_text: str):
    styles = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontSize=15,
            leading=18,
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "Heading3Custom",
            parent=styles["Heading3"],
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=10,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=styles["Italic"],
            fontSize=8.5,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=6,
        ),
    }

    story = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 0.12 * inch))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(md_inline_to_html(stripped[2:].strip()), custom["title"]))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(md_inline_to_html(stripped[3:].strip()), custom["h2"]))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(md_inline_to_html(stripped[4:].strip()), custom["h3"]))
            i += 1
            continue

        if stripped.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                alt_text, rel_path = match.groups()
                img_path = (REPORT_MD.parent / rel_path).resolve()
                if img_path.exists():
                    story.append(fit_image(img_path, max_width=6.8 * inch, max_height=5.8 * inch))
                    if alt_text:
                        story.append(Paragraph(md_inline_to_html(alt_text), custom["caption"]))
                    story.append(Spacer(1, 0.08 * inch))
                else:
                    story.append(Paragraph(md_inline_to_html(f"[Missing image: {rel_path}]"), custom["body"]))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(parse_table(table_lines, custom))
            story.append(Spacer(1, 0.10 * inch))
            continue

        if stripped.startswith("- "):
            story.append(Paragraph(md_inline_to_html("• " + stripped[2:].strip()), custom["bullet"]))
            i += 1
            continue

        story.append(Paragraph(md_inline_to_html(stripped), custom["body"]))
        i += 1

    return story


def main() -> None:
    md_text = REPORT_MD.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title="MoDiff Benchmark Report",
        author="GitHub Copilot",
    )
    story = build_story(md_text)
    doc.build(story)
    print(f"PDF written to: {REPORT_PDF}")


if __name__ == "__main__":
    main()

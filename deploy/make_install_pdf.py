#!/usr/bin/env python3
"""Render docs/INSTALL_CLIENT.md to a print-ready A4 PDF (dist/ANPR-Install-Guide.pdf).

Converts the small subset of Markdown the guide uses (headings, lists, tables, fenced code,
blockquotes, hr, **bold**, `code`, links) to a styled standalone HTML, then prints it to PDF with
headless Google Chrome. No third-party Python packages needed.
"""
from __future__ import annotations

import base64
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "INSTALL_CLIENT.md"
LOGO = ROOT / "anpr" / "web" / "static" / "logicclutch-logo.png"
OUT_HTML = ROOT / "dist" / "ANPR-Install-Guide.html"
OUT_PDF = ROOT / "dist" / "ANPR-Install-Guide.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def inline(text: str) -> str:
    """Escape, then apply inline Markdown: `code`, **bold**, [text](url)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "`":
            j = text.find("`", i + 1)
            if j != -1:
                out.append("<code>" + html.escape(text[i + 1 : j]) + "</code>")
                i = j + 1
                continue
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j != -1:
                out.append("<strong>" + html.escape(text[i + 2 : j]) + "</strong>")
                i = j + 2
                continue
        m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", text[i:])
        if m:
            out.append(f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>')
            i += m.end()
            continue
        out.append(html.escape(c))
        i += 1
    return "".join(out)


def render(md: str) -> str:
    lines = md.splitlines()
    html_parts: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        # fenced code block
        if line.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            html_parts.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue
        # table (header row | --- row | body rows)
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            def cells(row: str) -> list[str]:
                return [c.strip() for c in row.strip().strip("|").split("|")]

            header = cells(line)
            i += 2
            rows = []
            while i < n and "|" in lines[i]:
                rows.append(cells(lines[i]))
                i += 1
            thead = "".join(f"<th>{inline(c)}</th>" for c in header)
            tbody = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            html_parts.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>")
            continue
        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            html_parts.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        # hr
        if re.match(r"^---+\s*$", line):
            html_parts.append("<hr>")
            i += 1
            continue
        # blockquote
        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(inline(lines[i].lstrip(">").strip()))
                i += 1
            html_parts.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue
        # ordered / unordered list (joins wrapped continuation lines into the item)
        list_m = re.match(r"^\s*(\d+\.|[-*])\s+", line)
        if list_m:
            ordered = line.lstrip()[0].isdigit()
            marker = r"^\s*\d+\.\s+" if ordered else r"^\s*[-*]\s+"
            cont_stop = r"^(\s*[-*]\s|\s*\d+\.\s|#{1,6}\s|```|>|\s*\|)"
            buf = []
            while i < n and re.match(marker, lines[i]):
                text = re.sub(marker, "", lines[i])
                i += 1
                while (i < n and lines[i].strip() and not re.match(cont_stop, lines[i])
                       and "|" not in lines[i]):
                    text += " " + lines[i].strip()
                    i += 1
                buf.append("<li>" + inline(text) + "</li>")
            tag = "ol" if ordered else "ul"
            html_parts.append(f"<{tag}>" + "".join(buf) + f"</{tag}>")
            continue
        # blank
        if not line.strip():
            i += 1
            continue
        # paragraph
        buf = [line]
        i += 1
        _pstop = r"^(#{1,6}\s|```|>|\s*[-*]\s|\s*\d+\.\s|---+\s*$)"
        while (i < n and lines[i].strip() and not re.match(_pstop, lines[i])
               and "|" not in lines[i]):
            buf.append(lines[i])
            i += 1
        html_parts.append("<p>" + inline(" ".join(buf)) + "</p>")
    return "\n".join(html_parts)


def main() -> int:
    md = MD.read_text()
    body = render(md)
    logo_b64 = base64.b64encode(LOGO.read_bytes()).decode() if LOGO.exists() else ""
    logo_tag = f'<img class="logo" src="data:image/png;base64,{logo_b64}" alt="">' if logo_b64 else ""
    css = """
    @page { size: A4; margin: 16mm 15mm 14mm; }
    * { box-sizing: border-box; }
    body { font: 10.6pt/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
           color: #1b2430; margin: 0; }
    .brandbar { display: flex; align-items: center; gap: 14px; border-bottom: 3px solid #0d6efd;
                padding-bottom: 12px; margin-bottom: 10px; }
    .brandbar .logo { width: 46px; height: 46px; border-radius: 9px; }
    .brandbar .who { font-weight: 700; font-size: 15pt; color: #0b1b33; }
    .brandbar .sub { color: #5b6b7f; font-size: 9pt; }
    h1 { font-size: 20pt; margin: 6px 0 2px; color: #0b1b33; }
    h2 { font-size: 13pt; margin: 18px 0 6px; color: #0d6efd; page-break-after: avoid; }
    h3 { font-size: 11pt; margin: 12px 0 4px; page-break-after: avoid; }
    p { margin: 5px 0; }
    ul, ol { margin: 5px 0 5px 0; padding-left: 20px; }
    li { margin: 3px 0; }
    code { font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; font-size: 9.2pt;
           background: #eef2f7; padding: 1px 5px; border-radius: 4px; color: #0b3d2e; }
    pre { background: #0f172a; color: #e8eef7; padding: 10px 13px; border-radius: 8px; overflow: hidden;
          margin: 7px 0; page-break-inside: avoid; }
    pre code { background: none; color: inherit; font-size: 9.2pt; padding: 0; white-space: pre-wrap;
               word-break: break-word; }
    blockquote { background: #fff7e6; border-left: 4px solid #f0a500; margin: 8px 0; padding: 7px 12px;
                 border-radius: 0 6px 6px 0; font-size: 9.6pt; color: #5a4a1a; page-break-inside: avoid; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.3pt;
            page-break-inside: avoid; }
    th, td { border: 1px solid #d6dee8; padding: 6px 9px; text-align: left; vertical-align: top; }
    th { background: #eef4ff; color: #0b1b33; }
    tr:nth-child(even) td { background: #f7f9fc; }
    hr { border: none; border-top: 1px solid #e0e6ee; margin: 16px 0; }
    a { color: #0d6efd; text-decoration: none; }
    h1 + p, h2:first-of-type { page-break-before: avoid; }
    """
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>ANPR Install Guide</title><style>{css}</style></head><body>
<div class="brandbar">{logo_tag}<div><div class="who">LogicClutch</div>
<div class="sub">Automatic Number Plate Recognition · Raspberry Pi 3B+</div></div></div>
{body}
</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(doc)

    if not Path(CHROME).exists():
        print(f"Chrome not found at {CHROME}; wrote HTML only: {OUT_HTML}")
        return 1
    OUT_PDF.unlink(missing_ok=True)
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
        check=True, capture_output=True, text=True, timeout=120,
    )
    if OUT_PDF.exists():
        print(f"wrote {OUT_PDF}  ({OUT_PDF.stat().st_size // 1024} KB)")
        return 0
    print("Chrome ran but no PDF was produced")
    return 1


if __name__ == "__main__":
    sys.exit(main())

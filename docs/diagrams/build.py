#!/usr/bin/env python3
"""Rebuild the .svg and .pdf exports from each figure's .html source.

    python3 docs/diagrams/build.py            # all figures
    python3 docs/diagrams/build.py data-model # one figure

The .html files are the source of truth; .svg and .pdf are generated. Run this
after editing any figure.

Two outputs, because they have different jobs:

  .svg  portable vector for the web docs. Pulls typography from Google Fonts at
        render time, so it is small but needs the network to look right. Fine
        for GitHub and any browser.

  .pdf  print-ready, for LaTeX \\includegraphics. Fonts are downloaded once and
        embedded as base64 before rendering, so the PDF carries subsetted
        CID TrueType faces and needs nothing at all. This matters: rendering
        against the Google Fonts *webfont* endpoint instead yields Type 3 fonts
        (which IEEE and ACM reject) and drops U+2192 to a fallback face, because
        the arrow is outside Google's "latin" subset.

Requires: chromium on PATH, and network access on the first run (fonts are
cached under .fontcache/).
"""

from __future__ import annotations

import base64
import re
import subprocess
import sys
import tempfile
import urllib.request
import xml.dom.minidom
from pathlib import Path
from xml.parsers.expat import ExpatError

HERE = Path(__file__).parent
CACHE = HERE / ".fontcache"

# Legacy UA so the Google Fonts API serves full static .ttf rather than
# unicode-range-subsetted .woff2 (the subsetting is what loses U+2192).
LEGACY_UA = "Mozilla/5.0 (Windows NT 6.1)"
FONT_CSS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Inter:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;500"
    "&family=Source+Serif+4:ital,opsz,wght@0,8..60,600;1,8..60,400"
)
# Same families, webfont form, for the .svg export's remote @import.
FONT_IMPORT_URL = FONT_CSS_URL.replace("&", "&amp;") + "&amp;display=swap"


def extract_svg(html: Path) -> tuple[str, float, float]:
    """Return the figure's <svg> block and its viewBox width/height."""
    match = re.search(r"<svg\b.*?</svg>", html.read_text(), re.S)
    if not match:
        sys.exit(f"{html.name}: no <svg> block found")
    svg = match.group(0)
    if 'xmlns="http://www.w3.org/2000/svg"' not in svg:
        sys.exit(f"{html.name}: <svg> is missing xmlns")
    box = re.search(r'viewBox="([\d.\- ]+)"', svg)
    if not box:
        sys.exit(f"{html.name}: <svg> is missing viewBox")
    _, _, width, height = (float(v) for v in box.group(1).split())
    return svg, width, height


def fetch_fonts() -> str:
    """Download the static faces once and return them as @font-face rules."""
    CACHE.mkdir(exist_ok=True)
    css_path = CACHE / "fonts.css"
    if not css_path.exists():
        request = urllib.request.Request(FONT_CSS_URL, headers={"User-Agent": LEGACY_UA})
        with urllib.request.urlopen(request) as response:
            css_path.write_bytes(response.read())

    rules = []
    for block in re.findall(r"@font-face\s*\{(.*?)\}", css_path.read_text(), re.S):
        family = re.search(r"font-family: '([^']+)'", block).group(1)
        style = re.search(r"font-style: (\w+)", block).group(1)
        weight = re.search(r"font-weight: (\d+)", block).group(1)
        url = re.search(r"url\(([^)]+)\)", block).group(1)

        ttf = CACHE / f"{family.replace(' ', '')}-{weight}-{style}.ttf"
        if not ttf.exists():
            urllib.request.urlretrieve(url, ttf)
        encoded = base64.b64encode(ttf.read_bytes()).decode()
        rules.append(
            f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};"
            f"src:url(data:font/ttf;base64,{encoded}) format('truetype');}}"
        )
    return "\n".join(rules)


def write_svg(html: Path, svg: str) -> Path:
    """Standalone .svg, typography pulled from Google Fonts at render time."""
    style = f"<style>@import url('{FONT_IMPORT_URL}');</style>"
    svg = svg.replace("<defs>", f"<defs>\n        {style}", 1)
    out = html.with_suffix(".svg")
    out.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n")

    # A standalone .svg is parsed as strict XML, which browsers are not. Catch
    # here what only Inkscape/Illustrator/LaTeX would otherwise catch -- most
    # often a bare "&", or a "--" inside an <!-- comment -->, both illegal.
    try:
        xml.dom.minidom.parse(str(out))
    except ExpatError as exc:
        sys.exit(f"{out.name}: not well-formed XML -- {exc}")
    return out


def write_pdf(html: Path, svg: str, width: float, height: float, font_css: str) -> Path:
    """Print-ready .pdf at the figure's natural size, fonts embedded."""
    w_in, h_in = width / 96, height / 96  # CSS px -> inches
    page = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>\n{font_css}\n'
        f"@page{{size:{w_in:.4f}in {h_in:.4f}in;margin:0;}}"
        f"html,body{{margin:0;padding:0;background:#fff;}}"
        f"svg{{display:block;width:{w_in:.4f}in;height:{h_in:.4f}in;}}"
        f"</style></head><body>{svg}</body></html>"
    )
    out = html.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "figure.html"
        staged.write_text(page)
        subprocess.run(
            [
                "chromium",
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--virtual-time-budget=6000",
                "--no-pdf-header-footer",
                f"--print-to-pdf={out}",
                str(staged),
            ],
            check=True,
            capture_output=True,
        )
    return out


def main() -> None:
    wanted = sys.argv[1:]
    sources = sorted(HERE.glob("*.html"))
    if wanted:
        sources = [s for s in sources if s.stem in wanted]
        if not sources:
            sys.exit(f"no figure matching {wanted}")

    font_css = fetch_fonts()
    for html in sources:
        svg, width, height = extract_svg(html)
        write_svg(html, svg)
        write_pdf(html, svg, width, height, font_css)
        print(f"{html.stem:<24} {width / 96:.2f} x {height / 96:.2f} in   -> .svg .pdf")


if __name__ == "__main__":
    main()

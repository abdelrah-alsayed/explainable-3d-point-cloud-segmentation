"""Build data/fonts.js: the fonts of the explorer, embedded so that the page and every saved image use them, also offline.

fonts/ holds the font files:
    IBM Plex Sans (variable) and IBM Plex Mono 400/500/600, latin subset, from Google Fonts (SIL Open Font Licence)
    Computer Modern = Latin Modern Roman 10 and Mono 10, from /usr/share/texmf/fonts/opentype/public/lm (GUST Font Licence)
data/fonts.js sets window.FONT_SETS = {plex, cm}: for each set the CSS font stacks of the page and the @font-face rules
that a saved chart needs (an SVG drawn as an image cannot see the page fonts). It also adds all @font-face rules to the page.
Run: python3 build_fonts.py      (standard library only)
"""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONTS = HERE/"fonts"
OUT = HERE/"data"/"fonts.js"


def face(family, file, weight="400", style="normal"):
    fmt = {"woff2": ("font/woff2", "woff2"), "otf": ("font/otf", "opentype")}[file.split(".")[-1]]
    data = base64.b64encode((FONTS/file).read_bytes()).decode()
    return (f"@font-face{{font-family:'{family}';font-style:{style};font-weight:{weight};font-display:block;"
            f"src:url(data:{fmt[0]};base64,{data}) format('{fmt[1]}');}}")


def main():
    plex_sans = face("IBM Plex Sans", "plex-sans-variable.woff2", "100 700")
    plex_mono = [face("IBM Plex Mono", f"plex-mono-{w}.woff2", w) for w in ("400", "500", "600")]
    cm = [face("Computer Modern", "lmroman10-regular.otf"), face("Computer Modern", "lmroman10-bold.otf", "700"),
          face("Computer Modern", "lmroman10-italic.otf", "400", "italic")]
    cm_mono = face("Computer Modern Mono", "lmmono10-regular.otf")
    sets = {
        "plex": {"label": "IBM Plex",
                 "body": '"IBM Plex Sans", "Segoe UI", system-ui, sans-serif',
                 "data": '"IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace',
                 "code": '"IBM Plex Mono", ui-monospace, monospace',
                 "svg": "".join(plex_mono)},                 # chart text uses the data font
        "cm": {"label": "Computer Modern",
               "body": '"Computer Modern", "Latin Modern Roman", "CMU Serif", serif',
               "data": '"Computer Modern", "Latin Modern Roman", "CMU Serif", serif',
               "code": '"Computer Modern Mono", "Latin Modern Mono", "CMU Typewriter Text", monospace',
               "svg": cm[0] + cm[1]},
    }
    all_faces = plex_sans + "".join(plex_mono) + "".join(cm) + cm_mono
    js = ("/* written by build_fonts.py: IBM Plex (SIL OFL) and Computer Modern / Latin Modern (GUST licence), embedded */\n"
          f"window.FONT_SETS = {json.dumps(sets)};\n"
          f"(function () {{ const s = document.createElement('style'); s.textContent = {json.dumps(all_faces)}; document.head.appendChild(s); }})();\n")
    OUT.write_text(js)
    print(f"{OUT.stat().st_size / 1e3:.0f} kB -> {OUT}")


if __name__ == "__main__":
    main()

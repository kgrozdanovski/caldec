"""Draw the three README charts from a numbers file. `assets/` is written output.

GitHub renders an SVG referenced by `<img>` in a sandbox with no external fonts,
scripts or CSS, so every file carries literal colours and a system font stack. A
light and a dark file are written for each chart; the README selects one with a
`<picture>` element.

    python scripts/charts.py --numbers runs/numbers.json --out assets

The numbers file (see `runs/numbers.json` in the repository for the released one):

    {"accuracy": {"assistant": {"CalDec Laya": 0.823, ...}, "public": {...}},
     "calibration": {"Laya base": 0.989, "CalDec Laya": 0.559, ...},
     "per_site": [{"site": "tools.injection", "base": 0.60, "ours": 0.96, "n": 54}, ...],
     "caption": {...}}
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from pathlib import Path
from typing import Dict, List

SANS = "ui-sans-serif,system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

THEMES = {
    "light": {"bg": "#FBFAF8", "fg": "#16171B", "fg2": "#56585F", "fg3": "#8A8C94",
              "track": "#EDEAE4", "rule": "#DEDBD4", "base": "#A9ADB8"},
    "dark": {"bg": "#0E1015", "fg": "#F2F3F6", "fg2": "#A4A7B2", "fg3": "#737784",
             "track": "#1B1E27", "rule": "#262A34", "base": "#6C707C"},
}
GRADIENT = ('<defs><linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">'
            '<stop offset="0" stop-color="#3B6CFF"/><stop offset="1" stop-color="#9B5CFF"/>'
            '</linearGradient></defs>')


def text(x, y, s, size=13, weight=None, fill="#000", anchor=None, mono=False, spacing=None):
    attrs = [f'x="{x}"', f'y="{y}"', f'font-family="{MONO if mono else SANS}"', f'font-size="{size}"']
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    if spacing:
        attrs.append(f'letter-spacing="{spacing}"')
    attrs.append(f'fill="{fill}"')
    return f"<text {' '.join(attrs)}>{escape(str(s), quote=False)}</text>"


def bars(y0, rows: List[tuple], t: Dict[str, str], left=256, width=516, height=22, gap=12,
         ours=("CalDec Laya", "CalDec GLiNER"), scale=1.0) -> List[str]:
    """One bar per (name, value). The bar is `value / scale` of the track wide; the
    label beside it is always the value itself."""
    out = []
    y = y0
    for name, label in rows:
        value = label / scale
        out.append(f'<rect x="{left}" y="{y}" width="{width}" height="{height}" rx="4" fill="{t["track"]}"/>')
        highlighted = name in ours
        fill = "url(#accent)" if highlighted else t["base"]
        out.append(f'<rect x="{left}" y="{y}" width="{width * value:.1f}" height="{height}" rx="4" fill="{fill}"/>')
        out.append(text(left - 24, y + 15.5, name, 13, 650 if highlighted else None,
                        t["fg"] if highlighted else t["fg2"], anchor="end"))
        shown = str(Decimal(str(label)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        out.append(text(left + width * value + 8, y + 15.5, shown, 12.5,
                        650 if highlighted else None, t["fg"] if highlighted else t["fg2"], mono=True))
        y += height + gap
    return out


def accuracy_chart(numbers: Dict, theme: str) -> str:
    t = THEMES[theme]
    a = numbers["accuracy"]
    cap = numbers["caption"]
    height = max(560, 106 + sum(22 + len(a[key]) * 34 + 40 for key in ("assistant", "public")))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="{height}" viewBox="0 0 880 {height}" '
           f'role="img" aria-label="Accuracy on two test sets">', GRADIENT,
           f'<rect width="880" height="{height}" fill="{t["bg"]}"/>',
           text(24, 34, "Accuracy on two test sets", 17, 650, t["fg"]),
           text(24, 54, cap["accuracy"], 12.5, None, t["fg2"])]
    y = 96
    for key, title in (("assistant", cap["assistant_split"]), ("public", cap["public_split"])):
        out.append(text(24, y, title.upper(), 12, 650, t["fg3"], spacing="0.6"))
        out.append(f'<line x1="24" y1="{y + 9}" x2="856" y2="{y + 9}" stroke="{t["rule"]}" stroke-width="1"/>')
        rows = list(a[key].items())
        out += bars(y + 22, rows, t)
        y += 22 + len(rows) * 34 + 40
    out.append(text(24, height - 14, cap["accuracy_note"], 11.5, None, t["fg3"]))
    out.append("</svg>")
    return "".join(out)


def calibration_chart(numbers: Dict, theme: str) -> str:
    t = THEMES[theme]
    c = numbers["calibration"]
    cap = numbers["caption"]
    rows = list(c.items())
    h = 150 + len(rows) * 34
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="{h}" viewBox="0 0 880 {h}" '
           f'role="img" aria-label="Soft negative log-likelihood against the teacher distribution, lower is better">', GRADIENT,
           f'<rect width="880" height="{h}" fill="{t["bg"]}"/>',
           text(24, 34, "Calibration: soft NLL against the teacher — lower is better", 17, 650, t["fg"]),
           text(24, 54, cap["calibration"], 12.5, None, t["fg2"]),
           text(24, 96, cap["assistant_all_split"].upper(), 12, 650, t["fg3"], spacing="0.6"),
           f'<line x1="24" y1="105" x2="856" y2="105" stroke="{t["rule"]}" stroke-width="1"/>']
    out += bars(118, rows, t, scale=max(v for _, v in rows) * 1.15)
    out.append(text(24, h - 14, cap["calibration_note"], 11.5, None, t["fg3"]))
    out.append("</svg>")
    return "".join(out)


def per_site_chart(numbers: Dict, theme: str) -> str:
    t = THEMES[theme]
    sites = sorted(numbers["per_site"], key=lambda s: -(s["ours"] - s["base"]))
    cap = numbers["caption"]
    row_h, top = 26, 110
    h = top + len(sites) * row_h + 40
    left, width = 236, 560
    x = lambda v: left + (v - 0.2) / 0.8 * width
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="{h}" viewBox="0 0 880 {h}" '
           f'role="img" aria-label="CalDec Laya accuracy per decision site versus Laya base">', GRADIENT,
           f'<rect width="880" height="{h}" fill="{t["bg"]}"/>',
           text(24, 34, "Every decision site, Laya base → CalDec Laya", 17, 650, t["fg"]),
           text(24, 54, cap["per_site"], 12.5, None, t["fg2"])]
    for tick in (0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(f'<line x1="{x(tick):.1f}" y1="{top - 14}" x2="{x(tick):.1f}" y2="{h - 36}" stroke="{t["rule"]}" stroke-width="1"/>')
        out.append(text(x(tick), top - 20, f"{tick:.1f}", 11, None, t["fg3"], anchor="middle", mono=True))
    y = top
    for s in sites:
        cy = y + row_h / 2
        out.append(text(left - 14, cy + 4, s["site"], 12, None, t["fg2"], anchor="end", mono=True))
        out.append(f'<line x1="{x(s["base"]):.1f}" y1="{cy}" x2="{x(s["ours"]):.1f}" y2="{cy}" stroke="{t["base"]}" stroke-width="2"/>')
        out.append(f'<circle cx="{x(s["base"]):.1f}" cy="{cy}" r="4.5" fill="{t["base"]}"/>')
        out.append(f'<circle cx="{x(s["ours"]):.1f}" cy="{cy}" r="5" fill="#3B6CFF"/>')
        out.append(text(x(max(s["base"], s["ours"])) + 12, cy + 4, f'{s["ours"]:.2f}  n={s["n"]}', 11, None, t["fg3"], mono=True))
        y += row_h
    out.append(text(24, h - 14, cap["per_site_note"], 11.5, None, t["fg3"]))
    out.append("</svg>")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", required=True)
    parser.add_argument("--out", default="assets")
    args = parser.parse_args()
    numbers = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"accuracy-{theme}.svg").write_text(accuracy_chart(numbers, theme), encoding="utf-8")
        (out / f"calibration-{theme}.svg").write_text(calibration_chart(numbers, theme), encoding="utf-8")
        (out / f"per-site-{theme}.svg").write_text(per_site_chart(numbers, theme), encoding="utf-8")
    print(f"wrote {len(list(out.glob('*.svg')))} charts to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

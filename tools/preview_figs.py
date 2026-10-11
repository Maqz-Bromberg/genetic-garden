"""Render every documentation figure onto one page and screenshot it.

    python tools/preview_figs.py [out.png] [name ...]
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_garden import diagrams as D  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def figure_names(argv):
    if argv:
        return argv
    return [n[4:] for n in sorted(dir(D)) if n.startswith("fig_")]


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else "figs.png"
    names = figure_names(argv[2:])
    css = (ROOT / "genetic_garden" / "web" / "docs.css").read_text()
    blocks = []
    for name in names:
        fig = getattr(D, "fig_" + name)()
        blocks.append(
            f'<figure class="plate"><div class="frame">{fig.to_svg()}</div>'
            f'<figcaption><b>{name}</b> — {fig.caption}</figcaption></figure>')
    html = (f'<!doctype html><meta charset="utf-8"><style>{css}'
            f'body{{padding:20px}}</style><div class="layout"><div>'
            f'{"".join(blocks)}</div></div>')

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page(viewport={"width": 960, "height": 900},
                          device_scale_factor=2)
        page.set_content(html, wait_until="load")
        page.screenshot(path=out, full_page=True)
        b.close()
    print(f"{out}  ({len(names)} figures)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

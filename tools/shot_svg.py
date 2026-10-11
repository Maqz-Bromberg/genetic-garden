"""Screenshot SVG files through Chromium, for features ImageMagick's SVG
renderer does not support (gradients, filters, precise text metrics).

    python tools/shot_svg.py out.png a.svg b.svg ...
"""
from __future__ import annotations

import pathlib
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    out, svgs = argv[1], argv[2:]
    from playwright.sync_api import sync_playwright

    blocks = "".join(
        f'<div>{pathlib.Path(p).read_text()}</div>' for p in svgs)
    html = (f'<!doctype html><meta charset="utf-8">'
            f'<style>body{{margin:0;background:#fff;display:flex;'
            f'flex-direction:column;gap:8px}}svg{{display:block}}</style>'
            f'{blocks}')
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        page.set_content(html, wait_until="load")
        page.locator("body").screenshot(path=out)
        browser.close()
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

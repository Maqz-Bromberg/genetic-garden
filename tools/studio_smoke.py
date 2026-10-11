"""Drive the studio in a real browser and prove it works.

    pip install playwright && python -m playwright install chromium
    python tools/studio_smoke.py [--headed] [--shots DIR]

Starts the dev server on an ephemeral port, loads the page, exercises every
control, and fails on any console error, page error or failed request. Takes
a screenshot of each tab on the way through, which is the only way to check
that the thing actually looks right.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_garden.server import Handler  # noqa: E402

TABS = ("strip", "icons", "variants", "divergence", "breed", "garden")


def start_server() -> tuple[ThreadingHTTPServer, str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--shots", default="shots")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright missing: pip install playwright && "
              "python -m playwright install chromium", file=sys.stderr)
        return 2

    os.makedirs(args.shots, exist_ok=True)
    httpd, base = start_server()
    problems: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1500, "height": 1000},
                                    device_scale_factor=2)

            page.on("console", lambda m: m.type == "error"
                    and problems.append(f"console: {m.text}"))
            page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            page.on("requestfailed", lambda r: problems.append(
                f"requestfailed: {r.url} {r.failure}"))

            page.goto(base, wait_until="networkidle")
            page.wait_for_selector("#plant svg", timeout=15000)

            check("initial plant rendered", page.locator("#plant svg").count() == 1)
            check("stats populated", "branches" in page.inner_text("#stats"))
            check("growth readout populated",
                  "stage" in page.inner_text("#growth-readout"))
            check("gene sliders built",
                  page.locator("#genes input[type=range]").count() >= 30,
                  f'{page.locator("#genes input[type=range]").count()} sliders')
            page.screenshot(path=f"{args.shots}/01-initial.png", full_page=True)

            # --- growth: dragging contributions must change the drawing ----
            def plant_svg() -> str:
                return page.inner_html("#plant")

            before = plant_svg()
            page.locator("#n").fill("21")
            page.locator("#n").dispatch_event("input")
            page.wait_for_function(
                "(old) => document.querySelector('#plant').innerHTML !== old",
                arg=before, timeout=10000)
            check("contributions slider changes the plant", True)
            check("stage advanced to mature",
                  "mature" in page.inner_text("#growth-readout"),
                  page.inner_text("#growth-readout"))

            # --- kingdom switch keeps the chromosome -----------------------
            for kingdom in ("bush", "flower", "tree"):
                before = plant_svg()
                page.click(f'#kingdoms button[data-kingdom="{kingdom}"]')
                page.wait_for_function(
                    "(old) => document.querySelector('#plant').innerHTML !== old",
                    arg=before, timeout=10000)
                active = page.locator("#kingdoms button.on").inner_text()
                check(f"kingdom -> {kingdom}", active == kingdom, active)
                page.screenshot(path=f"{args.shots}/02-{kingdom}.png",
                                full_page=True)

            # --- gene editing ----------------------------------------------
            page.click("#kingdoms button[data-kingdom='tree']")
            page.wait_for_timeout(600)
            slider = page.locator("#genes input[type=range]").first
            before = plant_svg()
            slider.fill("0.03")
            slider.dispatch_event("input")
            page.wait_for_function(
                "(old) => document.querySelector('#plant').innerHTML !== old",
                arg=before, timeout=10000)
            low = plant_svg()
            slider.fill("0.97")
            slider.dispatch_event("input")
            page.wait_for_function(
                "(old) => document.querySelector('#plant').innerHTML !== old",
                arg=low, timeout=10000)
            check("gene slider reshapes the plant", plant_svg() != low)

            # --- every tab paints ------------------------------------------
            for tab in TABS:
                page.click(f'#tabs button[data-tab="{tab}"]')
                if tab == "breed":
                    page.wait_for_selector("#breed figure svg", timeout=20000)
                    count = page.locator("#breed figure").count()
                    check("breed shows both parents and a child", count == 3,
                          f"{count} figures")
                elif tab == "garden":
                    page.wait_for_selector("#garden svg", timeout=40000)
                    plants = page.locator("#garden svg > g").count()
                    check("garden plants a scene", plants >= 8,
                          f"{plants} plants")
                    check("garden reports what it planted",
                          "planted" in page.inner_text("#garden-stats"))
                else:
                    page.wait_for_selector(f'[data-pane="{tab}"] svg',
                                           timeout=20000)
                    n = page.locator(f'[data-pane="{tab}"] svg').count()
                    check(f"tab '{tab}' painted {n} svg", n > 0, f"{n} svg")
                page.wait_for_timeout(350)
                page.screenshot(path=f"{args.shots}/03-{tab}.png",
                                full_page=True)

            # --- buttons ----------------------------------------------------
            for button, label in (("#randomize", "randomize"),
                                  ("#reseed", "reseed"),
                                  ("#mutate", "mutate"),
                                  ("#reset", "reset")):
                page.click('#tabs button[data-tab="strip"]')
                page.wait_for_timeout(200)
                before = plant_svg()
                page.click(button)
                try:
                    page.wait_for_function(
                        "(old) => document.querySelector('#plant').innerHTML !== old",
                        arg=before, timeout=6000)
                    check(f"{label} redraws", True)
                except Exception:
                    # reseed/mutate may legitimately land on the same drawing
                    check(f"{label} redraws", label in ("reseed", "mutate"),
                          "no visible change")

            # --- the explanation pages are reachable and render ---------------
            page.set_viewport_size({"width": 1200, "height": 900})
            page.click('.docslink')
            page.wait_for_selector(".cards .card", timeout=15000)
            check("studio links to the explanations",
                  "How it works" in page.inner_text("h1"))
            page.click('.cards .card >> nth=2')          # the skeleton page
            page.wait_for_selector("figure.plate svg.fig", timeout=20000)
            plates = page.locator("figure.plate").count()
            check("a docs page renders its figures", plates >= 5,
                  f"{plates} figures")
            captioned = page.locator("figure.plate figcaption").count()
            check("every figure is captioned", captioned == plates,
                  f"{captioned}/{plates}")
            page.screenshot(path=f"{args.shots}/05-docs.png")
            page.go_back()
            page.go_back()
            page.wait_for_selector("#plant svg", timeout=15000)

            # --- narrow viewport --------------------------------------------
            page.set_viewport_size({"width": 430, "height": 900})
            page.wait_for_timeout(400)
            overflow = page.evaluate(
                "document.documentElement.scrollWidth - "
                "document.documentElement.clientWidth")
            check("no horizontal overflow at 430px", overflow <= 1,
                  f"{overflow}px")
            page.screenshot(path=f"{args.shots}/04-narrow.png", full_page=True)

            browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  {name:<{width}}  {detail}")
    for p in problems:
        print(f"  FAIL  {p}")

    failed = [c for c in checks if not c[1]] + problems
    print(f"\n{len(checks) - len([c for c in checks if not c[1]])}/{len(checks)}"
          f" checks passed, {len(problems)} page errors")
    print(f"screenshots in {os.path.abspath(args.shots)}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line interface.

    python -m genetic_garden serve
    python -m genetic_garden render apollo --kingdom tree -n 8 -o apollo.svg
    python -m genetic_garden series apollo --kingdom tree -o out/
    python -m genetic_garden scene -o garden.svg --count 26
    python -m genetic_garden docs -o genetic_garden/web/docs
    python -m genetic_garden sheet -o gallery.html
    python -m genetic_garden breed apollo borealis --kingdom flower
    python -m genetic_garden genome apollo --kingdom bush
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Optional

from .genome import KINGDOMS, Genome
from .growth import DEFAULT_FULL
from .plant import Plant

SERIES_DEFAULT = (0, 1, 2, 3, 5, 8, 13, 21)


def _write(path: str, text: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _plant(args) -> Plant:
    if getattr(args, "genome_file", None):
        with open(args.genome_file, encoding="utf-8") as fh:
            return Plant.from_genome(Genome.from_dict(json.load(fh)),
                                     full=args.full)
    return Plant.from_id(args.id, args.kingdom, full=args.full)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_render(args) -> int:
    plant = _plant(args)
    svg = plant.svg(args.contributions, size=args.size, detail=args.detail,
                    ground=not args.no_ground, background=args.background,
                    fit=args.fit)
    out = args.out or f"{args.id}-{args.contributions}.svg"
    _write(out, svg)
    print(f"{out}  {plant.kingdom}  {plant.stage(args.contributions)}  "
          f"{len(svg):,} bytes")
    return 0


def cmd_series(args) -> int:
    """Every growth stage of one plant -- the thing to eyeball when tuning."""
    plant = _plant(args)
    counts = args.counts or list(SERIES_DEFAULT)
    for n in counts:
        svg = plant.svg(n, size=args.size, detail=args.detail,
                        background=args.background, fit=args.fit)
        path = os.path.join(args.out, f"{args.id}-{plant.kingdom}-{n:02d}.svg")
        _write(path, svg)
    print(f"wrote {len(counts)} stages to {args.out}/")
    return 0


def cmd_icons(args) -> int:
    plant = _plant(args)
    for n in (args.counts or list(SERIES_DEFAULT)):
        svg = plant.icon(n, size=args.size)
        _write(os.path.join(args.out, f"{args.id}-icon-{n:02d}.svg"), svg)
    print(f"wrote icons to {args.out}/")
    return 0


def cmd_genome(args) -> int:
    print(_plant(args).genome.to_json())
    return 0


def cmd_breed(args) -> int:
    rng = random.Random(args.seed)
    a = Plant.from_id(args.a, args.kingdom, full=args.full)
    b = Plant.from_id(args.b, args.kingdom, full=args.full)
    child = a.cross(b, rng)
    if args.mutate:
        child = child.mutate(args.mutate, rng=rng)
    if args.out:
        _write(args.out, child.svg(args.contributions, size=args.size,
                                   background=args.background))
        print(f"{args.out}  {child.kingdom}")
    else:
        print(child.genome.to_json())
    return 0


def cmd_docs(args) -> int:
    """Generate the explanation pages, figures and all."""
    from .docs import write_site
    written = write_site(args.out)
    total = sum(os.path.getsize(p) for p in written)
    print(f"wrote {len(written)} files to {args.out}/  "
          f"({total / 1024:.0f} KB)")
    return 0


def cmd_scene(args) -> int:
    """A whole garden: plants strewn over a plane at random ages."""
    from .garden import render_scene, scatter
    places = scatter(args.count, args.width, args.height, seed=args.seed,
                     horizon=args.horizon, full=args.full,
                     id_prefix=args.prefix)
    svg = render_scene(places, args.width, args.height, horizon=args.horizon,
                       sky=not args.no_sky, haze=not args.no_haze)
    _write(args.out, svg)
    kinds = {}
    for pl in places:
        kinds[pl.plant.kingdom] = kinds.get(pl.plant.kingdom, 0) + 1
    spread = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
    print(f"{args.out}  {len(places)} plants ({spread})  {len(svg):,} bytes")
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_garden(args) -> int:
    """Render a whole garden from a project manifest.

    Input is JSON: a list of {"id", "contributions", optional "kingdom"},
    or a bare {"id": contributions} mapping. Output is one SVG plus one
    icon per project, and an index the site can consume directly.
    """
    with open(args.manifest, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        entries = [{"id": k, "contributions": v} for k, v in data.items()]
    else:
        entries = list(data)

    index = []
    for entry in entries:
        pid = entry["id"]
        n = int(entry.get("contributions", 0))
        plant = Plant.from_id(pid, entry.get("kingdom"), full=args.full)
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in pid)

        svg_name = f"{slug}.svg"
        icon_name = f"{slug}.icon.svg"
        _write(os.path.join(args.out, svg_name),
               plant.svg(n, size=args.size, background=args.background,
                         fit=args.fit))
        _write(os.path.join(args.out, icon_name),
               plant.icon(n, size=args.icon_size))
        index.append({
            "id": pid,
            "kingdom": plant.kingdom,
            "contributions": n,
            "stage": plant.stage(n),
            "maturity": round(plant.growth(n).value, 4),
            "svg": svg_name,
            "icon": icon_name,
        })

    _write(os.path.join(args.out, "index.json"),
           json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(index)} plants + index.json to {args.out}/")
    return 0


def cmd_sheet(args) -> int:
    """A browsable contact sheet: kingdoms down, growth across, plus a
    variation row and a divergence row."""
    counts = args.counts or list(SERIES_DEFAULT)
    kingdoms = args.kingdoms or list(KINGDOMS)
    ids = args.ids or ["apollo", "borealis", "cinder", "delta", "ember",
                       "fathom", "gallium", "hollow"]
    parts: List[str] = [_SHEET_HEAD]

    for kingdom in kingdoms:
        plant = Plant.from_id(ids[0], kingdom, full=args.full)
        parts.append(f"<h2>{kingdom} &mdash; growth of one genome "
                     f"(&ldquo;{ids[0]}&rdquo;)</h2><div class='row'>")
        for n in counts:
            parts.append(_cell(plant.svg(n, size=args.size), n,
                               plant.stage(n)))
        parts.append("</div>")

    for kingdom in kingdoms:
        parts.append(f"<h2>{kingdom} &mdash; variation between genomes "
                     f"at {args.contributions} contributions</h2>"
                     f"<div class='row'>")
        for pid in ids:
            p = Plant.from_id(pid, kingdom, full=args.full)
            parts.append(_cell(p.svg(args.contributions, size=args.size),
                               pid, p.stage(args.contributions)))
        parts.append("</div>")

    # divergence: one genome, small mutations, shown as they mature
    parts.append("<h2>divergence &mdash; one genome plus small mutations, "
                 "at three ages</h2>")
    for kingdom in kingdoms:
        base = Plant.from_id(ids[0], kingdom, full=args.full)
        family = [base] + [base.mutate(0.45, 0.07, random.Random(100 + i))
                           for i in range(4)]
        for n in (3, 8, 21):
            parts.append(f"<div class='row'><span class='tag'>{kingdom} "
                         f"n={n}</span>")
            for member in family:
                parts.append(_cell(member.svg(n, size=args.size), "", ""))
            parts.append("</div>")

    parts.append("<h2>icons</h2><div class='row icons'>")
    for kingdom in kingdoms:
        p = Plant.from_id(ids[0], kingdom, full=args.full)
        for n in counts:
            for px in (16, 24, 32):
                parts.append(f"<span title='{kingdom} n={n} {px}px'>"
                             f"{p.icon(n, size=px)}</span>")
    parts.append("</div></body></html>")

    _write(args.out, "".join(parts))
    print(f"wrote {args.out}")
    return 0


def _cell(svg: str, label, stage) -> str:
    cap = f"<figcaption>{label}<br><em>{stage}</em></figcaption>" if stage else ""
    return f"<figure>{svg}{cap}</figure>"


_SHEET_HEAD = """<!doctype html><meta charset="utf-8">
<title>genetic-garden contact sheet</title><style>
 body{background:#f4f1e8;color:#3a3630;font:14px/1.5 system-ui,sans-serif;
      margin:28px;max-width:100%}
 h2{font-size:15px;font-weight:600;margin:30px 0 8px;letter-spacing:.02em}
 .row{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-end;
      background:#fffdf7;border:1px solid #e3ddcd;border-radius:8px;padding:10px}
 figure{margin:0;text-align:center}
 figcaption{font-size:11px;color:#8a8175;margin-top:2px}
 .tag{font-size:11px;color:#8a8175;align-self:center;width:70px}
 .icons{align-items:center;gap:10px}
 @media (prefers-color-scheme:dark){
   body{background:#1d1c19;color:#e8e3d7}
   .row{background:#26241f;border-color:#3a362e}
 }
</style><body>"""


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="genetic_garden",
        description="Procedural 2D plants that grow with a project.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp, want_id=True):
        if want_id:
            sp.add_argument("id", help="project id; seeds the genome")
        sp.add_argument("-k", "--kingdom", choices=KINGDOMS, default=None,
                        help="default: derived from the id")
        sp.add_argument("--full", type=int, default=DEFAULT_FULL,
                        help="contributions to full maturity "
                             f"(default {DEFAULT_FULL})")
        sp.add_argument("--size", type=int, default=256)
        sp.add_argument("--background", default=None)
        sp.add_argument("--genome-file", default=None,
                        help="load a genome JSON instead of seeding from id")

    sp = sub.add_parser("render", help="one plant at one growth stage")
    common(sp)
    sp.add_argument("-n", "--contributions", type=int, default=8)
    sp.add_argument("-o", "--out", default=None)
    sp.add_argument("-d", "--detail", choices=("full", "compact", "icon"),
                    default="full")
    sp.add_argument("--no-ground", action="store_true")
    sp.add_argument("--fit", choices=("mature", "bbox", "world"),
                    default="mature",
                    help="mature: scale against this plant full-grown "
                         "(default); bbox: fill the frame; world: shared "
                         "cross-kingdom scale")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("series", help="one plant across growth stages")
    common(sp)
    sp.add_argument("-o", "--out", default="out")
    sp.add_argument("-d", "--detail", choices=("full", "compact", "icon"),
                    default="full")
    sp.add_argument("--counts", type=int, nargs="+", default=None)
    sp.add_argument("--fit", choices=("mature", "bbox", "world"),
                    default="mature")
    sp.set_defaults(func=cmd_series)

    sp = sub.add_parser("icons", help="icon-detail renders across stages")
    common(sp)
    sp.add_argument("-o", "--out", default="out")
    sp.add_argument("--counts", type=int, nargs="+", default=None)
    sp.set_defaults(func=cmd_icons)

    sp = sub.add_parser("genome", help="print a genome as JSON")
    common(sp)
    sp.set_defaults(func=cmd_genome)

    sp = sub.add_parser("breed", help="cross two genomes")
    common(sp, want_id=False)
    sp.add_argument("a")
    sp.add_argument("b")
    sp.add_argument("-n", "--contributions", type=int, default=13)
    sp.add_argument("-o", "--out", default=None)
    sp.add_argument("--mutate", type=float, default=0.0,
                    help="mutation rate applied after crossover")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_breed)

    sp = sub.add_parser("docs", help="generate the how-it-works pages")
    sp.add_argument("-o", "--out", default="genetic_garden/web/docs")
    sp.set_defaults(func=cmd_docs)

    sp = sub.add_parser("scene", help="a garden of plants on a plane")
    common(sp, want_id=False)
    sp.add_argument("-o", "--out", default="garden.svg")
    sp.add_argument("-c", "--count", type=int, default=26)
    sp.add_argument("--width", type=int, default=900)
    sp.add_argument("--height", type=int, default=500)
    sp.add_argument("--horizon", type=float, default=0.20,
                    help="fraction of the frame above the ground plane")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--prefix", default="plot",
                    help="project-id prefix the plants are seeded from")
    sp.add_argument("--no-sky", action="store_true")
    sp.add_argument("--no-haze", action="store_true")
    sp.set_defaults(func=cmd_scene)

    sp = sub.add_parser("serve", help="interactive studio in the browser")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("-p", "--port", type=int, default=8753)
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("garden", help="render a whole garden from a manifest")
    common(sp, want_id=False)
    sp.add_argument("manifest", help="JSON list of projects, or {id: count}")
    sp.add_argument("-o", "--out", default="garden")
    sp.add_argument("--icon-size", type=int, default=24)
    sp.add_argument("--fit", choices=("mature", "bbox", "world"),
                    default="mature")
    sp.set_defaults(func=cmd_garden)

    sp = sub.add_parser("sheet", help="HTML contact sheet for eyeballing")
    common(sp, want_id=False)
    sp.add_argument("-o", "--out", default="gallery.html")
    sp.add_argument("-n", "--contributions", type=int, default=13)
    sp.add_argument("--counts", type=int, nargs="+", default=None)
    sp.add_argument("--kingdoms", nargs="+", choices=KINGDOMS, default=None)
    sp.add_argument("--ids", nargs="+", default=None)
    sp.set_defaults(func=cmd_sheet)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

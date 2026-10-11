"""A local dev server for looking at plants while you tune the algorithm.

    python -m genetic_garden serve

Stdlib only, binds to localhost. It exists so you can drag a gene slider
and watch a mature tree change shape, which is the only reliable way to
tell whether a change to the recursion actually improved anything.

Endpoints
---------
GET  /api/meta            kingdoms, gene groups, expressed ranges, stages
POST /api/render          the workhorse: one genome in, every view out
POST /api/breed           cross two ids, optionally mutate, return the child
POST /api/garden          scatter plants over a plane and compose one scene
"""

from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import random
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import urlparse

from . import garden as garden_mod
from .genome import GENE_GROUPS, GENE_NAMES, KINGDOMS, Genome, gene_ranges
from .growth import DEFAULT_FULL, STAGE_NAMES
from .plant import Plant

WEB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

DEFAULT_STAGES = (0, 1, 2, 3, 5, 8, 13, 21, 34)
ICON_SIZES = (16, 24, 32)


# --------------------------------------------------------------------------
# payload builders
# --------------------------------------------------------------------------

def meta() -> Dict[str, Any]:
    return {
        "kingdoms": list(KINGDOMS),
        "genes": list(GENE_NAMES),
        "groups": {k: list(v) for k, v in GENE_GROUPS.items()},
        "ranges": {k: {g: list(r) for g, r in gene_ranges(k).items()}
                   for k in KINGDOMS},
        "stages": list(STAGE_NAMES),
        "default_stages": list(DEFAULT_STAGES),
        "icon_sizes": list(ICON_SIZES),
        "full": DEFAULT_FULL,
    }


def _genome_from(req: Dict[str, Any]) -> Genome:
    """Either an explicit genome (the editor's working copy) or one seeded
    from a project id. A supplied kingdom always wins, so you can express
    the same chromosome as a tree and as a flower and see the family
    resemblance."""
    kingdom = req.get("kingdom") or None
    if req.get("genome"):
        genome = Genome.from_dict(req["genome"])
    else:
        genome = Genome.from_id(req.get("id") or "apollo", kingdom)
    if kingdom and kingdom != genome.kingdom:
        if kingdom not in KINGDOMS:
            raise ValueError(f"unknown kingdom {kingdom!r}")
        genome = Genome(kingdom, dict(genome.genes), label=genome.label,
                        seed=genome.seed)
    if "seed" in req and req["seed"] is not None:
        genome = Genome(genome.kingdom, dict(genome.genes),
                        label=genome.label, seed=int(req["seed"]))
    return genome


def render_payload(req: Dict[str, Any]) -> Dict[str, Any]:
    genome = _genome_from(req)
    full = max(1, int(req.get("full") or DEFAULT_FULL))
    plant = Plant.from_genome(genome, full=full)

    n = max(0, int(req.get("contributions", 8)))
    size = max(32, min(1024, int(req.get("size", 340))))
    detail = req.get("detail", "full")
    fit = req.get("fit", "mature")
    gamma = float(req.get("gamma", 0.55))
    ground = bool(req.get("ground", True))
    background = req.get("background") or None
    include = set(req.get("include") or ["main"])

    def draw(count: int, px: int) -> str:
        return plant.svg(count, size=px, detail=detail, fit=fit,
                         ground=ground, background=background,
                         growth_gamma=gamma)

    growth = plant.growth(n)
    out: Dict[str, Any] = {
        "genome": genome.to_dict(),
        "traits": {k: round(v, 5) for k, v in sorted(genome.traits().items())},
        "kingdom": genome.kingdom,
        "stage": growth.stage,
        "maturity": round(growth.value, 4),
        "depth": round(growth.depth, 3),
        "leafiness": round(growth.leafiness, 3),
        "bloom": round(growth.bloom, 3),
        "woodiness": round(growth.woodiness, 3),
    }

    if "main" in include:
        svg = draw(n, size)
        _, sk = plant.build(n)
        out["main"] = {
            "svg": svg,
            "bytes": len(svg),
            "branches": len(sk.branches),
            "leaves": len(sk.leaves),
            "masses": len(sk.masses),
            "blooms": len(sk.blooms),
        }

    if "strip" in include:
        counts = [int(c) for c in (req.get("stages") or DEFAULT_STAGES)]
        strip_px = max(48, min(240, int(req.get("strip_size", 132))))
        out["strip"] = [{
            "n": c,
            "stage": plant.stage(c),
            "svg": draw(c, strip_px),
        } for c in counts]

    if "icons" in include:
        counts = [int(c) for c in (req.get("stages") or DEFAULT_STAGES)]
        out["icons"] = [{
            "n": c,
            "sizes": [{"px": px, "svg": plant.icon(c, size=px)}
                      for px in ICON_SIZES],
        } for c in counts]

    if "variants" in include:
        count = max(1, min(24, int(req.get("variant_count", 8))))
        salt = str(req.get("variant_salt", "v0"))
        variant_px = max(48, min(320, int(req.get("variant_size", 150))))
        items: List[Dict[str, Any]] = []
        for i in range(count):
            label = f"{salt}-{i}"
            sibling = Plant.from_id(label, genome.kingdom, full=full)
            items.append({
                "id": label,
                "svg": sibling.svg(n, size=variant_px, detail=detail, fit=fit,
                                   ground=ground, background=background,
                                   growth_gamma=gamma),
                "genome": sibling.genome.to_dict(),
            })
        out["variants"] = items

    if "divergence" in include:
        rate = float(req.get("div_rate", 0.5))
        scale = float(req.get("div_scale", 0.07))
        count = max(2, min(8, int(req.get("div_count", 5))))
        px = max(48, min(240, int(req.get("div_size", 132))))
        salt = int(req.get("div_salt", 0))
        family = [plant] + [
            plant.mutate(rate, scale, random.Random(salt * 1000 + i))
            for i in range(count - 1)]
        out["divergence"] = [{
            "n": c,
            "svgs": [m.svg(c, size=px, detail=detail, fit=fit, ground=ground,
                           background=background, growth_gamma=gamma)
                     for m in family],
        } for c in (int(req.get("div_young", 3)),
                    int(req.get("div_mid", 8)),
                    int(req.get("div_old", 21)))]

    return out


def breed_payload(req: Dict[str, Any]) -> Dict[str, Any]:
    kingdom = req.get("kingdom") or "tree"
    rng = random.Random(int(req.get("seed", 0)))
    a = Genome.from_id(req.get("a") or "apollo", kingdom)
    b = Genome.from_id(req.get("b") or "borealis", kingdom)
    child = a.crossover(b, rng)
    rate = float(req.get("mutate", 0.0))
    if rate > 0:
        child = child.mutate(rate, float(req.get("mutate_scale", 0.1)), rng)

    n = max(0, int(req.get("contributions", 13)))
    px = max(48, min(512, int(req.get("size", 170))))
    full = max(1, int(req.get("full") or DEFAULT_FULL))
    bg = req.get("background") or None

    def one(g: Genome) -> str:
        return Plant.from_genome(g, full=full).svg(n, size=px, background=bg)

    return {
        "parents": [
            {"id": a.label, "svg": one(a), "genome": a.to_dict()},
            {"id": b.label, "svg": one(b), "genome": b.to_dict()},
        ],
        "child": {"svg": one(child), "genome": child.to_dict()},
    }


def garden_payload(req: Dict[str, Any]) -> Dict[str, Any]:
    count = max(1, min(80, int(req.get("count", 26))))
    width = max(240, min(2000, int(req.get("width", 900))))
    height = max(160, min(1400, int(req.get("height", 500))))
    horizon = min(0.6, max(0.05, float(req.get("horizon", 0.20))))
    seed = int(req.get("seed", 0))
    full = max(1, int(req.get("full") or DEFAULT_FULL))

    places = garden_mod.scatter(count, width, height, seed=seed,
                                horizon=horizon, full=full)
    svg = garden_mod.render_scene(places, width, height, horizon=horizon)
    return {
        "svg": svg,
        "bytes": len(svg),
        "seed": seed,
        "planted": len(places),
        "requested": count,
        "plants": [{
            "id": pl.plant.label,
            "kingdom": pl.plant.kingdom,
            "contributions": pl.contributions,
            "stage": pl.plant.stage(pl.contributions),
            "depth": round(pl.depth, 3),
        } for pl in places],
    }


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "genetic-garden"

    def log_message(self, fmt, *args):      # quieter than the default
        if self.path.startswith("/api/"):
            return
        super().log_message(fmt, *args)

    # -- helpers ---------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/meta":
            return self._json(meta())
        if path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, f"no route {path}")
        return self._static(path)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        routes = {"/api/render": render_payload,
                  "/api/breed": breed_payload,
                  "/api/garden": garden_payload}
        handler = routes.get(path)
        if handler is None:
            return self._error(HTTPStatus.NOT_FOUND, f"no route {path}")
        try:
            body = self._read_json()
        except json.JSONDecodeError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, f"bad JSON: {exc}")
        try:
            return self._json(handler(body))
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST,
                               f"{type(exc).__name__}: {exc}")

    def _static(self, path: str) -> None:
        rel = posixpath.normpath(path).lstrip("/")
        if rel in ("", "."):
            rel = "index.html"
        target = os.path.normpath(os.path.join(WEB_ROOT, rel))
        # never serve outside web/
        if not target.startswith(os.path.join(WEB_ROOT, "")):
            return self._error(HTTPStatus.FORBIDDEN, "outside web root")
        if not os.path.isfile(target):
            return self._error(HTTPStatus.NOT_FOUND, f"no file {rel}")
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith(("javascript", "json")):
            ctype += "; charset=utf-8"
        with open(target, "rb") as fh:
            self._send(HTTPStatus.OK, fh.read(), ctype)


def serve(host: str = "127.0.0.1", port: int = 8753,
          open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"genetic-garden studio on {url}  (ctrl-c to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()

"""SVG rendering.

The renderer is deliberately dumb about botany: it takes a `Skeleton` and a
`Palette` and draws them. Its one interesting job is *scale-aware detail*.

Line weights are specified in **output pixels** and converted into plant
space once the fit scale is known. That single trick is what lets the same
plant render as a 512px illustration and a 20px list-header icon without
the twigs vanishing or the outlines turning to mud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from . import geometry, organs, palette as palette_mod
from .genome import BUSH, FLOWER, TREE, Genome
from .geometry import (Vec, add, bbox, blob, clamp, lerp, mul, normalize,
                       offset_side, path_polygon, path_smooth_closed,
                       path_smooth_open, perp, tangents, taper_outline)
from .growth import Growth
from .skeleton import Branch, Leaf, Mass, Skeleton

# Light comes from the upper left, consistently, for every plant.
LIGHT: Vec = normalize((-0.45, 1.0))

# Draw layers, back to front. The plant is stacked like a flat
# illustration: a band of foliage, then wood, then more foliage, so that
# limbs and leaves interleave instead of every twig sitting on top of the
# trunk.
L_BACKDROP, L_GROUND = 0, 5
L_MASS_BACK = 10
L_LEAF_DEEP = 12         # leaves lost inside the crown
L_BRANCH = 20            # all wood, in the order the skeleton emitted it
L_BRANCH_LIGHT = 22      # lit edges, above the wood
L_MASS_MID = 30
L_LEAF_MID = 32          # leaves sandwiched in the foliage
L_MASS_FRONT = 34
L_LEAF_BACK = 40
L_LEAF = 50
L_LEAF_DETAIL = 52       # highlights and veins, above every blade
L_BLOOM, L_ACCENT = 60, 70



DETAIL_LEVELS = {"icon": 0, "compact": 1, "full": 2}


@dataclass
class Style:
    """Line weights and thresholds, all in output pixels."""
    detail: int
    min_feature_px: float     # below this on-screen length, do not draw at all
    min_branch_px: float
    outline_px: float
    leaf_outline_px: float
    vein_px: float
    leaf_keep: float          # fraction of leaves to draw
    mass_boost: float         # masses grow slightly at small sizes
    mass_cap: int             # most blobs worth drawing at this size


def style_for(detail: int, size: float) -> Style:
    if detail <= 0:
        return Style(0, 4.5, 1.9, 0.0, 0.0, 0.0, 0.30, 1.14, 9)
    if detail == 1:
        return Style(1, 2.6, 1.3, 0.9, 0.0, 0.0, 0.55, 1.05, 14)
    # Full detail still respects the canvas: a 64px "full" render needs
    # heavier minimums than a 512px one.
    k = clamp((size - 48.0) / 200.0)
    return Style(2, lerp(2.3, 0.9, k), lerp(1.4, 0.75, k), lerp(1.0, 1.4, k),
                 lerp(0.0, 0.7, k), lerp(0.0, 0.85, k), 1.0, 1.0, 64)


# --------------------------------------------------------------------------
# element accumulation
# --------------------------------------------------------------------------

class _Canvas:
    """Collects paths and emits them grouped by shared presentation.

    Almost every path in a plant shares its fill and stroke with dozens of
    others, and the repeated attribute soup costs more than the geometry
    does. Paths are bucketed by (layer, style) and emitted inside one `<g>`
    per bucket, which roughly halves the output.
    """

    def __init__(self) -> None:
        self._buckets: Dict[Tuple[int, str], List[str]] = {}
        self._order: List[Tuple[int, int, str]] = []
        self._raw: List[Tuple[int, int, str]] = []
        self._seq = 0

    def _bucket(self, layer: int, attrs: str) -> List[str]:
        key = (layer, attrs)
        if key not in self._buckets:
            self._buckets[key] = []
            self._order.append((layer, self._seq, attrs))
            self._seq += 1
        return self._buckets[key]

    def add(self, layer: int, svg: str) -> None:
        """Escape hatch for a one-off element."""
        self._raw.append((layer, self._seq, svg))
        self._seq += 1

    def path(self, layer: int, d: str, fill: str, *, opacity: float = 1.0,
             stroke: str = "", stroke_width: float = 0.0) -> None:
        if not d:
            return
        attrs = f'fill="{fill}"'
        if opacity < 0.999:
            attrs += f' opacity="{opacity:.2f}"'
        if stroke and stroke_width > 0:
            attrs += (f' stroke="{stroke}" stroke-width="{stroke_width:.2f}"'
                      f' stroke-linejoin="round"')
        self._bucket(layer, attrs).append(d)

    def stroke(self, layer: int, d: str, color: str, width: float,
               opacity: float = 1.0) -> None:
        if not d or width <= 0:
            return
        attrs = (f'fill="none" stroke="{color}" stroke-width="{width:.2f}"'
                 f' stroke-linecap="round"')
        if opacity < 0.999:
            attrs += f' opacity="{opacity:.2f}"'
        self._bucket(layer, attrs).append(d)

    def emit(self) -> str:
        parts: List[Tuple[int, int, str]] = list(self._raw)
        for layer, seq, attrs in self._order:
            ds = self._buckets[(layer, attrs)]
            if not ds:
                continue
            if len(ds) == 1:
                parts.append((layer, seq, f'<path {attrs} d="{ds[0]}"/>'))
            else:
                body = "".join(f'<path d="{d}"/>' for d in ds)
                parts.append((layer, seq, f'<g {attrs}>{body}</g>'))
        return "".join(svg for _, _, svg in
                       sorted(parts, key=lambda it: (it[0], it[1])))


# --------------------------------------------------------------------------
# extent
# --------------------------------------------------------------------------

def content_bounds(sk: Skeleton) -> Tuple[float, float, float, float]:
    pts: List[Vec] = [(0.0, 0.0)]
    if sk.seed_radius:
        r = sk.seed_radius
        pts.extend([(-r, 0.0), (r, r * 1.3)])
    for br in sk.branches:
        hw = max(br.widths) * 0.5
        for p in br.points:
            pts.append((p[0] - hw, p[1] - hw))
            pts.append((p[0] + hw, p[1] + hw))
    for m in sk.masses:
        pts.append((m.center[0] - m.radius, m.center[1] - m.radius * m.squash))
        pts.append((m.center[0] + m.radius, m.center[1] + m.radius * m.squash))
    for lf in list(sk.leaves) + list(sk.cotyledons):
        pts.append((lf.pos[0] - lf.size, lf.pos[1] - lf.size))
        pts.append((lf.pos[0] + lf.size, lf.pos[1] + lf.size))
    for bl in sk.blooms:
        r = bl.size * (2.8 if bl.form == organs.SPIKE else 1.45)
        pts.append((bl.pos[0] - r, bl.pos[1] - r))
        pts.append((bl.pos[0] + r, bl.pos[1] + r))
    x0, y0, x1, y1 = bbox(pts)
    return x0, min(y0, 0.0), x1, y1


# --------------------------------------------------------------------------
# drawing the parts
# --------------------------------------------------------------------------

def _buried(br: Branch, masses: Sequence[Mass]) -> bool:
    """True if every part of this branch lies inside the foliage.

    On a mature tree most twigs are completely hidden by the crown. Drawing
    them costs a third of the file and changes not one pixel.
    """
    # Primary limbs are never hidden: a crown you cannot see any wood
    # inside reads as a lollipop, and the reference trees all show their
    # structure through the leaves.
    if br.order <= 2 or br.kind in ("trunk", "stem") or not masses:
        return False
    probes = (br.points[0], br.points[len(br.points) // 2], br.points[-1])
    for p in probes:
        for m in masses:
            rx = m.radius * 0.82
            ry = m.radius * m.squash * 0.82
            dx = (p[0] - m.center[0]) / rx
            dy = (p[1] - m.center[1]) / ry
            if dx * dx + dy * dy <= 1.0:
                break
        else:
            return False
    return True


def _draw_branches(c: _Canvas, sk: Skeleton, pal, st: Style, px: float,
                   masses: Sequence[Mass]) -> None:
    """Draw the wood in the order the skeleton emitted it.

    That order is the whole trick behind the junctions: limbs passing
    behind a branch are emitted before it, its own continuation after it.
    A branch that has something continuing from its tip is therefore left
    blunt, because the next segment's collar covers the cut; only a genuine
    terminal twig tapers to a point. Sorting here would undo all of it.
    """
    cutoff = st.min_feature_px * px
    for br in sk.branches:
        if _buried(br, masses):
            continue
        # A branch shorter than a couple of pixels contributes nothing but
        # file size. Dropping it beats clamping it up to a visible width,
        # which is what turns small renders into a ball of fat twigs.
        if _span(br.points) < cutoff:
            continue

        widths = [max(w, st.min_branch_px * px) for w in br.widths]
        # A twig only a couple of pixels wide is indistinguishable from a
        # round-capped stroke, and a stroke costs a quarter of the path data.
        if max(widths) <= 3.0 * px:
            c.stroke(L_BRANCH, path_smooth_open(br.points, tension=0.6),
                     pal.bark, max(widths))
            continue
        outline = taper_outline(br.points, widths,
                                tip_ratio=0.0 if br.terminal else 1.0)
        if not outline:
            continue
        # Only the trunk is wide enough for the smoothing to be visible;
        # on a limb the straight-edged outline is identical and a third
        # of the path data.
        d = (path_smooth_closed(outline, tension=0.55)
             if br.kind in ("trunk", "stem") else path_polygon(outline))
        c.path(L_BRANCH, d, pal.bark,
               stroke=pal.outline if st.outline_px else "",
               stroke_width=st.outline_px * px)
        # A lit edge along whichever flank faces the light. Worth it only
        # on genuinely broad wood: at 40% opacity on a five-pixel twig it
        # is invisible, and it was costing more than the branches did.
        if st.detail >= 1 and max(widths) > 9.0 * px:
            tans = tangents(br.points)
            mid = tans[len(tans) // 2]
            side = 1.0 if (perp(mid)[0] * LIGHT[0] +
                           perp(mid)[1] * LIGHT[1]) > 0 else -1.0
            ribbon = offset_side(br.points, widths, side, inset=0.30)
            d_light = (path_smooth_closed(ribbon, tension=0.55)
                       if br.kind in ("trunk", "stem")
                       else path_polygon(ribbon))
            c.path(L_BRANCH_LIGHT, d_light, pal.bark_light, opacity=0.40)


def _span(pts: Sequence[Vec]) -> float:
    return math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])


def _draw_masses(c: _Canvas, sk: Skeleton, pal, st: Style, px: float,
                 genome: Genome, blossom: float = 0.0) -> List[Mass]:
    """Three passes over the same blobs: a dark shell offset away from the
    light, the body, then a small lit cap on the sunward side. Cheap, and
    it gives the crown volume without any gradients."""
    masses = sk.masses
    if not masses:
        return []
    # Foliage is the main thing identifying a plant at small sizes, so
    # masses survive well below the cutoff that prunes twigs.
    masses = [m for m in masses if m.radius > st.min_feature_px * px * 0.30]
    if not masses:
        masses = sorted(sk.masses, key=lambda m: -m.radius)[:3]
    if len(masses) > st.mass_cap:
        # keep the biggest: they carry the silhouette, the rest are texture
        masses = sorted(masses, key=lambda m: -m.radius)[:st.mass_cap]
    rng = genome.rng("mass-render")
    fluff = genome.trait("canopy_fluff")
    lobes = 9 if st.detail >= 1 else 6
    shade_off = mul(LIGHT, -1.0)
    # A crab-apple in flower is a cloud of blossom with a little leaf in it,
    # so the canopy masses themselves take the petal colour.
    back = palette_mod.mix(pal.mass_back, pal.petal_shadow, blossom)
    body = palette_mod.mix(pal.mass_mid, pal.petal, blossom)
    cap = palette_mod.mix(pal.mass_front, pal.petal_light, blossom)
    tops = sorted(m.center[1] for m in masses)
    median_y = tops[len(tops) // 2]

    for m in masses:
        r = m.radius * st.mass_boost
        jitter = [1.0 + rng.uniform(-fluff, fluff) for _ in range(lobes)]
        phase = rng.uniform(0.0, math.tau)

        shell = blob(add(m.center, mul(shade_off, r * 0.13)), r * 1.06,
                     lobes, jitter, squash=m.squash, phase=phase)
        c.path(L_MASS_BACK, path_smooth_closed(shell), back,
               stroke=pal.outline if st.outline_px else "",
               stroke_width=st.outline_px * px * 0.9)

        disc = blob(m.center, r, lobes, jitter, squash=m.squash, phase=phase)
        c.path(L_MASS_MID, path_smooth_closed(disc), body)

        if st.detail >= 1 and m.center[1] >= median_y:
            crest = blob(add(m.center, mul(LIGHT, r * 0.34)), r * 0.54,
                         lobes, jitter, squash=m.squash, phase=phase + 0.7)
            c.path(L_MASS_FRONT, path_smooth_closed(crest), cap,
                   opacity=0.85)
    return masses


# Only a few leaves go right to the back: below the canopy masses a blade
# is invisible, so a deep band any wider than this is pure file size.
_LEAF_BANDS = (
    (0.08, L_LEAF_DEEP, "blade_deep"),
    (0.30, L_LEAF_MID, "blade_shadow"),
    (0.65, L_LEAF_BACK, "blade"),
    (2.00, L_LEAF, "blade"),
)


def _leaf_depth(lf: Leaf, pal) -> Tuple[int, str]:
    """Which depth band a blade sits in, and how dark it is there.

    Scattering leaves through the stack -- some lost behind the branches,
    some in the middle of the foliage, some out in front -- is what gives
    the crown thickness. A single layer of leaves always reads as a decal.
    """
    for cut, layer, colour in _LEAF_BANDS:
        if lf.depth < cut:
            return layer, getattr(pal, colour)
    return L_LEAF, pal.blade


def _draw_leaves(c: _Canvas, leaves: Sequence[Leaf], t: Dict[str, float],
                 c_pal, st: Style, px: float, feature_size: float,
                 kingdom: str) -> None:
    """`feature_size` is the blade size above which a leaf is a feature of
    the drawing rather than texture in a mass. Only features get veins and
    highlights -- nobody draws veins on an individual leaf of an oak."""
    for lf in leaves:
        if lf.size < 1.6 * px:      # smaller than a couple of pixels: skip
            continue
        layer, fill = _leaf_depth(lf, c_pal)
        # A toothed margin costs three times a smooth one to describe and
        # is invisible below roughly a thumbnail's worth of pixels.
        det = st.detail if lf.size / px > 24.0 else min(st.detail, 1)
        paths = organs.leaf_paths(lf.pos, lf.angle, lf.size, lf.flip, t,
                                  detail=det, kingdom=kingdom)
        blades = paths["blades"]
        # A dark edge is what makes a blade read against the canopy, but a
        # frond's leaflets overlap, so theirs is drawn finer to stop the
        # overlaps turning into scribble.
        outline_px = st.leaf_outline_px * (1.0 if len(blades) == 1 else 0.65)
        for blade in blades:
            c.path(layer, blade, fill,
                   stroke=c_pal.outline if outline_px else "",
                   stroke_width=outline_px * px)
        rachis = paths.get("rachis")
        if rachis and st.detail >= 1:
            c.stroke(layer, rachis, c_pal.blade_shadow,
                     max(0.7 * px, lf.size * 0.035))
        if (det >= 2 and layer >= L_LEAF_BACK
                and lf.size > max(7.0 * px, feature_size)):
            c.path(L_LEAF_DETAIL, paths.get("light", ""), c_pal.blade_light,
                   opacity=0.38)
            c.stroke(L_LEAF_DETAIL, paths.get("vein", ""), c_pal.vein,
                     st.vein_px * px, opacity=0.55)


def _is_scatter(bl: Bloom, kingdom: str) -> bool:
    """Blossom on a woody plant is texture over a crown and gets the plain
    five-petal treatment. Every bloom on a flower keeps the plant's own
    archetype, terminal or not -- a campanula does not grow one bell and
    two daisies."""
    return kingdom != FLOWER and bl.form in (organs.SCATTER, organs.BLOSSOM)

_BLOOM_FILL = {
    "bud": "petal", "sepal": "blade_shadow",
    "petal_back": "petal_shadow", "petal": "petal",
    "petal_light": "petal_light",
    "center": "bloom_center", "center_light": "bloom_center_light",
    "stamen": "petal_light",
    "globe_back": "petal_shadow", "globe": "petal",
    "floret": "petal_light", "floret_shade": "petal_shadow",
    "floret_center": "bloom_center",
}


def _draw_blooms(c: _Canvas, sk: Skeleton, t: Dict[str, float], pal,
                 st: Style, px: float, kingdom: str) -> None:
    for bl in sk.blooms:
        if bl.size < 1.8 * px:
            continue
        for role, d in organs.bloom_shapes(bl.pos, bl.angle, bl.size,
                                           bl.openness, t, detail=st.detail,
                                           simple=_is_scatter(bl, kingdom),
                                           form=bl.form):
            fill = getattr(pal, _BLOOM_FILL[role])
            outline = st.outline_px * px * 0.75 if role in (
                "petal", "petal_back", "bud") else 0.0
            c.path(L_BLOOM, d, fill,
                   opacity=0.42 if role == "petal_light" else 1.0,
                   stroke=pal.outline if outline else "",
                   stroke_width=outline)
    for ac in sk.accents:
        if ac.size < 1.2 * px:
            continue
        for role, d in organs.accent_paths(ac.pos, ac.size):
            c.path(L_ACCENT, d, getattr(pal, role))


def _draw_seed(c: _Canvas, sk: Skeleton, pal, st: Style, px: float) -> None:
    """A sown seed: no contributions yet, so there is nothing above ground
    but the thing itself, sitting in the soil."""
    r = max(sk.seed_radius, 1.6 * px)
    cy = r * 0.52
    c.path(L_LEAF, _ellipse((0.0, cy), r, r * 0.74), pal.seed,
           stroke=pal.outline if st.outline_px else "",
           stroke_width=st.outline_px * px)
    if st.detail >= 1:
        c.path(L_LEAF_DETAIL, _ellipse((-r * 0.26, cy + r * 0.22), r * 0.42,
                                       r * 0.30), pal.seed_light, opacity=0.65)


def _ellipse(c0: Vec, rx: float, ry: float) -> str:
    return (f"M{c0[0] - rx:.2f},{c0[1]:.2f}"
            f"a{rx:.2f},{ry:.2f} 0 1,0 {2 * rx:.2f},0"
            f"a{rx:.2f},{ry:.2f} 0 1,0 {-2 * rx:.2f},0Z")


def _draw_ground(c: _Canvas, sk: Skeleton, pal, st: Style, px: float,
                 x0: float, x1: float) -> None:
    half = max((x1 - x0) * 0.34, sk.base_width * 1.8)
    h = half * 0.17
    c.path(L_GROUND,
           f"M{-half:.2f},0 a{half:.2f},{h:.2f} 0 1,0 {2 * half:.2f},0"
           f" a{half:.2f},{h:.2f} 0 1,0 {-2 * half:.2f},0Z",
           pal.ground, opacity=0.30)
    mound = [(-half * 0.8, 0.0), (-half * 0.3, h * 0.9), (half * 0.25, h * 1.1),
             (half * 0.75, 0.0)]
    c.stroke(L_GROUND, path_smooth_open(mound), pal.ground_light,
             max(1.0 * px, 0.6), opacity=0.5)


# --------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------

def render(genome: Genome, growth: Growth, skeleton_obj: Skeleton, *,
           width: int = 256, height: int | None = None,
           detail: str | int = "full", ground: bool = True,
           background: str | None = None, padding: float = 0.06,
           world_height: float | None = None,
           reference: Tuple[float, float] | None = None,
           growth_gamma: float = 0.55) -> str:
    """Render a built skeleton to a standalone SVG string.

    Scaling has three modes, and the choice matters more than it sounds:

    * `reference` -- fit against a span you supply, normally this plant's
      own mature size. A seedling then occupies a corner of the frame and
      a mature tree fills it, so growth is legible at a glance. This is
      what `Plant.svg` uses, and it is the reason a garden reads as a
      garden rather than as a row of equally large plants.
    * `world_height` -- fit against a shared world unit, so a mature tree
      towers over a mature flower.
    * neither -- fit the bounding box, filling the frame. Right for icons,
      where legibility beats conveying scale.
    """
    height = height or width
    det = DETAIL_LEVELS[detail] if isinstance(detail, str) else int(detail)
    st = style_for(det, min(width, height))
    sk = skeleton_obj

    x0, y0, x1, y1 = content_bounds(sk)
    pad = padding * min(width, height)
    avail_w, avail_h = width - 2 * pad, height - 2 * pad
    span_x = max(x1 - x0, 1e-3)
    span_y = max(y1 - y0, 1e-3)
    if world_height:
        scale = avail_h / world_height
    elif reference:
        # How big is this stage compared with the same plant full-grown?
        # Raising that ratio to a power under 1 compresses the range: a
        # sprout still reads as small, but stays big enough to recognise
        # instead of becoming three pixels in the corner of the frame.
        ref_x, ref_y = max(reference[0], 1e-3), max(reference[1], 1e-3)
        ratio = clamp(max(span_x / ref_x, span_y / ref_y), 0.0, 1.0)
        scale = (min(avail_w / span_x, avail_h / span_y)
                 * ratio ** growth_gamma)
    else:
        scale = min(avail_w / span_x, avail_h / span_y)

    # centre horizontally on the content, sit the base on the ground line
    cx = (x0 + x1) * 0.5
    tx = width * 0.5 - cx * scale
    ty = height - pad - max(0.0, -y0) * scale

    body = draw_body(genome, growth, sk, st=st, scale=scale, ground=ground)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_label(genome, growth)}">'
        + (f'<rect width="{width}" height="{height}" fill="{background}"/>'
           if background else "")
        + f'<g transform="translate({tx:.3f},{ty:.3f}) scale({scale:.5f},'
        f'{-scale:.5f})">{body}</g></svg>'
    )


def draw_body(genome: Genome, growth: Growth, sk: Skeleton, *, st: Style,
              scale: float, ground: bool = True) -> str:
    """Draw one plant as SVG path data in plant space (origin at the base,
    y up). The caller supplies the transform, which is what lets several
    plants share one scene as well as each having its own frame."""
    pal = palette_mod.build(genome, growth)
    t = genome.traits()
    x0, y0, x1, y1 = content_bounds(sk)
    span_x = max(x1 - x0, 1e-3)
    span_y = max(y1 - y0, 1e-3)
    px = 1.0 / scale                      # one output pixel, in plant space

    # ~0.3px of positional error is invisible and roughly halves the file
    decimals = int(clamp(math.ceil(-math.log10(max(px * 0.3, 1e-6))), 0, 3))

    c = _Canvas()
    leaves = list(sk.leaves)
    if st.leaf_keep < 1.0 and len(leaves) > 8:
        step = max(1, int(round(1.0 / st.leaf_keep)))
        leaves = leaves[::step]

    with geometry.precision(decimals):
        if ground:
            _draw_ground(c, sk, pal, st, px, x0, x1)
        if sk.seed_radius:
            _draw_seed(c, sk, pal, st, px)
        blossom = 0.0
        if (genome.kingdom == TREE
                and organs.bloom_form(TREE, t) == organs.BLOSSOM):
            blossom = clamp(growth.bloom * 0.85)
        masses = _draw_masses(c, sk, pal, st, px, genome, blossom)
        _draw_branches(c, sk, pal, st, px, masses)
        feature_size = max(span_x, span_y) * 0.085
        _draw_leaves(c, leaves, t, pal, st, px, feature_size, genome.kingdom)
        for lf in sk.cotyledons:
            round_ness = {TREE: 0.25, BUSH: 0.55, FLOWER: 0.85}[genome.kingdom]
            paths = organs.cotyledon_paths(lf.pos, lf.angle, lf.size, lf.flip,
                                           round_ness)
            c.path(L_LEAF, paths["blade"], pal.blade,
                   stroke=pal.outline if st.outline_px else "",
                   stroke_width=st.outline_px * px)
            if st.detail >= 1:
                c.path(L_LEAF, paths["light"], pal.blade_light, opacity=0.34)
        _draw_blooms(c, sk, t, pal, st, px, genome.kingdom)
    return c.emit()


def plant_group(genome: Genome, growth: Growth, sk: Skeleton, *,
                scale: float, x: float, y: float,
                detail: str | int = "auto", ground: bool = True) -> str:
    """One plant as a positioned `<g>`, for compositing into a scene.

    `detail="auto"` picks the level from how big the plant will actually
    be on screen, which is what keeps a garden of thirty plants from
    weighing several megabytes.
    """
    on_screen = max(sk.canonical_height * scale, 4.0)
    if detail == "auto":
        # A scene is a view, not a portrait. Nothing in it earns full
        # detail: veins and per-leaf outlines are invisible at these sizes
        # and a garden of them runs to megabytes.
        det = 1 if on_screen >= 78 else 0
    else:
        det = DETAIL_LEVELS[detail] if isinstance(detail, str) else int(detail)
    st = style_for(det, on_screen)
    body = draw_body(genome, growth, sk, st=st, scale=scale, ground=ground)
    return (f'<g transform="translate({x:.2f},{y:.2f}) '
            f'scale({scale:.5f},{-scale:.5f})">{body}</g>')


def _label(genome: Genome, growth: Growth) -> str:
    name = genome.label or genome.kingdom
    return (f"{name}: {growth.stage} {genome.kingdom}, "
            f"{growth.contributions} contributions")

"""Leaves, petals and blooms.

Every blade in the system -- leaf, cotyledon, petal, sepal, bud -- is the
same parametric shape with different constants:

    half_width(u) = w * sin(pi * u**e) * (1 - tooth * triangle(u * n))

`e` slides the widest point from base (ovate, e<1) to tip (obovate, e>1);
`tooth` serrates the margin; a curved midline gives the blade its droop.
One function, and the genome picks a leaf out of the whole continuum.

Shapes are generated in a local frame -- base at the origin, tip at (1, 0),
unit length -- then placed. Outputs are SVG path data strings in plant
space; colour is applied by the renderer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .genome import BUSH, FLOWER, TREE
from .geometry import (TAU, Vec, _fmt, add, blob, clamp, cubic_through,
                       curve_to, from_angle, lerp, move_to,
                       path_smooth_closed, path_smooth_open, rotate)


UP = math.pi / 2


def _triangle(x: float) -> float:
    """0..1 triangle wave with period 1."""
    f = x - math.floor(x)
    return 2.0 * abs(f - 0.5)


def blade_local(width: float, tip_shift: float, *, teeth: int = 0,
                tooth_depth: float = 0.0, curl: float = 0.0,
                notch: float = 0.0, samples: int = 7
                ) -> Tuple[List[Vec], List[Vec]]:
    """Return (outline loop, midrib polyline) in the unit local frame."""
    def mid(u: float) -> Vec:
        return (u, curl * math.sin(math.pi * u))

    def hw(u: float) -> float:
        w = width * math.sin(math.pi * max(u, 0.0) ** tip_shift)
        if teeth and tooth_depth:
            w *= 1.0 - tooth_depth * _triangle(u * teeth)
        return w

    us = [i / (samples + 1) for i in range(1, samples + 1)]
    left = [(mid(u)[0], mid(u)[1] + hw(u)) for u in us]
    right = [(mid(u)[0], mid(u)[1] - hw(u)) for u in us]

    base = mid(0.0)
    tip = mid(1.0)
    if notch > 0.0:
        # pull the tip back along the midline to open a cleft
        tip = (1.0 - 0.17 * notch, mid(1.0 - 0.17 * notch)[1])
    outline = [base] + left + [tip] + list(reversed(right))
    midrib = [mid(u) for u in (0.0, 0.25, 0.5, 0.75, 0.94)]
    return outline, midrib


def blade_fast(pos: Vec, angle: float, size: float, flip: int, *,
               width: float, tip_shift: float, curl: float = 0.0,
               tip_u: float = 1.0) -> str:
    """A smooth blade as exactly two cubic segments.

    Used for every un-serrated leaf and petal, which is nearly all of them.
    Serrated margins fall back to the sampled spline in `blade_local`.
    """
    def mid(u: float) -> Vec:
        return (u, curl * math.sin(math.pi * u))

    def hw(u: float) -> float:
        return width * math.sin(math.pi * max(u, 0.0) ** tip_shift)

    def at(u: float, side: int) -> Vec:
        m = mid(u)
        return (m[0], m[1] + side * hw(u))

    a, b = tip_u / 3.0, tip_u * 2.0 / 3.0
    base, tip = mid(0.0), mid(tip_u)

    def put(v: Vec) -> Vec:
        return add(pos, rotate((v[0] * size, v[1] * size * flip), angle))

    up1, up2 = cubic_through(base, at(a, 1), at(b, 1), tip)
    dn1, dn2 = cubic_through(tip, at(b, -1), at(a, -1), base)
    return (move_to(put(base))
            + curve_to(put(up1), put(up2), put(tip))
            + curve_to(put(dn1), put(dn2), put(base)) + "Z")


def blade_fast_half(pos: Vec, angle: float, size: float, flip: int, *,
                    width: float, tip_shift: float, curl: float = 0.0) -> str:
    """The lit margin of a blade: one curve out, the midline straight back."""
    def mid(u: float) -> Vec:
        return (u, curl * math.sin(math.pi * u))

    def hw(u: float) -> float:
        return width * math.sin(math.pi * max(u, 0.0) ** tip_shift) * 0.62

    def at(u: float) -> Vec:
        m = mid(u)
        return (m[0], m[1] + hw(u))

    def put(v: Vec) -> Vec:
        return add(pos, rotate((v[0] * size, v[1] * size * flip), angle))

    base, tip = mid(0.0), mid(1.0)
    c1, c2 = cubic_through(base, at(1 / 3), at(2 / 3), tip)
    return (move_to(put(base)) + curve_to(put(c1), put(c2), put(tip))
            + f"L{_fmt(put(base)[0])},{_fmt(put(base)[1])}Z")


def place(pts: Sequence[Vec], pos: Vec, angle: float, size: float,
          flip: int = 1) -> List[Vec]:
    return [add(pos, rotate((p[0] * size, p[1] * size * flip), angle))
            for p in pts]


# --------------------------------------------------------------------------
# leaves
# --------------------------------------------------------------------------

# Leaf architecture. One blade is the common case, but a frond, a hand of
# leaflets or a needle tuft change a plant's texture more than any amount
# of recolouring, and they are what separates a locust from a maple from a
# cedar at a glance.
SIMPLE, COMPOUND, PALMATE, NEEDLE = "simple", "compound", "palmate", "needle"


def leaf_form(kingdom: str, t: Dict[str, float]) -> str:
    v = t["leaf_form"]
    if kingdom == TREE:
        return SIMPLE if v < 0.56 else (COMPOUND if v < 0.83 else NEEDLE)
    if kingdom == BUSH:
        return SIMPLE if v < 0.50 else (PALMATE if v < 0.80 else NEEDLE)
    return SIMPLE if v < 0.72 else PALMATE


def leaflet_count(form: str) -> int:
    """Blades drawn per leaf. The foliage budget is denominated in blades,
    not leaves, so a compound-leaved tree does not cost five times as much
    to draw as a simple-leaved one."""
    return {SIMPLE: 1, COMPOUND: 5, PALMATE: 5, NEEDLE: 5}[form]


# A frond or a needle tuft has to be drawn bigger than a single blade or
# its leaflets shrink to invisible specks.
# A budget denominated in blades buys far fewer compound leaves than simple
# ones, so each has to be correspondingly larger to read -- which is also
# what they look like in life. A locust frond really is that big.
_FORM_SCALE = {SIMPLE: 1.0, COMPOUND: 2.30, PALMATE: 1.50, NEEDLE: 1.95}


def form_size_scale(form: str) -> float:
    return _FORM_SCALE[form]


def _blade_geometry(t: Dict[str, float]) -> Tuple[float, float, float]:
    w = t["leaf_slender"]
    return w, lerp(0.62, 1.05, w / 0.62), 0.10 + 0.14 * w


def leaf_paths(pos: Vec, angle: float, size: float, flip: int,
               t: Dict[str, float], *, detail: int = 2,
               kingdom: str = TREE) -> Dict[str, object]:
    """A leaf as {blades: [path, ...], rachis, light, vein}.

    `blades` is a list because a compound or palmate leaf is several. The
    interior detail keys are dropped at low detail, which is what keeps
    icons clean.
    """
    form = leaf_form(kingdom, t)
    w, tip_shift, curl = _blade_geometry(t)

    if form == COMPOUND:
        return _compound(pos, angle, size, flip, t, w, detail)
    if form == PALMATE:
        return _palmate(pos, angle, size, flip, t, w, tip_shift, detail)
    if form == NEEDLE:
        return _needle(pos, angle, size, flip, t)

    # --- a single blade -------------------------------------------------
    lobes = t["leaf_lobes"]
    teeth = 0 if lobes < 0.45 else (5 if lobes < 0.75 else 7)
    tooth_depth = 0.0 if not teeth else lerp(0.06, 0.20, (lobes - 0.45) / 0.55)

    out: Dict[str, object] = {}
    if teeth and detail >= 2:
        outline, midrib = blade_local(w, tip_shift, teeth=teeth,
                                      tooth_depth=tooth_depth, curl=curl,
                                      samples=7)
        out["blades"] = [path_smooth_closed(
            place(outline, pos, angle, size, flip), tension=0.9)]
    else:
        out["blades"] = [blade_fast(pos, angle, size, flip, width=w,
                                    tip_shift=tip_shift, curl=curl)]
        midrib = [(u, curl * math.sin(math.pi * u))
                  for u in (0.0, 0.32, 0.64, 0.94)]
    if detail >= 1:
        out["vein"] = path_smooth_open(place(midrib, pos, angle, size, flip))
        out["light"] = blade_fast_half(pos, angle, size, flip, width=w,
                                       tip_shift=tip_shift, curl=curl)
    return out


def _compound(pos: Vec, angle: float, size: float, flip: int,
              t: Dict[str, float], w: float, detail: int) -> Dict[str, object]:
    """Leaflets in opposite pairs up a rachis, plus a terminal one."""
    pairs = 2
    spread = lerp(0.88, 1.18, t["leaf_pitch"] / 1.05)
    leaflet = size * lerp(0.34, 0.44, t["leaf_size"] / 1.35)
    lw = clamp(w * 0.95, 0.24, 0.52)
    blades = []
    for i in range(pairs):
        u = 0.30 + 0.56 * (i / max(pairs - 1, 1))
        anchor = add(pos, from_angle(angle, size * u))
        scale = leaflet * lerp(1.12, 0.82, u)
        for side in (1, -1):
            blades.append(blade_fast(anchor, angle + side * spread * flip,
                                     scale, side * flip, width=lw,
                                     tip_shift=0.85))
    tip = add(pos, from_angle(angle, size * 0.98))
    blades.append(blade_fast(tip, angle, leaflet * 0.95, flip, width=lw,
                             tip_shift=0.85))
    out: Dict[str, object] = {"blades": blades}
    if detail >= 1:
        out["rachis"] = path_smooth_open([pos, tip])
    return out


def _palmate(pos: Vec, angle: float, size: float, flip: int,
             t: Dict[str, float], w: float, tip_shift: float,
             detail: int) -> Dict[str, object]:
    """A hand of leaflets radiating from one point."""
    n = 5
    # A wide fan and narrow leaflets; overlapping leaflets outlined
    # individually read as a scribble rather than a leaf.
    fan = lerp(0.58, 0.95, t["leaf_pitch"] / 1.05)
    lw = clamp(w * 0.80, 0.18, 0.40)
    blades = []
    for i in range(n):
        f = (i / (n - 1)) - 0.5
        blades.append(blade_fast(pos, angle + f * 2.0 * fan,
                                 size * lerp(1.0, 0.72, abs(f * 2.0)), flip,
                                 width=lw, tip_shift=tip_shift))
    return {"blades": blades}


def _needle(pos: Vec, angle: float, size: float, flip: int,
            t: Dict[str, float]) -> Dict[str, object]:
    """A tuft of narrow needles. Deliberately pointed: tip_shift below 1
    puts the widest part near the base and tapers to a spike."""
    n = 5
    fan = lerp(0.32, 0.58, t["leaf_pitch"] / 1.05)
    blades = []
    for i in range(n):
        f = (i / (n - 1)) - 0.5
        blades.append(blade_fast(pos, angle + f * 2.0 * fan,
                                 size * lerp(1.05, 0.78, abs(f * 2.0)), flip,
                                 width=0.13, tip_shift=0.70))
    return {"blades": blades}


def cotyledon_paths(pos: Vec, angle: float, size: float, flip: int,
                    kingdom_round: float) -> Dict[str, str]:
    """Seed leaves: fatter and blunter than true leaves, which is most of
    what makes a sprout read as a sprout."""
    w = lerp(0.46, 0.72, kingdom_round)
    e = lerp(0.95, 1.35, kingdom_round)
    return {
        "blade": blade_fast(pos, angle, size, flip, width=w, tip_shift=e,
                            curl=0.10),
        "light": blade_fast_half(pos, angle, size, flip, width=w,
                                 tip_shift=e, curl=0.10),
    }


# --------------------------------------------------------------------------
# blooms
# --------------------------------------------------------------------------

def _petal_local(t: Dict[str, float], n_petals: int, samples: int) -> List[Vec]:
    """A single petal, narrowed to suit the whorl it sits in.

    Petal width has to fall as petal count rises or the whorl closes into a
    solid disc and the flower stops reading as a flower. Tying the two
    together means the `petals` and `petal_width` genes can vary freely
    without ever producing a pom-pom by accident.
    """
    width = lerp(0.20, 0.40, t["petal_width"]) * (7.0 / max(n_petals, 1))
    outline, _ = blade_local(
        width=clamp(width, 0.11, 0.42),
        tip_shift=1.18,                     # widest near the tip
        curl=0.0,
        notch=max(0.0, (t["petal_notch"] - 0.55) / 0.45),
        samples=samples)
    return outline


# --------------------------------------------------------------------------
# inflorescence archetypes
# --------------------------------------------------------------------------

DAISY, STAR, CROSS, POMPOM, BELL, SPIKE = (
    "daisy", "star", "cross", "pompom", "bell", "spike")
GLOBE, SCATTER, BLOSSOM = "globe", "scatter", "blossom"

# (petal count override, half-width, tip_shift, layers)
# tip_shift below 1 puts the widest point near the base and tapers to a
# spike -- that is what makes a star pointed rather than a rounded daisy.
_FORM_SPEC = {
    DAISY:  (None, None, 1.18, None),
    STAR:   (None, 0.17, 0.62, 1),
    CROSS:  (4, 0.36, 1.30, 1),
    POMPOM: (None, 0.13, 1.05, 2),
}


# Archetypes are not all the same size relative to their stem. A bell hangs
# small and in numbers; a pompom is a single showy head.
_BLOOM_SCALE = {DAISY: 1.0, STAR: 1.0, CROSS: 0.92, POMPOM: 0.95,
                BELL: 0.52, SPIKE: 0.80, GLOBE: 1.0, SCATTER: 1.0,
                BLOSSOM: 1.0}


def bloom_size_scale(form: str) -> float:
    return _BLOOM_SCALE.get(form, 1.0)


def bloom_form(kingdom: str, t: Dict[str, float]) -> str:
    """Which inflorescence a plant carries."""
    v = t["bloom_form"]
    if kingdom == FLOWER:
        for cut, name in ((0.28, DAISY), (0.48, STAR), (0.62, CROSS),
                          (0.76, POMPOM), (0.89, BELL)):
            if v < cut:
                return name
        return SPIKE
    if kingdom == BUSH:
        return GLOBE if v > 0.62 else SCATTER
    return BLOSSOM if v > 0.80 else SCATTER


def bloom_shapes(pos: Vec, angle: float, size: float, openness: float,
                 t: Dict[str, float], *, detail: int = 2,
                 simple: bool = False, form: str = DAISY
                 ) -> List[Tuple[str, str]]:
    """Return [(role, path_d), ...] back-to-front.

    A bud is drawn side-on along the stem; an open bloom is drawn face-on,
    which is the idiom that reads instantly at any size. `form` picks the
    archetype -- see `bloom_form`.
    """
    o = clamp(openness)
    if simple:
        detail = min(detail, 1)
        form = DAISY

    if form == GLOBE:
        return _globe(pos, size, o, detail)
    if form == SPIKE:
        return _spike(pos, angle, size, o, t, detail)
    if form == BELL:
        return _bell(pos, angle, size, o, t, detail)

    if o < 0.15:
        return _bud(pos, angle, size, o)
    return _rosette(pos, angle, size, o, t, detail, simple, form)


def _bud(pos: Vec, angle: float, size: float, o: float
         ) -> List[Tuple[str, str]]:
    b = o / 0.15
    length = size * lerp(0.85, 1.05, b)
    shapes = [("bud", blade_fast(pos, angle, length, 1,
                                 width=lerp(0.34, 0.46, b), tip_shift=1.30))]
    for side in (1, -1):
        shapes.append(("sepal", blade_fast(
            pos, angle + side * 0.55, length * 0.62, side,
            width=0.30, tip_shift=0.85, curl=0.16)))
    return shapes


def _rosette(pos: Vec, angle: float, size: float, o: float,
             t: Dict[str, float], detail: int, simple: bool, form: str
             ) -> List[Tuple[str, str]]:
    """A whorl of petals seen face-on: daisy, star, cross or pompom."""
    shapes: List[Tuple[str, str]] = []
    f = (o - 0.15) / 0.85
    center = add(pos, from_angle(angle, size * 0.22))

    n_over, width_over, tip_shift, layers_over = _FORM_SPEC[form]
    n = 5 if simple else (n_over or max(4, int(round(t["petals"]))))
    if form == POMPOM:
        n = min(18, max(8, n * 2))

    if layers_over is not None:
        layers = layers_over
    else:
        layers = 1 if (n >= 9 or simple) else min(
            2, max(1, int(round(t["bloom_layers"]))))

    notch = max(0.0, (t["petal_notch"] - 0.55) / 0.45)
    petal_w = (width_over if width_over is not None
               else clamp(lerp(0.20, 0.40, t["petal_width"]) * (7.0 / n),
                          0.11, 0.42))
    # the sampled outline only exists to show a notched tip, which no form
    # but the rounded daisy has room for
    sampled = notch > 0.3 and not simple and detail >= 2 and form == DAISY
    petal = _petal_local(t, n, samples=7) if sampled else None

    petal_len = size * lerp(0.62, 1.0, f) * (1.25 if form == STAR else 1.0)
    step = math.tau / n
    r = size * 0.26 * lerp(0.55, 1.0, f) * (0.55 if form == POMPOM else 1.0)
    hub = r * 0.55

    for layer in range(layers - 1, -1, -1):
        scale = petal_len * (1.0 + 0.22 * layer)
        phase = angle + step * 0.5 * layer
        role = "petal" if layer == 0 else "petal_back"
        for i in range(n):
            a = phase + step * i
            root = add(center, from_angle(a, hub))
            if sampled:
                d = path_smooth_closed(place(petal, root, a, scale),
                                       tension=0.95)
            else:
                d = blade_fast(root, a, scale, 1, width=petal_w,
                               tip_shift=tip_shift)
            shapes.append((role, d))
            if layer == 0 and detail >= 2 and f > 0.5 and form != POMPOM:
                shapes.append(("petal_light", blade_fast(
                    add(root, from_angle(a, scale * 0.10)), a, scale * 0.58,
                    1, width=petal_w, tip_shift=tip_shift)))

    shapes.append(("center", _circle(center, r)))
    if detail >= 2:
        shapes.append(("center_light",
                       _circle(add(center, (-r * 0.22, r * 0.22)), r * 0.52)))
        if f > 0.6 and form != POMPOM:
            for i in range(5):
                a = angle + math.tau * i / 5 + 0.4
                shapes.append(("stamen", _circle(
                    add(center, from_angle(a, r * 0.72)), r * 0.16)))
    return shapes


def _floret(pos: Vec, angle: float, size: float, petals: int = 4
            ) -> List[Tuple[str, str]]:
    """The small four-petalled unit that builds a globe or a spike."""
    step = math.tau / petals
    return [("floret", blade_fast(pos, angle + step * i, size, 1,
                                  width=0.42, tip_shift=1.25))
            for i in range(petals)]


def _globe(pos: Vec, size: float, o: float, detail: int
           ) -> List[Tuple[str, str]]:
    """A hydrangea truss: a dense dome of florets.

    Drawn as a soft mound with florets scattered over it rather than as a
    hundred individual flowers, which would cost more than the rest of the
    plant put together. `size` is a radius taken from the plant's span, so
    a bush in full bloom really is mostly flower.

    The edge is deliberately irregular and the florets are unoutlined: a
    clean circle full of outlined florets reads as a blackberry.
    """
    r = size * lerp(0.45, 1.0, o)
    if detail <= 0:
        # At icon size a nine-lobe spline and a circle are the same picture,
        # and the circle is a fifth of the path data.
        return [("globe_back", _circle(add(pos, (r * 0.06, -r * 0.06)),
                                       r * 1.03)),
                ("globe", _circle(pos, r * 0.93))]
    if detail == 1:
        shapes = [("globe_back", _circle(add(pos, (r * 0.06, -r * 0.06)),
                                         r * 1.03)),
                  ("globe", _circle(pos, r * 0.93))]
        for i in range(4):
            a = TAU * i / 4 + 0.5
            at = add(pos, from_angle(a, r * 0.48))
            shapes.extend(_floret(at, a * 1.7, r * 0.32))
        return shapes
    lobes = 9
    # deterministic wobble -- no RNG needed, and stable across renders
    jitter = [1.0 + 0.06 * math.sin(i * 2.399 + pos[0] * 0.71 + pos[1] * 1.27)
              for i in range(lobes)]
    shapes: List[Tuple[str, str]] = [
        ("globe_back", path_smooth_closed(
            blob(add(pos, (r * 0.06, -r * 0.06)), r * 1.03, lobes, jitter,
                 squash=0.94))),
        ("globe", path_smooth_closed(
            blob(pos, r * 0.93, lobes, jitter, squash=0.92, phase=0.4))),
    ]
    n = 11
    for i in range(n):
        a = TAU * i / n + 0.5
        ring = r * (0.30 if i % 2 else 0.60)
        role = "floret_shade" if i % 3 == 0 else "floret"
        at = add(pos, from_angle(a, ring))
        step = TAU / 4
        for k in range(4):
            shapes.append((role, blade_fast(at, a * 1.7 + step * k,
                                            r * 0.30, 1, width=0.44,
                                            tip_shift=1.25)))
    if detail >= 2:
        shapes.append(("floret_center", _circle(pos, r * 0.09)))
    return shapes


def _spike(pos: Vec, angle: float, size: float, o: float,
           t: Dict[str, float], detail: int) -> List[Tuple[str, str]]:
    """A raceme: many small florets up an axis, lavender-fashion. Opens
    from the bottom, so a half-open spike is green at the tip."""
    shapes: List[Tuple[str, str]] = []
    n = 7 if detail >= 2 else 5
    length = size * 1.55
    for i in range(n):
        u = (i + 0.5) / n
        at = add(pos, from_angle(angle, length * u))
        # florets open bottom-up
        local = clamp((o - u * 0.45) / 0.55)
        if local <= 0.02:
            shapes.append(("sepal", blade_fast(at, angle, size * 0.22, 1,
                                               width=0.42, tip_shift=1.2)))
            continue
        scale = size * 0.34 * lerp(0.45, 1.0, local) * lerp(1.1, 0.72, u)
        shapes.extend(_floret(at, angle + i * 1.1, scale, petals=4))
    return shapes


def _cup(pos: Vec, angle: float, length: float, mouth: float) -> str:
    """An open cup: narrow at the base, flaring to a wide rounded mouth.

    A blade pinches shut at both ends, so it can make a bud but never a
    bell. This is the one organ that needs its own outline.
    """
    def put(u: float, v: float) -> Vec:
        return add(pos, rotate((u * length, v * length), angle))

    # A concave flare: a narrow neck at the calyx opening out to the mouth.
    # Interpolating the profile rather than guessing control points is what
    # keeps this a bell instead of a triangle.
    def side(sign: int) -> Tuple[Vec, Vec]:
        pts = [put(u, sign * mouth * w)
               for u, w in ((0.0, 0.17), (1 / 3, 0.34),
                            (2 / 3, 0.62), (1.0, 1.0))]
        return cubic_through(*pts)

    base, lip_l, lip_r = put(0.0, 0.0), put(1.0, mouth), put(1.0, -mouth)
    l1, l2 = side(1)
    r1, r2 = side(-1)
    return (move_to(base)
            + curve_to(l1, l2, lip_l)
            # a shallow bulge across the mouth, so the rim reads as an
            # opening rather than a cut edge
            + curve_to(put(1.20, mouth * 0.58), put(1.20, -mouth * 0.58),
                       lip_r)
            + curve_to(r2, r1, base) + "Z")


def _bell(pos: Vec, angle: float, size: float, o: float,
          t: Dict[str, float], detail: int) -> List[Tuple[str, str]]:
    """A nodding bell. Bells hang, so this one ignores most of the stem's
    direction and points down, keeping only a little of its lean."""
    f = clamp(o / 0.9)
    nod = -UP + (angle - UP) * 0.28
    length = size * lerp(0.62, 1.20, f)
    mouth = lerp(0.24, 0.50, f) * lerp(0.85, 1.20, t["petal_width"])

    shapes: List[Tuple[str, str]] = [
        ("petal_back", _cup(pos, nod, length * 1.06, mouth * 1.10)),
        ("petal", _cup(pos, nod, length, mouth)),
    ]
    if f > 0.35:
        for side in (-1, 0, 1):
            rim = add(pos, from_angle(nod + side * 0.34, length * 1.02))
            shapes.append(("petal_light", blade_fast(
                rim, nod + side * 0.85, length * mouth * 0.9, 1,
                width=0.46, tip_shift=1.20)))
    if detail >= 2 and f > 0.6:
        shapes.append(("stamen", _circle(
            add(pos, from_angle(nod, length * 1.10)), size * 0.07)))
    for side in (1, -1):
        shapes.append(("sepal", blade_fast(
            pos, nod + side * 2.15, size * 0.30, side,
            width=0.32, tip_shift=0.85, curl=0.14)))
    return shapes


def _circle(c: Vec, r: float) -> str:
    # path form rather than <circle> so every drawable shares one emitter
    return (f"M{_fmt(c[0] - r)},{_fmt(c[1])}"
            f"a{_fmt(r)},{_fmt(r)} 0 1,0 {_fmt(2 * r)},0"
            f"a{_fmt(r)},{_fmt(r)} 0 1,0 {_fmt(-2 * r)},0Z")


def accent_paths(pos: Vec, size: float) -> List[Tuple[str, str]]:
    """A berry or small fruit, with a specular dot."""
    return [
        ("accent", _circle(pos, size)),
        ("accent_light", _circle(add(pos, (-size * 0.3, size * 0.3)),
                                 size * 0.34)),
    ]

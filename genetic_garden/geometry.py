"""Small 2D geometry kit.

Everything here works in *plant space*: origin at the base of the plant,
+x to the right, +y **up**. The renderer flips y on the way out to SVG, so
nothing else in the codebase has to think about screen coordinates.

Angles are radians, measured counter-clockwise from +x.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Sequence, Tuple

Vec = Tuple[float, float]

TAU = math.tau


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------

def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k)


def length(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def normalize(a: Vec) -> Vec:
    n = math.hypot(a[0], a[1])
    if n < 1e-12:
        return (0.0, 1.0)
    return (a[0] / n, a[1] / n)


def perp(a: Vec) -> Vec:
    """Left-hand normal."""
    return (-a[1], a[0])


def from_angle(theta: float, r: float = 1.0) -> Vec:
    return (math.cos(theta) * r, math.sin(theta) * r)


def angle_of(a: Vec) -> float:
    return math.atan2(a[1], a[0])


def rotate(a: Vec, theta: float) -> Vec:
    c, s = math.cos(theta), math.sin(theta)
    return (a[0] * c - a[1] * s, a[0] * s + a[1] * c)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_vec(a: Vec, b: Vec, t: float) -> Vec:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def smoothstep(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float, power: float = 2.0) -> float:
    return 1.0 - (1.0 - clamp(t)) ** power


# --------------------------------------------------------------------------
# polyline helpers
# --------------------------------------------------------------------------

def tangents(pts: Sequence[Vec]) -> List[Vec]:
    """Unit tangent at every vertex, via central differences."""
    n = len(pts)
    if n < 2:
        return [(0.0, 1.0)] * n
    out = []
    for i in range(n):
        if i == 0:
            d = sub(pts[1], pts[0])
        elif i == n - 1:
            d = sub(pts[-1], pts[-2])
        else:
            d = sub(pts[i + 1], pts[i - 1])
        out.append(normalize(d))
    return out


def resample(pts: Sequence[Vec], count: int) -> List[Vec]:
    """Resample a polyline to `count` evenly-spaced-by-arclength points."""
    if len(pts) < 2 or count < 2:
        return list(pts)
    seg = [0.0]
    for i in range(1, len(pts)):
        seg.append(seg[-1] + length(sub(pts[i], pts[i - 1])))
    total = seg[-1]
    if total < 1e-9:
        return [pts[0]] * count
    out = []
    j = 0
    for i in range(count):
        target = total * i / (count - 1)
        while j < len(seg) - 2 and seg[j + 1] < target:
            j += 1
        span = seg[j + 1] - seg[j]
        t = 0.0 if span < 1e-9 else (target - seg[j]) / span
        out.append(lerp_vec(pts[j], pts[j + 1], t))
    return out


def taper_outline(
    pts: Sequence[Vec],
    widths: Sequence[float],
    tip_ratio: float = 0.0,
) -> List[Vec]:
    """Turn a spine polyline + per-vertex widths into a closed outline loop.

    The loop runs up the left side and back down the right side, so the
    result can be stroked/filled directly. `tip_ratio` pinches the final
    width; 0 gives a true point.
    """
    n = len(pts)
    if n < 2:
        return []
    tans = tangents(pts)
    ws = list(widths)
    ws[-1] = ws[-1] * tip_ratio
    left: List[Vec] = []
    right: List[Vec] = []
    for i in range(n):
        nrm = perp(tans[i])
        half = ws[i] * 0.5
        left.append(add(pts[i], mul(nrm, half)))
        right.append(sub(pts[i], mul(nrm, half)))
    return left + list(reversed(right))


def offset_side(pts: Sequence[Vec], widths: Sequence[float], side: float,
                inset: float = 0.55) -> List[Vec]:
    """A thin ribbon hugging one side of a tapered stroke (for highlights)."""
    tans = tangents(pts)
    outer, inner = [], []
    for i, p in enumerate(pts):
        nrm = mul(perp(tans[i]), side)
        half = widths[i] * 0.5
        outer.append(add(p, mul(nrm, half * 0.92)))
        inner.append(add(p, mul(nrm, half * inset)))
    return outer + list(reversed(inner))


# --------------------------------------------------------------------------
# path emission
# --------------------------------------------------------------------------

# Emitted coordinate precision, in decimal places. Path data is generated
# in plant space but consumed at a known pixel size, so there is no point
# writing three decimals of a coordinate that lands inside one pixel --
# it is pure file size. `render` sets this from the actual render scale.
_PRECISION = 2


@contextmanager
def precision(decimals: int) -> Iterator[None]:
    """Temporarily set emitted coordinate precision."""
    global _PRECISION
    old = _PRECISION
    _PRECISION = max(0, min(4, int(decimals)))
    try:
        yield
    finally:
        _PRECISION = old


def _fmt(v: float) -> str:
    s = f"{v:.{_PRECISION}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def path_polygon(pts: Sequence[Vec]) -> str:
    """Straight-edged closed path."""
    if not pts:
        return ""
    head = f"M{_fmt(pts[0][0])},{_fmt(pts[0][1])}"
    body = "".join(f"L{_fmt(p[0])},{_fmt(p[1])}" for p in pts[1:])
    return head + body + "Z"


def path_smooth_closed(pts: Sequence[Vec], tension: float = 1.0) -> str:
    """Closed Catmull-Rom spline through `pts`, emitted as cubic beziers."""
    n = len(pts)
    if n < 3:
        return path_polygon(pts)
    k = tension / 6.0
    out = [f"M{_fmt(pts[0][0])},{_fmt(pts[0][1])}"]
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = add(p1, mul(sub(p2, p0), k))
        c2 = sub(p2, mul(sub(p3, p1), k))
        out.append(
            f"C{_fmt(c1[0])},{_fmt(c1[1])} {_fmt(c2[0])},{_fmt(c2[1])} "
            f"{_fmt(p2[0])},{_fmt(p2[1])}"
        )
    return "".join(out) + "Z"


def path_smooth_open(pts: Sequence[Vec], tension: float = 1.0) -> str:
    """Open Catmull-Rom spline (used for midribs, tendrils, ground lines)."""
    n = len(pts)
    if n < 3:
        if n < 2:
            return ""
        return (f"M{_fmt(pts[0][0])},{_fmt(pts[0][1])}"
                f"L{_fmt(pts[1][0])},{_fmt(pts[1][1])}")
    k = tension / 6.0
    out = [f"M{_fmt(pts[0][0])},{_fmt(pts[0][1])}"]
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, n - 1)]
        c1 = add(p1, mul(sub(p2, p0), k))
        c2 = sub(p2, mul(sub(p3, p1), k))
        out.append(
            f"C{_fmt(c1[0])},{_fmt(c1[1])} {_fmt(c2[0])},{_fmt(c2[1])} "
            f"{_fmt(p2[0])},{_fmt(p2[1])}"
        )
    return "".join(out)


def cubic_through(p0: Vec, p1: Vec, p2: Vec, p3: Vec) -> Tuple[Vec, Vec]:
    """Control points of the cubic that passes through p0..p3, with p1 and
    p2 at t = 1/3 and 2/3.

    A smooth leaf margin is well inside the error of this fit, so a whole
    blade becomes two curve segments instead of sixteen -- which is most of
    the difference between a 500KB tree and a 60KB one.
    """
    c1 = ((-5 * p0[0] + 18 * p1[0] - 9 * p2[0] + 2 * p3[0]) / 6.0,
          (-5 * p0[1] + 18 * p1[1] - 9 * p2[1] + 2 * p3[1]) / 6.0)
    c2 = ((2 * p0[0] - 9 * p1[0] + 18 * p2[0] - 5 * p3[0]) / 6.0,
          (2 * p0[1] - 9 * p1[1] + 18 * p2[1] - 5 * p3[1]) / 6.0)
    return c1, c2


def curve_to(c1: Vec, c2: Vec, end: Vec) -> str:
    return (f"C{_fmt(c1[0])},{_fmt(c1[1])} {_fmt(c2[0])},{_fmt(c2[1])} "
            f"{_fmt(end[0])},{_fmt(end[1])}")


def move_to(p: Vec) -> str:
    return f"M{_fmt(p[0])},{_fmt(p[1])}"


def bbox(points: Iterable[Vec]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for p in points:
        xs.append(p[0])
        ys.append(p[1])
    if not xs:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs), min(ys), max(xs), max(ys))


def blob(center: Vec, radius: float, lobes: int, jitter: Sequence[float],
         squash: float = 1.0, phase: float = 0.0) -> List[Vec]:
    """A rounded, slightly irregular closed shape.

    `jitter` supplies one multiplier per lobe (from the genome's RNG), which
    is what keeps foliage masses from looking like circles.
    """
    pts = []
    for i in range(lobes):
        a = phase + TAU * i / lobes
        r = radius * jitter[i % len(jitter)]
        pts.append((center[0] + math.cos(a) * r,
                    center[1] + math.sin(a) * r * squash))
    return pts

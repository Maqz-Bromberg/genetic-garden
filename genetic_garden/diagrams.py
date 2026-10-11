"""Annotated figures for the documentation pages.

Every figure here is drawn from the *real* algorithm -- the same `_trace`,
the same blade function, the same palette -- rather than hand-drawn to
resemble it. A diagram that is generated from the code cannot drift away
from the code, and when a constant changes the picture changes with it.

Figures carry CSS classes rather than literal colours wherever they are
structural, so the same SVG reads correctly on a light or a dark page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .geometry import Vec, add, angle_of, from_angle, lerp, mul, normalize, sub

Point = Tuple[float, float]

ARROW_ID = "gg-arrow"


def _f(v: float) -> str:
    s = f"{v:.2f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _pts(points: Sequence[Point]) -> str:
    return " ".join(f"{_f(x)},{_f(y)}" for x, y in points)


@dataclass
class Fig:
    """A small annotated SVG figure.

    Figure space is plain SVG space: x right, y **down**, origin top-left.
    Plant geometry is y-up, so `place()` exists to flip and scale it in.
    """

    width: float
    height: float
    caption: str = ""
    _parts: List[str] = field(default_factory=list)
    _defs: List[str] = field(default_factory=list)

    # -- raw ------------------------------------------------------------

    def add(self, svg: str) -> "Fig":
        self._parts.append(svg)
        return self

    def defs(self, svg: str) -> "Fig":
        self._defs.append(svg)
        return self

    # -- primitives ------------------------------------------------------

    def path(self, d: str, cls: str = "fig-line", **attrs) -> "Fig":
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return self.add(f'<path class="{cls}" d="{d}"{extra}/>')

    def line(self, a: Point, b: Point, cls: str = "fig-line", **attrs) -> "Fig":
        return self.path(f"M{_f(a[0])},{_f(a[1])}L{_f(b[0])},{_f(b[1])}",
                         cls, **attrs)

    def poly(self, points: Sequence[Point], cls: str = "fig-line",
             close: bool = False, **attrs) -> "Fig":
        if len(points) < 2:
            return self
        d = "M" + "L".join(f"{_f(x)},{_f(y)}" for x, y in points)
        if close:
            d += "Z"
        return self.path(d, cls, **attrs)

    def rect(self, x: float, y: float, w: float, h: float,
             cls: str = "fig-fill", **attrs) -> "Fig":
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return self.add(f'<rect class="{cls}" x="{_f(x)}" y="{_f(y)}" '
                        f'width="{_f(w)}" height="{_f(h)}"{extra}/>')

    def circle(self, c: Point, r: float, cls: str = "fig-dot",
               **attrs) -> "Fig":
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return self.add(f'<circle class="{cls}" cx="{_f(c[0])}" '
                        f'cy="{_f(c[1])}" r="{_f(r)}"{extra}/>')

    def text(self, p: Point, s: str, cls: str = "fig-text",
             anchor: str = "start", dy: float = 0.0, **attrs) -> "Fig":
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return self.add(
            f'<text class="{cls}" x="{_f(p[0])}" y="{_f(p[1] + dy)}" '
            f'text-anchor="{anchor}"{extra}>{_esc(s)}</text>')

    # -- annotation ------------------------------------------------------

    def arrow(self, a: Point, b: Point, cls: str = "fig-arrow",
              label: str = "", bend: float = 0.0) -> "Fig":
        """A straight or gently bent arrow, optionally labelled."""
        self._need_arrowhead()
        if abs(bend) < 1e-6:
            d = f"M{_f(a[0])},{_f(a[1])}L{_f(b[0])},{_f(b[1])}"
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        else:
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            nx, ny = -(b[1] - a[1]), (b[0] - a[0])
            n = math.hypot(nx, ny) or 1.0
            cx, cy = mx + nx / n * bend, my + ny / n * bend
            d = (f"M{_f(a[0])},{_f(a[1])}Q{_f(cx)},{_f(cy)} "
                 f"{_f(b[0])},{_f(b[1])}")
            mid = (mx + nx / n * bend * 0.5, my + ny / n * bend * 0.5)
        self.add(f'<path class="{cls}" d="{d}" marker-end="url(#{ARROW_ID})"/>')
        if label:
            up = b[1] < a[1]
            self.text((mid[0], mid[1] + (-6 if up else 13)), label,
                      "fig-text fig-small", "middle")
        return self

    # a rough advance width for the sans label font, enough to keep text
    # inside the frame without measuring it
    CHAR_W = 6.1

    def tag(self, p: Point, s: str, dx: float = 26.0, dy: float = -18.0,
            anchor: Optional[str] = None, cls: str = "fig-text") -> "Fig":
        """A dot on a feature with a leader line out to a label.

        The label flips to the other side rather than running off the
        frame: a diagram whose annotations are clipped is worse than one
        with none.
        """
        w = len(s) * self.CHAR_W
        if anchor is None and dx >= 0 and p[0] + dx + w + 8 > self.width:
            dx = -abs(dx)
        if anchor is None and dx < 0 and p[0] + dx - w - 8 < 0:
            dx = abs(dx)
        end = (p[0] + dx, min(max(p[1] + dy, 12.0), self.height - 8.0))
        self.line(p, end, "fig-leader")
        self.circle(p, 2.6, "fig-dot")
        a = anchor or ("start" if dx >= 0 else "end")
        pad = 4 if a == "start" else -4
        return self.text((end[0] + pad, end[1]), s, cls, a, dy=3.5)

    def bracket(self, a: Point, b: Point, label: str = "",
                offset: float = 10.0) -> "Fig":
        """A dimension bracket between two points."""
        d = normalize(sub(b, a))
        n = (-d[1], d[0])
        a2 = add(a, mul(n, offset))
        b2 = add(b, mul(n, offset))
        self.line(a, a2, "fig-leader")
        self.line(b, b2, "fig-leader")
        self.line(a2, b2, "fig-dim")
        if label:
            mid = ((a2[0] + b2[0]) / 2, (a2[1] + b2[1]) / 2)
            self.text((mid[0] + n[0] * 9, mid[1] + n[1] * 9 + 3.5), label,
                      "fig-text fig-small", "middle")
        return self

    def note(self, p: Point, s: str, anchor: str = "start") -> "Fig":
        return self.text(p, s, "fig-text fig-small fig-muted-ink", anchor)

    def title(self, p: Point, s: str, anchor: str = "start") -> "Fig":
        return self.text(p, s, "fig-text fig-label", anchor)

    # -- plant space -----------------------------------------------------

    def place(self, points: Sequence[Vec], origin: Point,
              scale: float) -> List[Point]:
        """Map y-up plant coordinates into y-down figure coordinates."""
        return [(origin[0] + p[0] * scale, origin[1] - p[1] * scale)
                for p in points]

    # -- plots -----------------------------------------------------------

    def axes(self, box: Tuple[float, float, float, float],
             xlim: Tuple[float, float], ylim: Tuple[float, float], *,
             xlabel: str = "", ylabel: str = "",
             xticks: Sequence[float] = (), yticks: Sequence[float] = (),
             xfmt: Callable[[float], str] = lambda v: _f(v),
             yfmt: Callable[[float], str] = lambda v: _f(v)) -> "Plot":
        plot = Plot(self, box, xlim, ylim)
        x, y, w, h = box
        self.line((x, y + h), (x + w, y + h), "fig-axis")
        self.line((x, y), (x, y + h), "fig-axis")
        for v in xticks:
            px = plot.x(v)
            self.line((px, y + h), (px, y + h + 4), "fig-axis")
            self.text((px, y + h + 15), xfmt(v), "fig-text fig-small",
                      "middle")
        for v in yticks:
            py = plot.y(v)
            self.line((x - 4, py), (x, py), "fig-axis")
            self.text((x - 7, py + 3.5), yfmt(v), "fig-text fig-small", "end")
        if xlabel:
            self.text((x + w, y + h + 28), xlabel,
                      "fig-text fig-small fig-muted-ink", "end")
        if ylabel:
            self.text((x, y - 9), ylabel, "fig-text fig-small fig-muted-ink")
        return plot

    def embed(self, svg: str, x: float, y: float, scale: float = 1.0,
              label: str = "") -> "Fig":
        """Drop a finished plant SVG into the figure. Nested `<svg>` keeps
        its own viewBox, so the plant is placed, never re-laid-out."""
        self.add(f'<g transform="translate({_f(x)},{_f(y)}) '
                 f'scale({_f(scale)})">{svg}</g>')
        if label:
            self.text((x, y - 7), label, "fig-text fig-small", "start")
        return self

    def swatch(self, x: float, y: float, w: float, h: float, colour: str,
               label: str = "", sub: str = "") -> "Fig":
        self.add(f'<rect class="fig-swatch" x="{_f(x)}" y="{_f(y)}" '
                 f'width="{_f(w)}" height="{_f(h)}" rx="3" fill="{colour}"/>')
        if label:
            self.text((x + w / 2, y + h + 14), label,
                      "fig-text fig-small", "middle")
        if sub:
            self.text((x + w / 2, y + h + 27), sub,
                      "fig-text fig-small fig-muted-ink", "middle")
        return self

    # -- output ----------------------------------------------------------

    def _need_arrowhead(self) -> None:
        if any(ARROW_ID in d for d in self._defs):
            return
        self.defs(
            f'<marker id="{ARROW_ID}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="fig-arrowhead"/></marker>')

    def to_svg(self) -> str:
        defs = f"<defs>{''.join(self._defs)}</defs>" if self._defs else ""
        return (f'<svg class="fig" xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {_f(self.width)} {_f(self.height)}" '
                f'role="img" aria-label="{_esc(self.caption or "diagram")}">'
                f'{defs}{"".join(self._parts)}</svg>')


@dataclass
class Plot:
    """Maps data coordinates into a figure box."""
    fig: Fig
    box: Tuple[float, float, float, float]
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]

    def x(self, v: float) -> float:
        x, _, w, _ = self.box
        lo, hi = self.xlim
        return x + (v - lo) / (hi - lo) * w

    def y(self, v: float) -> float:
        _, y, _, h = self.box
        lo, hi = self.ylim
        return y + h - (v - lo) / (hi - lo) * h

    def at(self, v: Tuple[float, float]) -> Point:
        return (self.x(v[0]), self.y(v[1]))

    def curve(self, fn: Callable[[float], float], samples: int = 120,
              cls: str = "fig-curve", **attrs) -> "Plot":
        lo, hi = self.xlim
        pts = []
        for i in range(samples + 1):
            v = lo + (hi - lo) * i / samples
            pts.append((self.x(v), self.y(fn(v))))
        self.fig.poly(pts, cls, **attrs)
        return self

    def series(self, values: Sequence[Tuple[float, float]],
               cls: str = "fig-curve", **attrs) -> "Plot":
        self.fig.poly([self.at(v) for v in values], cls, **attrs)
        return self

    def band(self, x0: float, x1: float, cls: str = "fig-band") -> "Plot":
        _, y, _, h = self.box
        self.fig.rect(self.x(x0), y, self.x(x1) - self.x(x0), h, cls)
        return self


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


# ==========================================================================
# shared helpers for drawing real plant geometry into a figure
# ==========================================================================

def _branch_shape(fig: Fig, points, widths, origin: Point, scale: float,
                  cls: str = "fig-wood", tip_ratio: float = 0.0) -> None:
    from .geometry import path_polygon, taper_outline
    loop = taper_outline(points, widths, tip_ratio=tip_ratio)
    if not loop:
        return
    fig.poly(fig.place(loop, origin, scale), cls, close=True)


def _skeleton_wood(fig: Fig, sk, origin: Point, scale: float,
                   cls: str = "fig-wood", min_width: float = 0.0) -> None:
    """A real skeleton's wood, in draw order.

    `min_width` drops the finest twigs: in a diagram they are a grey haze
    that obscures the structure being pointed at, and they are most of the
    file size.
    """
    for br in sk.branches:
        if max(br.widths) < min_width:
            continue
        _branch_shape(fig, br.points, br.widths, origin, scale, cls,
                      tip_ratio=0.0 if br.terminal else 1.0)


def _fit_scale(sk, target_h: float) -> float:
    """Scale that makes a skeleton `target_h` figure units tall."""
    from .render import content_bounds
    x0, y0, x1, y1 = content_bounds(sk)
    return target_h / max(y1 - y0, 1e-3)


def _demo_branch(**kw):
    """A single traced branch with sane defaults, for illustration."""
    import random

    from .skeleton import _trace
    params = dict(curve=0.30, upward=0.22, droop=0.10, sway=0.0,
                  thinness=0.4, steps=8)
    params.update(kw)
    rng = random.Random(7)
    return _trace(rng, (0.0, 0.0), math.pi / 2 - 0.30, 100.0, 12.0, 6.0,
                  **params)


# ==========================================================================
# skeleton figures
# ==========================================================================

def fig_trace_forces() -> Fig:
    """The three forces that bend a branch as it is walked forward."""
    from .skeleton import UP

    fig = Fig(680, 360, "Forces acting at each step of a traced branch")
    origin = (210.0, 320.0)
    scale = 2.3

    straight = [(0.0, i * 12.5) for i in range(9)]
    fig.poly(fig.place(straight, origin, scale), "fig-ghost")
    fig.note((210, 34), "un-forced: a stick", "middle")

    pts, ws = _demo_branch()
    placed = fig.place(pts, origin, scale)
    fig.poly(placed, "fig-spine")
    for q in placed:
        fig.circle(q, 2.2, "fig-dot")
    fig.line(placed[0], (placed[0][0] - 54, placed[0][1] - 6), "fig-leader")
    fig.line(placed[1], (placed[1][0] - 54, placed[1][1] + 6), "fig-leader")
    fig.note((placed[0][0] - 58, placed[1][1] - 2), "one", "end")
    fig.note((placed[0][0] - 58, placed[0][1] - 2), "step", "end")

    # blow one step up and show what acts on it
    i = 5
    here = placed[i]
    theta = angle_of(normalize(sub(pts[i + 1], pts[i])))
    fig.circle(here, 4.5, "fig-dot fig-dot-big")

    def ray(a: float, arm: float) -> Point:
        d = from_angle(a)
        return (here[0] + d[0] * arm, here[1] - d[1] * arm)

    for ang, arm, cls, label, side, ly in (
            (theta, 96, "fig-arrow", "heading", "start", -6),
            (UP, 58, "fig-arrow fig-accent-stroke", "phototropism", "end", -8),
            (-UP, 58, "fig-arrow fig-warn-stroke", "gravity", "start", 14),
            (theta + 1.35, 48, "fig-arrow fig-muted-stroke", "curvature",
             "end", -6)):
        tip = ray(ang, arm)
        fig.arrow(here, tip, cls)
        off = 8 if side == "start" else -8
        fig.text((tip[0] + off, tip[1] + ly), label,
                 "fig-text fig-small", side)

    fig.title((402, 34), "what one step does")
    for k, line in enumerate([
            "theta += curve / steps",
            "theta += upward * angle_to(UP) / steps",
            "theta += droop * thin * angle_to(-UP)",
            "theta += gauss(0, sway * 0.055)",
            "pos   += from_angle(theta) * step"]):
        fig.text((402, 60 + k * 20), line, "fig-text fig-code")
    fig.note((402, 178), "The constant curvature is the branch's own")
    fig.note((402, 195), "handwriting. The other two are the world")
    fig.note((402, 212), "acting on it, and gravity is weighted by")
    fig.note((402, 229), "how slender the wood is, so a trunk resists")
    fig.note((402, 246), "what a twig cannot.")
    fig.caption = ("Each step turns the heading by a constant curvature, a "
                   "pull toward vertical, a pull toward the ground weighted "
                   "by slenderness, and a little noise. Summed over eight "
                   "steps that is the difference between a branch and a "
                   "line segment.")
    return fig


def fig_taper_outline() -> Fig:
    """Spine plus per-vertex widths becomes a closed outline."""
    from .geometry import perp, tangents, taper_outline

    fig = Fig(680, 320, "Turning a spine into a tapered outline")
    scale = 1.9
    pts, ws = _demo_branch(steps=6)

    tans_all = None
    for panel, (ox, label) in enumerate([
            (100.0, "1. spine, width per vertex"),
            (330.0, "2. offset along the normals"),
            (556.0, "3. close the loop")]):
        origin = (ox, 272.0)
        fig.title((ox, 30), label, "middle")
        placed = fig.place(pts, origin, scale)
        if panel == 0:
            fig.poly(placed, "fig-spine")
            tn = tangents(pts)
            for q, w, t_i in zip(placed, ws, tn):
                n = perp(t_i)
                half = w * scale / 2
                fig.line((q[0] - n[0] * half, q[1] + n[1] * half),
                         (q[0] + n[0] * half, q[1] - n[1] * half), "fig-dim")
                fig.circle(q, 2.2, "fig-dot")
        else:
            tans = tangents(pts)
            left, right = [], []
            for p, w, tn in zip(pts, ws, tans):
                n = perp(tn)
                left.append(add(p, mul(n, w / 2)))
                right.append(sub(p, mul(n, w / 2)))
            if panel == 1:
                fig.poly(placed, "fig-spine")
                for a, b in zip(fig.place(left, origin, scale),
                                fig.place(right, origin, scale)):
                    fig.circle(a, 2.0, "fig-dot fig-accent-fill")
                    fig.circle(b, 2.0, "fig-dot fig-accent-fill")
                fig.poly(fig.place(left, origin, scale), "fig-thin")
                fig.poly(fig.place(right, origin, scale), "fig-thin")
            else:
                loop = taper_outline(pts, ws, tip_ratio=0.0)
                fig.poly(fig.place(loop, origin, scale), "fig-wood",
                         close=True)
                tip = fig.place([pts[-1]], origin, scale)[0]
                fig.tag(tip, "tip_ratio pinches this end", -26, 22)
    fig.caption = ("The outline runs up one flank and back down the other. "
                   "A branch with something growing on from its tip skips "
                   "the pinch, so the next segment's collar can cover it.")
    return fig


def fig_collar() -> Fig:
    """The branch collar and the root wedge that buries the cut end."""
    from .skeleton import (COLLAR_FLARE, COLLAR_SPAN, ROOT_DEPTH, ROOT_WIDTH,
                           _collar)

    fig = Fig(680, 350, "The branch collar")
    fig.title((58, 28), "width multiplier along the branch")
    plot = fig.axes((72, 48, 232, 116), (0.0, 1.0), (0.9, 1.6),
                    xlabel="u", xticks=(0.0, COLLAR_SPAN, 0.5, 1.0),
                    yticks=(1.0, 1.4),
                    xfmt=lambda v: f"{v:g}")
    plot.band(0.0, COLLAR_SPAN)
    plot.curve(lambda u: _collar(u, COLLAR_FLARE, COLLAR_SPAN))
    fig.note((72, 212), f"flare {COLLAR_FLARE}, decaying over the")
    fig.note((72, 229), f"first {COLLAR_SPAN:g} of the branch's length")
    fig.note((72, 253), "_collar(u) = 1 + flare * k^2,")
    fig.note((72, 270), "k = 1 - u / span")

    pts, ws = _demo_branch(steps=8)
    plain = list(ws)
    collared = [w * _collar(i / (len(ws) - 1), COLLAR_FLARE, COLLAR_SPAN)
                for i, w in enumerate(ws)]
    first_dir = angle_of(normalize(sub(pts[1], pts[0])))
    root_pt = add(pts[0], from_angle(first_dir, -ws[0] * ROOT_DEPTH))

    scale = 1.15
    for ox, widths, points, label in [
            (410.0, plain, list(pts), "butted on"),
            (585.0, [ws[0] * ROOT_WIDTH] + collared, [root_pt] + list(pts),
             "collar + root wedge")]:
        origin = (ox, 300.0)
        fig.rect(ox - 66, 214, 132, 30, "fig-parent")
        _branch_shape(fig, points, widths, origin, scale, "fig-wood")
        fig.title((ox, 28), label, "middle")
    fig.note((497, 262), "parent's wood", "middle")

    fig.tag((408, 228), "flat cut", -30, -44, "end")
    fig.tag((594, 243), "wedge, buried", -34, 54, "end")
    fig.tag((588, 208), "widest here", -40, -46, "end")
    fig.caption = ("A limb swells where it leaves its parent and starts from "
                   "a point behind its origin, inside the parent's wood. A "
                   "leader gets almost no flare -- it is the same axis "
                   "carrying on -- and the same flare at ground level is a "
                   "trunk's root buttress.")
    return fig


def fig_draw_order() -> Fig:
    """Why the emit order of the branch list is the whole junction."""
    from .skeleton import COLLAR_FLARE, COLLAR_SPAN, ROOT_DEPTH, ROOT_WIDTH, _collar

    fig = Fig(680, 300, "Draw order at a fork")
    scale = 1.45

    trunk = [(0.0, y) for y in (0, 25, 50, 75, 100)]
    trunk_w = [16.0, 15.0, 14.0, 13.2, 12.5]
    limb_dir = math.pi / 2 - 0.85
    limb = [from_angle(limb_dir, d) for d in (0, 16, 32, 48, 64)]
    limb = [add((0.0, 52.0), p) for p in limb]
    limb_w = [9.0, 8.0, 7.0, 6.0, 5.0]

    panels = [
        (110.0, "parent first", False, False),
        (340.0, "child first", True, False),
        (570.0, "child first + collar", True, True),
    ]
    for ox, label, child_first, collar in panels:
        origin = (ox, 262.0)
        fig.title((ox, 30), label, "middle")
        pts, ws = list(limb), list(limb_w)
        if collar:
            ws = [w * _collar(i / (len(ws) - 1), COLLAR_FLARE, COLLAR_SPAN)
                  for i, w in enumerate(ws)]
            root = add(pts[0], from_angle(limb_dir, -limb_w[0] * ROOT_DEPTH))
            pts = [root] + pts
            ws = [limb_w[0] * ROOT_WIDTH] + ws
        if child_first:
            _branch_shape(fig, pts, ws, origin, scale, "fig-wood")
            _branch_shape(fig, trunk, trunk_w, origin, scale, "fig-wood",
                          tip_ratio=1.0)
        else:
            _branch_shape(fig, trunk, trunk_w, origin, scale, "fig-wood",
                          tip_ratio=1.0)
            _branch_shape(fig, pts, ws, origin, scale, "fig-wood")

    fig.tag((136, 185), "pasted on top", 30, -46)
    fig.tag((352, 196), "emerges from behind", 22, -70)
    fig.tag((580, 206), "and swells into it", -28, 58, "end")
    fig.caption = ("`skeleton.branches` is a draw list, not a bag. Limbs "
                   "passing behind a branch are emitted before it, its own "
                   "continuation after it. Sorting the list undoes this.")
    return fig


def fig_kingdom_rules() -> Fig:
    """The three branching rules, wood only."""
    from .growth import profile
    from .plant import Plant
    from .skeleton import build

    fig = Fig(680, 330, "One branching rule per kingdom")
    notes = {
        "tree": ["one dominant trunk", "laterals start high",
                 "a leader carries on"],
        "bush": ["no trunk at all", "several stems from the ground",
                 "tips fork, nothing leads"],
        "flower": ["a single slender stem", "few, large leaves",
                   "one terminal bloom"],
    }
    for panel, kingdom in enumerate(("tree", "bush", "flower")):
        ox = 110.0 + panel * 230.0
        plant = Plant.from_id("apollo", kingdom)
        growth, sk = plant.build(21)
        origin = (ox, 246.0)
        _skeleton_wood(fig, sk, origin, _fit_scale(sk, 176.0),
                       min_width=0.55 if kingdom == "tree" else 0.0)
        fig.title((ox, 30), kingdom, "middle")
        for k, line in enumerate(notes[kingdom]):
            fig.note((ox, 272 + k * 17), line, "middle")
    fig.caption = ("Wood only, at full maturity. The silhouettes differ "
                   "before a single leaf is drawn, which is what makes the "
                   "kingdoms readable at icon size.")
    return fig


def fig_addressed_rng() -> Fig:
    """Why each branch derives its own RNG from its address."""
    from .geometry import lerp_vec

    fig = Fig(680, 320, "Addressed randomness")
    origin = (150.0, 286.0)
    scale = 1.55

    def seg(start: Vec, ang: float, length: float, label: str,
            at: float, dx: float, dy: float, anchor: str = "start"):
        end = add(start, from_angle(ang, length))
        fig.poly(fig.place([start, end], origin, scale), "fig-spine")
        mid = fig.place([lerp_vec(start, end, at)], origin, scale)[0]
        fig.text((mid[0] + dx, mid[1] + dy), label,
                 "fig-text fig-code", anchor)
        return end

    a = seg((0.0, 0.0), math.pi / 2, 64, '"0"', 0.5, 11, 4)
    b = seg(a, math.pi / 2 - 0.62, 46, '"0.0"', 0.9, 10, -4)
    c = seg(a, math.pi / 2 + 0.55, 44, '"0.1"', 0.9, -10, -4, "end")
    d = seg(a, math.pi / 2 - 0.06, 50, '"0.L"', 0.55, 11, 2)
    seg(d, math.pi / 2 - 0.5, 34, '"0.L.0"', 0.95, 10, -6)

    fig.title((386, 36), 'rng = Random(hash(f"{seed}:{path}"))')
    for k, line in enumerate([
            "Each branch derives its own stream from where it sits",
            "in the tree, never from a shared sequence.",
            "",
            "Growing the plant one level deeper appends branches in",
            "the middle of a traversal. With a shared RNG that",
            "reshuffles every branch drawn after it, and the plant",
            "stops being the same individual from one contribution",
            "to the next."]):
        fig.note((386, 66 + k * 19), line)
    fig.caption = ("Reproducibility is addressed, not streamed. This is what "
                   "lets growth add structure without disturbing what was "
                   "already there.")
    return fig


# ==========================================================================
# growth figures
# ==========================================================================

def fig_maturity_curve() -> Fig:
    """Contributions to maturity, and where the named stages fall."""
    from .growth import CURVE, DEFAULT_FULL, STAGE_CUTS, STAGE_NAMES, maturity

    fig = Fig(680, 332, "Contributions mapped onto maturity")
    plot = fig.axes((66, 44, 470, 190), (0, 34), (0, 1.05),
                    xlabel="sequential contributions", ylabel="maturity",
                    xticks=(0, 1, 3, 5, 8, 13, 21, 34), yticks=(0, 0.5, 1.0),
                    xfmt=lambda v: f"{v:g}")
    plot.series([(n, maturity(n)) for n in range(0, 35)])
    plot.curve(lambda n: min(1.0, n / DEFAULT_FULL), cls="fig-curve-3")
    fig.note((300, 228), "linear, for comparison")

    for name, cut in zip(STAGE_NAMES, list(STAGE_CUTS) + [1.0]):
        y = plot.y(cut)
        fig.line((66, y), (536, y), "fig-leader")
        fig.text((541, y + 3.5), name, "fig-text fig-small fig-muted-ink")
    for n in (1, 3, 8, 21):
        fig.circle(plot.at((n, maturity(n))), 3.2, "fig-dot fig-dot-big")

    fig.title((66, 274), "maturity(n) = (log1p(n) / log1p(full)) ** "
                         f"{CURVE}")
    fig.note((66, 293), "Logarithmic, so the first contributions move the "
                        "needle and later ones refine.")
    fig.note((66, 310), "The exponent is what keeps a sprout looking like a "
                        "sprout rather than a small adult.")
    fig.caption = ("The named stages are bands on this one curve. `full` "
                   "(default 21) is where a project counts as fully grown, "
                   "and a genome's `vigor` shifts it a little either way.")
    return fig


def fig_growth_channels() -> Fig:
    """The five things maturity actually drives."""
    from .growth import profile

    fig = Fig(680, 320, "What maturity drives")
    plot = fig.axes((66, 40, 470, 200), (0, 34), (0, 1.12),
                    xlabel="contributions", xticks=(0, 5, 13, 21, 34),
                    yticks=(0, 0.5, 1.0), xfmt=lambda v: f"{v:g}")
    channels = [
        ("depth / 5.4", lambda g: g.depth / 5.4, "fig-curve"),
        ("scale", lambda g: g.scale, "fig-curve-2"),
        ("leafiness", lambda g: g.leafiness, "fig-curve-3"),
        ("bloom", lambda g: g.bloom, "fig-curve-4"),
        ("woodiness", lambda g: g.woodiness, "fig-curve-5"),
    ]
    for k, (label, fn, cls) in enumerate(channels):
        pts = [(n, fn(profile(n, "tree"))) for n in range(0, 35)]
        plot.series(pts, cls)
        fig.text((548, 56 + k * 22), label, "fig-text fig-small")
        fig.line((540, 52 + k * 22), (546, 52 + k * 22), cls)

    fig.title((66, 276), "Growth extends the plant; it never rearranges it")
    fig.note((66, 296), "Depth is fractional on purpose: the newest level of "
                        "branches emerges scaled-down from its parent tips.")
    fig.caption = ("One number drives everything, read through a per-kingdom "
                   "schedule. Depth keeps climbing after size has plateaued, "
                   "so structure is still arriving at the end.")
    return fig


def fig_fractional_depth() -> Fig:
    """A new level of branches emerging rather than appearing."""
    from .geometry import lerp as _lerp
    from .geometry import smoothstep
    from .skeleton import COLLAR_FLARE, COLLAR_SPAN, _collar

    fig = Fig(680, 300, "Fractional depth")
    scale = 1.5
    parent = [(0.0, y) for y in (0, 22, 44, 66)]
    parent_w = [11.0, 10.0, 9.2, 8.6]

    for k, emergence in enumerate((0.0, 0.25, 0.6, 1.0)):
        ox = 108.0 + k * 158.0
        origin = (ox, 250.0)
        _branch_shape(fig, parent, parent_w, origin, scale, "fig-wood",
                      tip_ratio=1.0 if emergence > 0.02 else 0.0)
        fig.title((ox, 34), f"remaining = {4 + emergence:.2f}", "middle")
        fig.note((ox, 52), f"emergence {emergence:.2f}", "middle")
        if emergence > 0.02:
            grow = _lerp(0.12, 1.0, smoothstep(emergence))
            w0 = 6.4 * _lerp(0.35, 1.0, emergence)
            for side in (1, -1):
                ang = math.pi / 2 + side * 0.62
                child = [add((0.0, 66.0), from_angle(ang, 34.0 * grow * t))
                         for t in (0.0, 0.34, 0.67, 1.0)]
                ws = [w0 * _collar(t, COLLAR_FLARE, COLLAR_SPAN) *
                      _lerp(1.0, 0.72, t) for t in (0.0, 0.34, 0.67, 1.0)]
                _branch_shape(fig, child, ws, origin, scale, "fig-wood")
    fig.note((340, 282),
             "Each contribution pushes the newest level a little further out, "
             "so the plant reads as one individual getting bigger.", "middle")
    fig.caption = ("Recursion depth is deliberately fractional. A whole-number "
                   "depth would pop a new layer of branches into existence "
                   "between one contribution and the next.")
    return fig


def fig_fit_modes() -> Fig:
    """The three ways a plant can be scaled into a frame."""
    from .plant import Plant

    fig = Fig(680, 330, "Fit modes")
    plant = Plant.from_id("apollo", "tree")
    box = 150
    for k, (mode, label, note) in enumerate([
            ("mature", "mature", "scaled against this plant full-grown"),
            ("bbox", "bbox", "fills the frame at every stage"),
            ("world", "world", "shared cross-kingdom units")]):
        ox = 40 + k * 215
        fig.title((ox + box, 30), label, "middle")
        for j, n in enumerate((3, 21)):
            svg = plant.svg(n, size=box, fit=mode, background=None,
                            detail="icon", ground=False)
            fig.embed(svg, ox + j * (box - 42), 46, 1.0)
            fig.note((ox + j * (box - 42) + box / 2, 218, ), f"n={n}", "middle")
        fig.note((ox + box, 244), note, "middle")
    fig.note((340, 286), "`mature` is the default, and the reason a garden "
                         "reads as a garden", "middle")
    fig.note((340, 303), "rather than a row of equally large plants.",
             "middle")
    fig.caption = ("Same plant, same two ages, three scalings. Only `mature` "
                   "makes the growth legible at a glance; `bbox` throws that "
                   "away for legibility, which is the right trade for icons.")
    return fig


# ==========================================================================
# genome figures
# ==========================================================================

def fig_id_to_genes() -> Fig:
    """A project id becomes a genome, the same way every time."""
    import hashlib

    from .genome import GENE_NAMES, Genome, stable_seed

    pid = "apollo-rewrite"
    digest = hashlib.blake2b(pid.encode(), digest_size=8).hexdigest()
    seed = stable_seed(pid)
    genome = Genome.from_id(pid, "tree")

    fig = Fig(680, 300, "From project id to genome")
    steps = [
        (f'"{pid}"', "the project id"),
        (f"{digest}", "blake2b, 8 bytes"),
        (f"{seed}", "a process-independent seed"),
    ]
    for k, (value, note) in enumerate(steps):
        y = 44 + k * 52
        fig.text((66, y), value, "fig-text fig-code")
        fig.note((66, y + 16), note)
        if k < len(steps) - 1:
            fig.arrow((74, y + 24), (74, y + 40), "fig-arrow")

    fig.title((66, 212), "36 genes, each a float in [0, 1]")
    x, w = 66.0, 15.0
    for k, name in enumerate(GENE_NAMES):
        v = genome.genes[name]
        h = 46.0 * v
        fig.rect(x + k * w, 268 - h, w - 3, h, "fig-fill")
        fig.line((x + k * w, 268), (x + k * w + w - 3, 268), "fig-axis")
    fig.note((66, 288), "the chromosome knows nothing about plants")
    fig.caption = ("`hash()` is salted per process, so the seed comes from "
                   "blake2b instead: a project has to grow the same plant "
                   "tomorrow as it did today.")
    return fig


def fig_gene_to_trait() -> Fig:
    """One gene value, three kingdoms, three different plants."""
    from .genome import COMMON, KINGDOMS, gene_ranges

    fig = Fig(680, 300, "Expression: the same gene in three kingdoms")
    gene = "branch_angle"
    value = 0.62
    fig.title((66, 30), f'gene "{gene}" = {value}')
    fig.note((66, 50), "The chromosome is kingdom-agnostic. Expression maps "
                       "each gene through the ranges its kingdom uses.")

    x0, w = 150.0, 380.0
    lo_all = min(gene_ranges(k)[gene][0] for k in KINGDOMS)
    hi_all = max(gene_ranges(k)[gene][1] for k in KINGDOMS)
    for k, kingdom in enumerate(KINGDOMS):
        y = 98 + k * 52
        lo, hi = gene_ranges(kingdom)[gene]
        px = lambda v: x0 + (v - lo_all) / (hi_all - lo_all) * w
        fig.text((136, y + 4), kingdom, "fig-text", "end")
        fig.line((x0, y), (x0 + w, y), "fig-axis")
        fig.rect(px(lo), y - 7, px(hi) - px(lo), 14, "fig-band")
        trait = lo + (hi - lo) * value
        fig.circle((px(trait), y), 4.5, "fig-dot fig-dot-big")
        fig.text((px(trait), y - 14), f"{trait:.2f} rad",
                 "fig-text fig-small", "middle")
        fig.note((x0 + w + 8, y + 4), f"{lo:.2f}–{hi:.2f}")

    fig.note((66, 268), "A bush forks wider than a tree at the same gene "
                        "value, because a bush's range starts wider.")
    fig.note((66, 285), "The same chromosome read as a tree and as a flower "
                        "gives two plants with a family resemblance.")
    fig.caption = ("Keeping the chromosome free of botany means crossover "
                   "and mutation can never produce an invalid plant, and a "
                   "genome can be re-expressed as any kingdom.")
    return fig


def fig_breeding() -> Fig:
    """Crossover and mutation, on the chromosome."""
    import random

    from .genome import GENE_NAMES, Genome

    a = Genome.from_id("apollo", "tree")
    b = Genome.from_id("borealis", "tree")
    rng = random.Random(4)
    child = a.crossover(b, rng)
    mutant = child.mutate(0.3, 0.14, random.Random(9))

    fig = Fig(680, 320, "Crossover and mutation")
    names = GENE_NAMES[:18]
    x0, w, h = 78.0, 30.0, 40.0
    rows = [("parent A", a, "fig-bar-a"), ("parent B", b, "fig-bar-b"),
            ("child", child, "fig-bar-c"), ("mutated", mutant, "fig-bar-c")]
    for k, (label, g, cls) in enumerate(rows):
        y = 56 + k * 62
        fig.text((70, y + h / 2), label, "fig-text fig-small", "end")
        for i, name in enumerate(names):
            v = g.genes[name]
            bh = h * v
            fig.rect(x0 + i * w, y + h - bh, w - 4, bh, cls)
            fig.line((x0 + i * w, y + h), (x0 + i * w + w - 4, y + h),
                     "fig-axis")
        if k in (1, 2):
            fig.arrow((x0 + 264, y + h + 6), (x0 + 264, y + h + 20))
            fig.note((x0 + 274, y + h + 18),
                     "crossover" if k == 1 else "mutation")
    mut_y = 56 + 3 * 62
    for i, name in enumerate(names):
        if abs(mutant.genes[name] - child.genes[name]) > 1e-9:
            fig.circle((x0 + i * w + 13, mut_y - 7), 3.2,
                       "fig-dot fig-dot-big")
    fig.note((78, 300), "Crossover blends each gene by a random amount; "
                        "mutation nudges a few (marked) and clamps to [0, 1].")
    fig.caption = ("Because most genes feed a recursive rule, a small nudge "
                   "is nearly invisible on a seedling and obvious on a "
                   "mature plant -- the divergence compounds once per level "
                   "of depth.")
    return fig


def fig_divergence() -> Fig:
    """Siblings start alike and grow apart."""
    import random

    from .plant import Plant

    fig = Fig(680, 330, "Mutation diverges with growth")
    base = Plant.from_id("apollo", "tree")
    family = [base] + [base.mutate(0.5, 0.07, random.Random(100 + i))
                       for i in range(3)]
    for row, n in enumerate((3, 8, 21)):
        y = 34 + row * 96
        fig.text((58, y + 46), f"n={n}", "fig-text fig-small", "end")
        for col, member in enumerate(family):
            svg = member.svg(n, size=92, detail="icon", ground=False,
                             background=None)
            fig.embed(svg, 68 + col * 100, y, 1.0)
    fig.note((470, 60), "one genome, plus three")
    fig.note((470, 77), "mutations at scale 0.07")
    fig.note((470, 112), "At three contributions the")
    fig.note((470, 129), "four are nearly the same")
    fig.note((470, 146), "plant. At twenty-one they")
    fig.note((470, 163), "have different crowns and")
    fig.note((470, 180), "different trunk splits.")
    fig.caption = ("The fractal payoff. Nothing here special-cases age: the "
                   "same gene delta simply passes through more levels of "
                   "recursion as the plant grows.")
    return fig


# ==========================================================================
# organ figures
# ==========================================================================

def _blade_at(fig: Fig, origin: Point, scale: float, *, width: float,
              tip_shift: float, curl: float = 0.0, angle: float = math.pi / 2,
              cls: str = "fig-leafy") -> None:
    from .organs import blade_fast
    d = blade_fast((0.0, 0.0), angle, 1.0, 1, width=width,
                   tip_shift=tip_shift, curl=curl)
    # a scaled group scales stroke-width with it, so the outline has to
    # opt out or a 0.8px edge becomes a 77px blob
    fig.add(f'<g transform="translate({_f(origin[0])},{_f(origin[1])}) '
            f'scale({_f(scale)},{_f(-scale)})">'
            f'<path class="{cls}" vector-effect="non-scaling-stroke" '
            f'd="{d}"/></g>')


def fig_blade_function() -> Fig:
    """One parametric function is every blade in the system."""
    fig = Fig(680, 360, "The blade function")
    fig.title((66, 28), "half_width(u) = w * sin(pi * u ** e)")
    fig.note((66, 48), "e slides the widest point from the base to the tip. "
                       "That one exponent is the difference")
    fig.note((66, 65), "between a pointed needle, an ovate leaf and a "
                       "spatulate petal.")

    plot = fig.axes((80, 92, 240, 116), (0.0, 1.0), (-0.55, 0.55),
                    xlabel="u (base to tip)", xticks=(0, 0.5, 1),
                    yticks=(-0.5, 0, 0.5), xfmt=lambda v: f"{v:g}")
    for e, cls in ((0.62, "fig-curve-2"), (1.0, "fig-curve"),
                   (1.5, "fig-curve-3")):
        for sign in (1, -1):
            plot.curve(lambda u, e=e, sign=sign:
                       sign * 0.5 * math.sin(math.pi * max(u, 0.0) ** e))
    fig.note((332, 118), "e = 0.62   pointed")
    fig.note((332, 136), "e = 1.00   ovate")
    fig.note((332, 154), "e = 1.50   widest near tip")

    for k, (e, label) in enumerate([(0.62, "needle"), (1.0, "leaf"),
                                    (1.5, "petal")]):
        ox = 476.0 + k * 62.0
        _blade_at(fig, (ox, 208.0), 96.0, width=0.20, tip_shift=e)
        fig.note((ox, 224), label, "middle")

    fig.title((66, 250), "a margin, serrated")
    fig.note((66, 270), "half_width *= 1 - tooth * triangle(u * n)")
    plot2 = fig.axes((80, 286, 240, 56), (0.0, 1.0), (0.0, 0.6),
                     xticks=(), yticks=())
    plot2.curve(lambda u: 0.5 * math.sin(math.pi * u) *
                (1.0 - 0.18 * (2.0 * abs((u * 5) % 1.0 - 0.5))))
    fig.note((330, 300), "A toothed margin costs three times a smooth one")
    fig.note((330, 317), "to describe, so it is only drawn when the leaf is")
    fig.note((330, 334), "big enough on screen for the teeth to show.")
    fig.caption = ("Every leaf, cotyledon, petal, sepal and bud in the "
                   "system is this function with different constants. The "
                   "genome picks a shape out of a continuum rather than "
                   "from a list.")
    return fig


def fig_leaf_forms() -> Fig:
    """Four leaf architectures, drawn by the real code."""
    from .genome import Genome
    from .organs import COMPOUND, NEEDLE, PALMATE, SIMPLE, leaf_paths

    fig = Fig(680, 290, "Leaf architecture")
    t = dict(Genome.from_id("apollo", "tree").traits())
    t["leaf_slender"] = 0.42
    t["leaf_pitch"] = 0.7
    t["leaf_lobes"] = 0.2
    forms = [(SIMPLE, 0.1, "simple", "one blade"),
             (COMPOUND, 0.7, "compound", "leaflets up a rachis"),
             (PALMATE, 0.9, "palmate", "a hand from one point"),
             (NEEDLE, 0.95, "needle", "a tuft of spikes")]
    kingdoms = ["tree", "tree", "bush", "bush"]
    for k, (form, v, label, note) in enumerate(forms):
        ox = 110.0 + k * 158.0
        tt = dict(t)
        tt["leaf_form"] = v
        paths = leaf_paths((0.0, 0.0), math.pi / 2, 54.0, 1, tt,
                           detail=2, kingdom=kingdoms[k])
        fig.add(f'<g transform="translate({_f(ox)},210) scale(1,-1)">' +
                "".join(f'<path class="fig-leafy" d="{d}"/>'
                        for d in paths["blades"]) +
                (f'<path class="fig-vein" d="{paths["rachis"]}"/>'
                 if paths.get("rachis") else "") + "</g>")
        fig.title((ox, 34), label, "middle")
        fig.note((ox, 52), note, "middle")
        fig.note((ox, 240), f"{len(paths['blades'])} blade"
                            f"{'s' if len(paths['blades']) > 1 else ''}",
                 "middle")
    fig.note((340, 272), "The foliage budget is counted in blades, not "
                         "leaves, so a compound tree gets fewer, larger "
                         "fronds instead of costing five times as much.",
             "middle")
    fig.caption = ("Which architectures a kingdom can grow differs: trees "
                   "get compound and needle, bushes palmate and needle, "
                   "flowers simple and palmate.")
    return fig


def fig_cubic_fit() -> Fig:
    """Two curve segments instead of sixteen."""
    from .geometry import cubic_through

    fig = Fig(680, 330, "Fitting a blade margin")
    w, e = 0.34, 1.0
    scale = 240.0

    def margin(u: float) -> Vec:
        return (u, w * math.sin(math.pi * max(u, 0.0) ** e))

    # --- sampled ---------------------------------------------------------
    origin = (86.0, 148.0)
    sampled = [margin(i / 12) for i in range(13)]
    fig.title((86, 40), "sampled spline")
    fig.poly(fig.place(sampled, origin, scale), "fig-curve-3")
    for q in fig.place(sampled, origin, scale):
        fig.circle(q, 2.4, "fig-dot")
    fig.note((86, 176), "12 segments per margin")
    fig.note((86, 193), "~560 bytes a leaf")

    # --- fitted ----------------------------------------------------------
    origin2 = (380.0, 148.0)
    p0, p1, p2, p3 = margin(0), margin(1 / 3), margin(2 / 3), margin(1)
    c1, c2 = cubic_through(p0, p1, p2, p3)
    knots = fig.place([p0, p1, p2, p3], origin2, scale)
    ctrl = fig.place([c1, c2], origin2, scale)
    fig.title((380, 40), "one interpolating cubic")
    d = (f"M{_f(knots[0][0])},{_f(knots[0][1])}"
         f"C{_f(ctrl[0][0])},{_f(ctrl[0][1])} {_f(ctrl[1][0])},"
         f"{_f(ctrl[1][1])} {_f(knots[3][0])},{_f(knots[3][1])}")
    fig.line(knots[0], ctrl[0], "fig-leader")
    fig.line(knots[3], ctrl[1], "fig-leader")
    fig.path(d, "fig-curve")
    for q in knots:
        fig.circle(q, 3.2, "fig-dot fig-dot-big")
    for q in ctrl:
        fig.circle(q, 2.8, "fig-dot fig-accent-fill")
    fig.note((380, 176), "2 segments for the whole blade")
    fig.note((380, 193), "~90 bytes a leaf")
    fig.text((knots[1][0], knots[1][1] - 12), "u = 1/3",
             "fig-text fig-small", "middle")
    fig.text((knots[2][0], knots[2][1] - 12), "u = 2/3",
             "fig-text fig-small", "middle")
    fig.text((ctrl[0][0] - 10, ctrl[0][1] + 16), "control points",
             "fig-text fig-small fig-muted-ink", "end")

    fig.title((86, 244), "c1 = (-5*p0 + 18*p1 - 9*p2 + 2*p3) / 6")
    fig.note((86, 266), "The cubic that passes through p0..p3 with p1 and p2 "
                        "at u = 1/3 and 2/3. On a smooth margin the fit "
                        "error is")
    fig.note((86, 283), "far inside a pixel, so nothing is lost and most of "
                        "the file is.")
    fig.note((86, 308), "This is most of the difference between a 500KB tree "
                        "and a 60KB one.")
    fig.caption = ("Serrated margins still need the sampled spline, because "
                   "the teeth are the point. Everything else -- which is "
                   "nearly every blade -- takes the two-curve path.")
    return fig


def fig_bloom_forms() -> Fig:
    """The inflorescence archetypes, drawn by the real code."""
    from .genome import Genome
    from .organs import (BELL, CROSS, DAISY, GLOBE, POMPOM, SPIKE, STAR,
                         bloom_shapes)

    fig = Fig(680, 300, "Inflorescence archetypes")
    t = dict(Genome.from_id("apollo", "flower").traits())
    t["petals"] = 8.0
    t["petal_width"] = 0.55
    t["petal_notch"] = 0.2
    t["bloom_layers"] = 2.0

    fills = {"petal": "fig-petal", "petal_back": "fig-petal-back",
             "petal_light": "fig-petal-light", "center": "fig-centre",
             "center_light": "fig-centre-light", "stamen": "fig-petal-light",
             "bud": "fig-petal", "sepal": "fig-leafy",
             "globe": "fig-petal", "globe_back": "fig-petal-back",
             "floret": "fig-petal-light", "floret_shade": "fig-petal-back",
             "floret_center": "fig-centre"}
    forms = [(DAISY, "daisy"), (STAR, "star"), (CROSS, "cross"),
             (POMPOM, "pompom"), (BELL, "bell"), (SPIKE, "spike"),
             (GLOBE, "globe")]
    for k, (form, label) in enumerate(forms):
        ox = 66.0 + k * 90.0
        oy = 150.0
        shapes = bloom_shapes((0.0, 0.0), math.pi / 2, 30.0, 1.0, t,
                              detail=2, form=form)
        body = "".join(f'<path class="{fills.get(role, "fig-petal")}" '
                       f'd="{d}"/>' for role, d in shapes)
        fig.add(f'<g transform="translate({_f(ox + 40)},{_f(oy)}) '
                f'scale(1,-1)">{body}</g>')
        fig.title((ox + 40, 42), label, "middle")
    fig.note((340, 232), "A bell is not a wide daisy and a spike is not a "
                         "small one: each archetype is its own construction,",
             "middle")
    fig.note((340, 249), "sampled from by a gene rather than reached by "
                         "scaling a single shape.", "middle")
    fig.note((340, 276), "Flowers draw from all six; a bush gets scattered "
                         "blossom or globes; a tree gets blossom.", "middle")
    fig.caption = ("The bell is the one organ that needs its own outline: a "
                   "blade pinches shut at both ends, so it can make a bud "
                   "but never a mouth.")
    return fig


# ==========================================================================
# palette figures
# ==========================================================================

def _luminance(hexcolour: str) -> float:
    import colorsys
    r, g, b = (int(hexcolour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[1]


def fig_tone_ladder() -> Fig:
    """Masses must sit a clear step below the blades drawn over them."""
    from .growth import profile
    from .palette import build
    from .plant import Plant

    fig = Fig(680, 320, "The foliage tone ladder")
    genome = Plant.from_id("apollo", "tree").genome
    pal = build(genome, profile(21, "tree"))
    rungs = [("mass_back", pal.mass_back, "the shell, offset from the light"),
             ("mass_mid", pal.mass_mid, "the body of the canopy"),
             ("mass_front", pal.mass_front, "the lit cap"),
             ("blade_deep", pal.blade_deep, "leaves lost in the crown"),
             ("blade_shadow", pal.blade_shadow, "leaves in the foliage"),
             ("blade", pal.blade, "leaves out in front"),
             ("blade_light", pal.blade_light, "the lit half of a blade")]
    for k, (name, colour, note) in enumerate(rungs):
        y = 44 + k * 34
        fig.swatch(70, y, 64, 26, colour)
        fig.text((144, y + 17), name, "fig-text fig-code")
        fig.text((272, y + 17), f"L {_luminance(colour):.3f}",
                 "fig-text fig-small")
        fig.note((334, y + 17), note)
    fig.line((70, 142), (620, 142), "fig-dim")
    fig.note((70, 296), "Masses below the line, blades above it. An earlier "
                        "version had a gap of 0.026 between blade and")
    fig.note((70, 313), "mass_front: every leaf was being drawn and then "
                        "swallowed by the canopy under it.")
    fig.caption = ("Colour is built in HSL from a few genome-driven anchors, "
                   "so a plant's parts always harmonise. The one thing that "
                   "cannot be left to taste is the gap between a mass and "
                   "the blades that lie on it.")
    return fig


def fig_hue_bands() -> Fig:
    """Petals and berries come from bands, not from the whole wheel."""
    from .palette import _BERRY_BANDS, _BLOOM_BANDS, hsl

    fig = Fig(680, 300, "Botanical hue bands")
    x0, w = 66.0, 548.0
    for row, (title, bands, sat, light) in enumerate([
            ("petals", _BLOOM_BANDS, 0.62, 0.62),
            ("berries", _BERRY_BANDS, 0.70, 0.46)]):
        y = 60 + row * 108
        fig.title((66, y - 14), title)
        for i in range(120):
            h = i / 120 * 360.0
            fig.add(f'<rect x="{_f(x0 + i / 120 * w)}" y="{_f(y)}" '
                    f'width="{_f(w / 120 + 0.6)}" height="26" '
                    f'fill="{hsl(h, sat, light)}"/>')
        for lo, hi in bands:
            a = (lo % 360.0) / 360.0 * w
            b = (hi % 360.0) / 360.0 * w
            if b < a:
                fig.rect(x0 + a, y - 5, w - a, 36, "fig-band")
                fig.rect(x0, y - 5, b, 36, "fig-band")
            else:
                fig.rect(x0 + a, y - 5, b - a, 36, "fig-band")
        for tick in (0, 90, 180, 270, 360):
            fig.text((x0 + tick / 360 * w, y + 48), f"{tick}",
                     "fig-text fig-small fig-muted-ink", "middle")
    fig.note((66, 258), "Shaded: the bands a gene can land in. Everything "
                        "else -- the greens, the cyans -- is excluded.")
    fig.note((66, 275), "A garden of randomly seeded plants then still looks "
                        "like a garden rather than a colour picker.")
    fig.caption = ("Flower centres are always amber or brown, never a tint "
                   "of the petal: the contrast is what makes a bloom read at "
                   "icon size.")
    return fig


def fig_woodiness() -> Fig:
    """A stem browning off as the plant puts on wood."""
    from .growth import profile
    from .palette import build
    from .plant import Plant

    fig = Fig(680, 244, "Stem colour follows woodiness")
    genome = Plant.from_id("apollo", "tree").genome
    counts = (1, 2, 3, 5, 8, 13, 21)
    for k, n in enumerate(counts):
        g = profile(n, "tree")
        pal = build(genome, g)
        x = 72.0 + k * 78.0
        fig.swatch(x, 58, 58, 46, pal.bark, f"n={n}",
                   f"wood {g.woodiness:.2f}")
        fig.swatch(x, 146, 58, 24, pal.blade)
    fig.note((72, 190), "top: stem", "start")
    fig.note((72, 186 + 17), "bottom: foliage", "start")
    fig.note((196, 190), "Young growth is yellower and lighter; it cools and "
                         "deepens with age, and the stem is")
    fig.note((196, 207), "interpolated from green toward bark as "
                         "`woodiness` climbs. A flower never gets there.")
    fig.caption = ("Nothing switches over. Every colour is a continuous "
                   "interpolation on one of the growth channels, so there is "
                   "no frame where a plant changes species.")
    return fig


# ==========================================================================
# render figures
# ==========================================================================

def fig_layer_stack() -> Fig:
    """The z-order a plant is stacked in."""
    from . import render as R

    fig = Fig(680, 380, "Draw layers, back to front")
    layers = [
        (R.L_GROUND, "ground", "shadow and mound"),
        (R.L_MASS_BACK, "mass_back", "dark shell, offset from the light"),
        (R.L_LEAF_DEEP, "leaf_deep", "blades lost inside the crown"),
        (R.L_BRANCH, "branch", "all wood, in the skeleton's own order"),
        (R.L_BRANCH_LIGHT, "branch_light", "lit edges on broad wood"),
        (R.L_MASS_MID, "mass_mid", "the body of the canopy"),
        (R.L_LEAF_MID, "leaf_mid", "blades among the foliage"),
        (R.L_MASS_FRONT, "mass_front", "the lit cap"),
        (R.L_LEAF_BACK, "leaf_back", "blades over the canopy"),
        (R.L_LEAF, "leaf", "blades out in front"),
        (R.L_LEAF_DETAIL, "leaf_detail", "veins and highlights"),
        (R.L_BLOOM, "bloom", "flowers and buds"),
        (R.L_ACCENT, "accent", "berries and fruit"),
    ]
    for k, (num, name, note) in enumerate(layers):
        y = 40 + k * 25
        skew = 26.0
        x = 90.0 + (len(layers) - k) * 1.6
        fig.add(f'<path class="fig-plate" d="M{_f(x)},{_f(y)} '
                f'L{_f(x + 108)},{_f(y - 9)} L{_f(x + 148)},{_f(y - 3)} '
                f'L{_f(x + 40)},{_f(y + 6)}Z"/>')
        fig.text((66, y + 4), str(num), "fig-text fig-small fig-muted-ink",
                 "end")
        fig.text((252, y + 2), name, "fig-text fig-code")
        fig.note((370, y + 2), note)
    fig.note((90, 378), "Foliage is a band, then wood, then more foliage, so "
                        "limbs and leaves interleave.")
    fig.caption = ("Within the branch layer the skeleton's own emit order is "
                   "preserved, which is how a fork emerges from behind its "
                   "parent. Every other layer is flat.")
    return fig


def fig_detail_ladder() -> Fig:
    """What gets dropped as the render gets smaller."""
    from .plant import Plant

    fig = Fig(680, 330, "Scale-aware detail")
    plant = Plant.from_id("apollo", "tree")
    sizes = [(256, "full", "veins, leaf outlines, every twig"),
             (96, "compact", "no veins, fewer leaves, thin twigs stroked"),
             (32, "icon", "silhouette: masses, trunk, primary limbs")]
    x = 66.0
    for px, detail, note in sizes:
        svg = plant.svg(21, size=px, detail=detail, background=None,
                        ground=False)
        fig.embed(svg, x, 56, 1.0)
        fig.title((x, 44), f"{px}px, {detail}")
        fig.note((x, 64 + px + 18), note)
        fig.note((x, 64 + px + 35), f"{len(svg) / 1024:.1f} KB")
        x += px + 46
    fig.note((66, 296), "Line weights are specified in output pixels and "
                        "converted into plant space once the fit scale is "
                        "known.")
    fig.note((66, 313), "Structure below a couple of pixels is dropped, not "
                        "clamped up to a visible width -- clamping is what "
                        "makes a ball of fat twigs.")
    fig.caption = ("The same plant and the same skeleton. Only the drawing "
                   "changes, and it changes by discarding features that "
                   "cannot be seen rather than by scaling them down.")
    return fig


def fig_path_budget() -> Fig:
    """Where the bytes of a finished plant actually go."""
    import collections

    from . import render as R
    from .plant import Plant

    names = {R.L_GROUND: "ground", R.L_MASS_BACK: "canopy",
             R.L_MASS_MID: "canopy", R.L_MASS_FRONT: "canopy",
             R.L_BRANCH: "wood", R.L_BRANCH_LIGHT: "wood highlights",
             R.L_LEAF_DEEP: "leaves", R.L_LEAF_MID: "leaves",
             R.L_LEAF_BACK: "leaves", R.L_LEAF: "leaves",
             R.L_LEAF_DETAIL: "leaf detail", R.L_BLOOM: "blooms",
             R.L_ACCENT: "fruit"}
    plant = Plant.from_id("apollo", "tree")
    growth, sk = plant.build(21)
    tally: "collections.Counter[str]" = collections.Counter()
    original = R._Canvas._bucket

    def spy(self, layer, attrs, _o=original):
        bucket = _o(self, layer, attrs)
        key = names.get(layer, "other")

        class Counting(list):
            def append(self, d):
                tally[key] += len(d)
                list.append(bucket, d)

        return Counting()

    R._Canvas._bucket = spy
    try:
        total = len(plant.svg(21, size=256))
    finally:
        R._Canvas._bucket = original

    fig = Fig(680, 300, "Where the bytes go")
    fig.title((66, 30), f"a mature tree at 256px: {total / 1024:.0f} KB")
    top = tally.most_common()
    scale = 470.0 / max(v for _, v in top)
    for k, (label, value) in enumerate(top):
        y = 56 + k * 28
        fig.text((150, y + 12), label, "fig-text fig-small", "end")
        fig.rect(160, y, value * scale, 18, "fig-bar-c")
        fig.text((166 + value * scale, y + 13), f"{value / 1024:.1f} KB",
                 "fig-text fig-small")
    fig.note((66, 268), "Measured, not estimated. Profiling like this is how "
                        "the branch highlights turned out to cost more")
    fig.note((66, 285), "than the branches: a 40% highlight was being drawn "
                        "on five-pixel twigs where nothing could see it.")
    fig.caption = ("Paths are bucketed by (layer, style) and emitted inside "
                   "one group per bucket. Almost every path shares its fill "
                   "with dozens of others, and the repeated attributes cost "
                   "more than the geometry.")
    return fig


# ==========================================================================
# garden figures
# ==========================================================================

def fig_depth_mapping() -> Fig:
    """One number sets position, size and haze together."""
    from .garden import FAR_HEIGHT, NEAR_HEIGHT, Placement

    fig = Fig(680, 320, "Depth on the plane")
    height = 200.0
    horizon = 0.20
    box = (90.0, 46.0, 300.0, height)
    fig.rect(box[0], box[1], box[2], height * horizon, "fig-sky")
    fig.rect(box[0], box[1] + height * horizon, box[2],
             height * (1 - horizon), "fig-soil")
    fig.text((box[0] - 8, box[1] + height * horizon + 4), "horizon",
             "fig-text fig-small fig-muted-ink", "end")

    for depth in (0.0, 0.35, 0.7, 1.0):
        y = box[1] + lerp(height * horizon, height * 0.97, depth)
        h = lerp(FAR_HEIGHT, NEAR_HEIGHT, depth) * height
        fig.line((box[0], y), (box[0] + box[2], y), "fig-leader")
        cx = box[0] + 60 + depth * 170
        fig.rect(cx - 4, y - h, 8, h, "fig-bar-c")
        fig.text((cx + 10, y - h / 2), f"d={depth:g}",
                 "fig-text fig-small")

    plot = fig.axes((452, 46, 180, 130), (0, 1), (0, 1),
                    xlabel="depth", xticks=(0, 1), yticks=(0, 1))
    plot.curve(lambda d: lerp(FAR_HEIGHT, NEAR_HEIGHT, d) / NEAR_HEIGHT)
    plot.curve(lambda d: lerp(0.80, 1.0, d), cls="fig-curve-2")
    fig.note((452, 198), "green: size")
    fig.note((452, 215), "orange: opacity (aerial haze)")

    fig.note((90, 268), "Depth sets how far down the frame a plant stands, "
                        "how large it is, and how much air is in front of "
                        "it.")
    fig.note((90, 285), "Sorting by baseline then gives the painter's order "
                        "for nothing: near plants overlap far ones because")
    fig.note((90, 302), "they are drawn later.")
    fig.caption = ("That is the entire perspective. Plants are measured "
                   "against shared world units rather than each fitted to "
                   "its own box, so a mature tree towers over a mature "
                   "flower.")
    return fig


def fig_world_scale() -> Fig:
    """Measured mature heights, not assumed ones."""
    from .plant import Plant
    from .render import content_bounds
    from .skeleton import _BASE

    fig = Fig(680, 300, "World scale")
    fig.title((66, 30), "a mature plant of each kingdom, at one scale")
    heights = {}
    for kingdom in ("tree", "bush", "flower"):
        _, sk = Plant.from_id("apollo", kingdom).build(34)
        x0, y0, x1, y1 = content_bounds(sk)
        heights[kingdom] = y1 - y0

    scale = 170.0 / heights["tree"]
    x = 150.0
    for kingdom in ("tree", "bush", "flower"):
        _, sk = Plant.from_id("apollo", kingdom).build(34)
        _skeleton_wood(fig, sk, (x, 240.0), scale,
                       min_width=0.9 if kingdom == "tree" else 0.0)
        h = heights[kingdom] * scale
        fig.bracket((x + 62, 240.0), (x + 62, 240.0 - h),
                    f"{heights[kingdom] / heights['tree']:.2f}", offset=10)
        fig.note((x, 262), kingdom, "middle")
        fig.note((x, 279), f"declared {_BASE[kingdom][2]:g}", "middle")
        x += 175
    fig.note((66, 300 - 6), "")
    fig.caption = ("Everything downstream of `_trace` changes a plant's "
                   "final height, so the declared world height is measured "
                   "from the built skeleton rather than assumed. An assumed "
                   "figure had mature flowers drawn taller than the shrubs "
                   "behind them.")
    return fig

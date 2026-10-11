"""Composing many plants into one scene.

A garden is not a row of plants at the same size: it is a *plane* seen at a
shallow angle. Three things do all the work here.

* **Depth.** Every plant gets a depth 0..1. Depth sets how far down the
  frame it stands, how big it is, and how hazy it looks. That is the whole
  of the perspective, and it is enough.
* **World scale.** Plants are drawn against shared world units rather than
  each fitted to its own box, so a mature tree towers over a mature flower
  the way it should. This is what `render`'s `world_height` mode exists for.
* **Painter's order.** Sort by baseline, far first, and near plants overlap
  far ones for free.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import render as render_mod
from .genome import BUSH, FLOWER, TREE
from .geometry import clamp, lerp
from .growth import DEFAULT_FULL
from .plant import Plant

# Fraction of the scene's height a fully grown tree fills, nearest and
# furthest. The gap between them is the strength of the perspective.
NEAR_HEIGHT = 0.44
FAR_HEIGHT = 0.12

# Rough half-width of a mature plant, as a fraction of its own height. Used
# only for spacing, so a guess per kingdom is plenty.
_FOOTPRINT = {TREE: 0.40, BUSH: 0.52, FLOWER: 0.18}

SKY_TOP = "#e4e9d8"
SKY_HORIZON = "#f1efe3"
# Aerial perspective: the plane is hazier and lighter toward the horizon,
# which is most of what sells it as distance rather than as a stripe.
GROUND_FAR = "#c4cca8"
GROUND_NEAR = "#89996a"


@dataclass
class Placement:
    """One plant, positioned on the plane."""
    plant: Plant
    contributions: int
    x: float          # scene px, at the base of the stem
    y: float          # scene px, the baseline it stands on
    depth: float      # 0 far .. 1 near
    scale: float      # scene px per world unit

    @property
    def haze(self) -> float:
        """Distant plants sit slightly back into the air."""
        return lerp(0.80, 1.0, self.depth)


def _pick_kingdom(rng: random.Random, depth: float) -> str:
    """Big things read better toward the back, small ones toward the front.

    Not a rule of gardening, a rule of drawing: a mature tree in the near
    foreground eats the whole frame, and a seedling on the horizon is two
    pixels of nothing.
    """
    weights = {
        TREE: lerp(0.48, 0.13, depth),
        BUSH: 0.34,
        FLOWER: lerp(0.18, 0.53, depth),
    }
    r = rng.random() * sum(weights.values())
    for kingdom, w in weights.items():
        r -= w
        if r <= 0.0:
            return kingdom
    return FLOWER


def _pick_contributions(rng: random.Random, full: int) -> int:
    """A believable spread of project ages: mostly established, a few just
    sown, a few fully grown."""
    if rng.random() < 0.16:
        return rng.randint(0, 2)
    return max(1, int(round(rng.triangular(2.0, full * 1.7, full * 0.65))))


def _clear(placed: Sequence[Placement], x: float, y: float, radius: float,
           height: float) -> bool:
    """Reject crowding, but only between plants at a similar depth -- two
    plants far apart in the plane are *meant* to overlap."""
    band = height * 0.075
    for other in placed:
        if abs(other.y - y) > band:
            continue
        other_r = other.scale * 100.0 * _FOOTPRINT[other.plant.kingdom]
        if abs(other.x - x) < (radius + other_r) * 0.62:
            return False
    return True


def scatter(count: int, width: float, height: float, *, seed: int = 0,
            horizon: float = 0.20, full: int = DEFAULT_FULL,
            id_prefix: str = "plot", kingdoms: Optional[Sequence[str]] = None,
            attempts: int = 48) -> List[Placement]:
    """Strew `count` plants over the plane, at random ages."""
    rng = random.Random(seed)
    horizon_y = height * horizon
    placed: List[Placement] = []

    for i in range(count):
        for _ in range(attempts):
            depth = rng.random()
            y = lerp(horizon_y, height * 0.97, depth)
            scale = lerp(FAR_HEIGHT, NEAR_HEIGHT, depth) * height / 100.0
            kingdom = (rng.choice(list(kingdoms)) if kingdoms
                       else _pick_kingdom(rng, depth))
            radius = scale * 100.0 * _FOOTPRINT[kingdom]
            # inset by the plant's own footprint, so nothing is half off
            # the edge of the plane
            margin = min(radius, width * 0.22)
            x = rng.uniform(margin, width - margin)
            if _clear(placed, x, y, radius, height):
                placed.append(Placement(
                    plant=Plant.from_id(f"{id_prefix}-{seed}-{i}", kingdom,
                                        full=full),
                    contributions=_pick_contributions(rng, full),
                    x=x, y=y, depth=depth, scale=scale))
                break
        # if every attempt was crowded out, that plant simply does not grow

    placed.sort(key=lambda p: p.y)        # painter's order: far first
    return placed


def render_scene(placements: Sequence[Placement], width: int, height: int, *,
                 horizon: float = 0.20, detail: str = "auto",
                 sky: bool = True, haze: bool = True) -> str:
    """Compose placed plants into one standalone SVG scene."""
    horizon_y = height * horizon
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="A garden of {len(placements)} plants">'
    ]
    if sky:
        parts.append(
            '<defs>'
            '<linearGradient id="gg-sky" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{SKY_TOP}"/>'
            f'<stop offset="1" stop-color="{SKY_HORIZON}"/></linearGradient>'
            '<linearGradient id="gg-soil" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{GROUND_FAR}"/>'
            f'<stop offset="1" stop-color="{GROUND_NEAR}"/></linearGradient>'
            '</defs>'
            f'<rect width="{width}" height="{horizon_y + 1:.1f}" '
            f'fill="url(#gg-sky)"/>'
            f'<rect y="{horizon_y:.1f}" width="{width}" '
            f'height="{height - horizon_y:.1f}" fill="url(#gg-soil)"/>')

    for p in placements:
        growth, sk = p.plant.build(p.contributions)
        group = render_mod.plant_group(
            p.plant.genome, growth, sk, scale=p.scale, x=p.x, y=p.y,
            detail=detail)
        if haze and p.haze < 0.995:
            group = f'<g opacity="{p.haze:.2f}">{group}</g>'
        parts.append(group)

    parts.append("</svg>")
    return "".join(parts)


def garden(count: int = 22, width: int = 900, height: int = 520, *,
           seed: int = 0, **kwargs) -> str:
    """Scatter and render in one call."""
    scatter_kwargs = {k: kwargs.pop(k) for k in
                      ("horizon", "full", "id_prefix", "kingdoms")
                      if k in kwargs}
    places = scatter(count, width, height, seed=seed, **scatter_kwargs)
    if "horizon" in scatter_kwargs:
        kwargs.setdefault("horizon", scatter_kwargs["horizon"])
    return render_scene(places, width, height, **kwargs)

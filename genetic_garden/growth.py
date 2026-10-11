"""Growth: contributions -> a maturity profile.

One number (`Growth.value`, 0..1) drives everything, but each kingdom reads
it through its own schedule so that a half-grown tree and a half-grown
flower are each recognisably half-grown *for their kind*.

The important design constraint is that growth must never reshuffle the
plant. The genome fixes the individual; growth only extends it. In
particular `depth` is deliberately **fractional**: the newest level of
branches emerges scaled-down from its parent tips rather than popping into
existence, so consecutive contributions read as the same plant getting
bigger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .genome import BUSH, FLOWER, TREE
from .geometry import clamp, lerp, smoothstep

# Contributions needed to reach full maturity. Logarithmic, so the first
# few contributions visibly move the needle and later ones refine.
DEFAULT_FULL = 21

# Shapes the maturity curve. >1 keeps the very first contributions modest
# (a sprout should look like a sprout) while still front-loading progress
# relative to a linear count.
CURVE = 1.6

STAGE_NAMES = ("seed", "sprout", "seedling", "juvenile", "maturing", "mature")
STAGE_CUTS = (0.001, 0.14, 0.32, 0.56, 0.82)


def maturity(contributions: int, full: int = DEFAULT_FULL,
             vigor: float = 1.0) -> float:
    """Map a count of sequential contributions onto 0..1.

    Logarithmic: contribution 1 is worth far more than contribution 20.
    `vigor` is a genome trait, so two projects at the same count are not
    quite at the same size -- a little life in the garden.
    """
    n = max(0, int(contributions))
    if n == 0:
        return 0.0
    raw = (math.log1p(n) / math.log1p(max(1, full))) ** CURVE
    return clamp(raw * vigor)


def stage_name(value: float) -> str:
    for name, cut in zip(STAGE_NAMES, STAGE_CUTS):
        if value < cut:
            return name
    return STAGE_NAMES[-1]


# (depth_min, depth_max, scale_min, bloom_start)
_SCHEDULE = {
    TREE:   (1.0, 5.4, 0.15, 0.78),
    BUSH:   (1.0, 4.3, 0.20, 0.72),
    FLOWER: (0.0, 2.6, 0.24, 0.26),
}


@dataclass(frozen=True)
class Growth:
    """Everything the builders need to know about how far along a plant is."""

    contributions: int
    value: float         # 0..1 overall maturity
    stage: str
    scale: float         # size relative to a mature plant of this genome
    depth: float         # fractional recursion depth
    leafiness: float     # 0..1, how much foliage is out
    bloom: float         # 0..1, blossom/fruit openness and abundance
    woodiness: float     # 0..1, bark vs green stem

    @property
    def is_seed(self) -> bool:
        return self.value <= 0.0

    @property
    def is_sprout(self) -> bool:
        """Below this, the plant is drawn as a cotyledon sprout, not a
        miniature adult -- real seedlings do not have branch structure."""
        return 0.0 < self.value < 0.14


def profile(contributions: int, kingdom: str, *, full: int = DEFAULT_FULL,
            vigor: float = 1.0, stature: float = 1.0) -> Growth:
    g = maturity(contributions, full=full, vigor=vigor)
    depth_lo, depth_hi, scale_lo, bloom_start = _SCHEDULE[kingdom]

    # Size grows fast early then eases off; depth climbs steadily so that
    # structure keeps arriving even once overall size has plateaued.
    scale = lerp(scale_lo, 1.0, g ** 0.72) * stature
    depth = lerp(depth_lo, depth_hi, smoothstep(clamp(g * 1.04)))

    leafiness = smoothstep(clamp((g - 0.05) / 0.32))
    bloom = smoothstep(clamp((g - bloom_start) / (1.0 - bloom_start + 1e-6)))
    woodiness = 0.0 if kingdom == FLOWER else smoothstep(clamp((g - 0.18) / 0.55))

    return Growth(
        contributions=max(0, int(contributions)),
        value=g,
        stage=stage_name(g),
        scale=scale,
        depth=depth,
        leafiness=leafiness,
        bloom=bloom,
        woodiness=woodiness,
    )

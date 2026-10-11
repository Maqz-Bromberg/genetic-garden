"""The front door: a `Plant` ties a genome to a growth curve and renders it.

    from genetic_garden import Plant

    p = Plant.from_id("apollo-rewrite", "tree")
    open("apollo.svg", "w").write(p.svg(contributions=8))
    open("apollo-icon.svg", "w").write(p.icon(contributions=8, size=24))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from . import render as render_mod
from . import skeleton as skeleton_mod
from .genome import Genome
from .growth import DEFAULT_FULL, Growth, profile
from .skeleton import Skeleton


@dataclass
class Plant:
    genome: Genome
    full: int = DEFAULT_FULL
    _cache: Dict[Tuple[int, int], Tuple[Growth, Skeleton]] = field(
        default_factory=dict, repr=False, compare=False)
    _ref: Optional[Tuple[float, float]] = field(
        default=None, repr=False, compare=False)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_id(cls, identifier: str, kingdom: Optional[str] = None,
                full: int = DEFAULT_FULL) -> "Plant":
        """The usual entry point: a stable plant for a project id."""
        return cls(Genome.from_id(identifier, kingdom), full=full)

    @classmethod
    def from_genome(cls, genome: Genome, full: int = DEFAULT_FULL) -> "Plant":
        return cls(genome, full=full)

    # -- growth ----------------------------------------------------------

    def growth(self, contributions: int) -> Growth:
        t = self.genome.traits()
        return profile(contributions, self.genome.kingdom, full=self.full,
                       vigor=t["vigor"], stature=t["stature"])

    def build(self, contributions: int) -> Tuple[Growth, Skeleton]:
        key = (int(contributions), self.full)
        if key not in self._cache:
            g = self.growth(contributions)
            self._cache[key] = (g, skeleton_mod.build(self.genome, g))
        return self._cache[key]

    # -- rendering -------------------------------------------------------

    def mature_span(self) -> Tuple[float, float]:
        """Bounding span of this plant at full maturity. Rendering every
        stage against it is what makes growth visible: a sprout sits small
        in the frame and a mature plant fills it."""
        if self._ref is None:
            _, sk = self.build(self.full)
            x0, y0, x1, y1 = render_mod.content_bounds(sk)
            self._ref = (x1 - x0, y1 - y0)
        return self._ref

    def svg(self, contributions: int, size: int = 256, *,
            height: Optional[int] = None, detail: str = "full",
            ground: bool = True, background: Optional[str] = None,
            fit: str = "mature", growth_gamma: float = 0.55) -> str:
        """`fit` is one of:

        mature  scale against this plant's own full-grown size (default)
        bbox    fill the frame whatever the stage
        world   scale against shared world units, for plants side by side

        `growth_gamma` compresses the mature-relative range. 1.0 is honest
        scaling, which leaves a sprout too small to recognise; lower values
        keep early stages legible while still reading as small.
        """
        growth, sk = self.build(contributions)
        reference = self.mature_span() if fit == "mature" else None
        world = sk.canonical_height if fit == "world" else None
        return render_mod.render(
            self.genome, growth, sk, width=size, height=height, detail=detail,
            ground=ground, background=background, world_height=world,
            reference=reference, growth_gamma=growth_gamma)

    def icon(self, contributions: int, size: int = 24) -> str:
        """A list-header icon: silhouette-first, no ground, heavier minimum
        line weights so nothing disappears at small sizes."""
        # Icons still scale with growth, but on a flatter curve: at 16px a
        # faithfully-scaled seedling would be a smudge, while filling the
        # frame at every stage would throw the growth signal away entirely.
        return self.svg(contributions, size=size, detail="icon",
                        ground=False, fit="mature", growth_gamma=0.34)

    def world_svg(self, contributions: int, size: int = 256,
                  **kwargs) -> str:
        """Render at honest cross-kingdom scale, so a mature tree towers
        over a mature flower. Use this for plants sharing a garden."""
        return self.svg(contributions, size=size, fit="world", **kwargs)

    # -- breeding --------------------------------------------------------

    def mutate(self, rate: float = 0.12, scale: float = 0.10,
               rng=None) -> "Plant":
        return Plant(self.genome.mutate(rate, scale, rng), full=self.full)

    def cross(self, other: "Plant", rng=None) -> "Plant":
        return Plant(self.genome.crossover(other.genome, rng), full=self.full)

    # -- misc ------------------------------------------------------------

    @property
    def kingdom(self) -> str:
        return self.genome.kingdom

    @property
    def label(self) -> str:
        return self.genome.label

    def stage(self, contributions: int) -> str:
        return self.growth(contributions).stage

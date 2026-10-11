"""Genomes: stable, breedable parameter sets.

A genome is a plain dict of *genes*, each a float in [0, 1]. Nothing in the
chromosome knows about plants -- expression happens in `traits()`, which
maps each gene through a per-kingdom range. Keeping the chromosome
kingdom-agnostic has two payoffs:

* crossover and mutation can never produce an invalid plant, and
* the same chromosome expressed as a tree and as a flower yields two
  plants that visibly share a "family resemblance" (same lean, same
  branch rhythm, same hue shift) across kingdoms.

Genomes are derived deterministically from a string id, so a given project
always grows the same plant.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

# --------------------------------------------------------------------------
# kingdoms
# --------------------------------------------------------------------------

TREE = "tree"
BUSH = "bush"
FLOWER = "flower"
KINGDOMS = (TREE, BUSH, FLOWER)


# --------------------------------------------------------------------------
# gene -> trait ranges, per kingdom
# --------------------------------------------------------------------------

# Each entry is (low, high) for the expressed trait. A gene value of 0 maps
# to `low`, 1 maps to `high`. Genes absent from a kingdom's table fall back
# to COMMON.
COMMON: Dict[str, Tuple[float, float]] = {
    # --- skeleton -------------------------------------------------------
    "lean":            (-0.20, 0.20),   # radians the whole plant tilts
    "curve":           (-0.55, 0.55),   # per-branch constant curvature
    "sway":            (0.10, 0.85),    # amplitude of per-node wobble
    "branch_angle":    (0.35, 0.85),    # radians, child vs parent
    "angle_falloff":   (0.55, 1.00),    # angle multiplier per extra depth
    "length_ratio":    (0.58, 0.80),    # child length / parent length
    "width_ratio":     (0.60, 0.78),    # child width / parent width
    "internodes":      (2.0, 4.0),      # branch points along a branch
    "upward":          (0.05, 0.55),    # phototropic straightening
    "droop":           (0.00, 0.35),    # gravity pull on thin branches
    "asymmetry":       (0.00, 0.45),    # left/right vigour imbalance
    # Overall habit. This one gene is worth more to how distinct two plants
    # of a kingdom look than any other, because it acts on the recursion
    # rather than on a single feature: columnar vs spreading for a tree,
    # upright vs mounding for a bush.
    "form":            (0.00, 1.00),
    "taper":           (0.55, 0.85),    # how fast a branch narrows
    # --- foliage --------------------------------------------------------
    "leaf_size":       (0.70, 1.35),    # multiplier on kingdom base size
    "leaf_slender":    (0.30, 0.62),    # blade width / blade length
    "leaf_pitch":      (0.25, 1.05),    # radians off the twig
    "leaf_density":    (0.45, 1.00),    # fraction of leaf slots filled
    "leaf_lobes":      (0.00, 1.00),    # 0 smooth ovate .. 1 toothed
    # Leaf architecture: one blade, a compound frond, a palmate hand or a
    # needle tuft. Thresholds differ per kingdom -- see organs.leaf_form.
    "leaf_form":       (0.00, 1.00),
    # Foliage texture. Low: a few bold leaves, and the canopy masses carry
    # the crown. High: many tiny leaves that carry it themselves, with the
    # masses pulled back. Fine grain costs more paths and simplifies worse
    # at icon size, which is the trade being exposed here.
    "leaf_grain":      (0.00, 1.00),
    # How much of the woody interior shows through the foliage.
    "crown_open":      (0.00, 1.00),
    # --- masses ---------------------------------------------------------
    "canopy_spread":   (0.75, 1.30),
    "canopy_fluff":    (0.10, 0.34),    # blob edge irregularity
    # --- bloom ----------------------------------------------------------
    "petals":          (5.0, 8.0),
    "petal_width":     (0.45, 0.95),
    "petal_notch":     (0.00, 1.00),
    "bloom_size":      (0.80, 1.20),
    "bloom_layers":    (1.0, 2.0),
    # Which inflorescence this plant carries. Thresholds are per kingdom --
    # see organs.bloom_form. Flowers get the widest choice; bushes choose
    # between scattered blossom and big hydrangea-style heads.
    "bloom_form":      (0.00, 1.00),
    "accent":          (0.00, 1.00),    # fruit / berry / bud abundance
    # --- colour ---------------------------------------------------------
    "hue":             (-16.0, 16.0),   # degrees off the kingdom hue
    "sat":             (-0.10, 0.12),
    "light":           (-0.07, 0.07),
    "bark_hue":        (-14.0, 14.0),
    "bloom_hue":       (0.0, 360.0),    # absolute; blooms roam free
    # --- growth ---------------------------------------------------------
    "vigor":           (0.80, 1.25),    # scales the growth curve
    "stature":         (0.85, 1.20),    # final size multiplier
}

PER_KINGDOM: Dict[str, Dict[str, Tuple[float, float]]] = {
    TREE: {
        "branch_angle":  (0.42, 0.80),
        "length_ratio":  (0.62, 0.79),
        "internodes":    (2.0, 4.0),
        "upward":        (0.12, 0.50),
        "leaf_size":     (0.55, 1.05),
        "leaf_grain":    (0.15, 1.00),   # trees carry the finest foliage
        "canopy_spread": (0.80, 1.25),
        "petals":        (5.0, 6.0),
        "bloom_size":    (0.45, 0.70),
    },
    BUSH: {
        "lean":          (-0.10, 0.10),
        "branch_angle":  (0.52, 1.05),   # wide, repeated forks
        "length_ratio":  (0.56, 0.72),
        "width_ratio":   (0.62, 0.80),
        "internodes":    (1.0, 3.0),
        "upward":        (0.05, 0.30),
        "droop":         (0.05, 0.40),
        "leaf_size":     (0.75, 1.25),
        "leaf_grain":    (0.00, 0.75),   # bushes read boldest up close
        "leaf_density":  (0.65, 1.00),
        "canopy_spread": (0.95, 1.45),
        "canopy_fluff":  (0.14, 0.40),
        "petals":        (5.0, 5.0),
        "bloom_size":    (0.30, 0.50),
        "stature":       (0.80, 1.10),
    },
    FLOWER: {
        "lean":          (-0.14, 0.14),
        "curve":         (-0.75, 0.75),  # a pronounced S in the stem
        "branch_angle":  (0.45, 0.95),
        "length_ratio":  (0.50, 0.70),
        "width_ratio":   (0.70, 0.88),
        "internodes":    (2.0, 4.0),
        "upward":        (0.20, 0.60),
        "leaf_size":     (1.05, 1.85),   # big leaves on a small plant
        "leaf_grain":    (0.00, 0.30),
        "leaf_slender":  (0.22, 0.48),
        "leaf_density":  (0.60, 1.00),
        "petals":        (5.0, 13.0),
        "petal_width":   (0.40, 1.00),
        "bloom_size":    (0.95, 1.45),
        "bloom_layers":  (1.0, 3.0),
        "stature":       (0.85, 1.15),
    },
}

GENE_NAMES = tuple(sorted(COMMON))

# Presentation order for tools that expose the genes for editing. Purely
# cosmetic, but a flat list of 30 sliders is unusable.
GENE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "skeleton": ("form", "lean", "curve", "sway", "branch_angle",
                 "angle_falloff", "length_ratio", "width_ratio", "internodes",
                 "upward", "droop", "asymmetry", "taper"),
    "foliage": ("leaf_size", "leaf_slender", "leaf_pitch", "leaf_density",
                "leaf_lobes", "leaf_form", "leaf_grain", "canopy_spread",
                "canopy_fluff", "crown_open"),
    "bloom": ("petals", "petal_width", "petal_notch", "bloom_size",
              "bloom_layers", "bloom_form", "accent"),
    "colour": ("hue", "sat", "light", "bark_hue", "bloom_hue"),
    "growth": ("vigor", "stature"),
}

assert set(GENE_NAMES) == {g for names in GENE_GROUPS.values() for g in names}, (
    "GENE_GROUPS must cover every gene exactly once")


def gene_ranges(kingdom: str) -> Dict[str, Tuple[float, float]]:
    """Expressed trait range per gene, for a kingdom. Used by tooling that
    wants to show what a gene actually means."""
    return _ranges(kingdom)


def _ranges(kingdom: str) -> Dict[str, Tuple[float, float]]:
    table = dict(COMMON)
    table.update(PER_KINGDOM.get(kingdom, {}))
    return table


# --------------------------------------------------------------------------
# genome
# --------------------------------------------------------------------------

def stable_seed(identifier: str) -> int:
    """A process-independent seed for a string (hash() is salted per-run)."""
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class Genome:
    kingdom: str
    genes: Mapping[str, float]
    label: str = ""
    seed: int = 0
    _traits: Dict[str, float] = field(
        default_factory=dict, repr=False, compare=False)

    # -- construction ----------------------------------------------------

    @classmethod
    def random(cls, kingdom: str, rng: random.Random,
               label: str = "", seed: int = 0) -> "Genome":
        if kingdom not in KINGDOMS:
            raise ValueError(
                f"unknown kingdom {kingdom!r}; expected one of {KINGDOMS}")
        genes = {name: rng.random() for name in GENE_NAMES}
        return cls(kingdom, genes, label=label, seed=seed)

    @classmethod
    def from_id(cls, identifier: str, kingdom: str | None = None) -> "Genome":
        """Deterministic genome for a project id.

        With `kingdom=None` the kingdom itself is drawn from the id, which is
        handy for filling a garden without curating every entry.
        """
        seed = stable_seed(identifier)
        rng = random.Random(seed)
        if kingdom is None:
            kingdom = KINGDOMS[rng.randrange(len(KINGDOMS))]
        return cls.random(kingdom, rng, label=identifier, seed=seed)

    # -- expression ------------------------------------------------------

    def traits(self) -> Dict[str, float]:
        """Genes mapped through this kingdom's ranges. Cached."""
        if not self._traits:
            table = _ranges(self.kingdom)
            for name, (lo, hi) in table.items():
                g = float(self.genes.get(name, 0.5))
                self._traits[name] = lo + (hi - lo) * g
        return self._traits

    def trait(self, name: str) -> float:
        return self.traits()[name]

    def rng(self, stream: str = "") -> random.Random:
        """A named, reproducible RNG stream for this genome.

        Separate streams mean that adding noise to (say) leaf placement will
        not reshuffle the branch structure -- important for keeping growth
        stages of one plant looking like the same individual.
        """
        return random.Random(stable_seed(f"{self.seed}:{self.kingdom}:{stream}"))

    # -- breeding --------------------------------------------------------

    def mutate(self, rate: float = 0.12, scale: float = 0.10,
               rng: random.Random | None = None,
               label: str | None = None) -> "Genome":
        """Small perturbations to a few genes.

        Note the interaction with growth: because most genes feed a
        *recursive* rule (branch angle, length ratio, curvature), a 0.05
        nudge is nearly invisible on a seedling and obvious on a mature
        plant -- the divergence compounds once per level of depth.
        """
        rng = rng or random.Random()
        genes = dict(self.genes)
        for name in GENE_NAMES:
            if rng.random() < rate:
                v = genes.get(name, 0.5) + rng.gauss(0.0, scale)
                genes[name] = min(1.0, max(0.0, v))
        return Genome(self.kingdom, genes,
                      label=label if label is not None else self.label + "*",
                      seed=rng.randrange(1 << 62))

    def crossover(self, other: "Genome", rng: random.Random | None = None,
                  label: str = "") -> "Genome":
        """Per-gene blend, biased toward one parent or the other."""
        rng = rng or random.Random()
        if other.kingdom != self.kingdom:
            raise ValueError("cannot cross genomes from different kingdoms")
        genes = {}
        for name in GENE_NAMES:
            a = self.genes.get(name, 0.5)
            b = other.genes.get(name, 0.5)
            t = rng.random()
            genes[name] = a + (b - a) * t
        return Genome(self.kingdom, genes,
                      label=label or f"{self.label}x{other.label}",
                      seed=rng.randrange(1 << 62))

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "kingdom": self.kingdom,
            "label": self.label,
            "seed": self.seed,
            # Deliberately unrounded: a saved genome has to reproduce the
            # identical drawing, and traits are read straight off these.
            "genes": {k: float(v) for k, v in sorted(self.genes.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Genome":
        return cls(data["kingdom"], dict(data["genes"]),
                   label=data.get("label", ""), seed=int(data.get("seed", 0)))

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

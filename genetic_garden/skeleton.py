"""Skeleton construction: genome + growth -> branch geometry and organ sites.

This module produces *structure only* -- polylines, widths, and a list of
places where leaves, foliage masses and blooms belong. No colour, no SVG.
`render.py` turns it into a drawing.

Three rules keep the kingdoms visually distinct at every size:

  tree    one dominant trunk, branching begins high, laterals along the
          trunk plus a continuing leader, foliage concentrated in a crown
  bush    no trunk at all -- several stems from ground level, wide repeated
          forks, foliage covering the whole silhouette
  flower  a single slender stem, few but large leaves, and one terminal
          bloom that is the focal point

Randomness is addressed rather than streamed: every branch derives its own
RNG from its position in the tree ("0.2.1"), so growing the plant one level
deeper never reshuffles the branches that were already there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from . import organs
from .genome import BUSH, FLOWER, TREE, Genome, stable_seed
from .geometry import (TAU, Vec, add, angle_of, clamp, from_angle, lerp,
                       normalize, smoothstep)
from .growth import Growth

import random

UP = math.pi / 2

# Below this, an emerging branch is too small to draw at all. A branch whose
# children would all fall under it is therefore the terminal one, and must
# carry the foliage itself -- otherwise a plant whose fractional depth lands
# just above a whole number sheds its entire outer layer and renders bald.
MIN_EMERGENCE = 0.02


# --------------------------------------------------------------------------
# output records
# --------------------------------------------------------------------------

@dataclass
class Branch:
    points: List[Vec]
    widths: List[float]
    order: int            # 0 = trunk / basal stem
    emergence: float      # 1.0 = fully grown, <1 = this level is still pushing out
    kind: str             # trunk | branch | twig | stem | peduncle
    front: bool = False   # does this limb pass in front of the trunk?
    terminal: bool = True # nothing continues from the tip, so taper it off


@dataclass
class Leaf:
    pos: Vec
    angle: float          # direction the blade points
    size: float
    flip: int             # +1 / -1, which way the blade curls
    shade: float          # 0..1, depth cue -- lower values sit behind
    depth: float = 1.0    # 0 buried in the crown .. 1 out in front of it


@dataclass
class Mass:
    """A soft foliage volume, anchored on a cluster of branch tips.

    Masses carry the crown: they are what survives downscaling, and they
    sit *behind* the drawn leaves so the leaves break their edge and stop
    the plant looking like broccoli."""
    center: Vec
    radius: float
    squash: float
    weight: float         # how many tips fed this blob


@dataclass
class Bloom:
    pos: Vec
    angle: float
    size: float
    openness: float       # 0 bud .. 1 fully open
    terminal: bool
    form: str = "daisy"   # see organs.bloom_form


@dataclass
class Accent:
    """Berry, hip or fruit."""
    pos: Vec
    size: float


@dataclass
class Skeleton:
    branches: List[Branch] = field(default_factory=list)
    leaves: List[Leaf] = field(default_factory=list)
    masses: List[Mass] = field(default_factory=list)
    blooms: List[Bloom] = field(default_factory=list)
    accents: List[Accent] = field(default_factory=list)
    cotyledons: List[Leaf] = field(default_factory=list)
    tips: List[Vec] = field(default_factory=list)
    seed_radius: float = 0.0     # non-zero only at the seed stage
    base_width: float = 1.0
    canonical_height: float = 100.0


# How many individual leaf blades to actually draw. Woody plants get their
# volume from foliage masses; the drawn leaves are edge detail that breaks up
# the blob silhouette. Keeping this bounded is what stops a mature tree from
# becoming a 9000-path SVG.
# Budget the foliage in *blades*, not leaves, so that a compound-leaved
# tree does not cost seven times a simple-leaved one. The range is wide on
# purpose: low `leaf_grain` means a few bold leaves over the canopy masses,
# high means a haze of tiny ones that carries the crown by itself.
BLADE_BUDGET = {TREE: (74, 300), BUSH: (78, 260), FLOWER: (10_000, 10_000)}


def _leaf_budget(genome: Genome, t: Dict[str, float]) -> int:
    lo, hi = BLADE_BUDGET[genome.kingdom]
    blades = lerp(lo, hi, t["leaf_grain"])
    per_leaf = organs.leaflet_count(organs.leaf_form(genome.kingdom, t))
    return max(6, int(blades / per_leaf))

# Blossom and fruit on a woody plant are texture scattered over a crown,
# not the subject of the drawing. Left unbudgeted a mature tree grows one
# full flower per twig -- hundreds of them, dominating both the image and
# the file.
BLOOM_BUDGET = {TREE: 14, BUSH: 18, FLOWER: 10_000}
ACCENT_BUDGET = {TREE: 16, BUSH: 22, FLOWER: 10_000}


# Base dimensions per kingdom, in shared world units, so that a garden can
# place a mature tree next to a mature flower at honest relative sizes. The
# third number is the height a mature plant actually reaches -- measured,
# not assumed, because everything downstream of `_trace` (curvature,
# gravity, the crown) changes it. Get it wrong and `fit="world"` quietly
# draws flowers taller than the shrubs behind them.
_BASE = {
    #        trunk/stem length, base width, world height
    TREE:   (46.0, 7.4, 114.0),
    BUSH:   (20.0, 3.4, 56.0),
    FLOWER: (18.0, 1.15, 32.0),
}

# A seedling is much the same size whatever it will become, but not
# identically so -- an acorn's shoot outgrows a poppy's from the start.
_SPROUT_SCALE = {TREE: 1.0, BUSH: 0.80, FLOWER: 0.58}


def _form_traits(genome: Genome) -> Dict[str, float]:
    """Expressed traits with the `form` gene folded in.

    `form` is applied here rather than in the genome tables because it
    scales several traits at once -- it describes a habit, not a
    measurement. Because every trait it touches feeds the recursion, a
    small shift in `form` is barely visible on a sapling and produces a
    completely different silhouette on a mature plant.
    """
    t = dict(genome.traits())
    f = t["form"]
    if genome.kingdom == TREE:
        t["branch_angle"] *= lerp(0.62, 1.34, f)      # columnar .. spreading
        t["upward"] *= lerp(1.55, 0.60, f)
        t["length_ratio"] *= lerp(1.06, 0.93, f)
        t["droop"] *= lerp(0.55, 1.50, f)
        t["canopy_spread"] *= lerp(0.88, 1.10, f)
    elif genome.kingdom == BUSH:
        t["branch_angle"] *= lerp(0.72, 1.24, f)      # upright .. mounding
        t["upward"] *= lerp(1.60, 0.58, f)
        t["canopy_spread"] *= lerp(0.84, 1.14, f)
    else:
        t["upward"] *= lerp(1.30, 0.82, f)
    return t


# --------------------------------------------------------------------------
# tracing a single branch
# --------------------------------------------------------------------------

# A branch does not meet its parent at a clean right-angled cut -- it
# swells into one. Flaring the first fraction of a limb gives the branch
# collar; doing the same at ground level gives a tree its root buttress.
COLLAR_SPAN = 0.20          # fraction of the branch the flare covers
COLLAR_FLARE = 0.40         # how much wider the very base gets
# A leader is the same axis carrying on, not a new limb, so it barely
# swells -- a full collar there makes every trunk joint bulge like a knuckle.
LEADER_FLARE = 0.12
BUTTRESS_SPAN = 0.16
BUTTRESS_FLARE = 0.55

# How far a limb is sunk into its parent, as a multiple of its own base
# width. The buried part is a wedge rather than a flat end, so where a limb
# crosses in front of the trunk the join still reads as growth.
ROOT_DEPTH = 1.15
ROOT_WIDTH = 0.52
# A limb that passes in front of its parent has nothing to hide its buried
# end, so it gets a thin spike instead of a blunt one: a narrow wedge reads
# as the limb emerging, a wide one reads as a seam drawn across the trunk.
FRONT_ROOT_WIDTH = 0.16
FRONT_ROOT_DEPTH = 0.55


def _collar(u: float, flare: float, span: float) -> float:
    """Width multiplier along a branch, swelling at the base."""
    if flare <= 0.0 or u >= span:
        return 1.0
    k = 1.0 - u / span
    return 1.0 + flare * k * k


def _trace(rng: random.Random, origin: Vec, theta: float, length: float,
           w0: float, w1: float, *, curve: float, upward: float,
           droop: float, sway: float, thinness: float, steps: int,
           flare: float = 0.0, flare_span: float = COLLAR_SPAN,
           root: float = 0.0, root_width: float = ROOT_WIDTH
           ) -> Tuple[List[Vec], List[float]]:
    """Walk a branch forward, bending it as it goes.

    Three forces act at every step: a constant per-branch curvature (what
    gives a branch its 'handwriting'), phototropism pulling toward vertical,
    and gravity pulling thin branches back down. The sum is what separates
    these from straight-line stick figures.

    `flare` swells the base into a collar, and `root` prepends a point
    behind the origin so the branch starts inside its parent's wood.
    """
    pts: List[Vec] = []
    ws: List[float] = []
    if root > 0.0:
        pts.append(add(origin, from_angle(theta, -root)))
        ws.append(w0 * root_width)
    pts.append(origin)
    ws.append(w0 * _collar(0.0, flare, flare_span))

    pos = origin
    step = length / steps
    for i in range(1, steps + 1):
        t = i / steps
        theta += curve / steps
        # phototropism: stiff, thick wood resists less the higher it is
        theta += upward * _angle_delta(theta, UP) / steps
        # gravity: only meaningfully bends slender wood
        theta += droop * thinness * _angle_delta(theta, -UP) / steps * 0.9
        theta += rng.gauss(0.0, sway * 0.055)
        pos = add(pos, from_angle(theta, step))
        pts.append(pos)
        ws.append(lerp(w0, w1, t) * _collar(t, flare, flare_span))
    return pts, ws


def _angle_delta(a: float, b: float) -> float:
    """Signed shortest rotation from a to b."""
    d = (b - a + math.pi) % TAU - math.pi
    return d


def _flare_for(order: int, leader: bool) -> float:
    if order == 0:
        return BUTTRESS_FLARE
    return LEADER_FLARE if leader else COLLAR_FLARE


def _root_depth(front: bool, leader: bool) -> float:
    if leader:
        return 0.80
    return FRONT_ROOT_DEPTH if front else ROOT_DEPTH


def _branch_rng(genome: Genome, path: str) -> random.Random:
    return random.Random(stable_seed(f"{genome.seed}:{genome.kingdom}:b:{path}"))


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def build(genome: Genome, growth: Growth) -> Skeleton:
    if growth.is_seed:
        return _build_seed(genome, growth)
    if growth.is_sprout:
        return _build_sprout(genome, growth)
    if genome.kingdom == TREE:
        return _build_tree(genome, growth)
    if genome.kingdom == BUSH:
        return _build_bush(genome, growth)
    return _build_flower(genome, growth)


# --------------------------------------------------------------------------
# earliest stages -- shared, because real seedlings look alike
# --------------------------------------------------------------------------

def _build_seed(genome: Genome, growth: Growth) -> Skeleton:
    """No contributions yet: a sown seed, barely breaking the soil."""
    t = genome.traits()
    sk = Skeleton(canonical_height=_BASE[genome.kingdom][2])
    r = 1.05 * t["stature"] * lerp(1.0, _SPROUT_SCALE[genome.kingdom], 0.6)
    sk.seed_radius = r
    sk.base_width = r
    return sk


def _build_sprout(genome: Genome, growth: Growth) -> Skeleton:
    """A cotyledon sprout. Kingdom shows through in the shoot's posture and
    the cotyledon shape rather than in branching, which does not exist yet."""
    t = genome.traits()
    rng = _branch_rng(genome, "sprout")
    sk = Skeleton(canonical_height=_BASE[genome.kingdom][2])

    h = 10.0 * growth.scale * t["stature"] * _SPROUT_SCALE[genome.kingdom]
    w = 1.15 * growth.scale * t["stature"]
    curve = t["curve"] * (0.5 if genome.kingdom != FLOWER else 1.1)
    pts, ws = _trace(rng, (0.0, 0.0), UP + t["lean"] * 0.6, h, w, w * 0.62,
                     curve=curve, upward=0.5, droop=0.0, sway=t["sway"] * 0.4,
                     thinness=0.4, steps=5)
    sk.branches.append(Branch(pts, ws, 0, 1.0, "stem", False, True))
    sk.base_width = w

    tip = pts[-1]
    tip_dir = angle_of(normalize((tip[0] - pts[-2][0], tip[1] - pts[-2][1])))
    spread = {TREE: 0.85, BUSH: 1.05, FLOWER: 0.74}[genome.kingdom]
    size = h * {TREE: 0.40, BUSH: 0.36, FLOWER: 0.46}[genome.kingdom]
    for side in (1, -1):
        sk.cotyledons.append(Leaf(
            pos=tip, angle=tip_dir + side * spread,
            size=size * (1.0 + 0.08 * side * t["asymmetry"]),
            flip=side, shade=1.0))
    # the pair of true leaves just showing between the cotyledons
    if growth.value > 0.07:
        sk.leaves.append(Leaf(pos=tip, angle=tip_dir, size=size * 0.45,
                              flip=1, shade=1.0))
    return sk


# --------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------

def _build_tree(genome: Genome, growth: Growth) -> Skeleton:
    t = _form_traits(genome)
    sk = Skeleton(canonical_height=_BASE[TREE][2])
    base_len, base_w, _ = _BASE[TREE]

    length = base_len * growth.scale * t["stature"]
    width = base_w * growth.scale * t["stature"] * lerp(0.6, 1.0, growth.woodiness)
    sk.base_width = width

    _tree_branch(sk, genome, t, growth, path="0", origin=(0.0, 0.0),
                 theta=UP + t["lean"], length=length, width=width,
                 remaining=growth.depth, order=0, base_width=width,
                 front=False)
    _add_masses(sk, genome, t, growth)
    _cull_leaves(sk, genome, _leaf_budget(genome, t))
    rng = _branch_rng(genome, "thin")
    sk.blooms = _thin(sk.blooms, BLOOM_BUDGET[TREE], rng)
    sk.accents = _thin(sk.accents, ACCENT_BUDGET[TREE], rng)
    return sk


def _tree_branch(sk: Skeleton, genome: Genome, t: Dict[str, float],
                 growth: Growth, *, path: str, origin: Vec, theta: float,
                 length: float, width: float, remaining: float, order: int,
                 base_width: float, front: bool = False,
                 leader: bool = False) -> None:
    rng = _branch_rng(genome, path)
    emergence = clamp(remaining)
    if emergence <= MIN_EMERGENCE:
        return

    # A newly emerging level pushes out of its parent rather than appearing.
    grow_len = length * lerp(0.12, 1.0, smoothstep(emergence))
    grow_w = width * lerp(0.35, 1.0, emergence)

    thinness = clamp(1.0 - width / max(base_width, 1e-6))
    steps = 7 if order == 0 else max(3, 6 - order)
    sign = 1.0 if (order + len(path)) % 2 == 0 else -1.0
    curve = t["curve"] * sign * (0.45 if order == 0 else 1.0)

    pts, ws = _trace(
        rng, origin, theta, grow_len, grow_w, grow_w * t["taper"],
        curve=curve, upward=t["upward"] * (0.8 if order == 0 else 1.0),
        droop=t["droop"], sway=t["sway"], thinness=thinness, steps=steps,
        flare=_flare_for(order, leader),
        flare_span=BUTTRESS_SPAN if order == 0 else COLLAR_SPAN,
        root=0.0 if order == 0 else grow_w * _root_depth(front, leader),
        root_width=FRONT_ROOT_WIDTH if (front and not leader) else ROOT_WIDTH)

    kind = "trunk" if order == 0 else ("branch" if order <= 2 else "twig")
    terminal = remaining - 1.0 <= MIN_EMERGENCE
    me = Branch(pts, ws, order, emergence, kind, front, terminal)

    if terminal:
        sk.branches.append(me)
        _foliage_on(sk, genome, t, growth, pts, ws, order, emergence,
                    tip_cluster=True)
        return

    # --- children ------------------------------------------------------
    # Laterals along the upper part of the branch, plus a leader that
    # carries the trunk onward. Branching starts high on the trunk, which
    # is most of what makes a tree read as a tree.
    n_lat = int(round(t["internodes"]))
    if order == 0:
        n_lat += 1
    else:
        # Fan out hard at the top of the trunk, then restrain: without this
        # the branch count explodes geometrically and the crown turns to mush.
        n_lat = max(1, n_lat - 1 - order // 3)
    start = 0.62 if order == 0 else 0.34
    angle = t["branch_angle"] * (t["angle_falloff"] ** order)
    child_len = length * t["length_ratio"]
    if order == 0:
        # First-order limbs must stay clearly subordinate to the trunk,
        # or the tree reads as a bush with a stick under it.
        child_len *= 0.70
    child_w = width * t["width_ratio"]

    # Work out every child first, so the parent's own RNG stream is
    # unaffected by the order they are later drawn in.
    specs = []
    side = 1.0 if rng.random() < 0.5 else -1.0
    for i in range(n_lat):
        f = start + (1.0 - start) * ((i + 0.5) / n_lat)
        idx = min(len(pts) - 1, max(1, int(round(f * (len(pts) - 1)))))
        anchor = pts[idx]
        local = angle_of(normalize((pts[idx][0] - pts[idx - 1][0],
                                    pts[idx][1] - pts[idx - 1][1])))
        jitter = rng.uniform(-0.16, 0.16)
        # vigour imbalance: one flank of the tree outgrows the other
        bias = 1.0 + t["asymmetry"] * (0.55 if side > 0 else -0.55)
        # branches lower on the trunk leave at a wider angle
        widen = lerp(1.22, 0.80, f)
        # A whole limb sits either in front of the trunk or behind it, and
        # keeps that choice for its own branches.
        kid_front = front if order > 0 else (rng.random() < 0.34)
        specs.append(dict(
            path=f"{path}.{i}", origin=anchor,
            theta=local + side * angle * widen + jitter,
            length=child_len * bias * lerp(1.12, 0.82, f),
            width=child_w * lerp(1.0, 0.82, f),
            remaining=remaining - 1.0, order=order + 1,
            base_width=base_width, front=kid_front))
        side = -side

    tip_dir = angle_of(normalize((pts[-1][0] - pts[-2][0],
                                  pts[-1][1] - pts[-2][1])))
    leader = dict(
        path=f"{path}.L", origin=pts[-1],
        theta=tip_dir + rng.uniform(-0.1, 0.1) + side * angle * 0.28,
        length=child_len * (1.05 if order == 0 else 0.92),
        width=child_w * 1.04, remaining=remaining - 1.0,
        order=order + 1, base_width=base_width, front=front, leader=True)

    # Draw order, and the whole reason for building the list first:
    #   limbs passing behind -> this branch -> its own continuation ->
    #   limbs passing in front.
    # Laterals emerge from behind their parent, while the leader is drawn
    # over it so that the leader's collar hides the parent's cut tip. Get
    # this backwards and every trunk grows a seam across it.
    for spec in specs:
        if not spec["front"]:
            _tree_branch(sk, genome, t, growth, **spec)
    sk.branches.append(me)
    _tree_branch(sk, genome, t, growth, **leader)
    for spec in specs:
        if spec["front"]:
            _tree_branch(sk, genome, t, growth, **spec)

    # a little foliage on inner branches keeps the crown from looking hollow
    if order >= 2 and growth.leafiness > 0.4:
        _foliage_on(sk, genome, t, growth, pts, ws, order, emergence,
                    tip_cluster=False, density=0.35)


# --------------------------------------------------------------------------
# bush
# --------------------------------------------------------------------------

def _build_bush(genome: Genome, growth: Growth) -> Skeleton:
    t = _form_traits(genome)
    sk = Skeleton(canonical_height=_BASE[BUSH][2])
    base_len, base_w, _ = _BASE[BUSH]

    habit = lerp(1.26, 0.86, t["form"])          # upright .. mounding
    length = base_len * growth.scale * t["stature"] * habit
    width = base_w * growth.scale * t["stature"]
    sk.base_width = width

    rng = _branch_rng(genome, "root")
    # Several stems straight from the ground -- no trunk. The fan is what
    # distinguishes a bush silhouette from a small tree.
    n_stems = 3 + int(round(growth.value * 3.2))
    spread = (0.30 + 0.55 * t["canopy_spread"] / 1.45) * lerp(0.80, 1.18,
                                                              t["form"])
    for i in range(n_stems):
        f = (i + 0.5) / n_stems - 0.5           # -0.5 .. 0.5
        theta = UP + t["lean"] + f * 2.0 * spread + rng.uniform(-0.07, 0.07)
        # outer stems are shorter, which rounds the crown
        drop = 1.0 - 0.30 * abs(f * 2.0) ** 1.6
        _bush_branch(
            sk, genome, t, growth, path=f"s{i}",
            origin=(f * width * 0.9, 0.0), theta=theta,
            length=length * drop * rng.uniform(0.92, 1.08),
            width=width * lerp(1.0, 0.72, abs(f * 2.0)),
            remaining=growth.depth, order=0, base_width=width,
            front=rng.random() < 0.45)
    _add_masses(sk, genome, t, growth)
    _cull_leaves(sk, genome, _leaf_budget(genome, t))
    rng = _branch_rng(genome, "thin")
    if organs.bloom_form(BUSH, t) == organs.GLOBE:
        _globe_heads(sk, genome, t, growth)
    else:
        sk.blooms = _thin(sk.blooms, BLOOM_BUDGET[BUSH], rng)
    sk.accents = _thin(sk.accents, ACCENT_BUDGET[BUSH], rng)
    return sk


def _bush_branch(sk: Skeleton, genome: Genome, t: Dict[str, float],
                 growth: Growth, *, path: str, origin: Vec, theta: float,
                 length: float, width: float, remaining: float, order: int,
                 base_width: float, front: bool = False) -> None:
    rng = _branch_rng(genome, path)
    emergence = clamp(remaining)
    if emergence <= MIN_EMERGENCE:
        return

    grow_len = length * lerp(0.15, 1.0, smoothstep(emergence))
    grow_w = width * lerp(0.4, 1.0, emergence)
    thinness = clamp(1.0 - width / max(base_width, 1e-6))

    pts, ws = _trace(
        rng, origin, theta, grow_len, grow_w, grow_w * t["taper"],
        curve=t["curve"] * (1.0 if order % 2 == 0 else -1.0),
        upward=t["upward"], droop=t["droop"] * (0.6 + order * 0.25),
        sway=t["sway"], thinness=thinness, steps=max(3, 5 - order),
        flare=BUTTRESS_FLARE if order == 0 else COLLAR_FLARE,
        flare_span=BUTTRESS_SPAN if order == 0 else COLLAR_SPAN,
        root=0.0 if order == 0 else grow_w * _root_depth(front, False),
        root_width=FRONT_ROOT_WIDTH if front else ROOT_WIDTH)

    terminal = remaining - 1.0 <= MIN_EMERGENCE
    me = Branch(pts, ws, order, emergence,
                "stem" if order == 0 else "twig", front, terminal)

    if terminal:
        sk.branches.append(me)
        _foliage_on(sk, genome, t, growth, pts, ws, order, emergence,
                    tip_cluster=True)
        return

    # Sympodial forking: the tip splits, no leader survives. Wide angles.
    tip_dir = angle_of(normalize((pts[-1][0] - pts[-2][0],
                                  pts[-1][1] - pts[-2][1])))
    angle = t["branch_angle"] * (t["angle_falloff"] ** order)
    n_kids = 2 if rng.random() > 0.22 else 3
    forks = []
    for k in range(n_kids):
        f = (k + 0.5) / n_kids - 0.5
        bias = 1.0 + t["asymmetry"] * (0.5 if f > 0 else -0.5)
        forks.append(dict(
            path=f"{path}.{k}", origin=pts[-1],
            theta=tip_dir + f * 2.0 * angle + rng.uniform(-0.12, 0.12),
            length=length * t["length_ratio"] * bias,
            width=width * t["width_ratio"],
            remaining=remaining - 1.0, order=order + 1,
            base_width=base_width, front=front))

    # a mid-branch lateral every so often, to break up the regular forking
    lateral = None
    if rng.random() < 0.45 and len(pts) > 3:
        idx = len(pts) // 2
        local = angle_of(normalize((pts[idx][0] - pts[idx - 1][0],
                                    pts[idx][1] - pts[idx - 1][1])))
        lateral = dict(
            path=f"{path}.m", origin=pts[idx],
            theta=local + (1 if rng.random() < 0.5 else -1) * angle * 1.15,
            length=length * t["length_ratio"] * 0.72,
            width=width * t["width_ratio"] * 0.85,
            remaining=remaining - 1.4, order=order + 1,
            base_width=base_width, front=front)

    # A mid-branch lateral emerges from behind its stem; the tip forks are
    # drawn over it, so their collars cover where it was cut off.
    if lateral is not None:
        _bush_branch(sk, genome, t, growth, **lateral)
    sk.branches.append(me)
    for fork in forks:
        _bush_branch(sk, genome, t, growth, **fork)

    if growth.leafiness > 0.35:
        _foliage_on(sk, genome, t, growth, pts, ws, order, emergence,
                    tip_cluster=False, density=0.5)


# --------------------------------------------------------------------------
# flower
# --------------------------------------------------------------------------

def _build_flower(genome: Genome, growth: Growth) -> Skeleton:
    t = _form_traits(genome)
    sk = Skeleton(canonical_height=_BASE[FLOWER][2])
    base_len, base_w, _ = _BASE[FLOWER]
    rng = _branch_rng(genome, "root")

    length = base_len * growth.scale * t["stature"]
    width = base_w * growth.scale * t["stature"]
    sk.base_width = width

    _flower_stem(sk, genome, t, growth, path="0", origin=(0.0, 0.0),
                 theta=UP + t["lean"], length=length, width=width,
                 remaining=growth.depth, order=0, rng=rng)

    # A basal rosette anchors the plant and reads clearly even as an icon.
    n_basal = 2 + int(round(growth.value * 2.4))
    for i in range(n_basal):
        side = 1 if i % 2 == 0 else -1
        f = (i // 2 + 1) / (n_basal // 2 + 1)
        sk.leaves.append(Leaf(
            pos=(side * width * 0.35, length * 0.04 * i),
            angle=side * (0.28 + 0.34 * f) + rng.uniform(-0.1, 0.1),
            size=length * 0.26 * t["leaf_size"] / 1.4 * lerp(1.0, 0.68, f),
            flip=side, shade=0.82 if i % 2 else 1.0))
    return sk


def _flower_stem(sk: Skeleton, genome: Genome, t: Dict[str, float],
                 growth: Growth, *, path: str, origin: Vec, theta: float,
                 length: float, width: float, remaining: float, order: int,
                 rng: random.Random) -> None:
    brng = _branch_rng(genome, path)
    emergence = clamp(remaining + 1.0)   # the main stem exists from the start
    grow_len = length * lerp(0.35, 1.0, smoothstep(emergence))

    pts, ws = _trace(
        brng, origin, theta, grow_len, width, width * 0.80,
        curve=t["curve"] * (1.0 if order == 0 else -0.8),
        upward=t["upward"], droop=t["droop"] * 0.5, sway=t["sway"] * 0.6,
        thinness=0.3 + 0.3 * order, steps=8,
        flare=0.35 if order == 0 else COLLAR_FLARE,
        flare_span=0.12 if order == 0 else COLLAR_SPAN,
        root=0.0 if order == 0 else width * ROOT_DEPTH)
    sk.branches.append(Branch(pts, ws, order, 1.0,
                              "stem" if order == 0 else "peduncle",
                              False, True))

    # Stem leaves: few, large, alternating.
    n_leaves = max(1, int(round(t["internodes"])) + 1)
    for i in range(n_leaves):
        if brng.random() > t["leaf_density"] * (0.45 + 0.55 * growth.leafiness):
            continue
        f = 0.18 + 0.52 * ((i + 0.5) / n_leaves)
        idx = max(1, int(round(f * (len(pts) - 1))))
        side = 1 if i % 2 == 0 else -1
        local = angle_of(normalize((pts[idx][0] - pts[idx - 1][0],
                                    pts[idx][1] - pts[idx - 1][1])))
        sk.leaves.append(Leaf(
            pos=pts[idx],
            angle=local + side * lerp(0.52, 1.00, t["leaf_pitch"] / 1.05),
            size=length * 0.21 * t["leaf_size"] / 1.4 * lerp(1.05, 0.7, f)
            * lerp(0.55, 1.0, growth.leafiness),
            flip=side, shade=1.0))

    tip = pts[-1]
    tip_dir = angle_of(normalize((tip[0] - pts[-2][0], tip[1] - pts[-2][1])))

    # Side stems only once the plant is well established.
    if remaining > 1.45:
        n_side = 1 + int(remaining - 1.0 > 0.6)
        if organs.bloom_form(FLOWER, t) == organs.BELL:
            n_side += 1
        angle = t["branch_angle"]
        for k in range(n_side):
            side = 1 if k % 2 == 0 else -1
            idx = max(1, int(0.62 * (len(pts) - 1)))
            local = angle_of(normalize((pts[idx][0] - pts[idx - 1][0],
                                        pts[idx][1] - pts[idx - 1][1])))
            _flower_stem(
                sk, genome, t, growth, path=f"{path}.{k}", origin=pts[idx],
                theta=local + side * angle,
                length=length * t["length_ratio"] * 0.85,
                width=width * t["width_ratio"],
                remaining=remaining - 1.6, order=order + 1, rng=rng)

    # Terminal bloom. Openness is the whole growth story for a flower.
    form = organs.bloom_form(FLOWER, t)
    size = length * 0.44 * t["bloom_size"] / 1.2 * organs.bloom_size_scale(form)
    if order > 0:
        size *= 0.78
    open_t = clamp((growth.bloom - 0.05 * order) / 0.95)
    sk.blooms.append(Bloom(pos=tip, angle=tip_dir,
                           size=size * lerp(0.55, 1.0, open_t),
                           openness=open_t, terminal=(order == 0),
                           form=form))


# --------------------------------------------------------------------------
# foliage placement (trees & bushes)
# --------------------------------------------------------------------------

def _foliage_on(sk: Skeleton, genome: Genome, t: Dict[str, float],
                growth: Growth, pts: List[Vec], ws: List[float], order: int,
                emergence: float, *, tip_cluster: bool,
                density: float = 1.0) -> None:
    if growth.leafiness <= 0.02:
        return
    rng = _branch_rng(genome, f"leaf{order}:{pts[0][0]:.2f},{pts[0][1]:.2f}")
    kingdom = genome.kingdom
    base = {TREE: 5.2, BUSH: 3.4}.get(kingdom, 3.6)
    form_scale = organs.form_size_scale(organs.leaf_form(kingdom, t))
    size = (base * t["leaf_size"] * growth.scale * form_scale
            * lerp(1.34, 0.62, t["leaf_grain"])
            * lerp(0.70, 1.0, growth.leafiness) * lerp(0.5, 1.0, emergence))

    slots = (3 if tip_cluster else 2) + int(round(t["leaf_grain"] * 2.0))
    for i in range(slots):
        for side in (1, -1):
            if rng.random() > t["leaf_density"] * density * growth.leafiness:
                continue
            f = 0.35 + 0.62 * ((i + 0.5) / slots)
            idx = max(1, min(len(pts) - 1, int(round(f * (len(pts) - 1)))))
            local = angle_of(normalize((pts[idx][0] - pts[idx - 1][0],
                                        pts[idx][1] - pts[idx - 1][1])))
            sk.leaves.append(Leaf(
                pos=pts[idx],
                angle=local + side * t["leaf_pitch"] + rng.uniform(-0.25, 0.25),
                size=size * rng.uniform(0.78, 1.22),
                flip=side,
                shade=rng.uniform(0.72, 1.0),
                depth=rng.random()))

    if tip_cluster:
        sk.tips.append(pts[-1])
        tip_dir = angle_of(normalize((pts[-1][0] - pts[-2][0],
                                      pts[-1][1] - pts[-2][1])))
        sk.leaves.append(Leaf(pos=pts[-1], angle=tip_dir,
                              size=size * 1.05, flip=1, shade=1.0,
                              depth=rng.uniform(0.55, 1.0)))

    # Fruit / berries appear only on a plant that has been tended a while.
    if growth.bloom > 0.15 and tip_cluster:
        chance = growth.bloom * t["accent"] * 0.55
        if rng.random() < chance:
            sk.accents.append(Accent(pos=pts[-1],
                                     size=size * 0.30 * (0.8 + 0.5 * t["accent"])))
        if kingdom == BUSH and rng.random() < chance * 0.7:
            sk.blooms.append(Bloom(pos=pts[-1], angle=UP,
                                   size=size * 0.62 * t["bloom_size"] / 0.4,
                                   openness=growth.bloom, terminal=False,
                                   form=organs.SCATTER))
        if kingdom == TREE and rng.random() < chance * 0.5:
            sk.blooms.append(Bloom(pos=pts[-1], angle=UP,
                                   size=size * 0.5 * t["bloom_size"] / 0.6,
                                   openness=growth.bloom, terminal=False,
                                   form=organs.SCATTER))


def _thin(items: list, budget: int, rng: random.Random) -> list:
    """Keep a random `budget` of items, deterministically."""
    if len(items) <= budget:
        return items
    idx = list(range(len(items)))
    rng.shuffle(idx)
    keep = sorted(idx[:budget])
    return [items[i] for i in keep]


def _cull_leaves(sk: Skeleton, genome: Genome, budget: int) -> None:
    """Thin the leaf set down to a drawing budget.

    Outermost leaves are kept preferentially -- they read against the sky
    and give the crown its ragged edge -- but only just over half, because
    a crown whose leaves all sit on the perimeter looks like sprigs stuck
    round a blob. The rest are scattered over the canopy face.
    """
    if len(sk.leaves) <= budget:
        return
    rng = _branch_rng(genome, "cull")
    cx = sum(lf.pos[0] for lf in sk.leaves) / len(sk.leaves)
    cy = sum(lf.pos[1] for lf in sk.leaves) / len(sk.leaves)
    ranked = sorted(sk.leaves,
                    key=lambda lf: -math.hypot(lf.pos[0] - cx, lf.pos[1] - cy))
    n_edge = int(budget * 0.45)
    keep = ranked[:n_edge]
    rest = ranked[n_edge:]
    rng.shuffle(rest)
    keep.extend(rest[:budget - n_edge])
    # restore draw order (back-to-front by shade) rather than by radius
    keep.sort(key=lambda lf: lf.shade)
    sk.leaves = keep


def _globe_heads(sk: Skeleton, genome: Genome, t: Dict[str, float],
                 growth: Growth) -> None:
    """Hydrangea-style heads: a few big round trusses over the crown.

    Sized from the plant's own span rather than from leaf size, so a bush
    in full bloom really is mostly flower -- around a third of its area at
    the largest.
    """
    sk.blooms = [b for b in sk.blooms if b.terminal]
    if growth.bloom < 0.04 or not sk.masses:
        return
    xs = [m.center[0] for m in sk.masses]
    ys = [m.center[1] for m in sk.masses]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)

    rng = _branch_rng(genome, "globes")
    n = 4 + int(round(t["accent"] * 3.0))
    radius = span * lerp(0.105, 0.175, (t["bloom_size"] - 0.30) / 0.20)

    # prefer the upper, outer masses -- trusses sit where the light is
    ranked = sorted(sk.masses, key=lambda m: -(m.center[1] + m.radius * 0.5))
    pool = ranked[:max(n, int(len(ranked) * 0.6))]
    rng.shuffle(pool)
    for m in pool[:n]:
        jitter = (rng.uniform(-0.3, 0.3) * m.radius,
                  rng.uniform(-0.2, 0.3) * m.radius)
        sk.blooms.append(Bloom(
            pos=add(m.center, jitter), angle=UP,
            size=radius * rng.uniform(0.82, 1.16),
            openness=growth.bloom, terminal=False, form=organs.GLOBE))


def _add_masses(sk: Skeleton, genome: Genome, t: Dict[str, float],
                growth: Growth) -> None:
    """Fit soft volumes to clusters of branch tips.

    Tips are snapped to a grid whose cell size scales with the crown, so a
    dense twiggy region yields one fat blob rather than fifty overlapping
    ones. The union of the blobs is the crown silhouette.
    """
    if growth.leafiness < 0.10 or len(sk.tips) < 2:
        return
    xs = [p[0] for p in sk.tips]
    ys = [p[1] for p in sk.tips]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)

    spread = t["canopy_spread"]
    cell = span * 0.155 / max(spread, 0.5)
    cells: Dict[Tuple[int, int], List[Vec]] = {}
    for p in sk.tips:
        key = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))
        cells.setdefault(key, []).append(p)

    squash = 0.92 if genome.kingdom == TREE else 0.84
    for group in cells.values():
        gx = sum(p[0] for p in group) / len(group)
        gy = sum(p[1] for p in group) / len(group)
        # a cell holding more tips earns a slightly larger blob
        bulk = 1.0 + 0.16 * math.log1p(len(group) - 1)
        # Fine foliage and an open crown both mean the blobs should do less
        # of the work -- the leaves and the visible wood carry it instead.
        r = (cell * 0.94 * bulk * spread
             * lerp(1.0, 0.82, t["leaf_grain"])
             * lerp(1.06, 0.86, t["crown_open"])
             * lerp(0.62, 1.0, growth.leafiness))
        sk.masses.append(Mass(center=(gx, gy), radius=r, squash=squash,
                              weight=float(len(group))))

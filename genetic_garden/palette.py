"""Colour. Everything is built in HSL from a small number of genome-driven
anchors, so a plant's parts always harmonise and mutation shifts the whole
family together rather than producing clashing parts.

Growth drives colour too: young growth is yellow-green and pale, mature
growth is deeper and cooler, and stems brown off as `woodiness` climbs.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

from . import organs
from .genome import BUSH, FLOWER, TREE, Genome
from .geometry import clamp, lerp
from .growth import Growth


def hsl(h: float, s: float, l: float) -> str:
    """HSL (h in degrees, s/l in 0..1) to #rrggbb."""
    h = (h % 360.0) / 360.0
    s = clamp(s)
    l = clamp(l)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255 + 0.5), int(g * 255 + 0.5),
                              int(b * 255 + 0.5))


def mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colours. Used to flood a crown with blossom."""
    t = clamp(t)
    parts = []
    for i in (1, 3, 5):
        ca, cb = int(a[i:i + 2], 16), int(b[i:i + 2], 16)
        parts.append(int(ca + (cb - ca) * t + 0.5))
    return "#%02x%02x%02x" % tuple(parts)


# Flowers do not come in cyan. Genes map into bands that real petals
# occupy -- warm reds, yellows, pinks and purples -- which keeps a garden
# of randomly seeded plants looking like a garden.
_BLOOM_BANDS = ((348.0, 400.0), (38.0, 62.0), (296.0, 344.0), (250.0, 292.0))


def _bloom_hue(gene: float) -> float:
    n = len(_BLOOM_BANDS)
    i = min(n - 1, int(clamp(gene) * n))
    lo, hi = _BLOOM_BANDS[i]
    return lerp(lo, hi, clamp(gene) * n - i) % 360.0


# Berries and hips: scarlet, orange, damson, sloe. Not the petal's
# complement, which lands on cyan about a quarter of the time.
_BERRY_BANDS = ((2.0, 16.0), (20.0, 34.0), (300.0, 330.0), (232.0, 252.0))


def _berry(gene: float) -> tuple:
    n = len(_BERRY_BANDS)
    i = min(n - 1, int(clamp(gene) * n))
    lo, hi = _BERRY_BANDS[i]
    h = lerp(lo, hi, clamp(gene) * n - i)
    dark = 232.0 <= h <= 252.0
    return (h, 0.55 if dark else 0.72, 0.36 if dark else 0.47)


def _center(bloom_h: float) -> tuple:
    """Flower centres are amber or brown, never a tint of the petal --
    the contrast is what makes a bloom read at icon size."""
    if 30.0 <= bloom_h <= 72.0:          # yellow petals need a dark centre
        return (28.0, 0.62, 0.34)
    return (46.0, 0.84, 0.50)


# Kingdom anchors: (foliage hue, bark hue, bark lightness)
_ANCHOR = {
    TREE:   (108.0, 28.0, 0.34),
    BUSH:   (118.0, 32.0, 0.36),
    FLOWER: (100.0, 96.0, 0.42),
}


@dataclass(frozen=True)
class Palette:
    bark: str
    bark_shadow: str
    bark_light: str
    mass_back: str
    mass_mid: str
    mass_front: str
    blade: str
    blade_light: str
    blade_shadow: str
    blade_deep: str
    vein: str
    petal: str
    petal_light: str
    petal_shadow: str
    bloom_center: str
    bloom_center_light: str
    accent: str
    accent_light: str
    seed: str
    seed_light: str
    outline: str
    ground: str
    ground_light: str


def build(genome: Genome, growth: Growth) -> Palette:
    t = genome.traits()
    leaf_h0, bark_h0, bark_l0 = _ANCHOR[genome.kingdom]

    # Young foliage is yellower and lighter; it cools and deepens with age.
    age = growth.value
    lh = leaf_h0 + t["hue"] - lerp(18.0, 0.0, age)
    ls = clamp(0.40 + t["sat"] + lerp(0.10, -0.02, age))
    ll = clamp(0.44 + t["light"] - lerp(0.0, 0.07, age))

    bh = bark_h0 + t["bark_hue"]
    bl = bark_l0 + t["light"] * 0.5
    # A young stem is green; it browns off as the plant puts on wood.
    w = growth.woodiness
    stem_h = lerp(lh - 4.0, bh, w)
    stem_s = lerp(ls * 0.92, 0.30, w)
    stem_l = lerp(ll * 0.86, bl, w)

    bloom_h = _bloom_hue(t["bloom_hue"] / 360.0)
    showy = organs.bloom_form(genome.kingdom, t) in (organs.GLOBE,)
    if genome.kingdom != FLOWER and not showy:
        # Scattered blossom on woody plants stays near white/cream/pink --
        # saturated primaries on a tree read as fruit, not flowers.
        bloom_h = lerp(330.0, 400.0, (bloom_h % 360.0) / 360.0)
        bloom_s, bloom_l = 0.45, 0.86
    elif showy:
        # A hydrangea truss is the point of the plant, and hydrangeas come
        # in blue, pink and violet. Pale blossom colours leave the florets
        # with nothing to read against.
        bloom_h = lerp(200.0, 340.0, (t["bloom_hue"] % 360.0) / 360.0)
        bloom_s, bloom_l = clamp(0.44 + t["sat"] * 1.4), 0.70
    else:
        pale = t["light"] > 0.045          # a white-flowered strain
        bloom_s = 0.14 if pale else clamp(0.62 + t["sat"] * 1.5)
        bloom_l = 0.90 if pale else 0.62

    return Palette(
        bark=hsl(stem_h, stem_s, stem_l),
        bark_shadow=hsl(stem_h - 4, stem_s * 1.05, stem_l * 0.68),
        bark_light=hsl(stem_h + 6, stem_s * 0.85, min(0.92, stem_l * 1.34)),

        # The masses sit clearly darker than the blades drawn over them.
        # Without that gap the individual leaves vanish into the canopy and
        # all the foliage detail is wasted work.
        mass_back=hsl(lh + 6, ls * 1.02, ll * 0.60),
        mass_mid=hsl(lh, ls, ll * 0.76),
        mass_front=hsl(lh - 5, ls * 0.97, min(0.86, ll * 0.93)),

        # Blades sit a clear step lighter than the mass they lie on. A gap
        # of a couple of percent luminance is invisible, and all the leaf
        # geometry underneath it is then wasted.
        blade=hsl(lh, ls, min(0.78, ll * 1.20)),
        blade_light=hsl(lh - 9, ls * 0.95, min(0.90, ll * 1.48)),
        blade_shadow=hsl(lh + 8, ls * 1.03, ll * 1.00),
        blade_deep=hsl(lh + 12, ls * 1.05, ll * 0.84),
        vein=hsl(lh - 6, ls * 0.75, min(0.9, ll * 1.38)),

        petal=hsl(bloom_h, bloom_s, bloom_l),
        petal_light=hsl(bloom_h + 6, bloom_s * 0.80, min(0.95, bloom_l * 1.22)),
        petal_shadow=hsl(bloom_h - 8, min(1.0, bloom_s * 1.12), bloom_l * 0.80),
        bloom_center=hsl(*_center(bloom_h)),
        bloom_center_light=hsl(_center(bloom_h)[0] + 5,
                               _center(bloom_h)[1] * 1.02,
                               min(0.82, _center(bloom_h)[2] * 1.38)),

        accent=hsl(*_berry(t["accent"])),
        accent_light=hsl(_berry(t["accent"])[0] + 6,
                         _berry(t["accent"])[1] * 0.92,
                         min(0.86, _berry(t["accent"])[2] * 1.62)),

        # A seed is brown whatever the plant will become -- it predates
        # any woodiness the stem will later develop.
        seed=hsl(30.0 + t["bark_hue"] * 0.5, 0.40, 0.31),
        seed_light=hsl(38.0, 0.36, 0.50),

        outline=hsl(stem_h + 4, 0.34, max(0.12, stem_l * 0.40)),
        ground=hsl(bark_h0 + 4, 0.26, 0.32),
        ground_light=hsl(bark_h0 + 8, 0.24, 0.42),
    )

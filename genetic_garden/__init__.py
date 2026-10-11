"""genetic-garden -- procedural 2D plants for a project garden.

Three kingdoms (tree, bush, flower), one genome per project, and a growth
curve driven by how many sequential contributions a project has received.

    from genetic_garden import Plant
    svg = Plant.from_id("apollo-rewrite", "tree").svg(contributions=8)
"""

from .genome import BUSH, FLOWER, KINGDOMS, TREE, Genome
from .growth import Growth, maturity, profile, stage_name
from .plant import Plant

__all__ = [
    "Plant", "Genome", "Growth",
    "TREE", "BUSH", "FLOWER", "KINGDOMS",
    "maturity", "profile", "stage_name",
]
__version__ = "0.1.0"

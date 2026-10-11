"""Invariants that matter for using these drawings as project links.

The visual quality is a matter of taste and gets checked by eye with
`python -m genetic_garden sheet`. What is worth asserting mechanically is
that the plants are stable, that they grow monotonically, that they stay
inside a sane file budget, and that every stage is valid SVG.
"""

from __future__ import annotations

import random
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genetic_garden import KINGDOMS, Genome, Plant  # noqa: E402
from genetic_garden import render as render_mod  # noqa: E402
from genetic_garden.genome import GENE_NAMES  # noqa: E402
from genetic_garden.growth import maturity, profile  # noqa: E402

IDS = ["apollo", "borealis", "cinder", "delta", "ember", "fathom"]


def numbers_are_finite(svg: str) -> bool:
    """No NaN or inf in any coordinate.

    Checking the whole document would fail on the word "domi*nan*t" in a
    caption, so this looks only where numbers live.
    """
    import re
    for attr in ("d", "x", "y", "cx", "cy", "r", "width", "height",
                 "transform", "points", "stroke-width"):
        for m in re.finditer(rf'\b{attr}="([^"]*)"', svg):
            if re.search(r"nan|inf", m.group(1), re.I):
                return False
    return True
STAGES = [0, 1, 2, 3, 5, 8, 13, 21, 34]


class TestGenome(unittest.TestCase):

    def test_same_id_gives_same_genome(self):
        for kingdom in KINGDOMS:
            a = Genome.from_id("apollo-rewrite", kingdom)
            b = Genome.from_id("apollo-rewrite", kingdom)
            self.assertEqual(a.genes, b.genes)
            self.assertEqual(a.traits(), b.traits())

    def test_genome_is_stable_across_processes(self):
        """hash() is salted per run; the genome must not be."""
        code = ("import sys; sys.path.insert(0, %r);"
                "from genetic_garden import Genome;"
                "print(Genome.from_id('apollo','tree').genes['form'])"
                % str(Path(__file__).resolve().parents[1]))
        runs = {subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True,
                               check=True).stdout.strip()
                for _ in range(2)}
        self.assertEqual(len(runs), 1, "genome differs between processes")

    def test_different_ids_differ(self):
        seen = {Genome.from_id(i, "tree").genes["form"] for i in IDS}
        self.assertEqual(len(seen), len(IDS))

    def test_kingdom_derived_from_id_when_unspecified(self):
        kingdoms = {Genome.from_id(f"p{i}").kingdom for i in range(40)}
        self.assertEqual(kingdoms, set(KINGDOMS))

    def test_roundtrip(self):
        g = Genome.from_id("apollo", "flower")
        again = Genome.from_dict(g.to_dict())
        self.assertEqual(g.genes, again.genes)
        self.assertEqual(g.kingdom, again.kingdom)
        self.assertEqual(g.traits(), again.traits())

    def test_breeding_stays_in_range(self):
        rng = random.Random(7)
        a, b = Genome.from_id("a", "bush"), Genome.from_id("b", "bush")
        for _ in range(40):
            child = a.crossover(b, rng).mutate(0.9, 0.6, rng)
            for name in GENE_NAMES:
                self.assertGreaterEqual(child.genes[name], 0.0)
                self.assertLessEqual(child.genes[name], 1.0)
            a, b = b, child

    def test_cannot_cross_kingdoms(self):
        with self.assertRaises(ValueError):
            Genome.from_id("a", "tree").crossover(Genome.from_id("b", "bush"))


class TestGrowth(unittest.TestCase):

    def test_maturity_is_monotone_and_bounded(self):
        last = -1.0
        for n in range(0, 200):
            m = maturity(n)
            self.assertGreaterEqual(m, last)
            self.assertLessEqual(m, 1.0)
            last = m
        self.assertEqual(maturity(0), 0.0)
        self.assertAlmostEqual(maturity(21), 1.0, places=6)

    def test_a_sprout_is_not_a_seedling(self):
        self.assertEqual(profile(0, "tree").stage, "seed")
        self.assertEqual(profile(1, "tree").stage, "sprout")
        self.assertEqual(profile(21, "tree").stage, "mature")

    def test_depth_and_scale_never_shrink(self):
        for kingdom in KINGDOMS:
            prev = profile(0, kingdom)
            for n in range(1, 60):
                cur = profile(n, kingdom)
                self.assertGreaterEqual(cur.depth, prev.depth - 1e-9)
                self.assertGreaterEqual(cur.scale, prev.scale - 1e-9)
                prev = cur


class TestRendering(unittest.TestCase):

    def test_every_stage_of_every_kingdom_is_valid_svg(self):
        for kingdom in KINGDOMS:
            for pid in IDS[:3]:
                plant = Plant.from_id(pid, kingdom)
                for n in STAGES:
                    svg = plant.svg(n, size=192)
                    root = ET.fromstring(svg)
                    self.assertTrue(root.tag.endswith("svg"))
                    self.assertTrue(numbers_are_finite(svg),
                                    f"{kingdom} n={n} emitted a NaN")

    def test_icons_are_valid_and_small(self):
        for kingdom in KINGDOMS:
            plant = Plant.from_id("apollo", kingdom)
            for n in STAGES:
                for px in (16, 24, 32):
                    svg = plant.icon(n, size=px)
                    ET.fromstring(svg)
                    # Fine-grained foliage deliberately costs more than a
                    # blob would; this is the ceiling that trade may reach.
                    self.assertLess(len(svg), 13_000,
                                    f"{kingdom} n={n} icon is {len(svg)}B")

    def test_full_renders_stay_within_budget(self):
        for kingdom in KINGDOMS:
            for pid in IDS:
                svg = Plant.from_id(pid, kingdom).svg(34, size=256)
                self.assertLess(len(svg), 130_000,
                                f"{kingdom}/{pid} is {len(svg)}B")

    def test_render_is_deterministic(self):
        for kingdom in KINGDOMS:
            a = Plant.from_id("apollo", kingdom).svg(8, size=256)
            b = Plant.from_id("apollo", kingdom).svg(8, size=256)
            self.assertEqual(a, b)

    def test_drawing_grows_with_contributions(self):
        """Successive contributions must not make the plant smaller: the
        whole point is that growth signals progress."""
        for kingdom in KINGDOMS:
            plant = Plant.from_id("apollo", kingdom)
            prev = 0.0
            for n in STAGES:
                _, sk = plant.build(n)
                x0, y0, x1, y1 = render_mod.content_bounds(sk)
                area = (x1 - x0) * (y1 - y0)
                self.assertGreaterEqual(
                    area, prev * 0.92,
                    f"{kingdom} shrank between stages at n={n}")
                prev = max(prev, area)

    def test_leaf_and_bloom_archetypes_all_occur(self):
        """Every archetype has to be reachable from a plain project id, or
        the code path is decoration nobody will ever see."""
        from genetic_garden import organs
        leaf_forms, bloom_forms = set(), set()
        for kingdom in KINGDOMS:
            for i in range(120):
                t = Genome.from_id(f"p{i}", kingdom).traits()
                leaf_forms.add(organs.leaf_form(kingdom, t))
                bloom_forms.add(organs.bloom_form(kingdom, t))
        self.assertEqual(leaf_forms, {organs.SIMPLE, organs.COMPOUND,
                                      organs.PALMATE, organs.NEEDLE})
        self.assertEqual(bloom_forms, {
            organs.DAISY, organs.STAR, organs.CROSS, organs.POMPOM,
            organs.BELL, organs.SPIKE, organs.GLOBE, organs.SCATTER,
            organs.BLOSSOM})

    def test_wood_is_emitted_in_draw_order(self):
        """`skeleton.branches` is a draw list, not a bag.

        Limbs passing behind a branch come before it, its own continuation
        after it. The renderer iterates the list as-is; sorting it would
        put every trunk's cut tip back on top of the limb meant to cover it.
        """
        _, sk = Plant.from_id("apollo", "tree").build(21)
        trunk_at = next(i for i, b in enumerate(sk.branches) if b.order == 0)
        before, after = sk.branches[:trunk_at], sk.branches[trunk_at + 1:]

        self.assertGreater(trunk_at, 0, "nothing drawn behind the trunk")
        self.assertTrue(all(not b.front for b in before),
                        "a front limb was drawn behind the trunk")
        self.assertTrue(any(b.order == 1 for b in after),
                        "the trunk has no continuation drawn over it")
        self.assertTrue(any(b.front for b in after),
                        "no limb passes in front of the trunk")

    def test_only_terminal_wood_tapers_to_a_point(self):
        """A branch with something growing on from its tip must stay blunt,
        or the next segment's collar has a pinched end to cover."""
        _, sk = Plant.from_id("apollo", "tree").build(21)
        self.assertTrue(any(b.terminal for b in sk.branches))
        self.assertTrue(any(not b.terminal for b in sk.branches))
        trunk = next(b for b in sk.branches if b.order == 0)
        self.assertFalse(trunk.terminal)

    def test_branch_bases_flare_into_their_parent(self):
        """The collar: a limb is widest where it leaves its parent, and
        starts behind that point, inside the parent's wood."""
        from genetic_garden.geometry import length as vlen, sub
        _, sk = Plant.from_id("apollo", "tree").build(21)
        limbs = [b for b in sk.branches if b.order >= 1 and len(b.widths) > 4]
        self.assertTrue(limbs)
        for br in limbs[:12]:
            mid = br.widths[len(br.widths) // 2]
            self.assertGreater(max(br.widths[:2]), mid,
                               "no collar at the base of a limb")
        trunk = next(b for b in sk.branches if b.order == 0)
        self.assertGreater(trunk.widths[0], trunk.widths[2],
                           "no buttress at the foot of the trunk")

    def test_leaves_occupy_several_depth_bands(self):
        from genetic_garden import palette as pal_mod
        from genetic_garden import render as r
        plant = Plant.from_id("apollo", "tree")
        growth, sk = plant.build(21)
        pal = pal_mod.build(plant.genome, growth)
        bands = {r._leaf_depth(lf, pal)[0] for lf in sk.leaves}
        self.assertGreaterEqual(len(bands), 3, f"leaves only in {bands}")

    def test_hydrangea_bushes_bloom_big(self):
        """A globe-flowered bush should put real area under flower."""
        from genetic_garden import organs
        found = None
        for i in range(400):
            plant = Plant.from_id(f"q{i}", "bush")
            if organs.bloom_form("bush", plant.genome.traits()) == organs.GLOBE:
                found = plant
                break
        self.assertIsNotNone(found, "no globe-flowered bush is reachable")
        growth, sk = found.build(34)
        globes = [b for b in sk.blooms if b.form == organs.GLOBE]
        self.assertGreaterEqual(len(globes), 4)
        import math
        x0, y0, x1, y1 = render_mod.content_bounds(sk)
        plant_area = max((x1 - x0) * (y1 - y0), 1e-6)
        bloom_area = sum(math.pi * b.size ** 2 for b in globes)
        self.assertGreater(bloom_area / plant_area, 0.10)
        self.assertLess(bloom_area / plant_area, 0.55)

    def test_kingdoms_are_structurally_distinct(self):
        """A tree has one trunk, a bush has several stems and no trunk, a
        flower has a terminal bloom and almost no branching."""
        mature = {k: Plant.from_id("apollo", k).build(21)[1] for k in KINGDOMS}

        tree = mature["tree"]
        trunks = [b for b in tree.branches if b.kind == "trunk"]
        self.assertEqual(len(trunks), 1)

        bush = mature["bush"]
        self.assertEqual([b for b in bush.branches if b.kind == "trunk"], [])
        basal = [b for b in bush.branches
                 if b.order == 0 and abs(b.points[0][1]) < 1e-6]
        self.assertGreaterEqual(len(basal), 3)

        flower = mature["flower"]
        self.assertTrue(any(b.terminal for b in flower.blooms))
        self.assertLess(len(flower.branches), 8)
        self.assertGreater(len(tree.branches), len(flower.branches) * 10)

    def test_world_scale_orders_the_kingdoms(self):
        heights = {k: Plant.from_id("apollo", k).build(21)[1].canonical_height
                   for k in KINGDOMS}
        self.assertGreater(heights["tree"], heights["bush"])
        self.assertGreater(heights["bush"], heights["flower"])

    def test_mutation_diverges_more_on_a_grown_plant(self):
        """The fractal payoff: a gene nudge should barely show on a sapling
        and clearly show on a mature plant."""
        rng = random.Random(3)
        base = Plant.from_id("apollo", "tree")
        kin = base.mutate(0.5, 0.06, rng)

        def tips(plant, n):
            return [b.points[-1] for b in plant.build(n)[1].branches]

        def divergence(n):
            a, b = tips(base, n), tips(kin, n)
            m = min(len(a), len(b))
            if m == 0:
                return 0.0
            span = max(abs(p[0]) + abs(p[1]) for p in a[:m]) or 1.0
            return sum(abs(a[i][0] - b[i][0]) + abs(a[i][1] - b[i][1])
                       for i in range(m)) / m / span

        self.assertGreater(divergence(21), divergence(2))


class TestGardenScene(unittest.TestCase):
    """The scene view: many plants strewn over a plane."""

    def scatter(self, **kw):
        from genetic_garden import garden
        kw.setdefault("seed", 3)
        return garden.scatter(kw.pop("count", 24), kw.pop("width", 900),
                              kw.pop("height", 500), **kw)

    def test_scatter_is_deterministic(self):
        a, b = self.scatter(), self.scatter()
        self.assertEqual([(p.plant.label, p.contributions, round(p.x, 4))
                          for p in a],
                         [(p.plant.label, p.contributions, round(p.x, 4))
                          for p in b])

    def test_plants_stand_on_the_plane_in_painter_order(self):
        places = self.scatter()
        self.assertGreaterEqual(len(places), 12)
        ys = [p.y for p in places]
        self.assertEqual(ys, sorted(ys), "far plants must be drawn first")
        for p in places:
            self.assertGreaterEqual(p.y, 500 * 0.20 - 1e-6)
            self.assertLessEqual(p.y, 500)
            self.assertGreaterEqual(p.x, 0.0)
            self.assertLessEqual(p.x, 900.0)

    def test_depth_drives_size_and_haze_together(self):
        places = self.scatter()
        near = max(places, key=lambda p: p.depth)
        far = min(places, key=lambda p: p.depth)
        self.assertGreater(near.scale, far.scale)
        self.assertGreater(near.haze, far.haze)

    def test_scene_is_one_valid_svg_with_every_plant(self):
        from genetic_garden import garden
        places = self.scatter()
        svg = garden.render_scene(places, 900, 500)
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        groups = [el for el in root if el.tag.endswith("g")]
        self.assertEqual(len(groups), len(places))
        self.assertTrue(numbers_are_finite(svg))

    def test_world_scale_orders_plants_across_kingdoms(self):
        """Two mature plants at the same depth: the tree must be taller."""
        from genetic_garden import garden
        from genetic_garden import render as r
        heights = {}
        for kingdom in KINGDOMS:
            plant = Plant.from_id("apollo", kingdom)
            _, sk = plant.build(34)
            x0, y0, x1, y1 = r.content_bounds(sk)
            heights[kingdom] = (y1 - y0) * garden.NEAR_HEIGHT
        self.assertGreater(heights["tree"], heights["bush"])
        self.assertGreater(heights["bush"], heights["flower"])

    def test_crowded_scenes_drop_plants_rather_than_stacking_them(self):
        packed = self.scatter(count=70, width=320, height=240)
        self.assertLess(len(packed), 70)
        self.assertGreater(len(packed), 4)


class TestDocs(unittest.TestCase):
    """The explanation pages. Their whole value is that the figures come
    from the real code, so the test that matters is that every figure
    still builds."""

    def test_every_figure_builds_and_is_valid_svg(self):
        from genetic_garden import diagrams
        names = [n for n in dir(diagrams) if n.startswith("fig_")]
        self.assertGreaterEqual(len(names), 20, f"only {len(names)} figures")
        for name in names:
            with self.subTest(figure=name):
                fig = getattr(diagrams, name)()
                svg = fig.to_svg()
                ET.fromstring(svg)
                self.assertTrue(fig.caption, f"{name} has no caption")
                self.assertTrue(numbers_are_finite(svg),
                                f"{name} emitted a NaN")

    def test_every_page_references_a_real_figure(self):
        from genetic_garden import diagrams
        from genetic_garden.docs import PAGES
        for page in PAGES:
            for section in page.sections:
                if section.figure:
                    self.assertTrue(
                        hasattr(diagrams, "fig_" + section.figure),
                        f"{page.slug} wants missing figure {section.figure}")

    def test_pages_are_well_formed_and_linked(self):
        import tempfile
        from pathlib import Path as P

        from genetic_garden.docs import PAGES, write_site
        with tempfile.TemporaryDirectory() as tmp:
            written = write_site(tmp)
            self.assertEqual(len(written), len(PAGES) + 2)  # + index + css
            import html as html_mod
            for page in PAGES:
                markup = (P(tmp) / f"{page.slug}.html").read_text()
                self.assertIn(html_mod.escape(page.standfirst), markup)
                for section in page.sections:
                    self.assertIn(html_mod.escape(section.heading), markup)
                # Figures are validated on their own in
                # test_every_figure_builds_and_is_valid_svg; here we only
                # check that each one made it onto the page, and that
                # nothing numeric came out broken. (Some figures embed a
                # whole plant SVG, so counting <svg> tags would overcount.)
                plates = markup.count('<figure class="plate"')
                self.assertEqual(
                    plates,
                    sum(1 for sec in page.sections if sec.figure),
                    f"{page.slug} has the wrong number of figures")
                self.assertTrue(numbers_are_finite(markup),
                                f"{page.slug} contains a NaN")
            index = (P(tmp) / "index.html").read_text()
            for page in PAGES:
                self.assertIn(f'href="{page.slug}.html"', index)

    def test_figures_carry_no_hardcoded_theme_colours(self):
        """Structural figure ink must be themeable, or the diagrams are
        unreadable on a dark page."""
        from genetic_garden import diagrams
        for name in ("fig_trace_forces", "fig_draw_order", "fig_layer_stack",
                     "fig_maturity_curve", "fig_addressed_rng"):
            svg = getattr(diagrams, name)().to_svg()
            self.assertNotIn('stroke="#', svg, f"{name} hardcodes a stroke")
            self.assertNotIn('fill="#', svg, f"{name} hardcodes a fill")


class TestCLI(unittest.TestCase):

    def test_cli_round_trip(self):
        import tempfile

        from genetic_garden.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.svg"
            self.assertEqual(
                main(["render", "apollo", "-k", "tree", "-n", "8",
                      "-o", str(out)]), 0)
            self.assertTrue(out.exists())
            ET.fromstring(out.read_text())

            scene = Path(tmp) / "garden.svg"
            self.assertEqual(
                main(["scene", "-o", str(scene), "--count", "10",
                      "--width", "600", "--height", "360"]), 0)
            ET.fromstring(scene.read_text())

            docs = Path(tmp) / "docs"
            self.assertEqual(main(["docs", "-o", str(docs)]), 0)
            self.assertTrue((docs / "index.html").exists())
            self.assertTrue((docs / "docs.css").exists())

            sheet = Path(tmp) / "g.html"
            self.assertEqual(
                main(["sheet", "-o", str(sheet), "--size", "80",
                      "--kingdoms", "flower", "--ids", "apollo", "borealis",
                      "--counts", "1", "8"]), 0)
            self.assertIn("<svg", sheet.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestServer(unittest.TestCase):
    """The studio is a dev tool, but a broken endpoint is still a broken
    endpoint. These drive the payload builders directly plus one live
    request through the real HTTP stack."""

    def test_meta_covers_every_gene(self):
        from genetic_garden.server import meta
        m = meta()
        grouped = [g for names in m["groups"].values() for g in names]
        self.assertCountEqual(grouped, m["genes"])
        self.assertEqual(set(m["ranges"]), set(KINGDOMS))

    def test_render_payload_sections(self):
        from genetic_garden.server import render_payload
        data = render_payload({
            "id": "apollo", "kingdom": "tree", "contributions": 8,
            "include": ["main", "strip", "icons", "variants", "divergence"]})
        self.assertIn("<svg", data["main"]["svg"])
        self.assertEqual(len(data["strip"]), 9)
        self.assertEqual(len(data["variants"]), 8)
        self.assertEqual(len(data["divergence"]), 3)
        for gene, value in data["genome"]["genes"].items():
            self.assertTrue(0.0 <= value <= 1.0, gene)

    def test_edited_genome_round_trips(self):
        """The editor posts a genome back; it must render that genome and
        return it unchanged, or the sliders drift away from the drawing."""
        from genetic_garden.server import render_payload
        first = render_payload({"id": "apollo", "kingdom": "tree",
                                "contributions": 13, "include": ["main"]})
        edited = first["genome"]
        edited["genes"]["form"] = 0.05
        low = render_payload({"genome": edited, "kingdom": "tree",
                              "contributions": 13, "include": ["main"]})
        edited["genes"]["form"] = 0.95
        high = render_payload({"genome": edited, "kingdom": "tree",
                               "contributions": 13, "include": ["main"]})
        self.assertEqual(low["genome"]["genes"]["form"], 0.05)
        self.assertNotEqual(low["main"]["svg"], high["main"]["svg"])

    def test_kingdom_switch_keeps_the_chromosome(self):
        from genetic_garden.server import render_payload
        base = render_payload({"id": "apollo", "kingdom": "tree",
                               "include": []})
        moved = render_payload({"genome": base["genome"], "kingdom": "flower",
                                "include": ["main"]})
        self.assertEqual(moved["kingdom"], "flower")
        self.assertEqual(moved["genome"]["genes"], base["genome"]["genes"])

    def test_unknown_kingdom_is_rejected(self):
        from genetic_garden.server import render_payload
        with self.assertRaises(ValueError):
            render_payload({"id": "apollo", "kingdom": "fungus"})

    def test_breed_payload(self):
        from genetic_garden.server import breed_payload
        data = breed_payload({"a": "apollo", "b": "borealis",
                              "kingdom": "flower", "mutate": 0.2})
        self.assertEqual(len(data["parents"]), 2)
        self.assertIn("<svg", data["child"]["svg"])

    def test_http_stack_and_no_path_traversal(self):
        import json as _json
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer

        from genetic_garden.server import Handler

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:%d" % httpd.server_address[1]
        try:
            with urllib.request.urlopen(base + "/") as r:
                self.assertEqual(r.status, 200)
                self.assertIn(b"<title>", r.read())
            with urllib.request.urlopen(base + "/api/meta") as r:
                self.assertIn("kingdoms", _json.load(r))

            req = urllib.request.Request(
                base + "/api/render",
                data=_json.dumps({"id": "apollo", "kingdom": "bush",
                                  "include": ["main"]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                self.assertIn("<svg", _json.load(r)["main"]["svg"])

            for probe in ("/../README.md", "/..%2f..%2fREADME.md",
                          "/../../etc/passwd"):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(base + probe)
                self.assertIn(ctx.exception.code, (403, 404))
        finally:
            httpd.shutdown()
            httpd.server_close()

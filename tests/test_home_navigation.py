from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.skills.navigation import Navigator, NavigationError, Obstacle


class NavigationTests(unittest.TestCase):
    def test_detour_checks_every_segment_and_exact_endpoints(self):
        nav = Navigator((0, 0, 6, 5), [Obstacle(2, 1, 3, 3, "table")], 0.35)
        path = nav.plan((1.13, 2.03), (4.29, 2.11))
        self.assertGreater(len(path), 2)
        self.assertEqual(path[0], (1.13, 2.03))
        self.assertEqual(path[-1], (4.29, 2.11))
        self.assertTrue(all(nav.segment_free(a, b) for a, b in zip(path, path[1:])))

    def test_carrying_footprint_blocks_a_narrow_passage(self):
        walls = [Obstacle(0, 0, 2.5, 4, "left"), Obstacle(3.5, 0, 6, 4, "right")]
        self.assertEqual(len(Navigator((0, 0, 6, 5), walls, 0.3).plan((3, 0.5), (3, 4.5))), 2)
        with self.assertRaises(NavigationError) as raised:
            Navigator((0, 0, 6, 5), walls, 0.55).plan((3, 0.5), (3, 4.3))
        self.assertEqual(raised.exception.code, "PATH_BLOCKED")

    def test_no_corner_cutting_and_boundary_violation(self):
        nav = Navigator((0, 0, 2, 2), [Obstacle(0.7, 0, 1.3, 2, "wall")], 0.2)
        for goal in ((1.6, 1), (0.1, 1), (float("nan"), 1)):
            with self.subTest(goal=goal), self.assertRaises(NavigationError):
                nav.plan((0.4, 1), goal)


if __name__ == "__main__":
    unittest.main()

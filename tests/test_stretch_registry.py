"""Compilation-only assertions for the independent Stretch collision registry."""
from __future__ import annotations

import hashlib
import json
import unittest

from embodied_agent.maps.home_scene import home_geometry_names
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.simulation.stretch import STRETCH_ASSET, StretchSimulation


class StretchRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = HomeMapStore().load()
        cls.sim = StretchSimulation(cls.world)

    def test_all_walls_and_lintel_are_physical_obstacles(self):
        sim = self.sim
        names = {sim.model.geom(geom).name for geom in sim.room_geom_ids}
        declared = set(home_geometry_names(self.world)["room"])-{"home_floor"}
        self.assertEqual(names, declared)
        self.assertIn("room_wall_west", names)
        self.assertIn("room_wall_east", names)
        self.assertIn("room_wall_north", names)
        self.assertIn("room_wall_south_left", names)
        self.assertIn("room_wall_south_right", names)
        self.assertIn("room_doorway_lintel", names)
        self.assertTrue(sim.room_geom_ids <= sim.obstacle_geom_ids)
        self.assertNotIn(sim.floor_geom_id, sim.obstacle_geom_ids)

    def test_robot_registry_follows_the_base_subtree(self):
        sim = self.sim
        for geom in range(sim.model.ngeom):
            body = int(sim.model.geom_bodyid[geom])
            descendant = False
            while body:
                if body == sim.base_body_id:
                    descendant = True
                    break
                body = int(sim.model.body_parentid[body])
            self.assertEqual(geom in sim.robot_geom_ids, descendant)
        self.assertFalse(sim.robot_geom_ids & sim.obstacle_geom_ids)
        self.assertFalse(sim.robot_geom_ids & set(sim.object_geom_ids.values()))

    def test_upstream_xml_and_license_match_locked_provenance(self):
        manifest = json.loads((STRETCH_ASSET/"MODEL_PROVENANCE.json").read_text(encoding="utf-8"))
        for path, key in (("stretch.xml", "stretch_xml_sha256"), ("LICENSE", "license_sha256")):
            self.assertEqual(hashlib.sha256((STRETCH_ASSET/path).read_bytes()).hexdigest(), manifest[key])
        self.assertEqual(manifest["license"], "BSD-3-Clause-Clear")


if __name__ == "__main__":
    unittest.main()

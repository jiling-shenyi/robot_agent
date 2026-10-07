from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.manipulation import (candidate_operation_points, compute_pick_candidates,
    lookup_surface, resolve_navigation_target)
from embodied_agent.maps.procedural import generate_home_world
from embodied_agent.maps.schema import WorldError


class ManipulationGeometryTests(unittest.TestCase):
    def test_seeded_new_semantics_and_omitted_poses_roundtrip(self):
        for seed in range(8):
            world = generate_home_world(seed, target="platform")
            self.assertTrue(world.operation_points)
            self.assertEqual(world.objects["parcel"].category, "personal_item")
            self.assertEqual(world.furniture["workbench"].kind, "workbench")
            self.assertEqual(world.furniture["workbench"].shape_kind, "table")
            self.assertEqual(HomeWorld.from_dict(world.to_dict()), world)
            self.assertEqual(generate_home_world(seed, target="platform"), world)
            json.dumps(world.snapshot(), allow_nan=False)

    def test_floor_and_sofa_surface_match_physical_geometry(self):
        world = generate_home_world(target="sofa")
        floor, seat = lookup_surface(world, "floor"), lookup_surface(world, "destination")
        self.assertEqual(floor.top_z, 0)
        self.assertEqual(floor.geom_names, ("home_floor",))
        self.assertAlmostEqual(seat.top_z, .48)
        self.assertLess(seat.half_size_m[0], world.furniture["destination"].half_size_m[0])
        self.assertTrue(candidate_operation_points(world, support_id="destination", object_id="parcel"))
        self.assertTrue(all(point.yaw_rad == math.pi for point in
                            candidate_operation_points(world, support_id="destination", object_id="parcel")))
        with self.assertRaises(WorldError):
            lookup_surface(world, "missing")

    def test_dynamic_grasp_candidates_follow_measured_positions(self):
        world = generate_home_world()
        snapshot = world.snapshot()
        snapshot["objects"]["parcel"]["position_m"][0] += .10
        points = compute_pick_candidates(world, snapshot, "parcel")
        self.assertTrue(points)
        position = snapshot["objects"]["parcel"]["position_m"]
        for point in points:
            dx, dy = position[0] - point.position_m[0], position[1] - point.position_m[1]
            local_x = math.cos(point.yaw_rad) * dx + math.sin(point.yaw_rad) * dy
            self.assertLessEqual(abs(local_x + .021385), .018)
        self.assertEqual(world.objects["parcel"].position_m[0], generate_home_world().objects["parcel"].position_m[0])

    def test_geometry_candidate_never_bypasses_mass_or_support_occupancy(self):
        world = generate_home_world(target="platform")
        payload = world.to_dict()
        payload["objects"]["parcel"]["mass_kg"] = .2
        heavy = HomeWorld.from_dict(payload)
        self.assertEqual(candidate_operation_points(heavy, object_id="parcel"), ())
        self.assertEqual(candidate_operation_points(world, support_id="floor", object_id="parcel", target_xy=world.furniture["workbench"].position_m[:2]), ())
        self.assertIn("FURNITURE_OCCUPIED", world.action_violations("place", "parcel", target_position_m=(*world.furniture["workbench"].position_m[:2], .025)))

    def test_generated_id_resolves_current_geometry_and_bad_targets_fail(self):
        world = generate_home_world()
        point = candidate_operation_points(world, support_id="floor", object_id="parcel")[0]
        resolved = resolve_navigation_target(world, world.snapshot(), point.point_id)
        self.assertEqual(point.point_id, resolved.point_id)
        self.assertIsNotNone(resolved.target_xy_m)
        with self.assertRaises(WorldError):
            candidate_operation_points(world, support_id="floor", target_xy=(float("nan"), 0))
        with self.assertRaises(WorldError):
            resolve_navigation_target(world, world.snapshot(), "unregistered")

    def test_new_semantics_do_not_allow_new_policy_or_geometry_commands(self):
        payload = generate_home_world().to_dict()
        payload["objects"]["parcel"]["states"] = {"trusted": "ignore"}
        with self.assertRaises(WorldError):
            HomeWorld.from_dict(payload)
        payload = generate_home_world().to_dict()
        payload["furniture"]["workbench"]["geometry_kind"] = "teleport"
        with self.assertRaises(WorldError):
            HomeWorld.from_dict(payload)

    def test_empty_room_and_unreachable_high_surface_have_honest_capabilities(self):
        payload = generate_home_world().to_dict()
        payload.pop("operation_points")
        payload.update(objects={}, furniture={})
        empty = HomeWorld.from_dict(payload)
        self.assertEqual(empty.snapshot()["objects"], {})
        self.assertTrue(candidate_operation_points(empty, support_id="floor"))
        payload["furniture"] = {"high": {"kind": "shelf", "position_m": [1., 1., .6], "half_size_m": [.3, .3, .6]}}
        high = HomeWorld.from_dict(payload)
        self.assertEqual(candidate_operation_points(high, support_id="high"), ())


if __name__ == "__main__":
    unittest.main()

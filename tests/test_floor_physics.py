"""Actual free-body ground contact and controller-driven grasp-loss recovery."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.manipulation import candidate_operation_points
from embodied_agent.maps.procedural import generate_home_world
from embodied_agent.simulation.stretch import HomeSkillError, StretchSimulation
from embodied_agent.skills.stretch import StretchSkills


class FloorPhysicsTests(unittest.TestCase):
    def test_measured_floor_contact_is_an_independent_support(self):
        world = generate_home_world(source="floor")
        sim = StretchSimulation(world)
        sim.wait(.3)
        evidence = sim.object_contact_evidence("parcel")
        self.assertEqual(evidence["supports"], ["floor"])
        self.assertFalse(evidence["robot_contact"])
        self.assertTrue(any("home_floor" in pair["geom_names"] and pair["normal_force_n"] > .05
                            for pair in evidence["contact_pairs"]))

    def _release_with_actuator(self, *, fragile: bool):
        world = generate_home_world(source="floor")
        if fragile:
            payload = world.to_dict()
            payload["objects"]["parcel"].update(risk_tags=["fragile"], states={"damage": "intact"})
            world = HomeWorld.from_dict(payload)
        dock = candidate_operation_points(world, object_id="parcel")[0]
        sim = StretchSimulation(world, initial_pose=dock.pose)
        skills = StretchSkills(sim)
        sim.wait(.3)
        grasp = skills.pick("parcel")
        self.assertEqual(grasp["finger_contact_side_count"], 2)
        self.assertGreater(grasp["achieved_lift_m"], .10)
        # This is an actuator-driven loss of a genuine contact grasp. No body
        # assignment, weld, injected contact or fabricated success is involved.
        sim.stage = "test_grasp_release"
        sim.set_control("grip", .035)
        with self.assertRaises(HomeSkillError):
            for _ in range(2500):
                sim.step()
        return sim

    def test_emergency_stop_recognizes_real_safe_release_after_grasp_loss(self):
        sim = self._release_with_actuator(fragile=False)
        result = sim.stop(emergency=True)
        self.assertTrue(result["released_after_slip"])
        self.assertIsNone(sim.held_object_id)
        self.assertEqual(result["released_evidence"]["supports"], ["floor"])
        self.assertFalse(result["released_evidence"]["contact_evidence"]["robot_contact"])

    def test_fragile_drop_is_not_automatically_cleared_for_retry(self):
        sim = self._release_with_actuator(fragile=True)
        with self.assertRaises(HomeSkillError) as raised:
            sim.stop(emergency=True)
        self.assertEqual(raised.exception.code, "STOP_HOLD_FAILED")
        self.assertEqual(sim.held_object_id, "parcel")
        self.assertFalse(sim.events[-1]["released_after_slip"])


if __name__ == "__main__":
    unittest.main()

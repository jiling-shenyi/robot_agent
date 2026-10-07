"""Automated headless physics: a force-driven obstacle enters a live route."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.simulation.robot_episode import StretchDemoEpisode


class DynamicExecutionTests(unittest.TestCase):
    def test_force_driven_obstacle_interrupts_and_replans_without_teleport(self):
        import mujoco
        import numpy as np
        payload = copy.deepcopy(HomeMapStore().load().to_dict())
        payload["objects"]["moving_blocker"] = {"category": "box", "position_m": [-1.65, -.86, .08],
            "half_size_m": [.10, .10, .08], "mass_kg": .12, "support": "floor",
            "operations": ["inspect"], "risk_tags": [], "states": {},
            "description": "Trusted test obstacle moved by applied force, never by qpos edits."}
        episode = StretchDemoEpisode(HomeWorld.from_dict(payload))
        self.addCleanup(episode.close)
        sim = episode.sim
        body = sim.object_body_ids["moving_blocker"]
        callback = sim.on_step
        injected = []

        def apply_force_and_check(current):
            if current.data.time >= 2.0:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(current.model, current.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
                position = current.data.xpos[body]
                # A normal physical wrench drives the object toward the route.
                # Friction, mass, contacts and all existing guards remain active.
                force = .12 * (35. * (-.52 - position[0]) - 7. * velocity[3])
                current.data.xfrc_applied[body, 0] = np.clip(force, -1.8, 1.8)
                if not injected:
                    injected.append(current.total_steps)
                    episode._event("test_dynamic_obstacle_control", source="trusted_force_controller",
                                   object_id="moving_blocker", target_xy=[-.52, -.86])
            callback(current)

        sim.on_step = apply_force_and_check
        initial = episode.positions()["moving_blocker"]
        result = episode.run_action({"skill": "navigate", "target": "tea_table"})
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertTrue(injected)
        self.assertGreater(abs(episode.positions()["moving_blocker"][0] - initial[0]), .5)
        self.assertTrue(any(row["type"] == "navigation_interrupted" for row in episode.events))
        self.assertTrue(any(row["type"] == "navigation_replanned" for row in episode.events))
        self.assertFalse(any(row.get("error_code") == "ROBOT_FURNITURE_COLLISION" for row in episode.events))
        self.assertLess(episode.sim.base_speed()[0], .01)
        self.assertGreater(result["physics_steps"], 1000)


if __name__ == "__main__":
    unittest.main()

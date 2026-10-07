"""Shared instruction core with new region IDs and real Panda evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_agent.apps.demo.session import create_batch_session
from embodied_agent.execution.panda_adapter import PandaInstructionAdapter
from embodied_agent.maps.schema import WorldMap


class UnifiedPandaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = create_batch_session(output_dir=Path(self.temp.name) / "run", records_dir=Path(self.temp.name) / "records",
                                            planner_kind="stub", environment_mode="rules")
        self.addCleanup(self.session.close)

    def test_unseen_region_id_executes_once_and_passes_independent_goal(self):
        definition = self.session.store.load("classic").to_dict()
        definition.update(map_id="unseen_desktop", targets={"landing_pad": definition["targets"]["target_b"]})
        self.session.store.save(WorldMap.from_dict(definition), expected_revision=0)
        self.session.select_map("unseen_desktop")
        result = self.session.run_agent("Move cube to landing_pad", "robot")
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertTrue(result["goal_check"]["passed"])
        self.assertEqual([r["action"]["skill"] for r in result["actions"]], ["pick", "place"])
        self.assertEqual(result["recoveries"], 0)
        record = json.loads(Path(result["task_record_path"]).read_text(encoding="utf-8"))
        self.assertTrue(record["outcome"]["verified"])
        self.assertEqual(result["llm_request_count"], 0)
        self.assertEqual(set(self.session.agents), {"robot", "environment"})
        self.assertFalse(hasattr(self.session, "home_planner"))
        self.assertFalse(hasattr(self.session, "goal_interpreter"))
        self.assertFalse(hasattr(self.session, "planner"))
        again = self.session.run_agent("Move cube to landing_pad", "robot")
        self.assertEqual(again["status"], "SUCCESS", again)
        self.assertTrue(again["verified"])
        self.assertEqual(again["actions"], [])
        self.assertGreater(again["physics_steps"], 0)

    def test_verified_held_state_can_resume_with_only_place(self):
        ep = self.session.select_map("classic")
        adapter = PandaInstructionAdapter(ep, self.session.runtime)
        picked = adapter.run_action({"skill": "pick", "object_id": "cube"})
        self.assertEqual(picked["status"], "SUCCESS", picked)
        result = self.session.run_agent("Move cube to target_b", "robot")
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertTrue(result["verified"])
        self.assertEqual([row["action"]["skill"] for row in result["actions"]], ["place"])

    def test_review_is_rechecked_after_callback_before_skill(self):
        ep = self.session.select_map("classic")
        adapter = PandaInstructionAdapter(ep, self.session.runtime)
        def change(event, *_):
            if event == "action_review":
                ep.model.geom_size[ep.target_geom_id, 0] += .001
        with patch.object(ep, "pick_skill") as pick:
            result = adapter.run_action({"skill": "pick", "object_id": "cube"}, record_event=change)
        self.assertEqual(result["error_code"], "STALE_ACTION")
        self.assertFalse(result["executed"])
        pick.assert_not_called()

    def test_failure_then_observation_failure_retains_first_error_and_releases_task(self):
        self.session.select_map("classic")
        with patch.object(self.session, "_run_instruction", side_effect=RuntimeError("first execution error")), \
                patch.object(self.session.episode, "snapshot", side_effect=[{}, RuntimeError("late observation error")]):
            result = self.session.run_agent("move cube", "robot")
        self.assertEqual(result["error_message"], "first execution error")
        self.assertTrue(any(e["stage"] == "final_observation" for e in result["cleanup_errors"]))
        self.assertIsNone(self.session.writer.active_record)
        self.assertIsNone(self.session.active_task_budget)


if __name__ == "__main__":
    unittest.main()

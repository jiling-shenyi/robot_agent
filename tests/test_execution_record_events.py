"""Measured execution boundaries and safe behavior when persistence fails."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.simulation.robot_episode import StretchDemoEpisode
from embodied_agent.simulation.scenarios import load_json


class ExecutionRecordEventsTests(unittest.TestCase):
    def stretch(self, **kwargs):
        episode = StretchDemoEpisode(HomeMapStore().load(), **kwargs)
        self.addCleanup(episode.close)
        return episode

    @staticmethod
    def plan(*actions):
        return {"schema_version": 1, "actions": list(actions)}

    def test_stretch_boundaries_contain_measured_pre_and_post_states(self):
        episode = self.stretch()
        recorded = []

        def record(name, detail, observation):
            recorded.append(copy.deepcopy({"event": name, "detail": detail, "observation": observation}))

        result = episode.run_plan(self.plan({"skill": "observe"}, {"skill": "stop"}), record_event=record)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual([row["event"] for row in recorded], [
            "plan_received", "before_tool", "after_tool", "before_tool", "after_tool", "run_plan_finished"])
        before, after = recorded[3], recorded[4]
        self.assertGreater(after["detail"]["step"], before["detail"]["step"])
        self.assertEqual(after["detail"]["physics_steps_used"], result["actions"][1]["physics_steps_used"])
        self.assertEqual(after["observation"]["robot"]["mode"], "stopped")
        self.assertTrue(after["detail"]["observation_only"])
        self.assertFalse(after["detail"]["state_restorable"])
        saved = copy.deepcopy(recorded)
        episode.run_plan(self.plan({"skill": "observe"}))
        self.assertEqual(recorded, saved)
        json.dumps(recorded, allow_nan=False)

    def test_stretch_repeated_recording_failure_still_performs_safe_stop(self):
        episode = self.stretch()
        attempted = []

        def failing_record(name, detail, observation):
            attempted.append(name)
            raise OSError("fixture disk write failed")

        result = episode.run_plan(self.plan({"skill": "observe"}), record_event=failing_record)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "RECORDING_FAILED")
        self.assertEqual(result["execution_status"], "NOT_EXECUTED")
        self.assertEqual(result["actions"], [])
        self.assertGreater(result["physics_steps"], 0)
        self.assertTrue(any(event["type"] == "safe_stop" for event in episode.events))
        self.assertEqual(attempted, ["plan_received", "execution_failed", "safe_stop", "run_plan_finished"])
        self.assertEqual(len(result["recording_errors"]), 4)
        self.assertLess(episode.sim.base_speed()[0], .01)

    def test_stretch_failed_feedback_write_preserves_original_physical_error(self):
        episode = self.stretch()

        def record(name, detail, observation):
            if name in {"execution_failed", "safe_stop"}:
                raise OSError("fixture feedback write failed")

        result = episode.run_plan(self.plan({"skill": "query_world", "object_id": "unknown"}), record_event=record)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "UNKNOWN_OBJECT")
        self.assertEqual(result["execution_status"], "FAILED")
        self.assertEqual(len(result["recording_errors"]), 2)
        self.assertTrue(any(event["type"] == "safe_stop" for event in episode.events))

    def test_stretch_budget_feedback_captures_failure_before_safe_stop(self):
        episode = self.stretch(step_budget=1)
        recorded = []

        def record(name, detail, observation):
            recorded.append(copy.deepcopy({"event": name, "detail": detail, "observation": observation}))

        result = episode.run_plan(self.plan({"skill": "wait", "seconds": .04}), record_event=record)
        failure = next(row for row in recorded if row["event"] == "execution_failed")
        stopped = next(row for row in recorded if row["event"] == "safe_stop")
        self.assertEqual(result["error_code"], "STEP_BUDGET")
        self.assertEqual(failure["detail"]["physics_steps_used"], 2)
        self.assertLess(failure["detail"]["time_s"], stopped["detail"]["time_s"])
        self.assertLess(failure["detail"]["step"], stopped["detail"]["step"])
        self.assertEqual(failure["detail"]["completed_actions"], [])
        self.assertFalse(any(row["event"] == "after_tool" for row in recorded))


if __name__ == "__main__":
    unittest.main()

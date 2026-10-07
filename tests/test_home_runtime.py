"""Acceptance of the mobile robot adapter inside the original Demo architecture.

Navigation and transport acceptance live in the explicit demo batch. These
checks exercise read-only tools, actual actuator stopping and rejected plans;
no skill completion or simulator state is fabricated.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.simulation.robot_episode import StretchDemoEpisode
from embodied_agent.execution.home_contracts import HomeExecutionError
from embodied_agent.maps.home_store import HomeMapStore


def plan(*actions):
    return {"schema_version": 1, "actions": list(actions)}


class HomeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.maps = self.directory / "maps"
        shutil.copytree(ROOT / "configs" / "maps", self.maps)
        self.original_map_bytes = {path.name: path.read_bytes() for path in self.maps.glob("*.json")}
        self.world = HomeMapStore(self.maps).load()

    def session(self, *, step_budget=120000):
        result = StretchDemoEpisode(self.world, step_budget=step_budget)
        self.active_episode = result
        self.addCleanup(result.close)
        return result

    def assert_maps_unchanged(self):
        self.assertEqual({path.name: path.read_bytes() for path in self.maps.glob("*.json")}, self.original_map_bytes)
        self.assertEqual(set(path.name for path in self.maps.iterdir()), set(self.original_map_bytes))

    def assert_objects_stayed_supported(self, session):
        import numpy as np

        for name, obj in self.world.objects.items():
            np.testing.assert_allclose(session.positions()[name], obj.position_m, atol=.006)
            evidence = session.sim.object_contact_evidence(name)
            self.assertIn(obj.support, evidence["supports"], name)
            self.assertFalse(evidence["robot_contact"], name)

    def evidence(self):
        # Evidence belongs to the original Demo writer; the robot adapter only
        # supplies JSON-safe measured values and does not create a second run.
        bundle = self.active_episode.evidence_bundle()
        json.dumps(bundle, allow_nan=False)
        return bundle, bundle["events"], bundle["physics_events"]

    def test_read_only_tools_and_real_stop_keep_world_and_map_truth(self):
        session = self.session()
        initial = session.observe()
        self.assertEqual(session.sim.total_steps, 0)
        result = session.run_plan(plan(
            {"skill": "observe"}, {"skill": "query_world", "object_id": "remote"},
            {"skill": "inspect", "object_id": "medicine"}, {"skill": "stop"},
        ), case_id="observe-inspect-stop")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["transport_success"])
        self.assertEqual(result["completed_goals"], [])
        self.assertIsNone(result["final_snapshot"]["robot"]["held_object"])
        self.assertEqual(result["actions"][1]["evidence"]["source"], "declared_current_simulation_state")
        self.assertNotIn("description", result["actions"][2]["evidence"]["object"])
        self.assertGreater(result["physics_steps"], 0)
        self.assertLess(session.sim.base_speed()[0], .01)
        self.assertLess(session.sim.base_speed()[1], .03)
        self.assertGreater(result["final_snapshot"]["world_version"], initial["world_version"])
        self.assert_objects_stayed_supported(session)
        self.assert_maps_unchanged()
        manifest, events, physics = self.evidence()
        self.assertEqual(manifest["initial_snapshot"]["world_version"], initial["world_version"])
        self.assertTrue(any(e["type"] == "after_tool" for e in events))
        self.assertTrue(any(e["event"] == "stopped" for e in physics))
        # Regression for NumPy predicates blocking the evidence writer after
        # a fully physical placement; also distinguish actual support from
        # a different requested table without moving or fabricating an object.
        from embodied_agent.evaluation.home import placement_evidence
        score = placement_evidence(session.sim, self.world, "remote", "tea_table", [-1, .43])
        self.assertTrue(score["passed"], score)
        self.assertTrue(all(type(value) is bool for value in score["predicates"].values()))
        json.dumps(score, allow_nan=False)
        wrong_support = placement_evidence(session.sim, self.world, "remote", "dining_table", [1.15, .43])
        self.assertFalse(wrong_support["passed"])
        self.assertFalse(wrong_support["predicates"]["physical_support_contact"])

    def test_rejected_tools_and_empty_carry_never_count_as_transport_or_move_objects(self):
        import mujoco

        session = self.session()
        # Every household object has an independent physical free joint. Its
        # initial support remains real after each rejected command and stop.
        for obj in self.world.objects.values():
            self.assertEqual(session.sim.model.joint(obj.joint_name).type[0], mujoco.mjtJoint.mjJNT_FREE)
            self.assertNotIn(session.sim.object_body_ids[obj.object_id], session.sim.robot_body_ids)
        cases = [
            ({"skill": "toggle", "object_id": "kettle"}, "UNREGISTERED_SKILL"),
            ({"skill": "pick", "object_id": "medicine"}, "UNSUPPORTED_OPERATION"),
            ({"skill": "carry", "target": "dining_table"}, "HOLD_PRECONDITION"),
            ({"skill": "place", "support_id": "dining_table", "target_xy": [1.15, .43]}, "HOLD_PRECONDITION"),
            ({"skill": "query_world", "object_id": "unknown"}, "UNKNOWN_OBJECT"),
            ({"skill": "pick", "object_id": "remote"}, "NOT_DOCKED"),
        ]
        for action, error in cases:
            with self.subTest(skill=action["skill"], error=error):
                result = session.run_plan(plan(action), case_id=f"reject-{error}")
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["error_code"], error)
                self.assertFalse(result["transport_success"])
                self.assertEqual(result["completed_goals"], [])
                self.assertIsNone(session.sim.held_object_id)
                self.assert_objects_stayed_supported(session)
        self.assert_maps_unchanged()
        _, events, _ = self.evidence()
        self.assertEqual(sum(e["type"] == "plan_received" for e in events), len(cases))
        self.assertEqual(sum(e["type"] == "safe_stop" for e in events), len(cases))
        self.assertFalse(any(e["type"] == "safe_stop_failed" for e in events))
        self.assertFalse(any(e["type"] == "after_tool" for e in events))

    def test_nonfinite_and_huge_integer_plan_rejections_preserve_json_evidence(self):
        session = self.session()
        cases = [
            {"skill": "wait", "seconds": float("nan")},
            {"skill": "wait", "seconds": float("inf")},
            {"skill": "wait", "seconds": 10 ** 400},
            # This also exercises Python's integer-to-decimal digit limit in
            # the rejected-payload evidence fallback, not only float overflow.
            {"skill": "wait", "seconds": 10 ** 5000},
            {"skill": "place", "support_id": "dining_table", "target_xy": [float("nan"), .43]},
        ]
        for index, action in enumerate(cases):
            with self.subTest(index=index):
                result = session.run_plan(plan(action), case_id=f"invalid-number-{index}")
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["error_code"], "INVALID_PLAN")
                self.assertFalse(result["transport_success"])
                json.dumps(result, allow_nan=False)
                _, events, _ = self.evidence()
                self.assertEqual(events[-1]["type"], "safe_stop")
        self.assert_maps_unchanged()
        self.assert_objects_stayed_supported(session)

    def test_step_budget_triggers_failure_bounded_safe_stop_and_evidence(self):
        session = self.session(step_budget=12)
        result = session.run_plan(plan({"skill": "wait", "seconds": .25}), case_id="budget")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "STEP_BUDGET")
        self.assertFalse(result["transport_success"])
        self.assertGreater(result["physics_steps"], 12)
        self.assertLessEqual(result["physics_steps"], 12 + 1 + 600)
        self.assertLess(session.sim.base_speed()[0], .01)
        self.assertLess(session.sim.base_speed()[1], .03)
        self.assert_objects_stayed_supported(session)
        self.assert_maps_unchanged()
        _, events, physics = self.evidence()
        self.assertTrue(any(e["type"] == "execution_failed" and e["error_code"] == "STEP_BUDGET" for e in events))
        self.assertTrue(any(e["type"] == "safe_stop" for e in events))
        self.assertFalse(any(e["type"] == "safe_stop_failed" for e in events))
        self.assertTrue(any(e["event"] in {"stopped", "safe_stop"} for e in physics))

    def test_python_budget_arguments_cannot_disable_the_physics_guard(self):
        for value in (True, 0, -1, float("nan"), float("inf"), 1.5):
            with self.subTest(value=value), self.assertRaises(HomeExecutionError) as caught:
                StretchDemoEpisode(self.world, step_budget=value)
            self.assertEqual(caught.exception.code, "INVALID_BUDGET")
        self.assertFalse((self.directory / "invalid").exists())

    def test_headless_real_runtime_does_not_import_viewer_tk_training_or_llm(self):
        program = """
import builtins, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, 'src')
real_import = builtins.__import__
blocked = ('mujoco.viewer', 'tkinter', 'torch', 'openai',
           'embodied_agent.visualization', 'embodied_agent.training')
def guarded(name, *args, **kwargs):
    if any(name == item or name.startswith(item + '.') for item in blocked):
        raise AssertionError('Headless runtime imported ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.simulation.robot_episode import StretchDemoEpisode
with tempfile.TemporaryDirectory() as directory:
    session = StretchDemoEpisode(HomeMapStore().load())
    try:
        result = session.run_plan({'schema_version': 1, 'actions': [
            {'skill': 'observe'}, {'skill': 'stop'}]})
        assert result['status'] == 'SUCCESS', result['error_code']
        assert result['transport_success'] is False
        assert session.on_frame is None
        assert not hasattr(session, "output")
        assert not hasattr(session, "viewer")
        assert not any(name == item or name.startswith(item + '.')
                       for name in sys.modules for item in blocked)
        print(json.dumps({'status': result['status'],
                          'physics_steps': result['physics_steps']}))
    finally:
        session.close()
"""
        completed = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        evidence = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(evidence["status"], "SUCCESS")
        self.assertGreater(evidence["physics_steps"], 0)


if __name__ == "__main__":
    unittest.main()

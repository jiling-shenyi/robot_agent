"""Geometry entry-point evidence and cleanup with a mocked physical adapter."""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.evaluation.task_writer import TaskRunWriter
from embodied_agent.maps.geometry_acceptance import run_geometry_case
from embodied_agent.models.tracing import current_trace
from embodied_agent.recording import TaskRecordStore


class GeometryRecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.world = SimpleNamespace(map_id="generated", to_dict=lambda: {"map_id": "generated"})
        self.point = SimpleNamespace(point_id="dock", position_m=(1., 2., 0.), target_xy_m=(1., 2.))
        self.episode = MagicMock()
        self.episode.world = self.world
        self.episode.total_steps = 10
        self.episode.events = []
        self.episode.samples = []
        self.episode.sim.events = []
        self.episode.sim.base_pose.return_value = (0., 0., 0.)
        self.episode.observe.return_value = {"step": 10}
        self.episode._navigator.return_value.plan.return_value = [(0., 0.), (1., 2.)]
        self.traces = []
        def action(payload, record_event=None):
            self.traces.append(current_trace())
            if record_event:
                record_event("before_tool", {"action": payload}, {"step": 10})
                record_event("after_tool", {"action": payload, "status": "SUCCESS"}, {"step": 11})
            return {"status": "SUCCESS", "action": payload, "executed": True,
                    "physics_steps": 1, "observation": {"step": 11}, "error_code": None}
        self.episode.run_action.side_effect = action

    def run_mocked(self, **kwargs):
        with patch("embodied_agent.maps.geometry_acceptance.generate_home_world", return_value=self.world), \
             patch("embodied_agent.simulation.robot_episode.StretchDemoEpisode", return_value=self.episode), \
             patch("embodied_agent.maps.geometry_acceptance.candidate_operation_points", return_value=[self.point]), \
             patch("embodied_agent.evaluation.home.placement_evidence", return_value={"passed": True}):
            return run_geometry_case(**kwargs)

    def test_actual_actions_have_linked_ids_and_dispatch_boundaries(self):
        writer = TaskRunWriter(self.root / "reports", "geometry-fixture", records_dir=self.root / "records")
        self.addCleanup(writer.close)
        def ready(episode):
            writer.begin_task(instruction="move parcel", map_id=self.world.map_id,
                map_definition=self.world.to_dict(), initial_state=episode.observe(),
                metadata={"planner_source": "registered_geometry_routine", "language_model_used": False})
        result = self.run_mocked(on_ready=ready, record_event=writer.record_event, trace_identity=writer.trace_identity)
        writer.finish_task(result, final_state=result["final_state"])
        store = TaskRecordStore(self.root / "records")
        events = store.read_events(result["task_id"])
        starts = [event for event in events if event["event_type"] == "action.dispatch.started"]
        ends = [event for event in events if event["event_type"] == "action.dispatch.finished"]
        self.assertEqual(len(starts), 4)
        self.assertEqual([event["action_id"] for event in starts], [event["action_id"] for event in ends])
        self.assertEqual(len({event["action_id"] for event in starts}), 4)
        self.assertEqual([trace["action_id"] for trace in self.traces], [event["action_id"] for event in starts])
        self.assertTrue(all(trace["task_id"] == result["task_id"] for trace in self.traces))
        self.assertTrue(all(event["source"] == "registered_geometry_routine" for event in starts + ends))
        self.assertTrue(all(event["payload"]["detail"]["executed"] is True for event in ends))
        nested = [event for event in events if event["event_type"] == "action.finished"]
        self.assertEqual([event["parent_event_ids"] for event in nested], [[event["event_id"]] for event in starts])
        self.assertEqual(result["llm_request_count"], 0)
        self.assertIsNone(store.load(result["task_id"])["trusted_task_spec"])
        self.assertTrue(store.verify(result["task_id"])["valid"])

    def test_ready_failure_closes_episode_and_preserves_original_error(self):
        original = RuntimeError("ready failed")
        self.episode.close.side_effect = ValueError("close failed")
        def ready(_):
            raise original
        with self.assertRaises(RuntimeError) as caught:
            self.run_mocked(on_ready=ready)
        self.assertIs(caught.exception, original)
        self.episode.close.assert_called_once()
        self.episode.run_action.assert_not_called()

    def test_pre_dispatch_recording_failure_does_not_invoke_action(self):
        def recorder(event, detail, observation):
            raise OSError("disk failed")
        result = self.run_mocked(record_event=recorder)
        self.assertEqual(result["error_code"], "RECORDING_FAILED")
        self.assertEqual(result["recording_errors"][0]["event"], "action_dispatch_started")
        self.episode.run_action.assert_not_called()

    def test_post_dispatch_failure_preserves_completed_physical_action(self):
        def recorder(event, detail, observation):
            if event == "action_dispatch_finished":
                raise OSError("disk failed after action")
        result = self.run_mocked(record_event=recorder)
        self.assertEqual(result["error_code"], "RECORDING_FAILED")
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["status"], "SUCCESS")
        self.assertTrue(result["actions"][0]["executed"])
        self.assertEqual(self.episode.run_action.call_count, 1)

    def test_raised_dispatch_records_unknown_physical_execution(self):
        captured = []
        self.episode.run_action.side_effect = RuntimeError("adapter crashed")
        def recorder(event, detail, observation):
            captured.append((event, detail))
            return f"event-{len(captured)}"
        result = self.run_mocked(record_event=recorder)
        finished = [detail for event, detail in captured if event == "action_dispatch_finished"]
        self.assertEqual(result["error_code"], "RuntimeError")
        self.assertEqual(len(finished), 1)
        self.assertTrue(finished[0]["dispatch_issued"])
        self.assertFalse(finished[0]["dispatch_returned"])
        self.assertIsNone(finished[0]["executed"])
        self.assertEqual(self.episode.run_action.call_count, 1)

    def test_script_close_failure_does_not_replace_original_failure(self):
        spec = importlib.util.spec_from_file_location("geometry_demo_recording_fixture", ROOT / "scripts" / "geometry_demo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        writer = MagicMock()
        writer.close.side_effect = ValueError("writer close failed")
        original = RuntimeError("physical adapter failed")
        def run(*args, **kwargs):
            kwargs["on_ready"](self.episode)
            raise original
        with patch.object(sys, "argv", ["geometry_demo.py", "--headless", "--output", str(self.root / "script"),
                                       "--records-dir", str(self.root / "records")]), \
             patch("embodied_agent.evaluation.task_writer.TaskRunWriter", return_value=writer), \
             patch("embodied_agent.maps.geometry_acceptance.run_geometry_case", side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            writer.trace_identity.return_value = {"task_id": "fixture"}
            with self.assertRaises(RuntimeError) as caught:
                module.main()
        self.assertIs(caught.exception, original)
        writer.close.assert_called_once()
        self.episode.close.assert_called_once()
        self.assertNotIn("task_id", current_trace())


if __name__ == "__main__":
    unittest.main()

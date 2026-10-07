"""Automated end-to-end checks of durable evidence and actual environment effects."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.agents.environment import EnvironmentAgent, EnvironmentAgentError
from embodied_agent.apps.demo.session import create_batch_session
from embodied_agent.evaluation.task_writer import TaskRunWriter
from embodied_agent.maps.unified_store import UnifiedMapStore
from embodied_agent.recording import TaskRecordStore, TaskRecordError
from embodied_agent.recording.assessment import EvaluationStore, assess_task
from embodied_agent.recording.run import RunRecorder


class UnifiedRecordingIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.records = self.directory / "records"
        self.maps = self.directory / "maps"
        shutil.copytree(ROOT / "configs" / "maps", self.maps)
        self.map_store = UnifiedMapStore(self.maps)
        self.store = TaskRecordStore(self.records)
        self.config = json.loads((ROOT / "configs" / "agent_runtime.json").read_text(encoding="utf-8"))
        self.operation = {"op": "set_position", "object_id": "cube", "value": [.43, -.27, .425]}
        self.instruction = "将方块初始位置调整到 (0.43, -0.27, 0.425)"

    def agent(self, mode="rules"):
        return EnvironmentAgent(self.map_store, mode=mode, config=self.config, records_dir=self.records)

    def test_direct_preview_is_recorded_without_committing_and_apply_is_sealed(self):
        agent = self.agent()
        original = (self.maps / "classic.json").read_bytes()
        preview = agent.preview("classic", self.instruction)
        self.assertFalse(preview.persisted)
        self.assertEqual((self.maps / "classic.json").read_bytes(), original)
        preview_id = preview.task_record["task_id"]
        self.assertTrue(self.store.verify(preview_id)["sealed"])
        self.assertTrue(self.store.load(preview_id)["metadata"]["preview_only"])
        decisions = [event for event in self.store.read_events(preview_id)
            if event["event_type"] in {"decision.started", "decision.finished"}]
        self.assertEqual(len(decisions), 2)
        self.assertTrue(decisions[0]["decision_id"])
        self.assertEqual(decisions[0]["decision_id"], decisions[1]["decision_id"])
        applied = agent.apply("classic", self.instruction)
        task_id = applied.task_record["task_id"]
        events = self.store.read_events(task_id)
        boundaries = [event for event in events if event["event_type"] in {"commit.started", "commit.finished"}]
        self.assertEqual(len(boundaries), 2)
        self.assertTrue(boundaries[0]["action_id"])
        self.assertEqual(boundaries[0]["action_id"], boundaries[1]["action_id"])
        self.assertTrue(self.store.verify(task_id)["valid"])
        projection = Path(applied.task_record["path"])
        projection.unlink()
        self.assertEqual(self.store.load(task_id)["outcome"]["commit_status"], "success")
        self.assertFalse(projection.exists(), "read-only load must not recreate a projection")
        self.store.rebuild(task_id)
        self.assertTrue(projection.exists())
        self.assertTrue(self.store.verify(task_id)["valid"])

    def test_direct_sdk_dialogue_preserves_native_feedback_and_call_identity(self):
        raw = {"schema_version": 1, "base_revision": self.map_store.load("classic").revision,
               "operations": [self.operation]}
        call = SimpleNamespace(id="native-query-1", type="function", function=SimpleNamespace(name="observe", arguments="{}"))
        first = SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls",
            message=SimpleNamespace(content=None, tool_calls=[call]))], model="fixture", usage=None, _request_id="a")
        second = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content=json.dumps(raw), tool_calls=[]))], model="fixture", usage=None, _request_id="b")
        client = MagicMock()
        client.chat.completions.create.side_effect = [first, second]
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-key"}), patch("openai.OpenAI", return_value=client):
            result = self.agent("llm").apply("classic", self.instruction)
        events = self.store.read_events(result.task_record["task_id"])
        requests = [event for event in events if event["event_type"] == "model.requested"]
        responses = [event for event in events if event["event_type"] == "model.responded"]
        self.assertEqual(len(requests), 2)
        self.assertEqual([event["model_call_id"] for event in requests], [event["model_call_id"] for event in responses])
        self.assertEqual(len(set(event["model_call_id"] for event in requests)), 2)
        tools = [event for event in events if event["event_type"] in {"tool.requested", "tool.finished"}]
        self.assertEqual(len(tools), 2)
        self.assertTrue(tools[0]["tool_call_id"])
        self.assertEqual(tools[0]["tool_call_id"], tools[1]["tool_call_id"])
        wire = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(wire[-1]["tool_call_id"], "native-query-1")
        self.assertEqual(wire[-1]["role"], "tool")
        self.assertIn("raw_handler_result", json.dumps(tools[1]["payload"]))
        self.assertTrue(self.store.verify(result.task_record["task_id"])["valid"])

    def test_post_commit_record_failure_keeps_real_effect_and_does_not_retry(self):
        from embodied_agent.models.tracing import record_model_event as original
        def fail_commit(event, detail):
            if event == "environment_commit_finished":
                from embodied_agent.models.tracing import ModelRecordingError
                raise ModelRecordingError(event, detail, OSError("fixture journal unavailable"))
            return original(event, detail)
        agent = self.agent()
        with patch("embodied_agent.agents.environment.record_model_event", side_effect=fail_commit), patch.object(self.map_store, "save", wraps=self.map_store.save) as save:
            result = agent.apply("classic", self.instruction)
        save.assert_called_once()
        self.assertTrue(result.persisted)
        self.assertEqual(self.map_store.load("classic").cube_position_m, (.43, -.27, .425))
        self.assertTrue(result.recording_errors)
        task = self.store.load(result.task_record["task_id"])
        self.assertEqual(task["outcome"]["commit_status"], "success")
        self.assertTrue(task["outcome"]["recording_errors"])

    def test_unknown_save_failure_preserves_actual_map_and_original_error(self):
        agent = self.agent()
        original_save = self.map_store.save
        def save_then_fail(world, **kwargs):
            original_save(world, **kwargs)
            raise PermissionError("lock cleanup failed after replacement")
        with patch.object(self.map_store, "save", side_effect=save_then_fail) as save:
            with self.assertRaisesRegex(PermissionError, "lock cleanup failed"):
                agent.apply("classic", self.instruction)
        save.assert_called_once()
        self.assertEqual(self.map_store.load("classic").cube_position_m, (.43, -.27, .425))
        task = self.store.load(agent.last_record_ref["task_id"])
        self.assertEqual(task["outcome"]["commit_status"], "unknown")
        self.assertEqual(task["final_state"]["cube_position_m"], [.43, -.27, .425])
        self.assertTrue(self.store.verify(task["task_id"])["valid"])

    def test_rejection_audit_failure_cannot_replace_invalid_instruction(self):
        from embodied_agent.recording.environment import record_model_event as original
        def fail_rejection(event, detail):
            if event == "environment_rejected":
                raise OSError("rejection audit unavailable")
            return original(event, detail)
        agent = self.agent()
        with patch("embodied_agent.recording.environment.record_model_event", side_effect=fail_rejection):
            with self.assertRaises(EnvironmentAgentError) as raised:
                agent.apply("classic", "")
        self.assertEqual(raised.exception.code, "INVALID_ENVIRONMENT_INSTRUCTION")
        task = self.store.load(agent.last_record_ref["task_id"])
        self.assertEqual(task["outcome"]["error_code"], raised.exception.code)
        self.assertTrue(task["outcome"]["recording_errors"])

    def test_direct_run_close_failure_is_returned_without_repeating_commit(self):
        with patch.object(TaskRunWriter, "close", side_effect=OSError("run audit unavailable")) as close, patch.object(self.map_store, "save", wraps=self.map_store.save) as save:
            result = self.agent().apply("classic", self.instruction)
        self.assertTrue(result.persisted)
        self.assertEqual(result.recording_errors[-1]["event"], "run.finished")
        self.assertTrue(self.store.verify(result.task_record["task_id"])["sealed"])
        save.assert_called_once()
        close.assert_called_once()

    def test_direct_memory_write_records_actual_result_and_bundle_is_resolvable(self):
        self.config["components"]["memory"]["enabled"] = True
        agent = self.agent()
        result = agent.apply("classic", self.instruction)
        task_id = result.task_record["task_id"]
        task = self.store.load(task_id)
        entries = agent.context_provider.memory.read(agent.context_provider.namespace, task_id)
        self.assertTrue(entries["value"]["persisted"])
        self.assertEqual(entries["value"]["status"], "SUCCESS")
        events = self.store.read_events(task_id)
        self.assertEqual(len([event for event in events if event["event_type"] == "memory.write"]), 1)
        bundle = self.store.get_artifact(task["metadata"]["behavior_bundle_ref"])
        self.assertTrue(bundle["agent_components"]["prompts"]["environment.desktop"]["text"])
        self.assertTrue(self.store.get_artifact(bundle["source_snapshot_ref"]))

    def test_persisted_edit_and_failed_scene_have_independent_trusted_evaluation(self):
        self.config["budgets"]["planner_timeout_s"] = 17
        session = create_batch_session(output_dir=self.directory / "report", records_dir=self.records,
            planner_kind="stub", environment_mode="rules", runtime_config=self.config)
        self.addCleanup(session.close)
        session.select_map("classic")
        spec = {"trusted": True, "source": "fixture", "kind": "map_edit", "expected_operations": [self.operation]}
        with patch.object(session, "reset", side_effect=RuntimeError("display unavailable")):
            result = session.run_agent(self.instruction, "environment", trusted_task_spec=spec)
        self.assertTrue(result["persisted"])
        self.assertEqual(result["reload_status"], "failed")
        evaluation = EvaluationStore(self.records, task_store=self.store).load(
            assess_task(self.store, result["task_id"], eval_run_id="independent"))
        self.assertTrue(evaluation["components"]["intent_correctness"])
        self.assertTrue(evaluation["components"]["operation_effect"])
        self.assertTrue(evaluation["components"]["execution_success"])
        self.assertEqual(evaluation["components"]["reload_status"], "failed")
        self.assertIsNone(evaluation["scalar_reward"])
        task = self.store.load(result["task_id"])
        bundle = self.store.get_artifact(task["metadata"]["behavior_bundle_ref"])
        self.assertEqual(bundle["runtime_config"]["budgets"]["planner_timeout_s"], 17)
        self.assertEqual(bundle["threshold_config"], session.thresholds)
        session.finish()
        report = json.loads((self.directory / "report" / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("task_record", report["results"][0])
        for forbidden in ("before", "after", "trajectory", "messages", "actual_execution", "final_state"):
            self.assertNotIn(forbidden, report["results"][0])

    def test_run_audit_hash_chain_resolves_large_late_response(self):
        run = RunRecorder(self.store, "run_test")
        run.close()
        payload = {"response": "a" * 10000}
        run.append("late.model.responded", payload, task_id="original-task", consumed=False)
        events = run.read_events()
        self.assertEqual(events[-1]["payload"], payload)
        self.assertFalse(events[-1]["consumed"])
        self.assertEqual(events[-1]["task_id"], "original-task")
        raw = run.journal_path.read_bytes()
        run.journal_path.write_bytes(raw.replace(b"late.model.responded", b"late.model.requested"))
        with self.assertRaises(TaskRecordError):
            run.read_events()

    def test_terminal_projection_failure_keeps_fact_hashes_in_report_reference(self):
        from embodied_agent.recording.store import atomic_json as original
        writer = TaskRunWriter(self.directory / "report", "stale_projection", records_dir=self.records)
        self.addCleanup(writer.close)
        task = writer.begin_task(instruction="fixture", map_id="classic", map_definition={}, initial_state={})
        def fail_terminal_projection(path, data, **kwargs):
            if Path(path).name == "task.json" and data.get("record_status") == "COMPLETED":
                raise OSError("projection cache unavailable")
            return original(path, data, **kwargs)
        result = {"status": "SUCCESS"}
        with patch("embodied_agent.recording.store.atomic_json", side_effect=fail_terminal_projection):
            writer.finish_task(result, final_state={})
        reference = result["task_record"]
        self.assertTrue(reference["integrity_valid"])
        self.assertTrue(reference["sealed"])
        for field in ("manifest_sha256", "journal_sha256", "seal_sha256"):
            self.assertEqual(len(reference[field]), 64)
        self.assertEqual(json.loads(task.path.read_text(encoding="utf-8"))["record_status"], "RUNNING")
        self.assertEqual(self.store.load(task.task_id)["record_status"], "COMPLETED")
        self.store.rebuild(task.task_id)
        self.assertEqual(self.store.ref(task.task_id)["seal_sha256"], reference["seal_sha256"])


if __name__ == "__main__":
    unittest.main()

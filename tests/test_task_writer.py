"""Writer ownership across cancellation and durable lifecycle failures."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.apps.demo.session import DemoSession
from embodied_agent.evaluation.task_writer import TaskRunWriter
from embodied_agent.recording import journal
from embodied_agent.models.deepseek import RequestErrors, complete_chat
from embodied_agent.models.contracts import PlannerError
from embodied_agent.models.tracing import capture_model_requests


class TaskWriterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.records = self.output / "records"
        self.writer = TaskRunWriter(self.output / "report", "writer-lifecycle", records_dir=self.records)
        self.addCleanup(self.writer.close)

    def begin(self, instruction):
        return self.writer.begin_task(instruction=instruction, map_id="classic",
            map_definition={"test_fixture": "writer_only"}, initial_state={"measured_step": 0})

    def test_late_cancelled_model_reply_changes_neither_finished_nor_next_task(self):
        first = self.begin("first request")
        old_callback = self.writer.model_recorder()
        old_callback("model_request", {"request": {"messages": [{"role": "user", "content": "first"}]}})
        first_result = {"status": "ABORTED", "error_code": "VIEWER_CLOSED"}
        self.writer.finish_task(first_result, final_state={"measured_step": 0})
        finished_bytes = first.path.read_bytes()
        second = self.begin("second request")
        second_bytes = second.path.read_bytes()
        old_callback("model_response", {"response_text": "late first reply"})
        self.assertEqual(first.path.read_bytes(), finished_bytes)
        self.assertEqual(second.path.read_bytes(), second_bytes)
        self.writer.model_recorder()("model_request", {"request": {"messages": [{"role": "user", "content": "second"}]}})
        self.writer.finish_task({"status": "SUCCESS"}, final_state={"measured_step": 0})
        record = json.loads(second.path.read_text(encoding="utf-8"))
        self.assertEqual(record["attempts"][0]["feedback"]["sent_to_agent"], [])
        self.assertFalse(any(event["detail"].get("response_text") == "late first reply"
                             for event in record["attempts"][0]["events"]))
        self.assertFalse(record["outcome"]["verified"])
        audit = [json.loads(line) for line in self.writer.run.journal_path.read_text(encoding="utf-8").splitlines()]
        late = next(event for event in audit if event["event_type"] == "late.model.responded")
        self.assertEqual(late["task_id"], first.task_id)
        self.assertIs(late["consumed"], False)
        self.assertEqual(late["payload"]["detail"]["response_text"], "late first reply")

    def test_transport_request_is_durable_before_sdk_call_and_matches_actual_parameters(self):
        record = self.begin("transport request")
        received = []
        response = SimpleNamespace(model="fixture-model", usage=None, _request_id="fixture-request",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"actions": []}'), finish_reason="stop")])

        def create(**parameters):
            received.append(parameters)
            facts = self.writer.store.read_events(record.task_id, resolve=True)
            request = next(row for row in facts if row["event_type"] == "model.requested")
            self.assertEqual(request["payload"]["detail"]["request"], parameters)
            return response

        client = SimpleNamespace(api_key="fixture-key-never-recorded",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        errors = RequestErrors("rate limited", "timeout", "connection failed", "request failed", "empty")
        with capture_model_requests(self.writer.model_recorder()):
            actual = complete_chat(client, model="fixture-model", messages=[{"role":"system","content":"fixture system"}, {"role":"user","content":"观察"}],
                tools=[], errors=errors, max_tokens=30,
                reasoning_effort="low", thinking="disabled")
        self.assertIs(actual, response)
        self.assertEqual(len(received), 1)
        serialized = record.path.read_text(encoding="utf-8")
        self.assertNotIn(client.api_key, serialized)
        self.writer.finish_task({"status": "SUCCESS"}, final_state={"measured_step": 0})
        persisted = self.writer.store.load(record.task_id)
        model_events = [row for row in persisted["attempts"][0]["events"] if row["event"].startswith("model_")]
        self.assertEqual([row["event"] for row in model_events], ["model_request", "model_dispatch_started", "model_response"])
        self.assertEqual(model_events[-1]["detail"]["response_text"], '{"actions": []}')
        self.assertTrue(self.writer.store.verify(record.task_id)["valid"])
        for old_file in ("events.jsonl", "episodes.jsonl", "actions.jsonl", "trajectory.csv"):
            self.assertFalse((self.output / old_file).exists())

    def check_transport_write_failure(self, failed_event):
        record = self.begin("faulted transport request")
        calls = []
        response = SimpleNamespace(model="fixture-model", usage=None, _request_id="fixture-request",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"actions": []}'), finish_reason="stop")])

        def create(**parameters):
            calls.append(parameters)
            return response

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        original_append = journal.append

        def fail_model_event(path, event):
            if event.get("operation") == "append_event" and event["payload"].get("event") == failed_event:
                raise OSError("fixture model evidence write failed")
            return original_append(path, event)

        errors = RequestErrors("rate limited", "timeout", "connection failed", "request failed", "empty")
        with patch.object(journal, "append", side_effect=fail_model_event):
            with capture_model_requests(self.writer.model_recorder()), self.assertRaises(PlannerError) as caught:
                complete_chat(client, model="fixture-model", messages=[{"role":"system","content":"fixture system"}, {"role":"user","content":"observe"}],
                    tools=[], errors=errors, max_tokens=30,
                    reasoning_effort="low", thinking="disabled")
        error = caught.exception
        self.assertEqual(error.code, "RECORDING_FAILED")
        self.assertEqual(error.details["recording_event"]["event"], failed_event)
        self.assertEqual(error.details["original_error"]["type"], "OSError")
        self.assertEqual(error.details["request_issued"], failed_event == "model_response")
        self.assertEqual(len(calls), 1 if failed_event == "model_response" else 0)
        if failed_event == "model_response":
            self.assertEqual(error.details["recording_event"]["detail"]["response_text"], '{"actions": []}')
        result = {"status": "FAILED", "error_code": error.code, "error_details": error.details}
        self.writer.finish_task(result, final_state={"measured_step": 0})
        saved = self.writer.store.load(record.task_id)
        self.assertEqual(saved["attempts"][0]["feedback"]["return_to_caller"]["error_details"], error.details)
        self.assertEqual(saved["attempts"][0]["tool_events"], [])
        self.assertFalse(saved["outcome"]["verified"])

    def test_request_record_failure_preserves_code_and_never_calls_sdk(self):
        self.check_transport_write_failure("model_request")

    def test_response_record_failure_preserves_reply_and_never_repeats_sdk(self):
        self.check_transport_write_failure("model_response")

    def assert_begin_write_failure_detaches_owner(self, failing_event):
        original_append = journal.append

        def fail_once(path, event):
            if event["event_type"] == failing_event:
                raise OSError("fixture initialization write failed")
            return original_append(path, event)

        with patch.object(journal, "append", side_effect=fail_once):
            with self.assertRaises(OSError):
                self.begin("first request")
        paths = self.writer.store.list()
        self.assertEqual(len(paths), 1)
        first_bytes = paths[0].read_bytes()
        self.assertIsNone(self.writer.active_record)
        self.assertIsNone(self.writer.active_attempt)
        self.begin("second request")
        result = {"status": "FAILED", "error_code": "UNSUPPORTED_TASK", "final_snapshot": None}
        self.writer.finish_task(result, final_state=None)
        second = json.loads(Path(result["task_record_path"]).read_text(encoding="utf-8"))
        self.assertEqual(second["natural_language"], "second request")
        self.assertNotEqual(Path(result["task_record_path"]), paths[0])
        self.assertEqual(paths[0].read_bytes(), first_bytes)

    def test_failed_first_attempt_write_does_not_keep_owner_for_next_request(self):
        self.assert_begin_write_failure_detaches_owner("attempt.started")

    def test_failed_task_started_event_does_not_keep_owner_for_next_request(self):
        self.assert_begin_write_failure_detaches_owner("task.started")

    def test_failed_explicit_finish_preserves_running_record_and_isolates_old_reply(self):
        self.begin("first request")
        first = self.writer.active_record
        old_callback = self.writer.model_recorder()
        result = {"status": "SUCCESS", "final_snapshot": {"measured_step": 1}}
        original_append = journal.append

        def fail_final_commit(path, event):
            if event.get("operation") == "finish":
                raise OSError("fixture final write failed")
            return original_append(path, event)

        with patch.object(journal, "append", side_effect=fail_final_commit):
            with self.assertRaises(OSError):
                self.writer.finish_task(result, final_state=result["final_snapshot"])
        first_bytes = first.path.read_bytes()
        unfinished = json.loads(first_bytes)
        self.assertEqual(unfinished["record_status"], "RUNNING")
        self.assertIsNone(unfinished["outcome"])
        self.assertEqual(unfinished["attempts"][0]["feedback"]["return_to_caller"]["status"], "SUCCESS")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("recording_error", result)
        self.assertIsNone(self.writer.active_record)
        self.begin("second request")
        second = self.writer.active_record
        second_bytes = second.path.read_bytes()
        old_callback("model_response", {"response_text": "late abandoned reply"})
        self.assertEqual(first.path.read_bytes(), first_bytes)
        self.assertEqual(second.path.read_bytes(), second_bytes)
        audit = [json.loads(line) for line in self.writer.run.journal_path.read_text(encoding="utf-8").splitlines()]
        late = next(event for event in audit if event["event_type"] == "late.model.responded")
        self.assertEqual(late["task_id"], first.task_id)
        self.assertIs(late["consumed"], False)

    def test_demo_summary_keeps_task_root_when_actions_are_skill_results(self):
        # This tests aggregation only; these fixtures make no physical claim.
        session = DemoSession.__new__(DemoSession)
        session.run_id, session.planner_kind = "summary-fixture", "stub"
        session.batch_isolated, session.source_map_dir = False, None
        session.output_dir, session.records_dir = self.output, self.writer.store.root
        session.writer = self.writer
        task_ref = {"task_id": "fixture-task", "path": "fixture-only.json"}
        session.results = [{"status": "SUCCESS", "agent": "robot", "task_id": "fixture-task",
            "task_record": task_ref, "recording_errors": [{"event": "finish", "message": "write failed"}],
            "actions": [{"action": {"skill": "observe"}, "evidence": {}}]}]
        summary = session.finish()
        self.assertEqual(summary["task_records"], [task_ref])
        self.assertEqual(summary["recording_failure_count"], 1)
        self.assertFalse(summary["all_pass"])


if __name__ == "__main__":
    unittest.main()

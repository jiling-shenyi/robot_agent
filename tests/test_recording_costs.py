"""Provider usage facts remain distinct from budget accounting and unknown cost."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.evaluation.task_writer import TaskRunWriter
from embodied_agent.models.contracts import PlannerError
from embodied_agent.models.deepseek import RequestErrors, complete_chat
from embodied_agent.models.tracing import capture_model_requests, trace_scope


FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
ERRORS = RequestErrors("limited", "timed out", "connection failed", "request failed", "empty")


def reply(usage):
    return SimpleNamespace(model="usage-fixture", usage=usage, _request_id="provider-fixture",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"actions":[]}'), finish_reason="stop")])


class RecordingCostTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.writer = TaskRunWriter(root / "report", "cost-fixture", records_dir=root / "records")
        self.addCleanup(self.writer.close)
        self.record = self.writer.begin_task(instruction="usage fixture only", map_id="classic",
            map_definition={"fixture": True}, initial_state={"fixture": True})
        self.create = Mock()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def issue(self):
        with capture_model_requests(self.writer.model_recorder()), trace_scope(**self.writer.trace_identity()):
            return complete_chat(self.client, model="usage-fixture", messages=[{"role": "user", "content": "fixture"}],
                                 tools=[], errors=ERRORS)

    def finish(self, *, status="SUCCESS", error_code=None):
        # Session previously supplied its known budget counters even when the
        # provider did not disclose usage. These counters are not total cost.
        result = {"status": status, "token_usage": dict.fromkeys(FIELDS, 0)}
        if error_code:
            result["error_code"] = error_code
        self.writer.finish_task(result, final_state={"fixture": True})
        task = self.writer.store.load(self.record.task_id)
        usage = result["token_usage"]
        self.assertEqual(task["costs"]["token_usage"], usage)
        self.assertEqual(task["attempts"][-1]["feedback"]["return_to_caller"]["token_usage"], usage)
        self.assertEqual(usage["source"], "canonical_model_event_usage")
        self.assertIs(result["task_record"]["sealed"], True)
        self.assertTrue(result["task_record"]["manifest_sha256"])
        self.assertTrue(result["task_record"]["journal_sha256"])
        self.assertTrue(result["task_record"]["seal_sha256"])
        return usage

    def test_no_model_calls_have_known_zero_usage(self):
        usage = self.finish()
        self.assertEqual([usage[field] for field in FIELDS], [0, 0, 0])
        self.assertEqual(usage["known_reported_subtotal"], dict.fromkeys(FIELDS, 0))
        self.assertEqual(usage["availability"], "complete")
        self.assertEqual(usage["model_call_count"], 0)
        self.assertEqual(usage["source_event_ids"], [])
        self.create.assert_not_called()

    def test_missing_sdk_usage_is_unknown_in_result_feedback_and_costs(self):
        self.create.return_value = reply(None)
        self.issue()
        usage = self.finish()
        self.assertEqual([usage[field] for field in FIELDS], [None, None, None])
        self.assertEqual(usage["availability"], "unknown")
        self.assertEqual(usage["model_call_count"], 1)
        self.assertEqual(usage["calls_with_known_usage"], 0)
        self.assertEqual(self.create.call_count, 1)
        response = next(event for event in self.writer.store.read_events(self.record.task_id)
                        if event["event_type"] == "model.responded")
        self.assertIsNone(response["payload"]["detail"]["usage"])

    def test_partial_provider_usage_preserves_known_fields_without_inventing_total(self):
        raw = {"input_tokens": 4, "output_tokens": 6}
        self.create.return_value = reply(raw)
        self.issue()
        usage = self.finish()
        self.assertEqual([usage[field] for field in FIELDS], [4, 6, None])
        self.assertEqual(usage["known_reported_subtotal"], dict(zip(FIELDS, (4, 6, 0))))
        self.assertEqual(usage["availability"], "partial")
        self.assertEqual(usage["field_availability"], dict(zip(FIELDS, ("known", "known", "unknown"))))
        response = next(event for event in self.writer.store.read_events(self.record.task_id)
                        if event["event_type"] == "model.responded")
        self.assertEqual(response["payload"]["detail"]["usage"], raw)
        self.assertEqual(self.create.call_count, 1)

    def test_missing_usage_in_recovery_retains_reported_subtotal_not_false_total(self):
        raw = dict(zip(FIELDS, (3, 2, 5)))
        self.create.side_effect = [reply(raw), reply(None)]
        self.issue()
        self.writer.begin_recovery_attempt(2, {"status": "INCOMPLETE"}, {"fixture": True})
        self.issue()
        usage = self.finish()
        self.assertEqual([usage[field] for field in FIELDS], [None, None, None])
        self.assertEqual(usage["known_reported_subtotal"], raw)
        self.assertEqual(usage["field_availability"], dict.fromkeys(FIELDS, "partial"))
        self.assertEqual(usage["availability"], "partial")
        self.assertEqual(usage["model_call_count"], 2)
        self.assertEqual(usage["calls_with_known_usage"], 1)
        self.assertEqual(self.create.call_count, 2)

    def test_failed_request_keeps_remote_token_cost_unknown_without_retry(self):
        from openai import APITimeoutError
        self.create.side_effect = APITimeoutError(request=object())
        with self.assertRaises(PlannerError) as caught:
            self.issue()
        self.assertEqual(caught.exception.code, "TIMEOUT")
        usage = self.finish(status="FAILED", error_code="TIMEOUT")
        self.assertEqual([usage[field] for field in FIELDS], [None, None, None])
        self.assertEqual(usage["availability"], "unknown")
        self.assertEqual(usage["model_call_count"], 1)
        self.assertEqual(self.create.call_count, 1)

    def test_pre_dispatch_budget_rejection_has_no_provider_cost(self):
        budget = SimpleNamespace(output_limit=Mock(side_effect=PlannerError("TASK_BUDGET_EXCEEDED", "fixture limit")))
        with patch("embodied_agent.models.deepseek.current_budget", return_value=budget), self.assertRaises(PlannerError):
            self.issue()
        usage = self.finish(status="FAILED", error_code="TASK_BUDGET_EXCEEDED")
        self.assertEqual([usage[field] for field in FIELDS], [0, 0, 0])
        self.assertEqual(usage["model_call_count"], 0)
        self.assertEqual(usage["availability"], "complete")
        self.create.assert_not_called()

    def test_one_call_is_not_double_counted_by_repeated_response_facts(self):
        self.writer.record_event("model_request", {"model_call_id": "call-fixture", "request": {"messages": []}})
        raw = dict(zip(FIELDS, (7, 2, 9)))
        for _ in range(2):
            self.writer.record_event("model_response", {"model_call_id": "call-fixture", "usage": raw})
        usage = self.finish()
        self.assertEqual([usage[field] for field in FIELDS], [7, 2, 9])
        self.assertEqual(usage["known_reported_subtotal"], raw)
        self.assertEqual(usage["model_call_count"], 1)
        self.assertEqual(usage["availability"], "complete")

    def test_invalid_provider_counts_remain_unknown(self):
        self.create.return_value = reply(dict(zip(FIELDS, (True, -2, 1.0))))
        self.issue()
        usage = self.finish()
        self.assertEqual([usage[field] for field in FIELDS], [None, None, None])
        self.assertEqual(usage["known_reported_subtotal"], dict.fromkeys(FIELDS, 0))
        self.assertEqual(usage["availability"], "unknown")
        self.assertEqual(self.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()

"""Shared cancellation/cost bounds and pre-execution proposal repair."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.models.budget import TaskBudget, budget_context
from embodied_agent.models.contracts import PlannerError
from embodied_agent.models.deepseek import RequestErrors
from embodied_agent.models.dialogue import run_tool_dialogue

ERRORS = RequestErrors("limited", "timeout", "connection", "failed", "empty")


def reply(text, *, calls=None, usage=None):
    return SimpleNamespace(model="fixture", usage=usage, choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=calls), finish_reason="tool_calls" if calls else "stop")])


class TaskBudgetTests(unittest.TestCase):
    def dialogue(self, client, *, validator=None):
        return run_tool_dialogue(client, instruction="原始指令", system_prompt="Return a plan.",
            user_payload={}, tools=[], tool_handler=lambda *_: {}, config={}, errors=ERRORS,
            final_validator=validator)

    def test_invalid_final_json_is_returned_for_repair_before_any_execution(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [reply("Explanation before JSON"), reply('{"actions":[]}')]
        def validate(raw):
            try:
                return json.dumps(json.loads(raw))
            except ValueError as exc:
                raise PlannerError("INVALID_JSON", "Expected JSON") from exc
        result = self.dialogue(client, validator=validate)
        self.assertEqual(result.request_count, 2)
        messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(json.loads(messages[-1]["content"])["error"]["code"], "INVALID_JSON")
        self.assertEqual(json.loads(messages[1]["content"])["instruction"], "原始指令")

    def test_repair_does_not_reset_request_budget(self):
        client = MagicMock()
        client.chat.completions.create.return_value = reply("invalid")
        budget = TaskBudget({"max_requests": 1})
        def invalid(_):
            raise PlannerError("INVALID_JSON", "Bad format")
        with budget_context(budget), self.assertRaises(PlannerError) as caught:
            self.dialogue(client, validator=invalid)
        self.assertEqual(caught.exception.code, "TASK_BUDGET_EXCEEDED")
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_cancelled_remote_reply_is_discarded_without_next_request(self):
        budget = TaskBudget()
        client = MagicMock()
        def create(**_):
            budget.cancel("User stopped")
            return reply('{"actions":[]}', usage={"prompt_tokens": 8, "completion_tokens": 3})
        client.chat.completions.create.side_effect = create
        with budget_context(budget), self.assertRaises(PlannerError) as caught:
            self.dialogue(client)
        self.assertEqual(caught.exception.code, "TASK_CANCELLED")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(budget.snapshot()["total_tokens"], 11)

    def test_deadline_and_action_recovery_limits_are_shared(self):
        clock = [0.0]
        budget = TaskBudget({"timeout_s": 5, "max_actions": 1, "max_replans": 1}, clock=lambda: clock[0])
        budget.record_action()
        with self.assertRaises(PlannerError):
            budget.record_action()
        budget.record_replan()
        with self.assertRaises(PlannerError):
            budget.record_replan()
        clock[0] = 6
        with self.assertRaises(PlannerError) as caught:
            budget.check()
        self.assertEqual(caught.exception.code, "TASK_BUDGET_EXCEEDED")

    def test_total_only_usage_is_counted_and_exhaustion_blocks_another_request(self):
        budget = TaskBudget({"max_tokens": 7})
        budget.record_usage({"total_tokens": 7})
        self.assertEqual(budget.snapshot()["total_tokens"], 7)
        with self.assertRaises(PlannerError):
            budget.record_request()


if __name__ == "__main__":
    unittest.main()

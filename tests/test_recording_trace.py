"""Causal IDs bind actual calls, feedback and action outcomes without retries."""
from __future__ import annotations

from contextvars import copy_context
import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.execution.supervisor import EmbodiedTaskRunner
from embodied_agent.models.contracts import PlannerError, PlannerResponse
from embodied_agent.models.deepseek import RequestErrors, complete_chat
from embodied_agent.models.dialogue import run_tool_dialogue
from embodied_agent.models.tracing import capture_model_requests, current_trace, trace_scope

ERRORS = RequestErrors("limited", "timeout", "connection", "request failed", "empty")
TOOLS = [{"type": "function", "function": {"name": "observe", "description": "Read state",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}]


def reply(content="{}", *, tool=False):
    calls = [SimpleNamespace(id="provider-call", type="function",
        function=SimpleNamespace(name="observe", arguments="{}"))] if tool else None
    return SimpleNamespace(model="fixture", usage=None, _request_id="provider-request",
        choices=[SimpleNamespace(finish_reason="tool_calls" if tool else "stop",
            message=SimpleNamespace(content=None if tool else content, tool_calls=calls))])


class RecordingTraceTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.client = MagicMock()

    def collect(self, event, detail):
        self.events.append((event, copy.deepcopy(detail)))

    def values(self, name):
        return [detail for event, detail in self.events if event == name]

    def call(self):
        return complete_chat(self.client, messages=[{"role": "user", "content": "raw"}],
                             tools=TOOLS, errors=ERRORS)

    def test_transport_ids_exact_request_and_unknown_provider_metadata(self):
        sdk_context = []

        def request(**kwargs):
            sdk_context.append(current_trace())
            return reply()

        self.client.chat.completions.create.side_effect = request
        with trace_scope(task_id="task", agent_run_id="agent", role="instruction", decision_id="decision"), \
             capture_model_requests(self.collect):
            self.call()
        prepared, dispatch, response = (self.values(name)[0] for name in
                                      ("model_request", "model_dispatch_started", "model_response"))
        self.assertEqual({row["model_call_id"] for row in (prepared, dispatch, response)},
                         {prepared["model_call_id"]})
        for row in (prepared, dispatch, response):
            self.assertEqual((row["task_id"], row["agent_run_id"], row["decision_id"]),
                             ("task", "agent", "decision"))
        self.assertEqual(prepared["request"], self.client.chat.completions.create.call_args.kwargs)
        self.assertEqual(sdk_context[0]["model_call_id"], prepared["model_call_id"])
        self.assertIsNone(prepared["sampling"]["temperature"]["value"])
        self.assertEqual(prepared["sampling"]["temperature"]["availability"], "provider_default_unknown")
        self.assertIsNone(response["token_ids"])
        self.assertIsNone(response["usage"])
        self.assertEqual(response["availability"]["usage"], "unknown")
        self.assertEqual(dispatch["remote_delivery"], "unknown")
        self.assertEqual(current_trace(), {})

    def test_timeout_and_failed_error_record_preserve_one_sdk_invocation(self):
        from openai import APITimeoutError
        self.client.chat.completions.create.side_effect = APITimeoutError(request=object())
        with capture_model_requests(self.collect), self.assertRaises(PlannerError) as error:
            self.call()
        self.assertEqual(error.exception.code, "TIMEOUT")
        failed = self.values("model_failed")[0]
        self.assertEqual(failed["model_call_id"], self.values("model_request")[0]["model_call_id"])
        self.assertTrue(failed["request_issued"])
        self.assertIsNone(failed["usage"])
        self.assertEqual(failed["remote_processing"], "unknown")
        self.client.chat.completions.create.assert_called_once()
        self.client.reset_mock()

        def failing(event, detail):
            if event == "model_failed":
                raise OSError("cannot write failure")

        with capture_model_requests(failing), self.assertRaises(PlannerError) as recorded:
            self.call()
        self.assertEqual(recorded.exception.code, "RECORDING_FAILED")
        self.assertEqual(recorded.exception.details["original_request_error"]["code"], "TIMEOUT")
        self.assertTrue(recorded.exception.details["request_issued"])
        self.client.chat.completions.create.assert_called_once()

    def test_dispatch_record_failure_prevents_sdk_and_does_not_retry(self):
        def failing(event, detail):
            if event == "model_dispatch_started":
                raise OSError("cannot persist dispatch boundary")

        with capture_model_requests(failing), self.assertRaises(PlannerError) as error:
            self.call()
        self.assertEqual(error.exception.code, "RECORDING_FAILED")
        self.assertFalse(error.exception.details["request_issued"])
        self.client.chat.completions.create.assert_not_called()

    def test_copied_worker_context_keeps_original_identity_after_parent_moves_on(self):
        entered, release = threading.Event(), threading.Event()
        failures = []

        def delayed(**kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("worker fixture was not released")
            return reply()

        def work():
            try:
                self.call()
            except Exception as error:
                failures.append(error)

        self.client.chat.completions.create.side_effect = delayed
        with capture_model_requests(self.collect), trace_scope(task_id="original", decision_id="old"):
            context = copy_context()
            worker = threading.Thread(target=lambda: context.run(work))
            worker.start()
            self.assertTrue(entered.wait(5))
        with trace_scope(task_id="new", decision_id="new"):
            release.set()
            worker.join(5)
            self.assertEqual(current_trace()["task_id"], "new")
        self.assertFalse(worker.is_alive())
        self.assertFalse(failures)
        self.assertEqual(self.values("model_response")[0]["task_id"], "original")
        self.assertEqual(self.values("model_response")[0]["decision_id"], "old")

    def test_query_raw_and_presented_result_and_actual_feedback_inclusion(self):
        raw = {"large": "x" * 1000}
        self.client.chat.completions.create.side_effect = [reply(tool=True), reply('{"actions":[]}')]
        with capture_model_requests(self.collect), trace_scope(agent_run_id="agent"):
            dialogue = run_tool_dialogue(self.client, instruction="observe", system_prompt="system",
                user_payload={}, tools=TOOLS, tool_handler=lambda name, arguments: raw,
                config={"agent": {"max_tool_result_chars": 200}}, errors=ERRORS, stage="instruction.plan")
        started, finished = self.values("agent_tool_call")[0], self.values("agent_tool_result")[0]
        self.assertEqual(started["tool_call_id"], finished["tool_call_id"])
        self.assertNotEqual(started["tool_call_id"], "provider-call")
        self.assertEqual(finished["native_tool_call_id"], "provider-call")
        self.assertEqual(finished["raw_handler_result"], raw)
        self.assertEqual(finished["result"]["error"]["code"], "TOOL_RESULT_TOO_LARGE")
        self.assertEqual(dialogue.messages[3]["tool_call_id"], "provider-call")
        included = self.values("feedback_included")[0]
        request = self.values("model_request")[1]
        self.assertEqual(included["model_call_id"], request["model_call_id"])
        self.assertEqual(included["feedback_id"], finished["feedback_id"])
        self.assertEqual(request["request"]["messages"][included["message_index"]]["content"], finished["wire_result"])
        self.assertEqual(included["stage"], "instruction.plan")

    def test_proposal_repair_preserves_raw_normalized_and_feedback_position(self):
        self.client.chat.completions.create.side_effect = [reply("bad raw proposal"), reply("accepted raw proposal")]

        def validate(content):
            if content.startswith("bad"):
                raise PlannerError("INVALID_PLAN", "repair this proposal")
            return '{"actions":[]}'

        with capture_model_requests(self.collect):
            run_tool_dialogue(self.client, instruction="raw task", system_prompt="system", user_payload={},
                tools=TOOLS, tool_handler=MagicMock(), config={}, errors=ERRORS,
                final_validator=validate, stage="intent")
        self.assertEqual(self.values("proposal_rejected")[0]["raw_proposal"], "bad raw proposal")
        accepted = self.values("proposal_accepted")[0]
        self.assertEqual(accepted["raw_proposal"], "accepted raw proposal")
        self.assertEqual(accepted["normalized_proposal"], '{"actions":[]}')
        included = self.values("feedback_included")[0]
        second = self.values("model_request")[1]["request"]
        self.assertEqual(json.loads(second["messages"][included["message_index"]]["content"])["error"]["code"], "INVALID_PLAN")

    def test_unexpected_validator_error_is_rejected_and_not_retried(self):
        self.client.chat.completions.create.side_effect = [reply("original proposal")]

        def validate(content):
            raise ValueError("validator fixture failed")

        with capture_model_requests(self.collect), self.assertRaises(PlannerError) as rejected:
            run_tool_dialogue(self.client, instruction="raw", system_prompt="system", user_payload={},
                tools=TOOLS, tool_handler=MagicMock(), config={}, errors=ERRORS,
                final_validator=validate, stage="plan")
        self.assertEqual(rejected.exception.code, "PLANNER_ERROR")
        self.assertEqual(self.values("proposal_rejected")[0]["raw_proposal"], "original proposal")
        self.assertEqual(self.values("proposal_rejected")[0]["error"]["code"], "ValueError")
        self.assertEqual(rejected.exception.details["stage"], "plan")
        self.client.chat.completions.create.assert_called_once()

    def test_illegal_tool_round_rejection_has_its_own_id_and_later_success_is_independent(self):
        invalid = reply(tool=True)
        invalid.choices[0].message.tool_calls[0].function.name = "physical_pick"
        self.client.chat.completions.create.side_effect = [invalid, reply('{"actions":[]}')]
        handler = MagicMock()
        arguments = dict(instruction="raw task", system_prompt="system", user_payload={},
            tools=TOOLS, tool_handler=handler, config={}, errors=ERRORS, stage="instruction.plan")
        with trace_scope(task_id="same-task", agent_run_id="same-agent", decision_id="first-decision"), \
             capture_model_requests(self.collect):
            with self.assertRaises(PlannerError) as failure:
                run_tool_dialogue(self.client, **arguments)
            with trace_scope(decision_id="recovery-decision"):
                success = run_tool_dialogue(self.client, **arguments)
        self.assertEqual(failure.exception.code, "UNKNOWN_TOOL")
        self.assertEqual(success.response.content, '{"actions":[]}')
        requests = self.values("model_request")
        self.assertEqual(len(requests), 2)
        rejected = self.values("proposal_rejected")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["model_call_id"], requests[0]["model_call_id"])
        self.assertEqual(rejected[0]["decision_id"], "first-decision")
        self.assertEqual(rejected[0]["task_id"], "same-task")
        self.assertEqual(rejected[0]["stage"], "instruction.plan")
        self.assertEqual(rejected[0]["native_payload"]["tool_calls"][0]["function"]["name"], "physical_pick")
        accepted = self.values("proposal_accepted")[0]
        self.assertEqual(accepted["model_call_id"], requests[1]["model_call_id"])
        self.assertNotEqual(accepted["model_call_id"], rejected[0]["model_call_id"])
        self.assertEqual(accepted["decision_id"], "recovery-decision")
        self.assertEqual(len(self.values("model_response_delivered")), 2)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        handler.assert_not_called()
        self.assertEqual(current_trace(), {})

    def test_supervisor_links_recovery_skips_and_unstarted_plan_suffix(self):
        class Episode:
            total_steps = 0
            held = False
            navigated = False
            failed_once = False
            calls = []

            def observe(self):
                return {"held": self.held, "world_version": self.total_steps}

            def action_satisfied(self, action, prior):
                return action["skill"] == "navigate" and self.navigated

            def run_action(self, action, record_event=None):
                self.calls.append(copy.deepcopy(action))
                record_event("action_reviewed", {"action": action}, self.observe())
                record_event("before_tool", {"action": action}, self.observe())
                self.total_steps += 1
                if action["skill"] == "navigate":
                    self.navigated = True
                if action["skill"] == "pick" and not self.failed_once:
                    self.failed_once = True
                    return {"status": "FAILED", "executed": False, "action": action,
                            "error_code": "NOT_DOCKED"}
                if action["skill"] == "pick":
                    self.held = True
                record_event("after_tool", {"action": action}, self.observe())
                return {"status": "SUCCESS", "action": action, "executed": True,
                        "observation": self.observe(), "physics_steps": 1}

            def safe_stop(self):
                pass

        class Planner:
            last_goals = [{"predicate": "held", "object_id": "remote"}]
            last_decision = "plan"
            calls = []

            def plan(self, instruction, observation, world, **kwargs):
                self.calls.append(copy.deepcopy(kwargs))
                return PlannerResponse(json.dumps({"schema_version": 1, "actions": [
                    {"skill": "navigate", "target": "remote"}, {"skill": "pick", "object_id": "remote"},
                    {"skill": "wait", "seconds": .1}]}), "stub", None, None, None, 0, "stop", None)

        episode, planner = Episode(), Planner()
        with trace_scope(agent_run_id="agent", role="instruction", attempt_id=1), capture_model_requests(self.collect):
            result = EmbodiedTaskRunner(planner, goal_evaluator=lambda goals, observation, **kwargs:
                {"passed": observation["held"]}).run("hold remote", episode, None)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual([action["skill"] for action in episode.calls], ["navigate", "pick", "pick"])
        decisions = self.values("decision_started")
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[1]["parent_decision_id"], decisions[0]["decision_id"])
        self.assertEqual(len(self.values("decision_finished")), 2)
        self.assertEqual(self.values("action_skipped")[0]["action"]["skill"], "navigate")
        self.assertEqual([value["action"]["skill"] for value in self.values("action_not_started")], ["wait", "wait"])
        self.assertEqual(self.values("feedback_delivered")[0]["feedback"], planner.calls[1]["feedback"])
        review = self.values("action_reviewed")
        dispatch = self.values("action_dispatch_started")
        self.assertEqual([value["action_id"] for value in review], [value["action_id"] for value in dispatch])
        self.assertTrue(self.values("goal_evaluated"))


if __name__ == "__main__":
    unittest.main()

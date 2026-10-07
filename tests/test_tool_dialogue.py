"""Native query dialogue protocol, budget, schema and recording boundaries."""
from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.models.contracts import PlannerError, PlannerResponse
from embodied_agent.models.deepseek import RequestErrors, complete_chat
from embodied_agent.models.dialogue import run_tool_dialogue
from embodied_agent.models.tracing import capture_model_requests

ERRORS = RequestErrors("limited", "timed out", "connection failed", "request failed", "empty reply", True)
TOOLS = [{"type": "function", "function": {"name": "query_world", "description": "Read entities",
    "parameters": {"type": "object", "properties": {"entity_id": {"type": "string", "minLength": 1}},
        "required": ["entity_id"], "additionalProperties": False}}}]


def tool_call(call_id="call_1", name="query_world", arguments='{"entity_id":"remote"}', call_type="function"):
    return SimpleNamespace(id=call_id, type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments))


def reply(content=None, *, calls=None, finish=None, reasoning=None):
    message = SimpleNamespace(content=content, tool_calls=calls)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(model="fixture-response-model", usage=None, _request_id="fixture-request",
        choices=[SimpleNamespace(message=message, finish_reason=finish or ("tool_calls" if calls else "stop"))])


class ToolDialogueTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.handler = MagicMock(return_value={"ok": True, "entity_id": "remote", "position": [1, 2, 3]})
        self.events = []
        self.config = {"planner": {"thinking": "disabled", "reasoning_effort": "low"}}

    def run_dialogue(self, responses, **kwargs):
        self.client.chat.completions.create.side_effect = responses
        values = dict(instruction="把茶几上的遥控器送到沙发", system_prompt="Call queries then return final JSON.",
            user_payload={"instruction": "do not use this stale text", "world_version": 2},
            tools=TOOLS, tool_handler=self.handler, config=self.config, errors=ERRORS, model="fixture-model")
        values.update(kwargs)
        with capture_model_requests(lambda event, detail: self.events.append((event, deepcopy(detail)))):
            return run_tool_dialogue(self.client, **values)

    def test_multiple_native_rounds_feed_results_and_preserve_raw_instruction(self):
        result = self.run_dialogue([
            reply(calls=[tool_call()], reasoning="Need object position"),
            reply(calls=[tool_call("call_2", arguments='{"entity_id":"sofa"}')]),
            reply('{"actions":[{"skill":"navigate"}]}'),
        ])
        self.assertIsInstance(result.response, PlannerResponse)
        self.assertEqual(result.response.content, '{"actions":[{"skill":"navigate"}]}')
        self.assertEqual(result.response.planner_kind, "llm")
        self.assertEqual(result.request_count, 3)
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(result.response.response_model, "fixture-response-model")
        self.assertEqual(self.handler.call_args_list[1].args, ("query_world", {"entity_id": "sofa"}))
        calls = self.client.chat.completions.create.call_args_list
        for call in calls:
            self.assertNotIn("response_format", call.kwargs)
            self.assertEqual(call.kwargs["tools"], TOOLS)
            self.assertEqual(call.kwargs["tool_choice"], "auto")
            self.assertEqual(call.kwargs["max_tokens"], 4096)
        first = json.loads(calls[0].kwargs["messages"][1]["content"])
        self.assertEqual(first["instruction"], "把茶几上的遥控器送到沙发")
        second = calls[1].kwargs["messages"]
        self.assertEqual(second[2]["reasoning_content"], "Need object position")
        self.assertEqual(second[3]["role"], "tool")
        self.assertEqual(second[3]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(second[3]["content"])["position"], [1, 2, 3])
        self.assertEqual([name for name, _ in self.events if name in {
            "model_request", "model_response", "agent_tool_call", "agent_tool_result"}], ["model_request", "model_response",
            "agent_tool_call", "agent_tool_result", "model_request", "model_response",
            "agent_tool_call", "agent_tool_result", "model_request", "model_response"])
        self.assertNotIn("api_key", json.dumps(self.events))

    def test_sdk_native_message_and_usage_are_supported(self):
        from openai.types.chat import ChatCompletion
        native = ChatCompletion.model_validate({"id": "fixture-id", "object": "chat.completion", "created": 0,
            "model": "sdk-model", "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "native-call", "type": "function", "function": {
                        "name": "query_world", "arguments": '{"entity_id":"remote"}'}}]}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}})
        result = self.run_dialogue([native, reply('{"actions":[]}')])
        self.assertEqual(result.request_count, 2)
        event = next(detail for name, detail in self.events if name == "model_response")
        self.assertEqual(event["tool_calls"][0]["id"], "native-call")
        self.assertEqual(event["usage"]["total_tokens"], 5)

    def test_whole_batch_validation_prevents_partial_queries(self):
        invalid_calls = [
            (tool_call("second", name="pick"), "UNKNOWN_TOOL"),
            (tool_call("call_1"), "INVALID_TOOL_CALL"),
            (tool_call("second", call_type="custom"), "INVALID_TOOL_CALL"),
            (tool_call("second", arguments='{"entity_id":true}'), "INVALID_TOOL_ARGUMENTS"),
            (tool_call("second", arguments='{"entity_id":"x","extra":1}'), "INVALID_TOOL_ARGUMENTS"),
            (tool_call("second", arguments='{"entity_id":"x","entity_id":"y"}'), "INVALID_TOOL_ARGUMENTS"),
            (tool_call("second", arguments='{"entity_id":NaN}'), "INVALID_TOOL_ARGUMENTS"),
            (tool_call("second", arguments='[]'), "INVALID_TOOL_ARGUMENTS"),
            (tool_call("second", arguments='{"entity_id":'), "INVALID_TOOL_ARGUMENTS"),
            (tool_call("second", arguments={"entity_id": "x"}), "INVALID_TOOL_ARGUMENTS"),
        ]
        for invalid, code in invalid_calls:
            with self.subTest(code=code, invalid=invalid):
                self.client.reset_mock()
                self.handler.reset_mock()
                self.events.clear()
                with self.assertRaises(PlannerError) as caught:
                    self.run_dialogue([reply(calls=[tool_call(), invalid])])
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.details["request_count"], 1)
                self.assertEqual(caught.exception.details["tool_call_count"], 0)
                self.assertEqual(len(caught.exception.details["received_response"]["tool_calls"]), 2)
                self.handler.assert_not_called()
                rejected = [detail for event, detail in self.events if event == "proposal_rejected"]
                self.assertEqual(len(rejected), 1)
                self.assertEqual(rejected[0]["error_code"], code)
                self.assertEqual(rejected[0]["reason"], "tool_call_schema")
                self.assertEqual(rejected[0]["source"], "llm")
                self.assertFalse(rejected[0]["tool_batch_executed"])
                response = next(detail for event, detail in self.events if event == "model_response")
                self.assertEqual(rejected[0]["model_call_id"], response["model_call_id"])
                self.assertEqual(rejected[0]["native_payload"]["tool_calls"], response["tool_calls"])

    def test_call_id_cannot_be_reused_in_later_turn(self):
        with self.assertRaises(PlannerError) as caught:
            self.run_dialogue([reply(calls=[tool_call()]), reply(calls=[tool_call()])])
        self.assertEqual(caught.exception.code, "INVALID_TOOL_CALL")
        self.assertEqual(caught.exception.details["request_count"], 2)
        self.assertEqual(caught.exception.details["tool_call_count"], 1)
        responses = [detail for event, detail in self.events if event == "model_response"]
        rejected = [detail for event, detail in self.events if event == "proposal_rejected"]
        self.assertEqual([row["model_call_id"] for row in rejected], [responses[1]["model_call_id"]])

    def test_malformed_native_tool_list_is_a_protocol_rejection(self):
        with self.assertRaises(PlannerError) as caught:
            self.run_dialogue([reply(calls="not-a-list", finish="tool_calls")])
        self.assertEqual(caught.exception.code, "INVALID_TOOL_CALL")
        rejected = [detail for event, detail in self.events if event == "proposal_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "response_protocol")
        self.assertEqual(rejected[0]["native_payload"]["tool_calls"], "not-a-list")
        self.handler.assert_not_called()
        self.client.chat.completions.create.assert_called_once()

    def test_tool_and_round_budgets_prevent_further_queries(self):
        for config, response, code in [
            ({"agent": {"max_rounds": 1}}, reply(calls=[tool_call()]), "DIALOGUE_BUDGET_EXCEEDED"),
            ({"agent": {"max_tool_calls": 1}}, reply(calls=[tool_call(), tool_call("second")]), "TOOL_BUDGET_EXCEEDED"),
            ({"agent": {"max_tool_arguments_chars": 8}}, reply(calls=[tool_call()]), "INVALID_TOOL_ARGUMENTS"),
        ]:
            self.events.clear()
            with self.subTest(code=code), self.assertRaises(PlannerError) as caught:
                self.run_dialogue([response], config=config)
            self.assertEqual(caught.exception.code, code)
            self.handler.assert_not_called()
            self.assertEqual(any(event == "proposal_rejected" for event, _ in self.events),
                             code == "INVALID_TOOL_ARGUMENTS")
        result = self.run_dialogue([reply('{"actions":[]}')], config={"agent": {"max_output_tokens": 8192}})
        self.assertEqual(result.request_count, 1)
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs["max_tokens"], 8192)

    def test_query_errors_return_to_model_and_do_not_stop_dialogue(self):
        self.handler.side_effect = PlannerError("ENTITY_NOT_FOUND", "Entity unavailable", {"entity_id": "remote"})
        result = self.run_dialogue([reply(calls=[tool_call()]), reply('{"actions":[]}')])
        self.assertEqual(result.tool_results[0]["result"]["error"]["code"], "ENTITY_NOT_FOUND")
        feedback = json.loads(self.client.chat.completions.create.call_args.kwargs["messages"][-1]["content"])
        self.assertFalse(feedback["ok"])
        self.assertFalse(any(event == "proposal_rejected" for event, _ in self.events))

    def test_rejection_record_failure_keeps_original_protocol_error_and_one_request(self):
        self.client.chat.completions.create.side_effect = [reply(calls=[tool_call(name="physical_pick")])]
        attempted = []
        def recorder(event, detail):
            attempted.append((event, deepcopy(detail)))
            if event == "proposal_rejected":
                raise OSError("cannot record invalid proposal")
        with capture_model_requests(recorder), self.assertRaises(PlannerError) as caught:
            run_tool_dialogue(self.client, instruction="raw", system_prompt="system", user_payload={},
                tools=TOOLS, tool_handler=self.handler, config={}, errors=ERRORS)
        error = caught.exception
        self.assertEqual(error.code, "UNKNOWN_TOOL")
        self.assertEqual(error.details["request_count"], 1)
        self.assertEqual(error.details["tool_call_count"], 0)
        self.assertEqual(error.details["recording_errors"][0]["event"], "proposal_rejected")
        self.assertEqual(error.details["received_response"]["tool_calls"][0]["function"]["name"], "physical_pick")
        self.assertEqual(sum(name == "proposal_rejected" for name, _ in attempted), 1)
        self.handler.assert_not_called()
        self.client.chat.completions.create.assert_called_once()

    def test_final_rejection_record_failure_preserves_validator_error_and_does_not_repair(self):
        self.client.chat.completions.create.side_effect = [reply("invalid final proposal"), reply("later reply")]
        original = PlannerError("INVALID_PLAN", "original validator rejection")
        def recorder(event, detail):
            if event == "proposal_rejected":
                raise OSError("rejection journal unavailable")
        with capture_model_requests(recorder), self.assertRaises(PlannerError) as caught:
            run_tool_dialogue(self.client, instruction="raw", system_prompt="system", user_payload={},
                tools=TOOLS, tool_handler=self.handler, config={}, errors=ERRORS,
                final_validator=MagicMock(side_effect=original))
        self.assertIs(caught.exception, original)
        self.assertEqual(caught.exception.code, "INVALID_PLAN")
        self.assertTrue(caught.exception.details["recording_errors"])
        self.assertEqual(caught.exception.details["request_count"], 1)
        self.client.chat.completions.create.assert_called_once()
        self.handler.assert_not_called()

    def test_invalid_and_oversized_results_return_bounded_error(self):
        for result, code in [(object(), "INVALID_TOOL_RESULT"), ({"value": float("nan")}, "INVALID_TOOL_RESULT"),
                ({"huge": "x" * 300}, "TOOL_RESULT_TOO_LARGE")]:
            with self.subTest(code=code):
                self.handler.return_value = result
                dialogue = self.run_dialogue([reply(calls=[tool_call()]), reply('{"actions":[]}')],
                    config={"agent": {"max_tool_result_chars": 200}})
                self.assertEqual(dialogue.tool_results[0]["result"]["error"]["code"], code)
                text = self.client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
                self.assertLessEqual(len(text), 200)

    def test_truncation_and_bad_finish_never_execute_queries(self):
        for response, code in [(reply("partial", calls=[tool_call()], finish="length"), "TRUNCATED_RESPONSE"),
                (reply(calls=[tool_call()], finish="stop"), "INVALID_RESPONSE"),
                (reply('{"actions":[]}', finish="content_filter"), "INVALID_RESPONSE"),
                (reply(""), "EMPTY_RESPONSE"),
                (SimpleNamespace(choices=[], usage=None), "EMPTY_RESPONSE"),
                (SimpleNamespace(), "EMPTY_RESPONSE")]:
            first_event = len(self.events)
            with self.subTest(code=code), self.assertRaises(PlannerError) as caught:
                self.run_dialogue([response])
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.details["request_count"], 1)
            self.handler.assert_not_called()
            if code == "TRUNCATED_RESPONSE":
                self.assertFalse(any(event == "proposal_rejected" for event, _ in self.events[first_event:]))
        self.assertEqual(next(detail for event, detail in self.events if event == "model_response")["response_text"], "partial")

    def test_recording_failure_stops_without_retry_and_preserves_received_content(self):
        for boundary in ("model_request", "model_response", "agent_tool_call", "agent_tool_result"):
            with self.subTest(boundary=boundary):
                self.client.reset_mock()
                self.handler.reset_mock()
                self.client.chat.completions.create.side_effect = [reply(calls=[tool_call()]), reply('{"actions":[]}')]

                def recorder(event, detail):
                    if event == boundary:
                        raise OSError("fixture evidence failure")

                with capture_model_requests(recorder), self.assertRaises(PlannerError) as caught:
                    run_tool_dialogue(self.client, instruction="raw", system_prompt="system", user_payload={},
                        tools=TOOLS, tool_handler=self.handler, config={}, errors=ERRORS)
                error = caught.exception
                self.assertEqual(error.code, "RECORDING_FAILED")
                issued = boundary != "model_request"
                self.assertEqual(error.details["request_count"], int(issued))
                self.assertEqual(error.details["request_issued"], issued)
                self.assertEqual(self.client.chat.completions.create.call_count, int(issued))
                self.assertEqual(self.handler.call_count, int(boundary == "agent_tool_result"))
                if issued:
                    self.assertEqual(error.details["received_response"]["tool_calls"][0]["id"], "call_1")
                if boundary == "agent_tool_result":
                    self.assertEqual(len(error.details["tool_results"]), 1)

    def test_transport_failure_retains_previous_tool_results_and_counts_issued_request(self):
        from openai import APITimeoutError
        with self.assertRaises(PlannerError) as caught:
            self.run_dialogue([reply(calls=[tool_call()]), APITimeoutError(request=object())])
        self.assertEqual(caught.exception.code, "TIMEOUT")
        self.assertEqual(caught.exception.details["request_count"], 2)
        self.assertEqual(caught.exception.details["tool_call_count"], 1)
        self.assertEqual(len(caught.exception.details["tool_results"]), 1)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)

    def test_invalid_config_and_unsupported_schema_issue_no_requests(self):
        for config in ({"agent": {"max_rounds": True}}, {"agent": {"max_output_tokens": "4096"}}):
            with self.subTest(config=config), self.assertRaises(PlannerError) as caught:
                self.run_dialogue([], config=config)
            self.assertEqual(caught.exception.code, "INVALID_AGENT_CONFIG")
        tools = deepcopy(TOOLS)
        tools[0]["function"]["parameters"]["not"] = {}
        with self.assertRaises(PlannerError) as caught:
            self.run_dialogue([], tools=tools)
        self.assertEqual(caught.exception.code, "INVALID_TOOL_SCHEMA")
        self.client.chat.completions.create.assert_not_called()

    def test_transport_missing_optional_metadata_does_not_hide_request(self):
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")])
        self.client.chat.completions.create.side_effect = None
        with capture_model_requests(lambda event, detail: self.events.append((event, detail))):
            response = complete_chat(self.client, messages=[{"role": "user", "content": "raw"}],
                tools=TOOLS, errors=ERRORS)
        self.assertIsNotNone(response)
        self.assertEqual([name for name, _ in self.events],
                         ["model_request", "model_dispatch_started", "model_response"])
        self.assertIsNone(self.events[-1][1]["response_model"])
        self.assertEqual(self.events[-1][1]["response_text"], "{}")


if __name__ == "__main__":
    unittest.main()

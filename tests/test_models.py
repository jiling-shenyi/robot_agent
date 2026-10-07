"""Behavioral regression for current instruction/environment model transport."""

from __future__ import annotations

import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.agents.environment import EnvironmentAgent
from embodied_agent.agents.instruction import InstructionAgent
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.maps.store import MapStore
from embodied_agent.models.contracts import PlannerError


class SharedModelTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "configs" / "agent_runtime.json").read_text(encoding="utf-8"))
        self.config["agent"]["interpret_goals_first"] = False
        self.instruction_world = HomeMapStore().load()
        self.world = MapStore().load("classic")
        self.response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"ok":true}'))],
            model="response-model", usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 23}),
            _request_id="response-request",
        )
        self.client = MagicMock()
        self.client.chat.completions.create.return_value = self.response
        environment = patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "request-model",
            "DEEPSEEK_BASE_URL": "https://example.invalid",
        })
        environment.start()
        self.addCleanup(environment.stop)
        sdk = patch("openai.OpenAI", return_value=self.client)
        self.sdk = sdk.start()
        self.addCleanup(sdk.stop)

    def invoke(self, role: str):
        if role == "instruction":
            return InstructionAgent(self.config).plan("观察房间", self.instruction_world.snapshot(), self.instruction_world)
        return EnvironmentAgent(MapStore(), config=self.config)._llm_plan("move cube", self.world)


    def test_sdk_errors_keep_role_codes_messages_details_and_single_request(self) -> None:
        from openai import (
            APIConnectionError, APIStatusError, APITimeoutError,
            AuthenticationError, RateLimitError,
        )

        request = object()
        status = SimpleNamespace(request=request, status_code=503, headers={"x-request-id": "failure-request"})
        cases = (
            (AuthenticationError("mock", response=status, body=None), "AUTHENTICATION_FAILED"),
            (RateLimitError("mock", response=status, body=None), "RATE_LIMITED"),
            (APITimeoutError(request=request), "TIMEOUT"),
            (APIConnectionError(request=request), "CONNECTION_FAILED"),
            (APIStatusError("mock", response=status, body=None), "API_STATUS_ERROR"),
            (RuntimeError("mock"), "PLANNER_ERROR"),
        )
        timeout_messages = {
            "instruction": "Instruction request timed out",
            "environment": "Environment planning request timed out",
        }
        for role in timeout_messages:
            for failure, expected_code in cases:
                with self.subTest(role=role, code=expected_code):
                    self.client.reset_mock()
                    self.client.chat.completions.create.side_effect = failure
                    with self.assertRaises(PlannerError) as raised:
                        self.invoke(role)
                    self.assertEqual(raised.exception.code, expected_code)
                    self.client.chat.completions.create.assert_called_once()
                    self.assertEqual(self.client.close.call_count, int(role == "environment"))
                    if expected_code == "TIMEOUT":
                        self.assertEqual(str(raised.exception), timeout_messages[role])
                    expected_details = {"http_status": 503, "request_id": "failure-request"} if (
                        role == "instruction" and expected_code == "API_STATUS_ERROR"
                    ) else {}
                    for key, value in expected_details.items():
                        self.assertEqual(raised.exception.details[key], value)
                    self.assertEqual(raised.exception.details["request_count"], 1)
                    self.assertEqual(raised.exception.details["tool_call_count"], 0)
                    self.assertEqual(raised.exception.details["tool_results"], [])

    def test_native_protocol_metadata_and_client_lifetime(self):
        self.response.choices[0].message.content = '{"schema_version":1,"goals":[],"actions":[],"decision":"clarify","question":"Which object?"}'
        for role in ("instruction", "environment"):
            with self.subTest(role=role):
                self.client.reset_mock()
                result = self.invoke(role)
                self.client.chat.completions.create.assert_called_once()
                request = self.client.chat.completions.create.call_args.kwargs
                self.assertEqual(request["max_tokens"], 4096)
                self.assertNotIn("response_format", request)
                self.assertEqual(request["tool_choice"], "auto")
                self.assertTrue(request["tools"])
                self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
                self.assertEqual(request["reasoning_effort"], "low")
                self.assertEqual(self.sdk.call_args.kwargs["max_retries"], 0)
                self.assertNotIn("required_plan", json.loads(request["messages"][1]["content"]))
                self.assertEqual(self.client.close.call_count, int(role == "environment"))
                if role == "instruction":
                    self.assertEqual(result.requested_model, "request-model")
                    self.assertEqual(result.response_model, "response-model")
                    self.assertEqual(result.usage, {"total_tokens": 23})
                    self.assertEqual(result.request_id, "response-request")

    def test_empty_and_truncated_responses_keep_evidence(self):
        for choices, code in (([], "EMPTY_RESPONSE"), ([SimpleNamespace(finish_reason="length",
                message=SimpleNamespace(content="partial"))], "TRUNCATED_RESPONSE")):
            self.response.choices = choices
            for role in ("instruction", "environment"):
                with self.subTest(role=role, code=code), self.assertRaises(PlannerError) as raised:
                    self.invoke(role)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.details["request_count"], 1)
                if code == "TRUNCATED_RESPONSE":
                    self.assertEqual(raised.exception.details["response_text"], "partial")

    def test_invalid_dialogue_config_never_issues_request(self):
        self.config["agent"]["max_output_tokens"] = "invalid"
        for role in ("instruction", "environment"):
            with self.subTest(role=role), self.assertRaises(PlannerError) as raised:
                self.invoke(role)
            self.assertEqual(raised.exception.code, "INVALID_AGENT_CONFIG")
        self.client.chat.completions.create.assert_not_called()


    def test_configured_prompt_overrides_reach_model_request(self) -> None:
        self.response.choices[0].message.content = '{"schema_version":1,"goals":[],"actions":[],"decision":"clarify","question":"Which object?"}'
        original = deepcopy(self.config)
        for role, prompt_id in (("instruction", "instruction.plan"), ("environment", "environment.desktop")):
            with self.subTest(role=role):
                self.client.reset_mock()
                self.config = deepcopy(original)
                text = f"replacement {role} prompt\nExact candidate text."
                self.config.setdefault("components", {}).setdefault("prompts", {}).setdefault("overrides", {})[
                    prompt_id] = {"text": text, "version": "transport-test-v2"}
                self.invoke(role)
                self.client.chat.completions.create.assert_called_once()
                self.assertEqual(self.client.chat.completions.create.call_args.kwargs["messages"][0]["content"], text)

    def test_missing_credential_never_opens_client_or_falls_back(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            for role in ("instruction", "environment"):
                with self.subTest(role=role), self.assertRaises(PlannerError) as raised:
                    self.invoke(role)
                self.assertEqual(raised.exception.code, "MISSING_CREDENTIAL")
        self.sdk.assert_not_called()
        self.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()

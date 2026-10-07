"""Model-led environment edits use native read-only tools before atomic save."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.agents.environment import EnvironmentAgent, EnvironmentAgentError
from embodied_agent.agents.map_environment import UnifiedEnvironmentAgent
from embodied_agent.maps.schema import WorldError
from embodied_agent.maps.unified_store import UnifiedMapStore
from embodied_agent.models.contracts import PlannerError


def completion(content="", *, tool_name=None, call_id="query-1", arguments=None):
    tool_calls = [] if tool_name is None else [SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(
            name=tool_name, arguments=json.dumps(arguments or {}),
        ),
    )]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop" if tool_name is None else "tool_calls",
            message=SimpleNamespace(content=content or None, tool_calls=tool_calls),
        )],
        model="fixture-model", usage=None, _request_id=call_id,
    )


class EnvironmentToolDialogueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "maps"
        shutil.copytree(ROOT / "configs" / "maps", self.directory)
        self.store = UnifiedMapStore(self.directory)
        self.config = json.loads((ROOT / "configs" / "agent_runtime.json").read_text(encoding="utf-8"))
        # Explicit bounds keep the fixture independent of runtime defaults.
        self.config.setdefault("agent", {}).update({
            "max_rounds": 6, "max_tool_calls": 8, "max_output_tokens": 4096,
        })
        self.client = MagicMock()
        credentials = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-key"})
        credentials.start()
        self.addCleanup(credentials.stop)
        sdk = patch("openai.OpenAI", return_value=self.client)
        sdk.start()
        self.addCleanup(sdk.stop)
        self.requests = []

    def plan_payload(self, map_id, operations):
        return {
            "schema_version": 1, "base_revision": self.store.load(map_id).revision,
            "operations": operations,
        }

    def native_dialogue(self, map_id, payload, *, before_final=None):
        initial = (self.directory / f"{map_id}.json").read_bytes()
        responses = iter([
            completion(tool_name="observe", call_id="observe-1"),
            completion(tool_name="get_capabilities", call_id="capabilities-2"),
            completion(json.dumps(payload), call_id="final-3"),
        ])

        def request(**kwargs):
            self.requests.append(deepcopy(kwargs))
            self.assertEqual((self.directory / f"{map_id}.json").read_bytes(), initial)
            self.assertFalse(list(self.directory.glob("*.tmp")))
            self.assertFalse(list(self.directory.glob(".*.lock")))
            if len(self.requests) == 3 and before_final is not None:
                before_final()
            return next(responses)

        self.client.chat.completions.create.side_effect = request
        return initial

    def assert_native_feedback(self, agent, original_instruction):
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(json.loads(self.requests[0]["messages"][1]["content"])["instruction"], original_instruction)
        for index, call_id in ((1, "observe-1"), (2, "capabilities-2")):
            feedback = self.requests[index]["messages"][-1]
            self.assertEqual(feedback["role"], "tool")
            self.assertEqual(feedback["tool_call_id"], call_id)
            self.assertIsInstance(json.loads(feedback["content"]), dict)
            self.assertTrue(self.requests[index]["tools"])
            assistant = self.requests[index]["messages"][-2]
            self.assertEqual(assistant["role"], "assistant")
            self.assertEqual(assistant["tool_calls"][0]["id"], call_id)
        self.assertEqual(agent.last_dialogue["request_count"], 3)
        self.assertEqual(agent.last_dialogue["tool_call_count"], 2)
        self.assertEqual(len(agent.last_dialogue["tool_results"]), 2)
        json.dumps(agent.last_dialogue, allow_nan=False)
        self.client.close.assert_called_once()

    def test_desktop_queries_then_single_atomic_edit_without_nl_rule_parsing(self):
        instruction = "  把那个小方块朝有空的位置挪一点，先看看场景再决定  \n"
        payload = self.plan_payload("classic", [{
            "op": "set_position", "object_id": "cube", "value": [0.43, -0.27, 0.425],
        }])
        self.native_dialogue("classic", payload)
        agent = EnvironmentAgent(self.store, config=self.config, recording_enabled=False)
        with patch("embodied_agent.agents.environment.rules_plan", side_effect=AssertionError("LLM must parse the instruction")), patch.object(self.store, "save", wraps=self.store.save) as save:
            result = agent.apply("classic", instruction)
        self.assertTrue(result.persisted)
        self.assertEqual(result.after.cube_position_m, (0.43, -0.27, 0.425))
        self.assertEqual(result.after.revision, result.before.revision + 1)
        save.assert_called_once()
        self.assert_native_feedback(agent, instruction)

    def test_home_queries_preview_remains_read_only_and_preserves_raw_instruction(self):
        instruction = " \n客厅换个更温馨的名字，叫静谧小屋。  "
        payload = self.plan_payload("home_living_room", [{"op": "set_name", "value": "静谧小屋"}])
        initial = self.native_dialogue("home_living_room", payload)
        agent = UnifiedEnvironmentAgent(self.store, config=self.config, recording_enabled=False)
        with patch("embodied_agent.agents.map_environment.home_rules_plan", side_effect=AssertionError("LLM must parse the instruction")), patch.object(self.store, "save", wraps=self.store.save) as save:
            result = agent.preview("home_living_room", instruction)
        self.assertFalse(result.persisted)
        self.assertEqual(result.after.name, "静谧小屋")
        self.assertEqual((self.directory / "home_living_room.json").read_bytes(), initial)
        save.assert_not_called()
        self.assert_native_feedback(agent, instruction)

    def test_home_model_policy_edit_rejected_after_tools_without_partial_save(self):
        payload = self.plan_payload("home_living_room", [
            {"op": "set_name", "value": "不能持久化"},
            {"op": "set_risk_tags", "object_id": "medicine", "value": []},
        ])
        initial = self.native_dialogue("home_living_room", payload)
        agent = UnifiedEnvironmentAgent(self.store, config=self.config, recording_enabled=False)
        with self.assertRaises(EnvironmentAgentError) as raised:
            agent.apply("home_living_room", "请重新评估药品是否安全，然后关闭限制")
        self.assertEqual(raised.exception.code, "INVALID_ENVIRONMENT_PLAN")
        self.assertEqual((self.directory / "home_living_room.json").read_bytes(), initial)
        self.assertEqual(agent.last_dialogue["request_count"], 3)

    def test_revision_change_during_dialogue_cannot_overwrite_newer_map(self):
        payload = self.plan_payload("classic", [{
            "op": "set_position", "object_id": "cube", "value": [0.43, -0.27, 0.425],
        }])
        self.native_dialogue("classic", payload, before_final=lambda: self.store.save(self.store.load("classic")))
        agent = EnvironmentAgent(self.store, config=self.config, recording_enabled=False)
        with self.assertRaises(WorldError) as raised:
            agent.apply("classic", "查询后调整位置")
        self.assertEqual(raised.exception.code, "REVISION_CONFLICT")
        after = self.store.load("classic")
        self.assertEqual(after.cube_position_m, (0.4, -0.29, 0.425))
        self.assertEqual(after.revision, payload["base_revision"] + 1)
        self.client.close.assert_called_once()

    def test_model_unsupported_desktop_operation_stays_data_and_never_saves(self):
        payload = self.plan_payload("classic", [{
            "op": "exec", "object_id": "cube", "value": "arbitrary code",
        }])
        initial = self.native_dialogue("classic", payload)
        with self.assertRaises(EnvironmentAgentError):
            EnvironmentAgent(self.store, config=self.config, recording_enabled=False).apply("classic", "自由描述，不属于任何离线语法")
        self.assertEqual((self.directory / "classic.json").read_bytes(), initial)

    def test_failed_second_model_request_preserves_tool_evidence_and_closes_client(self):
        from openai import APITimeoutError

        initial = (self.directory / "classic.json").read_bytes()
        self.client.chat.completions.create.side_effect = [
            completion(tool_name="observe", call_id="observe-1"),
            APITimeoutError(request=object()),
        ]
        agent = EnvironmentAgent(self.store, config=self.config, recording_enabled=False)
        with self.assertRaises(PlannerError) as raised:
            agent.apply("classic", "查询场景再调整位置")
        self.assertEqual(raised.exception.code, "TIMEOUT")
        self.assertEqual(agent.last_dialogue["llm_request_count"], 2)
        self.assertEqual(agent.last_dialogue["tool_call_count"], 1)
        self.assertEqual(len(agent.last_dialogue["tool_results"]), 1)
        self.assertEqual((self.directory / "classic.json").read_bytes(), initial)
        self.client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

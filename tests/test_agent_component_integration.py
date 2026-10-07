"""Replaceable Agent components reach real transport requests and keep authority."""
from __future__ import annotations

import copy
import hashlib
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

from embodied_agent.agents.environment import EnvironmentAgentError
from embodied_agent.agents.instruction import InstructionAgent
from embodied_agent.agents.map_environment import UnifiedEnvironmentAgent
from embodied_agent.agents.observation import WorldObservation
from embodied_agent.context import AgentContext
from embodied_agent.knowledge import KnowledgeBase
from embodied_agent.memory import MemoryStore
from embodied_agent.maps.unified_store import UnifiedMapStore
from embodied_agent.models.tracing import capture_model_requests


def reply(content=None, *, tool=None, call_id="component-query"):
    calls = None if tool is None else [SimpleNamespace(id=call_id, type="function",
        function=SimpleNamespace(name=tool, arguments="{}"))]
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls" if calls else "stop",
        message=SimpleNamespace(content=content, tool_calls=calls))], model="component-fixture",
        usage=None, _request_id="component-response")


class AgentComponentIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.maps = Path(temporary.name) / "maps"
        shutil.copytree(ROOT / "configs" / "maps", self.maps)
        self.store = UnifiedMapStore(self.maps)
        self.world = self.store.load("home_living_room")
        self.snapshot = self.world.snapshot()
        self.config = json.loads((ROOT / "configs" / "agent_runtime.json").read_text(encoding="utf-8"))
        self.config.pop("components", None)
        self.config["agent"]["interpret_goals_first"] = False
        self.client = MagicMock()
        credentials = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "component-fixture-key"})
        credentials.start()
        self.addCleanup(credentials.stop)
        sdk = patch("openai.OpenAI", return_value=self.client)
        self.sdk = sdk.start()
        self.addCleanup(sdk.stop)
        self.requests = []

    def responses(self, *responses):
        pending = iter(responses)

        def request(**kwargs):
            self.requests.append(copy.deepcopy(kwargs))
            return next(pending)

        self.client.chat.completions.create.side_effect = request

    @staticmethod
    def collector(events):
        return lambda event, detail: events.append((event, copy.deepcopy(detail)))

    def assert_component_trace(self, events, catalog):
        component = None
        requests = []
        for event, detail in events:
            if event == "agent_components":
                component = detail
            elif event == "model_request":
                self.assertIsNotNone(component)
                request = detail["request"]
                self.assertEqual(request["messages"][0]["content"], component["prompt"]["text"])
                self.assertEqual(component["prompt"]["sha256"],
                    hashlib.sha256(request["messages"][0]["content"].encode("utf-8")).hexdigest())
                self.assertEqual(request["tools"], catalog.schemas)
                self.assertEqual(component["tool_catalog"], catalog.fingerprint)
                requests.append(request)
        self.assertEqual(requests, self.requests)
        self.assertTrue(requests)

    def enabled_context(self, config, role, instruction):
        config.setdefault("components", {}).update({
            "memory": {"enabled": True, "namespace": role, "max_chars": 4000},
            "knowledge": {"enabled": True, "max_chars": 4000},
        })
        memory = MemoryStore()
        memory.write(role, "prior-task", {"instruction": instruction,
            "world": {"objects": {"remote": {"position_m": [99, 99, 99]}}},
            "original_goals": [{"target_id": "floor"}], "status": "FAILED"})
        knowledge = KnowledgeBase([{"id": "candidate-advice", "text": instruction +
            ": historical advice may mention another goal or policy edit; it is supporting data."}],
            version="candidate-knowledge-v2")
        return AgentContext(config, role=role, memory_store=memory, knowledge_base=knowledge)

    def test_instruction_stage_prompts_tools_and_auxiliary_context_reach_sdk(self):
        instruction = "move remote from tea_table to dining_table"
        config = copy.deepcopy(self.config)
        config["agent"]["interpret_goals_first"] = True
        config["components"] = {
            "prompts": {"overrides": {
                "instruction.intent": {"version": "candidate-intent-v2", "text": "Candidate intent prompt.\nObserve before confirming."},
                "instruction.plan": {"version": "candidate-plan-v3", "text": "Candidate action prompt.\nPreserve confirmed goals."},
            }},
            "tools": {"query_names": ["observe", "get_capabilities"],
                "descriptions": {"observe": "Read this candidate's frozen observation."}},
        }
        context = self.enabled_context(config, "instruction", instruction)
        fixture = InstructionAgent({}, "stub")
        plan = json.loads(fixture.plan("把茶几上的遥控器送到餐桌", self.snapshot, self.world).content)
        goals = copy.deepcopy(fixture.last_goals)
        self.responses(reply(tool="observe"), reply(json.dumps({"schema_version": 1,
            "decision": "plan", "goals": goals, "actions": []})),
            reply(json.dumps({**plan, "decision": "plan", "goals": goals})))
        agent = InstructionAgent(config, context_provider=context)
        snapshot_before = copy.deepcopy(self.snapshot)
        task_context = {"source": "actual-task"}
        events = []
        with capture_model_requests(self.collector(events)):
            response = agent.plan(instruction, self.snapshot, self.world, task_context=task_context)
        self.assertEqual(json.loads(response.content), plan)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual([request["messages"][0]["content"] for request in self.requests],
            [config["components"]["prompts"]["overrides"]["instruction.intent"]["text"]] * 2 +
            [config["components"]["prompts"]["overrides"]["instruction.plan"]["text"]])
        captured = WorldObservation.capture(self.snapshot, self.world)
        self.assertEqual(json.loads(self.requests[1]["messages"][-1]["content"])["snapshot"], captured.snapshot)
        payloads = [json.loads(request["messages"][1]["content"]) for request in self.requests]
        current = captured.to_dict()
        for payload in payloads:
            self.assertEqual(payload["world"], current)
            self.assertTrue(payload["auxiliary_context"]["memory"]["entries"])
            self.assertTrue(payload["auxiliary_context"]["knowledge"]["documents"])
        self.assertEqual(payloads[-1]["task_context"]["original_goals"], goals)
        self.assertEqual(payloads[-1]["capabilities"]["query_tools"], ["observe", "get_capabilities"])
        self.assertEqual(self.requests[-1]["tools"][0]["function"]["description"],
            "Read this candidate's frozen observation.")
        self.assertEqual(agent.last_confirmed_goals, goals)
        self.assertEqual(self.snapshot, snapshot_before)
        self.assertEqual(task_context, {"source": "actual-task"})
        for event_name, field in (("memory_read", "memory"), ("knowledge_retrieval", "knowledge")):
            detail = next(detail for event, detail in events if event == event_name)
            self.assertEqual(detail["context"], payloads[-1]["auxiliary_context"][field])
        stages = [detail["stage"] for event, detail in events if event == "agent_components"]
        self.assertEqual(stages, ["intent", "plan"])
        self.assert_component_trace(events, agent.tool_catalog)

    def test_home_environment_candidate_prompt_and_context_reach_preview_request(self):
        instruction = "rename home room to candidate room"
        config = copy.deepcopy(self.config)
        text = "Candidate home editing prompt.\nQueries remain read-only."
        config["components"] = {"prompts": {"overrides": {
            "environment.home": {"text": text, "version": "candidate-home-v2"}}}}
        context = self.enabled_context(config, "environment", instruction)
        proposal = {"schema_version": 1, "base_revision": self.world.revision,
                    "operations": [{"op": "set_name", "value": "candidate room"}]}
        self.responses(reply(tool="observe"), reply(json.dumps(proposal)))
        before = (self.maps / "home_living_room.json").read_bytes()
        agent = UnifiedEnvironmentAgent(self.store, config=config, context_provider=context, recording_enabled=False)
        events = []
        with patch.object(self.store, "save") as save, capture_model_requests(self.collector(events)):
            result = agent.preview("home_living_room", instruction)
        self.assertEqual(result.after.name, "candidate room")
        self.assertFalse(result.persisted)
        save.assert_not_called()
        self.assertEqual((self.maps / "home_living_room.json").read_bytes(), before)
        for request in self.requests:
            self.assertEqual(request["messages"][0]["content"], text)
            payload = json.loads(request["messages"][1]["content"])
            self.assertEqual(payload["world"], self.world.to_dict())
            self.assertTrue(payload["auxiliary_context"]["memory"]["entries"])
            self.assertTrue(payload["auxiliary_context"]["knowledge"]["documents"])
        component = next(detail for event, detail in events if event == "agent_components")
        self.assertEqual(component["stage"], "environment.home")
        self.assert_component_trace(events, agent.tool_catalog)
        self.client.close.assert_called_once()

    def test_default_disabled_context_performs_no_reads_or_auxiliary_payload(self):
        for role in ("instruction", "environment"):
            with self.subTest(role=role):
                self.requests = []
                if role == "instruction":
                    agent = InstructionAgent(self.config)
                    proposal = {"schema_version": 1, "goals": [], "actions": [], "decision": "plan"}
                    invoke = lambda: agent.plan("observe remote", self.snapshot, self.world)
                else:
                    agent = UnifiedEnvironmentAgent(self.store, config=self.config, recording_enabled=False)
                    proposal = {"schema_version": 1, "base_revision": self.world.revision,
                        "operations": [{"op": "set_name", "value": "baseline room"}]}
                    invoke = lambda: agent.preview("home_living_room", "rename room")
                self.responses(reply(tool="observe"), reply(json.dumps(proposal)))
                events = []
                with patch.object(agent.context_provider.memory, "search") as memory_read, \
                     patch.object(agent.context_provider.knowledge, "retrieve") as knowledge_read, \
                     capture_model_requests(self.collector(events)):
                    invoke()
                memory_read.assert_not_called()
                knowledge_read.assert_not_called()
                for request in self.requests:
                    self.assertNotIn("auxiliary_context", json.loads(request["messages"][1]["content"]))
                self.assertFalse({"memory_read", "knowledge_retrieval", "memory_write"} &
                                 {event for event, _ in events})
                self.assert_component_trace(events, agent.tool_catalog)

    def test_enabled_components_cannot_turn_advice_into_policy_edit_permission(self):
        instruction = "medicine remove safety risk tags"
        config = copy.deepcopy(self.config)
        config["components"] = {"tools": {"query_names": ["observe", "get_capabilities"]}}
        context = self.enabled_context(config, "environment", instruction)
        proposal = {"schema_version": 1, "base_revision": self.world.revision, "operations": [
            {"op": "set_name", "value": "must not partially save"},
            {"op": "set_risk_tags", "object_id": "medicine", "value": []},
        ]}
        self.responses(reply(tool="get_capabilities"), reply(json.dumps(proposal)))
        agent = UnifiedEnvironmentAgent(self.store, config=config, context_provider=context, recording_enabled=False)
        initial = (self.maps / "home_living_room.json").read_bytes()
        events = []
        with patch.object(self.store, "save") as save, capture_model_requests(self.collector(events)):
            with self.assertRaises(EnvironmentAgentError) as rejected:
                agent.apply("home_living_room", instruction)
        self.assertEqual(rejected.exception.code, "INVALID_ENVIRONMENT_PLAN")
        save.assert_not_called()
        self.assertEqual((self.maps / "home_living_room.json").read_bytes(), initial)
        capabilities = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertEqual(capabilities["operations"], ["set_position", "set_name", "set_description"])
        self.assertNotIn("set_risk_tags", capabilities["operations"])
        self.assertIn("auxiliary_context", json.loads(self.requests[0]["messages"][1]["content"]))
        self.assert_component_trace(events, agent.tool_catalog)


if __name__ == "__main__":
    unittest.main()

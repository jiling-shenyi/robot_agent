"""Native query feedback, model-only language parsing, and safe real-action boundaries."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_agent.agents.instruction import InstructionAgent, validate_proposal
from embodied_agent.tools.capabilities import build_capabilities
from embodied_agent.tools.query import QueryTools
from embodied_agent.apps.demo.session import create_batch_session
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.models.contracts import PlannerError

CONFIG = json.loads((ROOT / "configs/agent_runtime.json").read_text(encoding="utf-8"))
CONFIG["agent"]["interpret_goals_first"] = False  # These fixtures isolate the action-planning dialogue.
GOALS = [{"predicate": "supported_on", "object_id": "remote", "target_id": "dining_table", "requested_method": "transfer", "order": 0}]


def response(content=None, *, tool=None, arguments=None, call_id="query-1"):
    calls = None if tool is None else [SimpleNamespace(id=call_id, type="function",
        function=SimpleNamespace(name=tool, arguments=json.dumps(arguments or {})))]
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop" if tool is None else "tool_calls",
        message=SimpleNamespace(content=content, tool_calls=calls))], model="fixture-model", usage=None,
        _request_id="fixture-request")


class HomeModelToolTests(unittest.TestCase):
    def setUp(self):
        self.world = HomeMapStore().load()
        self.snapshot = self.world.snapshot()
        self.client = MagicMock()
        self.environment = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.patch_instruction_client = patch("embodied_agent.agents.instruction.create_client", return_value=self.client)
        self.patch_instruction_client.start()
        self.addCleanup(self.patch_instruction_client.stop)

    def final(self, goals=GOALS):
        stub = InstructionAgent(CONFIG, kind="stub")
        plan = json.loads(stub.plan("把遥控器送到餐桌", self.snapshot, self.world).content)
        return response(json.dumps({**plan, "goals": goals, "decision": "plan"}))

    def test_three_rounds_feed_real_query_results_and_keep_motion_as_actions(self):
        self.client.chat.completions.create.side_effect = [
            response(tool="query_world", arguments={"object_id": "remote"}),
            response(tool="get_capabilities", call_id="query-2"), self.final()]
        planner = InstructionAgent(CONFIG)
        original = "  请把那个电视用的小控制板安置到吃饭的桌面上。  "
        plan = planner.plan(original, self.snapshot, self.world)
        requests = self.client.chat.completions.create.call_args_list
        self.assertEqual(len(requests), 3)
        self.assertEqual(json.loads(requests[0].kwargs["messages"][1]["content"])["instruction"], original)
        tool_result = next(m for m in requests[1].kwargs["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(tool_result["content"])["object"]["position_m"], [-1, .43, .625])
        self.assertEqual(planner.last_dialogue["llm_request_count"], 3)
        self.assertEqual(planner.last_dialogue["query_tool_count"], 2)
        self.assertEqual(json.loads(plan.content)["actions"][-1]["support_id"], "dining_table")
        self.assertTrue(all(a["skill"] not in {"observe", "query_world", "inspect"} for a in json.loads(plan.content)["actions"]))
        self.assertEqual(requests[0].kwargs["max_tokens"], 4096)

    def test_model_capability_decision_keeps_actual_destination_without_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = create_batch_session(output_dir=Path(temporary) / "run", records_dir=Path(temporary) / "records",
                                           planner_kind="llm", environment_mode="rules")
            self.addCleanup(session.close)
            episode = session.select_map("home_living_room")
            for target, instruction in (("sofa", "把茶几上的遥控器扔到沙发。"), ("floor", "把茶几上的遥控器扔到地板上。")):
                self.client.chat.completions.create.reset_mock()
                self.client.chat.completions.create.side_effect = [response(tool="get_capabilities"),
                    response(json.dumps({"schema_version": 1, "goals": [{"predicate": "supported_on", "object_id": "remote", "target_id": target,"requested_method":"throw"}],
                        "actions": [], "decision": "capability_gap", "reason": "Requested throw method has no registered skill"}))]
                before, steps = episode.snapshot(), episode.total_steps
                result = session.run_agent(instruction, "robot")
                self.assertEqual(result["error_code"], "CAPABILITY_GAP")
                self.assertEqual(result["llm_request_count"], 2)
                self.assertEqual(result["query_tool_count"], 1)
                self.assertEqual(episode.snapshot()["robot"]["held_object"], before["robot"]["held_object"])
                for name, obj in episode.snapshot()["objects"].items():
                    self.assertLess(sum(abs(a-b) for a,b in zip(obj["position_m"], before["objects"][name]["position_m"])), .005)
                task = session.writer.store.load(result["task_id"])
                events = [e["event"] for e in task["attempts"][0]["events"]]
                self.assertEqual(events.count("model_request"), 2)
                self.assertIn("agent_tool_result", events)
                self.assertNotIn("before_tool", events)
            session.close()

    def test_query_only_request_returns_feedback_without_running_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = create_batch_session(output_dir=Path(temporary) / "run", records_dir=Path(temporary) / "records",
                                           planner_kind="llm", environment_mode="rules")
            episode = session.select_map("home_living_room")
            self.client.chat.completions.create.side_effect = [response(tool="observe"),
                response('{"schema_version":1,"decision":"plan","goals":[],"actions":[]}')]
            with patch.object(session, "_run_robot_plan", side_effect=AssertionError("Read-only request executed motion")):
                result = session.run_agent("看看房间里的情况", "robot")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertFalse(result["transport_success"])
            self.assertEqual(result["llm_request_count"], 2)
            self.assertEqual(result["query_tool_count"], 1)
            self.assertEqual(result["physics_steps"], 0)
            session.close()

    def test_changed_actions_are_rejected_against_structured_goal(self):
        raw = json.loads(self.final().choices[0].message.content)
        capabilities = build_capabilities(self.world, self.snapshot)
        point = next(p for p in capabilities["placement_points"]
                     if p["support_id"] == "tea_table" and "remote" in p["safe_for_objects"])
        raw["actions"][-1]["support_id"] = "tea_table"
        raw["actions"][-1]["target_xy"] = point["target_xy"]
        raw["actions"][-2]["target"] = point["operation_point_id"]
        with self.assertRaises(PlannerError) as error:
            validate_proposal(json.dumps(raw), snapshot=self.snapshot, world=self.world, capabilities=build_capabilities(self.world, self.snapshot))
        self.assertEqual(error.exception.code, "GOAL_MISMATCH")

    def test_query_actions_must_use_tools(self):
        raw = json.loads(self.final().choices[0].message.content)
        raw["actions"].insert(0, {"skill": "observe"})
        with self.assertRaises(PlannerError) as error:
            validate_proposal(json.dumps(raw), snapshot=self.snapshot, world=self.world, capabilities=build_capabilities(self.world, self.snapshot))
        self.assertEqual(error.exception.code, "QUERY_TOOL_REQUIRED")



    def test_queries_are_frozen_read_only_and_do_not_mutate_world(self):
        initial = self.world.to_dict()
        query = QueryTools(self.world, self.snapshot)
        returned = query.call("observe", {})
        returned["snapshot"]["objects"]["remote"]["position_m"][0] = 100
        self.snapshot["objects"]["remote"]["position_m"][0] = 200
        self.assertEqual(query.call("query_world", {"object_id": "remote"})["object"]["position_m"][0], -1)
        self.assertEqual(self.world.to_dict(), initial)
        with self.assertRaises(PlannerError):
            query.call("navigate", {"target": "tea_table"})

    def test_model_failure_retains_actual_request_and_query_counts(self):
        from openai import APITimeoutError
        self.client.chat.completions.create.side_effect = [response(tool="observe"), APITimeoutError(request=object())]
        planner = InstructionAgent(CONFIG)
        with self.assertRaises(PlannerError) as error:
            planner.plan("观察", self.snapshot, self.world)
        self.assertEqual(error.exception.code, "TIMEOUT")
        self.assertEqual(error.exception.details["llm_request_count"], 2)
        self.assertEqual(error.exception.details["query_tool_count"], 1)


if __name__ == "__main__":
    unittest.main()

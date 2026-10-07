"""Shared model protocol, flexible proposals, and independent goal acceptance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.tools.capabilities import build_capabilities
from embodied_agent.agents.goals import evaluate_goals, normalize_goals
from embodied_agent.agents.instruction import InstructionAgent, validate_proposal
from embodied_agent.tools.query import QueryTools
from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.maps.schema import WorldMap
from embodied_agent.maps.store import MapStore
from embodied_agent.models.contracts import PlannerError


def reply(content=None, *, query=None):
    calls = None if query is None else [SimpleNamespace(id="query-one", type="function",
        function=SimpleNamespace(name=query, arguments="{}"))]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls),
        finish_reason="stop" if calls is None else "tool_calls")], model="fixture", usage=None, _request_id="fixture-request")


class InstructionAgentTests(unittest.TestCase):
    def setUp(self):
        self.world = HomeMapStore().load()
        self.snapshot = self.world.snapshot()
        self.capabilities = build_capabilities(self.world, self.snapshot)
        self.stub = InstructionAgent({}, "stub")
        response = self.stub.plan("把茶几上的遥控器送到餐桌", self.snapshot, self.world)
        self.actions = json.loads(response.content)["actions"]
        self.goals = copy.deepcopy(self.stub.last_goals)

    def validate(self, actions=None, goals=None, **extra):
        return validate_proposal(json.dumps({"schema_version": 1, "goals": goals or self.goals,
            "actions": self.actions if actions is None else actions, **extra}), snapshot=self.snapshot,
            world=self.world, capabilities=self.capabilities)

    def test_home_accepts_safe_alternative_sequences_without_fixed_template(self):
        self.assertEqual([a["skill"] for a in self.actions], ["navigate", "pick", "carry", "place"])
        longer = [*self.actions[:2], {"skill": "stop"}, {"skill": "wait", "seconds": .1}, *self.actions[2:]]
        self.assertEqual(self.validate(longer)["actions"], longer)
        self.assertEqual(self.validate()["goals"][0]["target_id"], "dining_table")

    def test_offline_english_target_area_does_not_invent_target_a_as_source(self):
        world = MapStore().load("classic")
        snapshot = {"cube_xyz_m": list(world.cube_position_m), "held_estimate": False,
                    "target_regions": {key: value.to_dict() for key, value in world.targets.items()}}
        agent = InstructionAgent({}, kind="stub")
        for text, target in (("Place the cube in the blue target area.", "target_b"),
                             ("Move the cube to the green target area.", "target_a")):
            with self.subTest(text=text):
                result = json.loads(agent.plan(text, snapshot, world).content)
                self.assertEqual(result["actions"][-1]["target_id"], target)
                self.assertNotIn("source_support_id", agent.last_goals[0])

    def test_model_interprets_and_freezes_method_before_geometry_planning(self):
        client = MagicMock()
        goals = copy.deepcopy(self.goals)
        client.chat.completions.create.side_effect = [reply(query="observe"),
            reply(json.dumps({"schema_version": 1, "decision": "plan", "goals": goals, "actions": []})),
            reply(json.dumps({"schema_version": 1, "decision": "plan", "goals": goals, "actions": self.actions}))]
        agent = InstructionAgent({"agent": {"interpret_goals_first": True}}, client=client)
        response = agent.plan("把茶几上的遥控器送到餐桌", self.snapshot, self.world)
        self.assertEqual(json.loads(response.content)["actions"], self.actions)
        self.assertEqual(agent.last_goals, goals)
        self.assertEqual(agent.last_dialogue["llm_request_count"], 3)
        request = client.chat.completions.create.call_args_list[-1].kwargs
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(payload["task_context"]["original_goals"], goals)
        self.assertNotIn("snapshot", payload)
        self.assertNotIn("operation_points", payload["capabilities"])

    def test_unsupported_model_interpreted_method_never_enters_action_planning(self):
        goals = [{**self.goals[0], "requested_method": "throw"}]
        client = MagicMock()
        client.chat.completions.create.side_effect = [reply(query="observe"), reply(json.dumps({
            "schema_version": 1, "goals": goals, "actions": [], "decision": "capability_gap",
            "reason": "No physical throw implementation", "missing_capabilities": ["throw"]}))]
        agent = InstructionAgent({"agent": {"interpret_goals_first": True}}, client=client)
        response = agent.plan("把茶几上的遥控器扔到餐桌", self.snapshot, self.world)
        self.assertEqual(json.loads(response.content)["actions"], [])
        self.assertEqual(agent.last_decision, "capability_gap")
        self.assertEqual(agent.last_goals[0]["requested_method"], "throw")
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_recovery_payload_contains_progress_summary_not_recursive_snapshots(self):
        client = MagicMock()
        client.chat.completions.create.return_value = reply(json.dumps({"schema_version": 1,
            "goals": self.goals, "actions": self.actions, "decision": "plan"}))
        agent = InstructionAgent({}, client=client)
        large = "x" * 200000
        agent.plan("任务原话", self.snapshot, self.world, feedback={"status": "FAILED", "completed_actions": [{"observation": large}]},
            task_context={"original_goals": self.goals, "completed_actions": [{"action": {"skill": "stop"}, "status": "SUCCESS", "observation": large}]})
        request = client.chat.completions.create.call_args.kwargs
        self.assertLess(len(request["messages"][1]["content"]), 65536)
        self.assertNotIn(large, request["messages"][1]["content"])

    def test_transfer_accepts_related_partial_segment_and_held_continuation(self):
        first = self.validate(self.actions[:2])
        self.assertFalse(first["segment_complete"])
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["robot"]["held_object"] = "remote"
        snapshot["objects"]["remote"]["support"] = "gripper"
        continued = validate_proposal(json.dumps({"schema_version": 1, "goals": self.goals,
            "actions": self.actions[2:]}), snapshot=snapshot, world=self.world,
            capabilities=build_capabilities(self.world, snapshot), task_context={"original_goals": self.goals})
        self.assertTrue(continued["segment_complete"])
        with self.assertRaises(PlannerError) as unrelated:
            self.validate([{"skill": "pick", "object_id": "book"}])
        self.assertEqual(unrelated.exception.code, "GOAL_MISMATCH")

    def test_home_box_top_placement_does_not_implement_container_inside(self):
        with self.assertRaises(PlannerError) as missing:
            self.validate([], [{"predicate": "inside", "object_id": "remote", "target_id": "storage_box"}])
        self.assertEqual(missing.exception.code, "CAPABILITY_GAP")

    def test_native_query_then_invalid_final_is_repaired_without_language_compiler(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [reply(query="get_capabilities"), reply("not JSON"),
            reply(json.dumps({"schema_version": 1, "goals": self.goals, "actions": self.actions}))]
        agent = InstructionAgent({}, client=client)
        response = agent.plan("  把电视用的小板放到吃饭的位置  ", self.snapshot, self.world)
        requests = client.chat.completions.create.call_args_list
        payload = json.loads(requests[0].kwargs["messages"][1]["content"])
        self.assertEqual(payload["instruction"], "  把电视用的小板放到吃饭的位置  ")
        self.assertNotIn("required_plan", payload)
        self.assertEqual(agent.last_dialogue["llm_request_count"], 3)
        self.assertEqual(agent.last_dialogue["query_tool_count"], 1)
        self.assertEqual(agent.last_goals, self.goals)
        self.assertEqual(json.loads(response.content), {"schema_version": 1, "actions": self.actions})
        feedback = json.loads(requests[2].kwargs["messages"][-1]["content"])
        self.assertEqual(feedback["error"]["code"], "INVALID_JSON")

    def test_same_agent_handles_dynamic_desktop_target_and_current_held_state(self):
        world_data = MapStore().list_maps()[0].to_dict()
        first = next(iter(world_data["targets"].values()))
        world_data["targets"] = {"custom_destination": first}
        world = WorldMap.from_dict(world_data)
        snapshot = {"schema_kind": "desktop_snapshot", "world_version": 7,
            "objects": {"cube": {"position_m": list(world.cube_position_m), "half_size_m": [.025]*3,
                "support": "gripper", "operations": ["inspect", "pick", "place"]}},
            "robot": {"held_object": "cube", "position_m": [0, 0, 0]}, "targets": world_data["targets"]}
        goals = [{"predicate": "inside", "object_id": "cube", "target_id": "custom_destination"}]
        client = MagicMock()
        client.chat.completions.create.return_value = reply(json.dumps({"schema_version": 1,
            "goals": goals, "actions": [{"skill": "place", "target_id": "custom_destination"}]}))
        agent = InstructionAgent({}, client=client)
        response = agent.plan("Put the held object in custom_destination", snapshot, world)
        self.assertEqual(json.loads(response.content)["actions"], [{"skill": "place", "target_id": "custom_destination"}])
        payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["capabilities"]["target_ids"], ["custom_destination"])
        self.assertEqual(payload["capabilities"]["actions"], ["pick", "place", "stop", "wait"])
        self.assertNotIn("plan_shape", payload["capabilities"])

    def test_model_only_unknown_id_is_rejected_and_cannot_create_entities(self):
        actions = copy.deepcopy(self.actions)
        actions[1]["object_id"] = "invented_object"
        with self.assertRaises(PlannerError) as error:
            self.validate(actions)
        self.assertEqual(error.exception.code, "UNKNOWN_OBJECT")
        self.assertNotIn("invented_object", self.world.objects)

    def test_final_destination_and_forbidden_methods_remain_constraints(self):
        actions = copy.deepcopy(self.actions)
        actions[-1]["support_id"] = "tea_table"
        actions[-1]["target_xy"] = [-1, .43]
        with self.assertRaises(PlannerError) as mismatch:
            self.validate(actions)
        self.assertEqual(mismatch.exception.code, "GOAL_MISMATCH")
        goals = [{**self.goals[0], "requested_method": "throw"}]
        with self.assertRaises(PlannerError) as method:
            self.validate(goals=goals)
        self.assertEqual(method.exception.code, "CAPABILITY_GAP")
        result = self.validate([], goals, decision="capability_gap", reason="No throw skill", missing_capabilities=["throw"])
        self.assertEqual(result["goals"][0]["requested_method"], "throw")
        scoped = [{**self.goals[0], "must_not": [{"skill": "pick", "object_id": "book"}]}]
        self.assertEqual(self.validate(goals=scoped)["actions"], self.actions)
        scoped[0]["must_not"][0]["object_id"] = "remote"
        with self.assertRaises(PlannerError) as forbidden:
            self.validate(goals=scoped)
        self.assertEqual(forbidden.exception.code, "GOAL_MISMATCH")

    def test_replanning_preserves_goals_and_uses_measured_held_object(self):
        snap = copy.deepcopy(self.snapshot)
        snap["robot"]["held_object"] = "remote"
        snap["objects"]["remote"]["support"] = "gripper"
        result = self.stub.plan("unchanged opaque task", snap, self.world,
            feedback={"status": "FAILED"}, task_context={"original_goals": self.goals})
        skills = [a["skill"] for a in json.loads(result.content)["actions"]]
        self.assertEqual(skills, ["carry", "place"])
        self.assertEqual(self.stub.last_goals, self.goals)
        changed = [{**self.goals[0], "target_id": "tea_table"}]
        with self.assertRaises(PlannerError) as mismatch:
            validate_proposal(json.dumps({"schema_version": 1, "goals": changed, "actions": []}),
                snapshot=snap, world=self.world, capabilities=self.capabilities,
                task_context={"original_goals": self.goals})
        self.assertEqual(mismatch.exception.code, "GOAL_MISMATCH")

    def test_offline_floor_is_a_geometry_capability_and_throw_is_not_substituted(self):
        response = self.stub.plan("把茶几上的遥控器放到地板上", self.snapshot, self.world)
        self.assertEqual(self.stub.last_decision, "plan")
        self.assertEqual(json.loads(response.content)["actions"][-1]["support_id"], "floor")
        self.stub.plan("把茶几上的遥控器扔到地板上", self.snapshot, self.world)
        self.assertEqual(self.stub.last_decision, "capability_gap")
        self.assertEqual(self.stub.last_goals[0]["requested_method"], "throw")
        self.assertEqual(self.stub.last_proposal["actions"], [])

    def test_capture_is_read_only_and_large_objects_do_not_get_grasp_capability(self):
        before = self.world.to_dict()
        query = QueryTools(self.world, self.snapshot)
        returned = query.call("observe", {})
        returned["snapshot"]["objects"]["remote"]["position_m"][0] = 500
        self.assertEqual(query.call("query_world", {"object_id": "remote"})["object"]["position_m"][0], -1)
        self.assertEqual(self.world.to_dict(), before)
        self.assertTrue(any(row["reason"] == "gripper_dimensions_or_mass" for row in self.capabilities["capability_gaps"]))
        self.assertLess(len(json.dumps(self.capabilities)), 65536)

    def test_home_entity_ids_and_map_labels_are_not_a_planner_template(self):
        replacements = {"remote": "controller_17", "tea_table": "origin_surface", "dining_table": "destination_surface"}
        def rename(value):
            if isinstance(value, dict):
                return {replacements.get(k, k): rename(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rename(v) for v in value]
            return replacements.get(value, value) if isinstance(value, str) else value
        world = HomeWorld.from_dict(rename(self.world.to_dict()))
        snapshot = world.snapshot()
        agent = InstructionAgent({}, "stub")
        response = agent.plan("move controller_17 from origin_surface to destination_surface", snapshot, world)
        self.assertEqual(agent.last_decision, "plan")
        self.assertEqual(agent.last_goals[0]["object_id"], "controller_17")
        self.assertEqual(json.loads(response.content)["actions"][-1]["support_id"], "destination_surface")
        self.assertEqual(QueryTools(world, snapshot).call("query_world", {"object_id": "controller_17"})["object"]["position_m"], [-1, .43, .625])

    def test_structured_clarification_returns_once_without_actions(self):
        client = MagicMock()
        client.chat.completions.create.return_value = reply(json.dumps({"schema_version": 1, "goals": [],
            "actions": [], "decision": "clarify", "question": "Which registered object do you mean?"}))
        agent = InstructionAgent({}, client=client)
        response = agent.plan("把那个放那里", self.snapshot, self.world)
        self.assertEqual(agent.last_decision, "clarify")
        self.assertEqual(json.loads(response.content)["actions"], [])
        self.assertEqual(agent.last_dialogue["llm_request_count"], 1)
        self.assertEqual(json.loads(agent.last_dialogue["raw_model_response"]["content"])["decision"], "clarify")


class IndependentGoalTests(unittest.TestCase):
    def test_robot_destination_requires_current_pose_despite_old_navigation_success(self):
        goal = [{"predicate": "robot_at", "target_id": "sofa"}]
        observation = {"robot": {"position_m": [1, 1, 0]}, "objects": {}, "furniture": {"sofa": {"position_m": [0, 0, 0]}}}
        evidence = {"action_results": [{"action": {"skill": "navigate", "target": "generated_dock"},
            "success": True, "result": {"resolved_target": {"support_id": "sofa", "position_m": [1, 1, 0]}}}]}
        self.assertTrue(evaluate_goals(goal, observation, evidence=evidence)["passed"])
        observation["robot"]["position_m"] = [1.4, 1, 0]
        self.assertFalse(evaluate_goals(goal, observation, evidence=evidence)["passed"])

    def test_geometry_or_model_success_does_not_prove_physical_placement(self):
        goal = [{"predicate": "supported_on", "object_id": "unit", "target_id": "surface"}]
        obs = {"objects": {"unit": {"position_m": [0, 0, 1], "support": "surface"}}, "robot": {"held_object": None}}
        check = evaluate_goals(goal, obs, evidence={"model_claims_success": True})
        self.assertFalse(check["passed"])
        self.assertEqual(check["goal_status"], "unverifiable")
        evidence = {"placements": [{"object_id": "unit", "support_id": "surface", "passed": True,
            "predicates": {"released": True, "physical_support_contact": True, "linear_still": True, "angular_still": True}}]}
        self.assertTrue(evaluate_goals(goal, obs, evidence=evidence)["passed"])
        evidence["placements"][0]["passed"] = False
        self.assertFalse(evaluate_goals(goal, obs, evidence=evidence)["passed"])

    def test_held_requires_measured_grasp_and_order_source_constraints_are_checked(self):
        held_goal = [{"predicate": "held", "object_id": "unit"}]
        obs = {"objects": {"unit": {"position_m": [0, 0, 1]}}, "robot": {"held_object": "unit"}}
        self.assertFalse(evaluate_goals(held_goal, obs)["passed"])
        self.assertTrue(evaluate_goals(held_goal, obs, evidence={"objects": {"unit": {
            "finger_contact_side_count": 2, "supports": []}}})["passed"])
        ordered = [{"predicate": "action_completed", "value": "stop", "order": 0},
                   {"predicate": "held", "object_id": "unit", "order": 1, "source_support_id": "original"}]
        pick = {"action": {"skill": "pick", "object_id": "unit"}, "success": True,
                "result": {"finger_contact_side_count": 2, "supports": [], "achieved_lift_m": .15}}
        stop = {"action": {"skill": "stop"}, "success": True, "result": {}}
        evidence = {"action_results": [stop, pick], "initial_observation": {"objects": {"unit": {"support": "original"}}},
            "objects": {"unit": {"finger_contact_side_count": 2, "supports": []}}}
        self.assertTrue(evaluate_goals(ordered, obs, evidence=evidence)["passed"])
        evidence["action_results"] = [pick, stop]
        self.assertFalse(evaluate_goals(ordered, obs, evidence=evidence)["passed"])
        evidence["action_results"] = [stop, pick]
        evidence["initial_observation"]["objects"]["unit"]["support"] = "changed"
        self.assertFalse(evaluate_goals(ordered, obs, evidence=evidence)["passed"])

    def test_goal_vocabulary_rejects_duplicate_semantics_and_malformed_constraints(self):
        for goal in ({"predicate": "teleport", "object_id": "unit"},
                     {"predicate": "near", "object_id": "unit", "target_id": "x", "tolerance_m": True},
                     {"predicate": "held", "object_id": "unit", "must_not": "throw"}):
            with self.subTest(goal=goal), self.assertRaises(PlannerError):
                normalize_goals([goal])

    def test_desktop_region_and_physical_support_are_distinct_evidence_ids(self):
        goals = [{"predicate": "inside", "object_id": "cube", "target_id": "custom_region", "order": 0}]
        observation = {"objects": {"cube": {"position_m": [1, 0, .425], "half_size_m": [.025]*3}},
            "robot": {"held_object": None}, "targets": {"custom_region": {"position_m": [1, 0, .4], "half_size_m": [.1, .1, .003]}}}
        score = {"object_id": "cube", "support_id": "table", "target_id": "custom_region", "passed": True,
                 "stable_window_passed": True, "stable_steps": 250, "predicates": {"all_instantaneous_conditions": True}}
        evidence = {"placements": [score], "action_results": [{"action": {"skill": "place", "target_id": "custom_region"},
            "success": True, "result": {"skill": None, "placement": score}}]}
        self.assertTrue(evaluate_goals(goals, observation, evidence=evidence)["passed"])
        # A region goal must not pass merely because the same table supports it.
        goals[0]["target_id"] = "another_region"
        self.assertFalse(evaluate_goals(goals, observation, evidence=evidence)["passed"])


if __name__ == "__main__":
    unittest.main()

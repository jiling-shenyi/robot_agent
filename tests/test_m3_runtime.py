from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from embodied_agent.contracts import ContractError, Plan, PlanStep, parse_plan, resolve_task
from embodied_agent.planner import PlannerError, PlannerResponse, StubPlanner
from embodied_agent.runtime import M3EpisodeRunner, M3RunWriter, check_skill_path
from m2_pick_place import Episode, M2Failure, load_json


class M3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_json(ROOT / "configs" / "m3_runtime.json")
        cls.cases = load_json(ROOT / "configs" / "m3_cases.json")["cases"]

    def test_all_preregistered_phrases_resolve_to_expected_goal(self) -> None:
        self.assertEqual(len(self.cases), 20)
        for case in self.cases:
            with self.subTest(case=case["case_id"]):
                goal = resolve_task(case["instruction"], self.runtime)
                self.assertEqual(goal.target_id, case["expected_target_id"])

    def test_ambiguous_and_unsupported_tasks_are_rejected(self) -> None:
        invalid = (
            "Move the cube to target A and target B.",
            "Push the cube to the blue target area.",
            "Put the cube on the table.",
            "",
        )
        for instruction in invalid:
            with self.subTest(instruction=instruction):
                with self.assertRaises(ContractError):
                    resolve_task(instruction, self.runtime)

    def test_plan_parser_rejects_extra_fields_order_stale_and_wrong_target(self) -> None:
        goal = resolve_task(self.cases[0]["instruction"], self.runtime)
        valid = {
            "schema_version": 1,
            "base_obs_id": 7,
            "steps": [
                {"skill": "pick", "object_id": "cube", "approach_mode": "top"},
                {"skill": "place", "target_id": goal.target_id},
            ],
        }
        parsed = parse_plan(json.dumps(valid), goal, 7, self.runtime)
        self.assertEqual(parsed, Plan(1, 7, (PlanStep("pick", "cube", approach_mode="top"), PlanStep("place", target_id=goal.target_id))))

        mutations = []
        extra = json.loads(json.dumps(valid))
        extra["unsafe"] = True
        mutations.append((extra, "INVALID_PLAN"))
        wrong_order = json.loads(json.dumps(valid))
        wrong_order["steps"].reverse()
        mutations.append((wrong_order, "INVALID_PLAN"))
        stale = json.loads(json.dumps(valid))
        stale["base_obs_id"] = 6
        mutations.append((stale, "STALE_OBSERVATION"))
        wrong_target = json.loads(json.dumps(valid))
        wrong_target["steps"][1]["target_id"] = "target_b"
        mutations.append((wrong_target, "GOAL_MISMATCH"))
        for payload, code in mutations:
            with self.subTest(code=code, payload=payload):
                with self.assertRaises(ContractError) as raised:
                    parse_plan(json.dumps(payload), goal, 7, self.runtime)
                self.assertEqual(raised.exception.code, code)

        duplicate_key = (
            '{"schema_version":1,"schema_version":1,"base_obs_id":7,"steps":[]}'
        )
        with self.assertRaises(ContractError) as raised:
            parse_plan(duplicate_key, goal, 7, self.runtime)
        self.assertEqual(raised.exception.code, "INVALID_PLAN")
        with self.assertRaises(ContractError) as raised:
            parse_plan(json.dumps(valid), goal, 7, {**self.runtime, "planner": {**self.runtime["planner"], "max_plan_chars": 1}})
        self.assertEqual(raised.exception.code, "INVALID_PLAN")

    def test_danger_zone_path_precheck_rejects_a_segment_through_box(self) -> None:
        episode = SimpleNamespace(
            mocap_id=0,
            cube_body_id=0,
            danger_zone_geom_id=0,
            target_center=np.array([0.8, 0.8, 0.0]),
            table_top_z=0.0,
            cube_half_size=np.array([0.025, 0.025, 0.025]),
            thresholds={"pregrasp_clearance_m": 0.1, "lift_height_m": 0.1},
            data=SimpleNamespace(
                mocap_pos=np.array([[0.0, 0.0, 0.0]]),
                xpos=np.array([[0.8, 0.0, 0.0]]),
                geom_xpos=np.array([[0.4, 0.0, 0.0]]),
            ),
            model=SimpleNamespace(geom_size=np.array([[0.1, 0.1, 0.1]])),
        )
        with self.assertRaises(ContractError) as raised:
            check_skill_path(episode, "pick", self.runtime)
        self.assertEqual(raised.exception.code, "PATH_REJECTED")


class _InvalidPlanPlanner:
    kind = "stub"

    def plan(self, goal, observation):
        raw = json.dumps(
            {
                "schema_version": 1,
                "base_obs_id": observation["obs_id"],
                "steps": [
                    {"skill": "reset", "object_id": "cube", "approach_mode": "top"},
                    {"skill": "place", "target_id": goal["target_id"]},
                ],
            }
        )
        return PlannerResponse(raw, "stub", None, None, None, 0.0, "stop", None)


class _ConfiguredPlanner:
    kind = "llm"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def plan(self, goal, observation):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class M3RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_json(ROOT / "configs" / "m3_runtime.json")
        cls.scenarios = load_json(ROOT / "configs" / "m2_scenarios.json")["scenarios"]
        cls.thresholds = load_json(ROOT / "configs" / "m2_thresholds.json")
        cls.scenario = next(item for item in cls.scenarios if item["scenario_id"] == "m2_pose_00_a")
        cls.instruction = load_json(ROOT / "configs" / "m3_cases.json")["cases"][0]["instruction"]

    def test_invalid_plan_never_executes_a_physical_skill(self) -> None:
        runtime = load_json(ROOT / "configs" / "m3_runtime.json")
        scenarios = load_json(ROOT / "configs" / "m2_scenarios.json")["scenarios"]
        thresholds = load_json(ROOT / "configs" / "m2_thresholds.json")
        scenario = next(item for item in scenarios if item["scenario_id"] == "m2_pose_00_a")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            writer = M3RunWriter(output, "test-invalid-plan")
            runner = M3EpisodeRunner(Episode, runtime, writer, _InvalidPlanPlanner(), "test-invalid-plan")
            result = runner.run_episode(
                instruction=self.cases_instruction(),
                scenario=scenario,
                thresholds=thresholds,
                output_dir=output,
                enable_viewer=False,
                show_ui=False,
                episode_id="invalid-plan",
                expected_target_id="target_a",
            )
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "INVALID_PLAN")
        self.assertEqual(result["completed_steps"], [])
        self.assertEqual(result["step_count"], thresholds["scene_settle_steps"])
        self.assertFalse(any(event["state"]["phase"].startswith("approach_") for event in result["episode_events"]))

    def test_skill_budget_stops_physics_at_the_configured_limit(self) -> None:
        runtime = load_json(ROOT / "configs" / "m3_runtime.json")
        runtime["budgets"]["pick_skill_steps"] = 1
        scenarios = load_json(ROOT / "configs" / "m2_scenarios.json")["scenarios"]
        thresholds = load_json(ROOT / "configs" / "m2_thresholds.json")
        scenario = next(item for item in scenarios if item["scenario_id"] == "m2_pose_00_a")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            writer = M3RunWriter(output, "test-step-budget")
            runner = M3EpisodeRunner(Episode, runtime, writer, StubPlanner(), "test-step-budget")
            result = runner.run_episode(
                instruction=self.cases_instruction(),
                scenario=scenario,
                thresholds=thresholds,
                output_dir=output,
                enable_viewer=False,
                show_ui=False,
                episode_id="step-budget",
                expected_target_id="target_a",
            )
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "BUDGET_EXHAUSTED")
        self.assertEqual(result["step_count"], thresholds["scene_settle_steps"] + 1)
        self.assertEqual(result["completed_steps"], [])

    def test_planner_failures_and_empty_or_truncated_responses_stop_before_skills(self) -> None:
        valid_plan = json.dumps(
            {
                "schema_version": 1,
                "base_obs_id": 1,
                "steps": [
                    {"skill": "pick", "object_id": "cube", "approach_mode": "top"},
                    {"skill": "place", "target_id": "target_a"},
                ],
            }
        )
        outcomes = (
            (PlannerError("TIMEOUT", "request timed out"), "TIMEOUT"),
            (PlannerError("AUTHENTICATION_FAILED", "request rejected"), "AUTHENTICATION_FAILED"),
            (PlannerError("RATE_LIMITED", "rate limited"), "RATE_LIMITED"),
            (PlannerResponse("", "llm", "model", "model", None, 0.0, "stop", None), "INVALID_PLAN"),
            (PlannerResponse(valid_plan, "llm", "model", "model", None, 0.0, "length", None), "TRUNCATED_RESPONSE"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, (outcome, expected_code) in enumerate(outcomes):
                with self.subTest(error=expected_code):
                    output = base / str(index)
                    output.mkdir()
                    planner = _ConfiguredPlanner(outcome)
                    writer = M3RunWriter(output, f"planner-failure-{index}")
                    runner = M3EpisodeRunner(Episode, self.runtime, writer, planner, f"planner-failure-{index}")
                    result = runner.run_episode(
                        instruction=self.instruction,
                        scenario=self.scenario,
                        thresholds=self.thresholds,
                        output_dir=output,
                        enable_viewer=False,
                        show_ui=False,
                        episode_id=f"planner-failure-{index}",
                        expected_target_id="target_a",
                    )
                    self.assertEqual(result["status"], "FAILED")
                    self.assertEqual(result["error_code"], expected_code)
                    self.assertEqual(result["completed_steps"], [])
                    self.assertEqual(result["step_count"], self.thresholds["scene_settle_steps"])
                    self.assertEqual(planner.calls, 1)

    def test_invalid_task_is_rejected_before_planner_or_simulator(self) -> None:
        planner = _ConfiguredPlanner(
            PlannerResponse("", "llm", "model", "model", None, 0.0, "stop", None)
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            writer = M3RunWriter(output, "invalid-task")
            runner = M3EpisodeRunner(Episode, self.runtime, writer, planner, "invalid-task")
            result = runner.run_episode(
                instruction="Push the cube to the blue target area.",
                scenario=None,
                thresholds=self.thresholds,
                output_dir=output,
                enable_viewer=False,
                show_ui=False,
                episode_id="invalid-task",
            )
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "INVALID_TASK")
        self.assertEqual(planner.calls, 0)
        self.assertEqual(result["step_count"], 0)

    def test_place_skill_requires_a_verified_grasp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode = Episode(self.scenario, self.thresholds, Path(temporary))
            with self.assertRaises(M2Failure) as raised:
                episode.place_skill()
        self.assertEqual(raised.exception.code, "PRECONDITION_FAILED")
        self.assertEqual(episode.total_steps, 0)

    @staticmethod
    def cases_instruction() -> str:
        return load_json(ROOT / "configs" / "m3_cases.json")["cases"][0]["instruction"]


if __name__ == "__main__":
    unittest.main()

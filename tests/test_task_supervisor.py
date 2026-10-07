"""Recovery uses measured progress, immutable goals and bounded fresh decisions."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.execution.supervisor import EmbodiedTaskRunner
from embodied_agent.models.contracts import PlannerError, PlannerResponse
from embodied_agent.skills.navigation import Navigator, Obstacle
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.simulation.robot_episode import StretchDemoEpisode


class FakeEpisode:
    def __init__(self):
        self.total_steps, self.held, self.at_remote = 0, None, False
        self.calls, self.stop_count, self.goals = [], 0, []
        self.fail_pick_once = True
        self.observation_error = False

    def observe(self):
        if self.observation_error:
            raise ValueError("secondary observation failure")
        return {"objects": {"remote": {"position_m": [0, 0, .025], "states": {}}},
                "robot": {"held_object": self.held, "position_m": [0, 0, 0]}, "world_version": self.total_steps}

    def begin_task_execution(self, budget):
        self.budget = budget

    def end_task_execution(self):
        self.budget = None

    def run_action(self, action, *, record_event=None):
        self.calls.append(copy.deepcopy(action))
        self.total_steps += 1
        if action["skill"] == "navigate":
            self.at_remote = True
        if action["skill"] == "pick":
            if self.fail_pick_once:
                self.fail_pick_once = False
                return {"status": "FAILED", "action": action, "executed": False, "error_code": "NOT_DOCKED",
                        "error_message": "fixture changed docking state", "physics_steps": 0}
            self.held = action["object_id"]
        return {"status": "SUCCESS", "action": action, "executed": True, "evidence": {}, "physics_steps": 1,
                "observation": self.observe()}

    def action_satisfied(self, action, prior):
        return action["skill"] == "navigate" and self.at_remote

    def safe_stop(self):
        self.stop_count += 1

    def evaluate_goal_evidence(self, goals=None):
        return {"objects": {"remote": {"finger_contact_side_count": 2 if self.held else 0,
                                      "supports": [] if self.held else ["floor"], "held_verified": bool(self.held)}}}


class FakePlanner:
    def __init__(self, proposals, goals=None):
        self.proposals, self.calls = iter(proposals), []
        self.last_goals = goals or [{"predicate": "held", "object_id": "remote"}]
        self.last_decision = {"kind": "plan"}

    def plan(self, instruction, observation, world, *, feedback=None, task_context=None):
        self.calls.append(copy.deepcopy({"instruction": instruction, "observation": observation,
                                        "feedback": feedback, "task_context": task_context}))
        proposal = next(self.proposals)
        if isinstance(proposal, Exception):
            raise proposal
        if callable(proposal):
            proposal = proposal(self)
        return PlannerResponse(json.dumps({"schema_version": 1, "actions": proposal}), "stub", None, None, None, 0., "stop", None)


class TaskSupervisorTests(unittest.TestCase):
    def test_failed_rejection_recording_stops_recovery_without_replacing_proposal_error(self):
        episode = FakeEpisode()
        error = PlannerError("INVALID_RESPONSE", "proposal rejected")
        error.recording_errors = [{"event": "proposal_rejected", "message": "journal unavailable"}]
        planner = FakePlanner([error, [{"skill": "pick", "object_id": "remote"}]])
        result = EmbodiedTaskRunner(planner).run("拿遥控器", episode, None)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "INVALID_RESPONSE")
        self.assertEqual(result["recording_errors"], error.recording_errors)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(episode.calls, [])
        self.assertEqual(episode.stop_count, 1)

    def test_confirmed_intent_survives_failure_of_first_action_proposal(self):
        episode = FakeEpisode()
        episode.fail_pick_once = False
        planner = FakePlanner([PlannerError("INVALID_JSON", "malformed action proposal"),
            [{"skill": "navigate", "target": "remote"}, {"skill": "pick", "object_id": "remote"}]])
        planner.last_confirmed_goals = copy.deepcopy(planner.last_goals)
        result = EmbodiedTaskRunner(planner).run("拿遥控器", episode, None)
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertEqual(planner.calls[1]["task_context"]["original_goals"], planner.last_confirmed_goals)
        self.assertEqual(result["first_error"]["error_code"], "INVALID_JSON")
        self.assertEqual(result["goals"], planner.last_confirmed_goals)

    def test_recovery_gets_actual_prefix_and_skips_still_satisfied_navigation(self):
        episode = FakeEpisode()
        navigate = {"skill": "navigate", "target": "remote"}
        pick = {"skill": "pick", "object_id": "remote"}
        planner = FakePlanner([[navigate, pick], [navigate, pick]])
        attempts = []
        result = EmbodiedTaskRunner(planner, on_attempt=lambda *args: attempts.append(copy.deepcopy(args))).run("拿遥控器", episode, None)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["verified"])
        self.assertEqual(episode.calls, [navigate, pick, pick])
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(planner.calls[1]["feedback"]["error_code"], "NOT_DOCKED")
        self.assertEqual(planner.calls[1]["task_context"]["original_goals"], planner.last_goals)
        self.assertEqual(planner.calls[1]["observation"]["world_version"], 2)
        self.assertEqual(result["attempts"][1]["actions"][0]["status"], "SKIPPED_COMPLETED")
        self.assertEqual(len(attempts), 2)

    def test_recovery_cannot_change_original_object(self):
        episode = FakeEpisode()
        def changed(planner):
            planner.last_goals = [{"predicate": "held", "object_id": "medicine"}]
            return [{"skill": "pick", "object_id": "medicine"}]
        planner = FakePlanner([[{"skill": "pick", "object_id": "remote"}], changed])
        result = EmbodiedTaskRunner(planner, max_rounds=2).run("拿遥控器", episode, None)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["feedback"]["error_code"], "GOAL_CHANGED")
        self.assertEqual(episode.calls, [{"skill": "pick", "object_id": "remote"}])

    def test_format_repair_feedback_precedes_any_physics(self):
        episode = FakeEpisode()
        episode.fail_pick_once = False
        planner = FakePlanner([PlannerError("INVALID_JSON", "fixture malformed proposal"),
                               [{"skill": "pick", "object_id": "remote"}]])
        result = EmbodiedTaskRunner(planner).run("拿遥控器", episode, None)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(planner.calls[1]["feedback"]["error_code"], "INVALID_JSON")
        self.assertEqual(len(episode.calls), 1)

    def test_cancel_blocks_later_action_but_allows_safety_stop(self):
        class Budget:
            cancelled = False
            def check(self):
                if self.cancelled:
                    raise PlannerError("TASK_CANCELLED", "cancel fixture")
            def record_action(self):
                self.cancelled = True
            def snapshot(self):
                return {}
        episode = FakeEpisode()
        original = episode.run_action
        def checked(action, **kwargs):
            budget.check()
            return original(action, **kwargs)
        budget = Budget()
        episode.run_action = checked
        planner = FakePlanner([[{"skill": "navigate", "target": "remote"}, {"skill": "pick", "object_id": "remote"}]])
        result = EmbodiedTaskRunner(planner, budget).run("拿遥控器", episode, None)
        self.assertEqual(result["status"], "ABORTED")
        self.assertEqual(result["error_code"], "TASK_CANCELLED")
        self.assertEqual(episode.calls, [])
        self.assertEqual(episode.stop_count, 1)
        self.assertEqual(len(planner.calls), 1)

    def test_primary_failure_survives_failed_observation_and_stop(self):
        episode = FakeEpisode()
        def primary(planner):
            episode.observation_error = True
            raise PlannerError("API_TIMEOUT", "original API failure")
        def failed_stop():
            raise RuntimeError("secondary stop failure")
        episode.safe_stop = failed_stop
        result = EmbodiedTaskRunner(FakePlanner([primary])).run("拿遥控器", episode, None)
        self.assertEqual(result["error_code"], "API_TIMEOUT")
        self.assertEqual(result["error_message"], "original API failure")
        self.assertEqual(result["first_error"]["error_code"], "API_TIMEOUT")
        self.assertIsNone(result["final_snapshot"])
        self.assertIn("stop", [row["phase"] for row in result["cleanup_errors"]])
        self.assertIn("observation", [row["phase"] for row in result["cleanup_errors"]])

    def test_window_close_exception_is_canonical_and_never_replanned(self):
        for exception_name in ("ViewerClosed",):
            with self.subTest(exception_name=exception_name):
                episode = FakeEpisode()
                closed = type(exception_name, (RuntimeError,), {})("fixture window closed")
                planner = FakePlanner([closed])
                result = EmbodiedTaskRunner(planner).run("stop when window closes", episode, None)
                self.assertEqual(result["status"], "ABORTED")
                self.assertEqual(result["error_code"], "VIEWER_CLOSED")
                self.assertEqual(result["first_error"]["error_code"], "VIEWER_CLOSED")
                self.assertEqual(result["feedback"]["error_code"], "VIEWER_CLOSED")
                self.assertEqual(episode.calls, [])
                self.assertEqual(episode.stop_count, 1)
                self.assertEqual(len(planner.calls), 1)

    def test_window_close_action_result_is_terminal_and_canonical(self):
        for code in ("ViewerClosed", "VIEWER_CLOSED"):
            with self.subTest(code=code):
                episode = FakeEpisode()
                episode.run_action = lambda action, **kwargs: {
                    "status": "FAILED", "action": action, "executed": False,
                    "error_code": code, "error_message": "fixture window closed"}
                planner = FakePlanner([[{"skill": "navigate", "target": "remote"}]])
                result = EmbodiedTaskRunner(planner).run("stop when window closes", episode, None)
                self.assertEqual(result["status"], "ABORTED")
                self.assertEqual(result["error_code"], "VIEWER_CLOSED")
                self.assertEqual(result["first_error"]["error_code"], "VIEWER_CLOSED")
                self.assertEqual(result["feedback"]["error_code"], "VIEWER_CLOSED")
                self.assertEqual(episode.stop_count, 1)
                self.assertEqual(len(planner.calls), 1)

    def test_explicit_wait_is_not_erased_as_a_completed_effect(self):
        episode = FakeEpisode()
        wait = {"skill": "wait", "seconds": .1}
        planner = FakePlanner([[wait], [wait]], goals=[{"predicate": "action_completed", "value": {"skill": "wait", "seconds": .1}}])
        result = EmbodiedTaskRunner(planner).run("等一下", episode, None)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(episode.calls, [wait])


class NavigationSupervisionTests(unittest.TestCase):
    def test_remaining_path_trims_travelled_segments_and_detects_new_obstacle(self):
        path = [(-1., 0.), (0., 0.), (1., 0.)]
        clear = Navigator((-2, -2, 2, 2), [], .2)
        self.assertEqual(clear.remaining_path((.1, 0.), path), [(.1, 0.), (1., 0.)])
        moved = Navigator((-2, -2, 2, 2), [Obstacle(.6, -.05, .7, .05, "moving")], .2)
        self.assertFalse(moved.route_free((.1, 0.), path))
        reroute = moved.plan((.1, 0.), (1., 0.))
        self.assertTrue(moved.route_free((.1, 0.), reroute))


class StretchPermitAndHoldTests(unittest.TestCase):
    def setUp(self):
        self.episode = StretchDemoEpisode(HomeMapStore().load())
        self.addCleanup(self.episode.close)

    def test_permit_single_use_and_compiled_action_mutation_cannot_execute(self):
        permit = self.episode.review_next_action({"skill": "wait", "seconds": 0})
        self.episode.execute_reviewed_action(permit)
        with self.assertRaisesRegex(ValueError, "unused"):
            self.episode.execute_reviewed_action(permit)
        permit = self.episode.review_next_action({"skill": "wait", "seconds": 0})
        permit["action"]["seconds"] = 1
        started = self.episode.total_steps
        with self.assertRaisesRegex(ValueError, "replaced"):
            self.episode.execute_reviewed_action(permit)
        self.assertEqual(self.episode.total_steps, started)

    def test_physical_progress_invalidates_old_permit(self):
        permit = self.episode.review_next_action({"skill": "wait", "seconds": 0})
        self.episode.hold(.02)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.episode.execute_reviewed_action(permit)

    def test_hold_keeps_every_physics_guard_and_suppresses_recursive_frame_callbacks(self):
        checked, frames = [], []
        original = self.episode.sim.check_physics
        def check():
            checked.append(self.episode.total_steps)
            original()
        self.episode.sim.check_physics = check
        self.episode.on_frame = lambda episode: frames.append(episode.total_steps)
        started = self.episode.total_steps
        self.episode.hold(.12)
        self.assertEqual(len(checked), self.episode.total_steps - started)
        self.assertGreater(len(checked), 0)
        self.assertEqual(frames, [])
        self.assertIsNotNone(self.episode.sim.on_step)

    def test_already_supported_goal_requires_actual_continuous_window_without_redundant_pick(self):
        planner = FakePlanner([[]], goals=[{"predicate": "supported_on", "object_id": "remote", "target_id": "tea_table"}])
        result = EmbodiedTaskRunner(planner).run("保持遥控器在茶几上", self.episode, self.episode.world)
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertTrue(result["verified"])
        self.assertEqual(result["actions"], [])
        self.assertGreaterEqual(result["physics_steps"], 250)
        self.assertEqual(len(planner.calls), 1)

    def test_record_callback_state_change_prevents_stale_action_invocation(self):
        seen = []
        original = self.episode._execute_action
        self.episode._execute_action = lambda *args: seen.append(args) or original(*args)
        def record(name, detail, observation):
            if name == "before_tool":
                self.episode.hold(.02)
        result = self.episode.run_action({"skill": "wait", "seconds": .1}, record_event=record)
        self.assertEqual(result["error_code"], "STALE_ACTION")
        self.assertFalse(result["executed"])
        self.assertEqual(seen, [])

    def test_control_change_invalidates_permit_without_needing_pose_motion(self):
        permit = self.episode.review_next_action({"skill": "wait", "seconds": 0})
        self.episode.sim.set_control("grip", .02)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.episode.execute_reviewed_action(permit)

    def test_cancel_in_record_callback_prevents_handler_but_keeps_braking(self):
        from embodied_agent.models.budget import TaskBudget
        budget = TaskBudget({})
        self.episode.begin_task_execution(budget)
        self.addCleanup(self.episode.end_task_execution)
        seen = []
        original = self.episode._execute_action
        self.episode._execute_action = lambda *args: seen.append(args) or original(*args)
        def record(name, detail, observation):
            if name == "before_tool":
                budget.cancel("fixture cancelled before invocation")
        result = self.episode.run_action({"skill": "wait", "seconds": .1}, record_event=record)
        self.assertEqual(result["error_code"], "TASK_CANCELLED")
        self.assertEqual(result["status"], "ABORTED")
        self.assertFalse(result["executed"])
        self.assertEqual(seen, [])
        self.assertIn("safe_stop", result["evidence"])


if __name__ == "__main__":
    unittest.main()

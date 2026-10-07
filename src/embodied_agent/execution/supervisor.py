"""Bounded task decisions over real single-action results and fresh observations.

Model decisions run through the caller's planner wrapper. This module never
touches MuJoCo from a language worker and never resets an episode to recover.
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from embodied_agent.maps.schema import strict_json
from embodied_agent.models.tracing import current_trace, trace_scope, new_trace_id, record_model_event


@dataclass
class ExecutionResult:
    status: str
    action: dict
    evidence: dict = field(default_factory=dict)
    observation: dict | None = None
    physics_steps: int = 0
    error_code: str | None = None
    error_message: str | None = None
    error_evidence: dict = field(default_factory=dict)
    cleanup_errors: list = field(default_factory=list)
    executed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class EmbodiedTaskRunner:
    """Ask for bounded segments, execute them, and feed actual progress back.

    ``on_attempt(index, feedback, observation)`` belongs to the record adapter;
    it can link attempts without this runner owning persistence or task IDs.
    """
    HARD_FAILURES = {"TASK_CANCELLED", "TASK_BUDGET_EXCEEDED", "STEP_BUDGET", "VIEWER_CLOSED",
        "RECORDING_FAILED", "ROBOT_FURNITURE_COLLISION", "HELD_OBJECT_COLLISION", "RISK_CLEARANCE",
        "OBJECT_DAMAGED", "PHYSICS_NONFINITE", "PHYSICS_WARNING", "BASE_UNSTABLE",
        "STOP_FAILED", "STOP_HOLD_FAILED", "UNEXPECTED_OBJECT_CONTACT"}
    REPAIRABLE_PROPOSALS = {"INVALID_JSON", "INVALID_PLAN", "INVALID_GOAL", "INVALID_SCHEMA",
        "INVALID_RESPONSE", "EMPTY_RESPONSE", "GOAL_CHANGED", "STALE_ACTION", "TRUNCATED_RESPONSE",
        "UNKNOWN_OBJECT", "UNKNOWN_TARGET", "UNREGISTERED_GOAL"}

    def __init__(self, planner: Any, budget: Any = None, record_event: Callable | None = None,
                 on_attempt: Callable | None = None, *, goal_evaluator: Callable | None = None,
                 max_rounds: int = 8, max_recoveries: int = 4):
        self.planner, self.budget = planner, budget
        self.record_event, self.on_attempt = record_event, on_attempt
        self.goal_evaluator = goal_evaluator
        self.max_rounds, self.max_recoveries = max_rounds, max_recoveries
        self._initial_observation = None
        self._last_evidence = {}

    def _check(self) -> None:
        if self.budget is not None:
            self.budget.check()

    @staticmethod
    def _error_code(error) -> str:
        code = error if isinstance(error, str) else getattr(error, "code", type(error).__name__)
        return "VIEWER_CLOSED" if code in {"ViewerClosed"} else code

    def _event(self, name: str, detail: dict, observation=None) -> None:
        detail = {**current_trace(), **copy.deepcopy(detail)}
        if self.record_event is not None:
            try:
                self.record_event(name, detail, observation)
            except Exception as exc:
                from embodied_agent.models.contracts import PlannerError
                raise PlannerError("RECORDING_FAILED", f"Cannot record {name}: {exc}") from exc
        else:
            record_model_event(name, {**detail, "observation": copy.deepcopy(observation)})

    def _evaluate(self, goals, observation, episode, world, executed) -> dict:
        if self.goal_evaluator is None:
            from embodied_agent.agents.goals import evaluate_goals
            evaluator = evaluate_goals
        else:
            evaluator = self.goal_evaluator
        evidence = {"completed_goals": copy.deepcopy(getattr(episode, "goals", [])),
                    "actions": copy.deepcopy(executed), "source": "measured_execution"}
        if hasattr(episode, "evaluate_goal_evidence"):
            evidence.update(episode.evaluate_goal_evidence(goals))
        evidence["action_results"] = [{"action": copy.deepcopy(row["action"]),
            "success": row.get("status") == "SUCCESS", "result": {**copy.deepcopy(row.get("evidence", {})),
                "observation": copy.deepcopy(row.get("observation")), "physics_steps": row.get("physics_steps")},
            "observation": copy.deepcopy(row.get("observation"))}
            for row in executed]
        evidence["performed_methods"] = ["controlled_place" for row in executed
            if row.get("status") == "SUCCESS" and row["action"].get("skill") == "place"]
        evidence["initial_observation"] = copy.deepcopy(self._initial_observation)
        self._last_evidence = evidence
        result = evaluator(goals, observation, evidence=evidence, world=world)
        self._event("goal_evaluated", {"goals": copy.deepcopy(goals), "result": copy.deepcopy(result),
            "evidence": copy.deepcopy(evidence), "source": "execution_goal_evaluator"}, observation)
        return result

    def _verify(self, goals, observation, episode, world, executed) -> dict:
        result = self._evaluate(goals, observation, episode, world, executed)
        # An already achieved geometric relation still needs an independently
        # measured temporal window. No second model call or fake success is
        # needed: advance guarded physics for this bounded verification window.
        if not result.get("passed") and self._last_evidence.get("pending_stability") and hasattr(episode, "hold"):
            for _ in range(7):
                self._check()
                episode.hold(.1)
                result = self._evaluate(goals, episode.observe(), episode, world, executed)
                if result.get("passed") or not self._last_evidence.get("pending_stability"):
                    break
        return result

    @staticmethod
    def _json_identity(value) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)

    def run(self, instruction: str, episode: Any, world: Any, task_context: dict | None = None) -> dict:
        with trace_scope(supervision_id=new_trace_id("supervision")):
            return self._run(instruction, episode, world, task_context)

    def _run(self, instruction: str, episode: Any, world: Any, task_context: dict | None = None) -> dict:
        context = copy.deepcopy(task_context or {})
        self._initial_observation = None
        executed, attempts, feedback, cleanup_errors = [], [], None, []
        recording_errors = []
        first_error = None
        goals = copy.deepcopy(context.get("original_goals"))
        recoveries = 0
        initial_steps = episode.total_steps
        status, error_code, error_message = "FAILED", "TASK_INCOMPLETE", "Task has not been independently verified"
        verified = False
        last_check = None
        observation = None
        previous_decision_id = None
        feedback_id = None
        try:
            if hasattr(episode, "begin_task_execution"):
                episode.begin_task_execution(self.budget)
            for index in range(1, self.max_rounds + 1):
                self._check()
                observation = episode.observe()
                if self._initial_observation is None:
                    self._initial_observation = copy.deepcopy(observation)
                if goals:
                    last_check = self._verify(goals, observation, episode, world, executed)
                    if last_check.get("passed"):
                        status, error_code, error_message, verified = "SUCCESS", None, None, True
                        break
                if index > 1 and feedback and feedback.get("status") in {"FAILED", "INCOMPLETE"}:
                    if recoveries >= self.max_recoveries:
                        error_code, error_message = "RECOVERY_BUDGET_EXCEEDED", "Task exhausted its decision/recovery budget"
                        break
                    if self.budget is not None and hasattr(self.budget, "record_replan"):
                        self.budget.record_replan()
                    recoveries += 1
                if self.on_attempt is not None:
                    self.on_attempt(index, copy.deepcopy(feedback), copy.deepcopy(observation))
                decision_context = {**context, "original_goals": copy.deepcopy(goals),
                    "completed_actions": copy.deepcopy(executed), "decision_index": index,
                    "budget": self.budget.snapshot() if self.budget is not None else None}
                decision_id = new_trace_id("decision")
                parent_decision_id, previous_decision_id = previous_decision_id, decision_id
                with trace_scope(decision_id=decision_id, parent_decision_id=parent_decision_id,
                                 attempt_index=index, feedback_id=feedback_id):
                    self._event("decision_started", {"decision_index": index, "feedback": feedback}, observation)
                    attempt = {"attempt": index, "feedback_in": copy.deepcopy(feedback), "actions": []}
                    attempts.append(attempt)
                    actions, action_ids, consumed_actions = [], [], set()
                    try:
                        try:
                            response = self.planner.plan(instruction, observation, world,
                                feedback=copy.deepcopy(feedback), task_context=decision_context)
                        finally:
                            if feedback is not None:
                                primary = sys.exc_info()[1]
                                try:
                                    self._event("feedback_delivered", {"feedback_id": feedback_id,
                                        "feedback": copy.deepcopy(feedback), "recipient": "local_planner_invocation",
                                        "remote_read": None}, observation)
                                except Exception as recording_error:
                                    if primary is None:
                                        raise
                                    cleanup_errors.append({"phase": "recording", "event": "feedback_delivered",
                                        "message": str(recording_error)})
                        self._check()
                        proposed_goals = copy.deepcopy(getattr(self.planner, "last_goals", None))
                        if goals is None and proposed_goals:
                            goals = proposed_goals
                        elif proposed_goals and goals and self._json_identity(goals) != self._json_identity(proposed_goals):
                            from embodied_agent.models.contracts import PlannerError
                            raise PlannerError("GOAL_CHANGED", "Recovery must preserve the confirmed original goals")
                        decision = getattr(self.planner, "last_decision", {}) or {}
                        kind = decision.get("kind", "plan") if isinstance(decision, dict) else decision if isinstance(decision, str) else "plan"
                        if kind in {"clarify", "capability_gap"}:
                            status, error_code = "FAILED", "CLARIFICATION_REQUIRED" if kind == "clarify" else "CAPABILITY_GAP"
                            details = decision if isinstance(decision, dict) else getattr(self.planner, "last_proposal", {}) or {}
                            error_message = details.get("message", details.get("reason", details.get("question", "More information or a supported skill is required")))
                            attempt.update(status=status, decision=copy.deepcopy(decision))
                            break
                        if getattr(response, "finish_reason", "stop") != "stop":
                            from embodied_agent.models.contracts import PlannerError
                            raise PlannerError("TRUNCATED_RESPONSE", "A complete proposal is required before execution")
                        payload = strict_json(response.content) if hasattr(response, "content") else response
                        if not isinstance(payload, dict) or set(payload) != {"schema_version", "actions"} or payload["schema_version"] != 1 or not isinstance(payload["actions"], list):
                            from embodied_agent.models.contracts import PlannerError
                            raise PlannerError("INVALID_PLAN", "Expected schema_version 1 and an actions array")
                        actions = copy.deepcopy(payload["actions"])
                        action_ids = [new_trace_id("action") for _ in actions]
                        attempt["proposal"] = copy.deepcopy(payload)
                        if goals:
                            last_check = self._verify(goals, episode.observe(), episode, world, executed)
                            if last_check.get("passed"):
                                status, error_code, error_message, verified = "SUCCESS", None, None, True
                                attempt["status"] = status
                                break
                        if not actions:
                            observation = episode.observe()
                            last_check = self._verify(goals, observation, episode, world, executed) if goals else None
                            if not goals or last_check.get("passed"):
                                status, error_code, error_message = "SUCCESS", None, None
                                verified = bool(goals)
                                attempt["status"] = status
                                break
                            feedback = {"status": "INCOMPLETE", "error_code": "GOAL_NOT_MET",
                                        "goal_check": last_check, "completed_actions": copy.deepcopy(executed)}
                            feedback_id = new_trace_id("feedback")
                            self._event("feedback_created", {"feedback_id": feedback_id,
                                "feedback": copy.deepcopy(feedback), "delivered": False})
                            attempt.update(status="INCOMPLETE", feedback_out=copy.deepcopy(feedback))
                            continue
                        interrupted = False
                        for action_index, action in enumerate(actions):
                            self._check()
                            # A repaired model often repeats a completed prefix. Skip
                            # only an actual earlier action whose effect still holds.
                            same = [row for row in executed if row.get("status") == "SUCCESS" and row["action"] == action]
                            if feedback is not None and same and hasattr(episode, "action_satisfied") and episode.action_satisfied(action, same[-1]):
                                attempt["actions"].append({"action": action, "status": "SKIPPED_COMPLETED", "executed": False})
                                self._event("action_skipped", {"action_id": action_ids[action_index],
                                    "action_index": action_index, "action": copy.deepcopy(action),
                                    "reason": "previous_success_still_satisfied", "executed": False})
                                consumed_actions.add(action_index)
                                continue
                            if self.budget is not None and hasattr(self.budget, "record_action"):
                                self.budget.record_action()
                            with trace_scope(action_id=action_ids[action_index], action_index=action_index):
                                self._event("action_dispatch_started", {"action": copy.deepcopy(action),
                                    "boundary": "before_episode_run_action", "executed": None})
                                consumed_actions.add(action_index)
                                try:
                                    result = episode.run_action(action, record_event=self._event)
                                except Exception as action_error:
                                    self._event("action_dispatch_finished", {"action": copy.deepcopy(action),
                                        "result_available": False, "executed": None,
                                        "error": {"code": self._error_code(action_error), "message": str(action_error)}})
                                    raise
                                self._event("action_dispatch_finished", {"action": copy.deepcopy(action),
                                    "result_available": True, "result": result.to_dict() if isinstance(result, ExecutionResult) else copy.deepcopy(result),
                                    "executed": getattr(result, "executed", None) if isinstance(result, ExecutionResult) else result.get("executed")})
                            if isinstance(result, ExecutionResult):
                                result = result.to_dict()
                            if result.get("error_code") is not None:
                                result["error_code"] = self._error_code(result["error_code"])
                                if result["error_code"] in {"TASK_CANCELLED", "VIEWER_CLOSED"}:
                                    result["status"] = "ABORTED"
                            attempt["actions"].append(copy.deepcopy(result))
                            if result.get("executed") or result.get("status") == "SUCCESS":
                                executed.append(copy.deepcopy(result))
                            cleanup_errors.extend(result.get("cleanup_errors", []))
                            if result.get("status") != "SUCCESS":
                                if first_error is None:
                                    first_error = {"phase": "action", "attempt": index, "action": copy.deepcopy(action),
                                        "error_code": result.get("error_code"), "error_message": result.get("error_message")}
                                feedback = {**copy.deepcopy(result), "completed_actions": copy.deepcopy(executed),
                                    "original_goals": copy.deepcopy(goals)}
                                feedback_id = new_trace_id("feedback")
                                self._event("feedback_created", {"feedback_id": feedback_id,
                                    "feedback": copy.deepcopy(feedback), "delivered": False})
                                attempt.update(status="FAILED", feedback_out=copy.deepcopy(feedback))
                                code = result.get("error_code", "ACTION_FAILED")
                                hard_failure = code in self.HARD_FAILURES or result.get("stop_failed") or (
                                    code == "OBJECT_SLIPPED" and not result.get("recoverable", False))
                                if hard_failure:
                                    status, error_code, error_message = result.get("status", "FAILED"), code, result.get("error_message")
                                    interrupted = True
                                    break
                                interrupted = True
                                break
                            if goals:
                                measured = result.get("observation") or episode.observe()
                                last_check = self._evaluate(goals, measured, episode, world, executed)
                                if last_check.get("passed"):
                                    # Additional proposed work cannot be necessary
                                    # once all terminal/process goals are verified.
                                    break
                        if interrupted:
                            if feedback.get("error_code") in self.HARD_FAILURES or feedback.get("stop_failed") or (
                                    feedback.get("error_code") == "OBJECT_SLIPPED" and not feedback.get("recoverable", False)):
                                break
                            continue
                        observation = episode.observe()
                        last_check = self._verify(goals, observation, episode, world, executed) if goals else None
                        feedback = {"status": "SUCCESS" if last_check and last_check.get("passed") else "SEGMENT_COMPLETED",
                            "observation": observation, "completed_actions": copy.deepcopy(executed), "goal_check": last_check,
                            "original_goals": copy.deepcopy(goals)}
                        feedback_id = new_trace_id("feedback")
                        self._event("feedback_created", {"feedback_id": feedback_id,
                            "feedback": copy.deepcopy(feedback), "delivered": False})
                        attempt.update(status="SUCCESS", feedback_out=copy.deepcopy(feedback))
                        if not goals or last_check.get("passed"):
                            status, error_code, error_message, verified = "SUCCESS", None, None, bool(goals)
                            break
                    except Exception as exc:
                        # Intent can be confirmed before the following action
                        # proposal fails; recovery must not reinterpret that goal.
                        confirmed = getattr(self.planner, "last_confirmed_goals", None)
                        if goals is None and confirmed:
                            goals = copy.deepcopy(confirmed)
                        code = self._error_code(exc)
                        audit_failures = (getattr(exc, "recording_errors", None) or
                            (getattr(exc, "details", None) or {}).get("recording_errors", []))
                        recording_errors.extend(copy.deepcopy(audit_failures))
                        if first_error is None:
                            first_error = {"phase": "decision", "attempt": index, "error_code": code, "error_message": str(exc)}
                        try:
                            failure_observation = episode.observe()
                        except Exception as observation_error:
                            failure_observation = None
                            cleanup_errors.append({"phase": "observation", "message": str(observation_error)})
                        feedback = {"status": "FAILED", "error_code": code, "error_message": str(exc),
                            "observation": failure_observation, "completed_actions": copy.deepcopy(executed),
                            "original_goals": copy.deepcopy(goals)}
                        feedback_id = new_trace_id("feedback")
                        self._event("feedback_created", {"feedback_id": feedback_id,
                            "feedback": copy.deepcopy(feedback), "delivered": False})
                        attempt.update(status="FAILED", feedback_out=copy.deepcopy(feedback))
                        if code not in self.REPAIRABLE_PROPOSALS or audit_failures:
                            raise
                    finally:
                        primary = sys.exc_info()[1]
                        try:
                            for remaining_index, remaining in enumerate(actions):
                                if remaining_index not in consumed_actions:
                                    self._event("action_not_started", {"action_id": action_ids[remaining_index],
                                        "action_index": remaining_index, "action": copy.deepcopy(remaining),
                                        "reason": "goal_verified" if last_check and last_check.get("passed") else "decision_ended_before_dispatch",
                                        "executed": False})
                            self._event("decision_finished", {"decision_index": index,
                                "status": attempt.get("status", "FAILED"),
                                "feedback_id": feedback_id, "attempt_summary": copy.deepcopy(attempt)})
                        except Exception as recording_error:
                            if primary is None:
                                raise
                            cleanup_errors.append({"phase": "recording", "event": "decision_finished",
                                "message": str(recording_error)})
            else:
                error_code, error_message = "DECISION_BUDGET_EXCEEDED", "Task exhausted its bounded decision rounds"
        except Exception as exc:
            error_code, error_message = self._error_code(exc), str(exc)
            if not recording_errors:
                recording_errors.extend(copy.deepcopy(getattr(exc, "recording_errors", None) or
                    (getattr(exc, "details", None) or {}).get("recording_errors", [])))
            if first_error is None:
                first_error = {"phase": "task", "error_code": error_code, "error_message": error_message}
            status = "ABORTED" if error_code in {"TASK_CANCELLED", "VIEWER_CLOSED"} else "FAILED"
        finally:
            if status != "SUCCESS" and hasattr(episode, "safe_stop"):
                try:
                    episode.safe_stop()
                except Exception as exc:
                    cleanup_errors.append({"phase": "stop", "code": self._error_code(exc), "message": str(exc)})
            try:
                observation = episode.observe()
            except Exception as exc:
                observation = None
                cleanup_errors.append({"phase": "observation", "message": str(exc)})
                if status == "SUCCESS":
                    status, error_code, error_message = "FAILED", "OBSERVATION_FAILED", str(exc)
            if hasattr(episode, "end_task_execution"):
                try:
                    episode.end_task_execution()
                except Exception as exc:
                    cleanup_errors.append({"phase": "cleanup", "message": str(exc)})
                    if status == "SUCCESS":
                        status, error_code, error_message = "FAILED", "CLEANUP_FAILED", str(exc)
        recording_errors.extend(copy.deepcopy(item) for item in cleanup_errors if item.get("phase") == "recording")
        if error_code == "RECORDING_FAILED" and not recording_errors:
            recording_errors.append({"error_code": error_code, "message": error_message})
        result = {"status": status, "error_code": error_code, "error_message": error_message,
            "first_error": first_error,
            "actions": executed, "attempts": attempts, "feedback": feedback, "goals": goals,
            "goal_check": last_check, "verified": verified and status == "SUCCESS", "final_snapshot": observation,
            "physics_steps": episode.total_steps - initial_steps, "recoveries": recoveries,
            "cleanup_errors": cleanup_errors, "recording_errors": recording_errors,
            "transport_success": status == "SUCCESS" and verified and
                any(row.get("action", {}).get("skill") == "place" for row in executed)}
        try:
            self._event("task_supervision_finished", {"result": result}, observation)
        except Exception as exc:
            cleanup_errors.append({"phase": "recording", "event": "task_supervision_finished", "message": str(exc)})
            recording_errors.append({"event": "task_supervision_finished", "message": str(exc)})
        return result

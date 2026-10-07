"""Execution-facing task recorder; lifecycle events are evidence, not permissions."""
from __future__ import annotations

from functools import wraps
import hashlib
import json
import math
import threading
import uuid
from pathlib import Path
from typing import Any

from embodied_agent.evaluation.evidence import jsonable
from embodied_agent.recording import TaskRecord, TaskRecordStore
from embodied_agent.recording.run import RunRecorder
from embodied_agent.recording.common import atomic_json
from embodied_agent.models.tracing import current_trace
from embodied_agent.paths import PROJECT_ROOT


EVENT_TYPES = {
    "task_started": "task.started", "initial_state_observed": "observation.captured",
    "model_request": "model.requested", "model_dispatch_started": "model.dispatch.started",
    "model_response": "model.responded", "model_failed": "model.failed",
    "model_response_delivered": "feedback.delivered",
    "agent_tool_call": "tool.requested", "agent_tool_result": "tool.finished",
    "agent_components": "components.selected", "context_assembled": "context.assembled",
    "memory_read": "memory.read", "memory_write": "memory.write",
    "knowledge_retrieval": "retrieval.finished", "proposal_received": "proposal.created",
    "proposal_validation_started": "proposal.validation.started", "proposal_accepted": "proposal.validated",
    "proposal_rejected": "proposal.rejected", "feedback_created": "feedback.created",
    "feedback_included": "feedback.included_in_request", "feedback_delivered": "feedback.delivered",
    "decision_started": "decision.started", "decision_finished": "decision.finished",
    "planning_started": "decision.planning.started", "planning_response": "proposal.created",
    "plan_received": "plan.accepted", "plan_validated": "plan.validated",
    "action_review": "action.reviewed", "action_reviewed": "action.reviewed",
    "skill_started": "action.started", "before_tool": "action.started",
    "post_skill_passed": "action.finished", "after_tool": "action.finished",
    "skill_failed": "action.failed", "execution_failed": "action.failed",
    "action_dispatch_started": "action.dispatch.started", "action_dispatch_finished": "action.dispatch.finished",
    "action_skipped": "action.skipped", "action_not_started": "action.not_started",
    "goal_evaluated": "goal.evaluated", "replanning_requested": "decision.replanning.requested",
    "recovery_attempt_started": "attempt.recovery.started", "environment_plan": "plan.accepted",
    "environment_proposal": "proposal.created", "environment_validated": "proposal.validated",
    "environment_rejected": "proposal.rejected", "environment_commit_started": "commit.started",
    "environment_commit_finished": "commit.finished", "environment_commit_rejected": "commit.rejected",
    "environment_reload_started": "scene.reload.started", "environment_reload_finished": "scene.reload.finished",
    "environment_reload_failed": "scene.reload.failed",
}


def recordable(value: Any) -> Any:
    """Preserve invalid proposals/telemetry without writing non-standard JSON."""
    value = jsonable(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"unavailable": "non_finite_number", "representation": repr(value)}
    if isinstance(value, dict):
        return {str(key): recordable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [recordable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unavailable": "unsupported_value", "type": type(value).__name__}


def _aggregate_token_usage(events: list[dict]) -> dict:
    """Sum provider-reported usage without treating budget counters as totals.

    A prepared request without a terminal event has unknown cost. An explicit
    pre-dispatch failure has no model cost. Repeated facts for one call do not
    double-count usage; conflicting values remain unknown for that field.
    """
    names = {"model.requested": "request", "model.responded": "response", "model.failed": "failed"}
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    aliases = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}
    calls, source_ids = {}, []
    for event in events:
        kind = names.get(event.get("event_type"))
        if kind is None:
            continue
        payload = event.get("payload", {})
        detail = payload.get("detail", payload) if isinstance(payload, dict) else {}
        if not isinstance(detail, dict):
            detail = {}
        call_id = event.get("model_call_id") or detail.get("model_call_id")
        if not call_id:
            # Legacy collector fixtures may omit identity. Only pair a reply
            # when exactly one anonymous request is pending; never guess among
            # concurrent requests.
            pending = [key for key, call in calls.items() if call["anonymous"] and not call["terminal"]]
            call_id = pending[0] if kind != "request" and len(pending) == 1 else "anonymous:" + event["event_id"]
        call = calls.setdefault(call_id, {"anonymous": not (event.get("model_call_id") or detail.get("model_call_id")),
            "terminal": False, "has_response": False, "not_issued": False, "reports": []})
        source_ids.append(event["event_id"])
        if kind in {"response", "failed"}:
            call["terminal"] = True
            call["has_response"] |= kind == "response"
            call["not_issued"] |= kind == "failed" and detail.get("request_issued") is False
            if isinstance(detail.get("usage"), dict):
                call["reports"].append(detail["usage"])
    chargeable = [call for call in calls.values() if call["has_response"] or not call["not_issued"]]
    subtotal = {field: 0 for field in fields}
    known_counts = {field: 0 for field in fields}
    calls_with_known_usage = 0
    for call in chargeable:
        any_known = False
        for field in fields:
            reported = set()
            for usage in call["reports"]:
                value = usage.get(field, usage.get(aliases.get(field)))
                if type(value) is int and value >= 0:
                    reported.add(value)
            if len(reported) == 1:
                subtotal[field] += reported.pop()
                known_counts[field] += 1
                any_known = True
        calls_with_known_usage += any_known
    count = len(chargeable)
    available = {field: "known" if known_counts[field] == count else
        "partial" if known_counts[field] else "unknown" for field in fields}
    availability = ("complete" if all(value == "known" for value in available.values()) else
        "unknown" if all(value == "unknown" for value in available.values()) else "partial")
    return {**{field: subtotal[field] if available[field] == "known" else None for field in fields},
        "known_reported_subtotal": subtotal, "availability": availability,
        "field_availability": available, "source": "canonical_model_event_usage",
        "model_call_count": count, "calls_with_known_usage": calls_with_known_usage,
        "source_event_ids": source_ids}


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class TaskRunWriter:
    """Assemble execution evidence over the single v2 journal/artifact store."""

    def __init__(self, output_dir: Path, run_id: str, *, records_dir: Path | None = None,
                 versions: dict[str, Any] | None = None, source_root: Path | None = None):
        self.output_dir, self.run_id = Path(output_dir).resolve(), run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.store = TaskRecordStore(records_dir or PROJECT_ROOT / "records")
        self.versions = recordable(versions or {})
        archive = {}
        base = Path(source_root or PROJECT_ROOT).resolve()
        for name, expected in self.versions.get("source_hashes", {}).items():
            source = (base / name).resolve()
            if source.is_relative_to(base) and source.is_file():
                content = source.read_bytes()
                if hashlib.sha256(content).hexdigest() == expected:
                    archive[name] = content.decode("utf-8")
        if archive:
            self.versions["source_snapshot_ref"] = self.store.put_artifact(archive)
        self.session_id = "session_" + uuid.uuid4().hex
        self.run = RunRecorder(self.store, run_id, session_id=self.session_id,
            manifest={"versions": self.versions, "report_directory": str(self.output_dir)})
        self.active_record: TaskRecord | None = None
        self.active_attempt: int | None = None

    @_serialized
    def begin_task(self, *, instruction: str, map_id: str, map_definition: Any,
                   initial_state: Any, metadata: dict[str, Any] | None = None) -> TaskRecord:
        if self.active_record is not None:
            raise RuntimeError("Finish the active task before starting another request")
        supplied = metadata or {}
        role = supplied.get("agent_role", "instruction" if supplied.get("agent", "robot") == "robot" else supplied.get("agent"))
        bundle = self.store.put_artifact(recordable(self.versions))
        record = self.store.begin(map_id=map_id, map_definition=recordable(map_definition),
            initial_state=recordable(initial_state), natural_language=instruction,
            versions=self.versions, metadata={**supplied, "run_id": self.run_id, "session_id": self.session_id,
                "agent_run_id": "agent_" + uuid.uuid4().hex, "agent_role": role,
                "task_kind": supplied.get("task_kind", "map_edit" if role == "environment" else "robot_task"),
                "behavior_bundle_id": bundle["sha256"], "behavior_bundle_ref": bundle,
                "trusted_task_spec": supplied.get("trusted_task_spec"),
                "episode_end_contract_ref": "application-task-v2",
                "checkpoint_availability": "unavailable", "policy_token_availability": "unavailable",
                "state_contract": {"kind": "declared_simulation_observation", "restorable": False}})
        self.active_record = record
        try:
            self.active_attempt = record.append_attempt()
            self.record_event("task_started", {"instruction": instruction}, initial_state)
            self.run.append("task.started", {"task_id": record.task_id,
                "uri": record.path.relative_to(self.store.root).as_posix()}, task_id=record.task_id)
        except BaseException:
            # Preserve the last durable RUNNING file, but release ownership so
            # a later request cannot append its evidence to this failed start.
            self.active_record, self.active_attempt = None, None
            raise
        return record

    def trace_identity(self) -> dict:
        record = self.active_record
        if record is None:
            return {}
        metadata = record.data["metadata"]
        return {"run_id": self.run_id, "session_id": self.session_id, "task_id": record.task_id,
                "agent_run_id": metadata["agent_run_id"], "agent_role": metadata["agent_role"],
                "attempt_id": self.active_attempt}

    def _fill_attempt(self, **fields: Any) -> None:
        record = self.active_record
        if record is None:
            return
        attempt = record.data["attempts"][self.active_attempt - 1]
        values = {key: recordable(value) for key, value in fields.items()
                  if value is not None and attempt.get(key) is None}
        if values:
            record.update_attempt(self.active_attempt, **values)

    @_serialized
    def record_event(self, event: str, detail: dict[str, Any] | None = None,
                     observation: Any = None, *, state: str | None = None,
                     episode_id: str | None = None) -> str:
        record = self.active_record
        if record is None:
            raise RuntimeError("Execution evidence requires an active task record")
        detail = recordable(detail or {})
        observation = recordable(observation)
        scope = {**current_trace(), **{key: detail[key] for key in ("decision_id", "model_call_id",
            "tool_call_id", "action_id", "agent_run_id", "agent_role", "parent_event_id", "caused_by_event_ids") if key in detail}}
        if detail.get("source"):
            source = detail["source"]
        elif event.startswith("model_") or event in {"agent_tool_call", "proposal_received"}:
            source = "llm"
        elif event in {"planner_response", "planning_response", "plan_received", "environment_proposal"}:
            source = record.data["metadata"].get("planner_source", "executor")
        elif event == "goal_evaluated":
            source = "evaluator"
        else:
            source = "executor"
        visible = "model_visible" if event in {"model_request", "model_response", "agent_tool_result",
            "feedback_included", "context_assembled", "memory_read", "knowledge_retrieval"} else "executor_only"
        row = {**scope, **self.trace_identity(), "event": event, "event_type": EVENT_TYPES.get(event, event.replace("_", ".")),
               "source": source, "visibility": detail.get("visibility", visible),
               "state": state, "episode_id": episode_id,
               "environment_instance_id": episode_id or record.data["metadata"].get("environment_instance_id"),
               "sim_step": (observation or {}).get("step") if isinstance(observation, dict) else None,
               "sim_time_s": (observation or {}).get("sim_time_s") if isinstance(observation, dict) else None,
               "detail": detail, "observation": observation}
        event_id = record.append_event(row, attempt_id=self.active_attempt)
        if event == "initial_state_observed" and record.data["initial_state"] is None:
            record.set_initial_state(observation if observation is not None else detail["snapshot"])
        if event in {"planning_started", "planner_request"}:
            self._fill_attempt(planning_observation=observation,
                               model_input=detail.get("input", detail))
        if event == "model_request":
            # Exact transport messages remain with every model_request event.
            # The first input field may also contain a business-level contract.
            self._fill_attempt(model_input=detail.get("request"))
        if event == "feedback_delivered" and detail.get("recipient") == "local_planner_invocation":
            attempt = record.data["attempts"][self.active_attempt - 1]
            record.set_attempt_feedback(detail.get("feedback"),
                attempt_id=attempt.get("parent_attempt_id") or self.active_attempt,
                destination="sent_to_agent")
        if event in {"planner_response", "planning_response"}:
            self._fill_attempt(proposal=detail)
        if event in {"plan_received", "plan_validated", "environment_plan"}:
            plan = detail.get("plan")
            self._fill_attempt(plan=plan)
            if plan is not None:
                encoded = json.dumps(plan, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(",", ":"))
                record.update_attempt(self.active_attempt,
                    metadata={"plan_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()})
        if event in {"before_tool", "after_tool", "skill_started", "post_skill_passed",
                     "skill_failed", "execution_failed", "safe_stop", "safe_stop_failed"}:
            record.append_tool_event({"source_event_id": event_id}, attempt_id=self.active_attempt)
        return event_id

    @_serialized
    def begin_recovery_attempt(self, index: int, feedback: dict | None, observation: Any) -> None:
        """Link real feedback to the previous attempt before the next decision."""
        if index <= 1 or self.active_record is None:
            return
        record, parent = self.active_record, self.active_attempt
        trigger = self.record_event("replanning_requested", {"feedback": feedback}, observation)
        self._fill_attempt(actual_execution={"completed_actions": (feedback or {}).get("completed_actions", []),
                                            "partial": True})
        record.update_attempt(parent, phase="FINISHED")
        self.active_attempt = record.append_attempt(parent_attempt_id=parent, trigger_event_id=trigger,
            planning_observation=recordable(observation), metadata={"decision_round": index})
        self.record_event("recovery_attempt_started", {"parent_attempt_id": parent, "feedback": feedback}, observation)

    def model_recorder(self):
        """Bind worker-thread evidence to this task, including cancelled replies."""
        target = self.active_record

        def capture(event: str, detail: dict[str, Any]) -> None:
            # A late reply after cancellation must not rewrite a finished task
            # or enter a subsequently started task's planning input.
            with self._lock:
                if target is not None and self.active_record is target and target.data["record_status"] == "RUNNING":
                    self.record_event(event, detail)
                elif target is not None:
                    identities = {key: detail[key] for key in ("model_call_id", "decision_id", "agent_run_id",
                        "agent_role", "tool_call_id", "action_id") if key in detail}
                    self.run.append("late." + EVENT_TYPES.get(event, event.replace("_", ".")),
                        {"original_event": event, "detail": recordable(detail), "consumed": False},
                        task_id=target.task_id, consumed=False, source="collector", **identities)
        return capture


    @_serialized
    def finish_task(self, result: dict[str, Any], *, final_state: Any,
                    actual_execution: dict[str, Any] | None = None) -> dict[str, Any]:
        record = self.active_record
        if record is None:
            raise RuntimeError("There is no active task to finish")
        result["token_usage"] = _aggregate_token_usage(self.store.read_events(record.task_id, resolve=True))
        feedback = recordable(result)
        for key in ("task_id", "task_record", "task_record_path"):
            feedback.pop(key, None)
        execution = actual_execution or {"episode_events": feedback.get("episode_events", []),
            "physics_events": [], "trajectory": [],
            "range": {"episode_id": feedback.get("episode_id"), "start_step": 0,
                      "end_step": feedback.get("step_count", 0), "events_start": 0,
                      "events_end": len(feedback.get("episode_events", [])),
                      "samples_start": 0, "samples_end": 0}}
        self._fill_attempt(actual_execution=execution)
        record.set_attempt_feedback(feedback, attempt_id=self.active_attempt)
        record.update_attempt(self.active_attempt, phase="FINISHED")
        task_predicates = feedback.get("postconditions", {})
        goal_check = result.get("goal_check")
        general_verified = (result.get("instruction_agent") == "universal-instruction-v1"
            and result.get("verified") is True and isinstance(goal_check, dict)
            and goal_check.get("passed") is True
            and any(row.get("predicate") in {"supported_on", "inside", "near", "robot_at", "held",
                    "state_equals", "released", "stable"} for row in goal_check.get("predicates", [])))
        if general_verified:
            task_predicates = goal_check
        # Transport and an independent evaluated Panda placement are distinct
        # from a normally returning query/stop or an environment edit.
        verified = result.get("status") == "SUCCESS" and not result.get("recording_errors") and (
            general_verified or result.get("transport_success") is True or (
                isinstance(task_predicates, dict) and
                task_predicates.get("all_instantaneous_conditions") is True and
                task_predicates.get("full_projected_footprint_inside_target_with_margin") is True
            )
        )
        costs = {key: feedback[key] for key in ("wall_time_s", "physics_steps", "step_count",
                 "llm_request_count", "token_usage")
                 if key in feedback}
        record.update(metadata={"initial_state_availability": (
            "observed" if record.data["initial_state"] is not None else "unavailable"),
            "final_state_available": final_state is not None})
        error_code = result.get("error_code")
        truncated = result["status"] == "ABORTED" or error_code in {"TASK_BUDGET_EXCEEDED", "TIMEOUT", "TASK_CANCELLED"}
        reason = ("clarification" if result.get("decision") == "clarify" else
                  "capability_rejection" if error_code == "CAPABILITY_GAP" else
                  error_code or ("completed" if result["status"] == "SUCCESS" else "task_failed"))
        try:
            record.finish(final_state=recordable(final_state), costs=costs,
                outcome={"status": result["status"], "error_code": result.get("error_code"),
                         "error_message": result.get("error_message"), "verified": verified,
                         "executor_verified": verified,
                         "agent_interpreted_goals": result.get("goals"), "goal_check": goal_check,
                         "persisted": result.get("persisted"), "proposal_valid": result.get("proposal_valid"),
                         "commit_status": result.get("commit_status", "success" if result.get("persisted") is True else
                             "not_started" if result.get("persisted") is False else "unknown"),
                         "reload_status": result.get("reload_status", "failed" if result.get("refresh_error") else "unknown"),
                         "effect_status": "committed" if result.get("persisted") is True else
                             "executed" if execution.get("physics_events") else "none_or_unverified",
                         "terminated": not truncated, "truncated": truncated, "termination_reason": reason,
                         "bootstrap_state_availability": "observation_only_not_restorable" if final_state is not None else "unavailable",
                         "record_integrity": "sealed", "episode_end_contract_ref": "application-task-v2",
                         "component_errors": result.get("component_errors", []),
                         "task_predicates": task_predicates,
                         "transport_success": result.get("transport_success"),
                         "recording_errors": result.get("recording_errors", [])})
        except (OSError, ValueError) as error:
            result["recording_error"] = {"type": type(error).__name__, "message": str(error)}
            result.update(task_id=record.task_id, task_record_path=str(record.path))
            self.active_record, self.active_attempt = None, None
            raise
        reference = self.store.ref(record.task_id)
        try:
            projection_hash = hashlib.sha256(record.path.read_bytes()).hexdigest()
        except OSError:
            projection_hash = None
        result.update(task_id=record.task_id, task_record_path=str(record.path),
                      task_record={**reference, "path": str(record.path),
                                   "uri": record.path.relative_to(self.store.root).as_posix(), "schema_version": 2,
                                   "sha256": projection_hash, "projection_sha256": projection_hash})
        try:
            self.run.append("task.ended", {"task_record": result["task_record"], "status": result["status"]}, task_id=record.task_id)
        finally:
            self.active_record, self.active_attempt = None, None
        return result

    def finish(self, results: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
        summary = {**manifest, "schema_version": 2, "run_id": self.run_id,
            "episode_count": len(results), "success_count": sum(row["status"] == "SUCCESS" for row in results),
            "failure_count": sum(row["status"] == "FAILED" for row in results),
            "aborted_count": sum(row["status"] == "ABORTED" for row in results),
            "all_pass": bool(results) and all(row["status"] == "SUCCESS" for row in results),
            "episodes": [self.report_result(row) for row in results]}
        summary["task_records"] = [row["task_record"] for row in results if "task_record" in row]
        summary["task_records_dir"] = str(self.store.root)
        summary["recording_failure_count"] = sum(bool(row.get("recording_error") or row.get("recording_errors")) for row in results)
        if summary["recording_failure_count"]:
            summary["all_pass"] = False
        atomic_json(self.output_dir / "summary.json", recordable(summary))
        self.run.append("run.summary.written", {"report": str(self.output_dir / "summary.json")})
        return summary

    @staticmethod
    def report_result(result: dict) -> dict:
        keys = {"task_id", "task_record", "task_record_path", "case_id", "name", "agent", "instruction",
            "status", "error_code", "error_message", "expected", "expected_pass", "expectation_failures",
            "passed", "planner_kind", "map_id", "map_revision", "robot_kind", "transport_success",
            "persisted", "commit_status", "reload_status", "wall_time_s", "llm_request_count", "query_tool_count",
            "recording_error", "recording_errors", "component_errors", "physical_evidence", "test_fault_protocol"}
        row = {key: value for key, value in result.items() if key in keys}
        if "agent" not in result and "actions" in result:
            row["actions"] = [TaskRunWriter.report_result(action) for action in result["actions"]]
        return row

    def close(self):
        self.run.close()

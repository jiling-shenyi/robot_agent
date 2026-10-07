"""Immutable, independently versioned assessments of sealed task evidence.

Observed goal verification and user-intent correctness remain separate. No
reward weights are chosen here; an absent trusted specification means no scalar
reward, even when the executor reported SUCCESS.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uuid

from embodied_agent.recording.common import decode


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _identifier(value: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}", value)
            or value in {".", ".."}):
        raise ValueError("Record identifiers must be single nonempty path components")
    return value


def write_once(path: Path, value: Any) -> None:
    """Publish a flushed file atomically without ever replacing an existing one."""
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def event_detail(event: dict) -> dict:
    """Extract the original execution payload from a resolved canonical event."""
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        return {}
    detail = payload.get("detail")
    return detail if isinstance(detail, dict) else payload


def event_name(event: dict) -> str:
    payload = event.get("payload", {})
    return event.get("event_type") or (payload.get("event") if isinstance(payload, dict) else "") or ""


def recording_evidence_complete(task: dict) -> bool:
    """Known collector failures differ from ordinary optional component failures."""
    outcome = task.get("outcome") or {}
    return (not bool(outcome.get("recording_errors")) and outcome.get("error_code") != "RECORDING_FAILED"
            and outcome.get("record_integrity") in (None, "sealed", "complete"))


def trusted_spec(task: dict) -> dict | None:
    spec = task.get("metadata", {}).get("trusted_task_spec", task.get("trusted_task_spec"))
    kinds = {"robot_goal", "map_edit", "query", "stop", "clarification", "rejection"}
    if (not isinstance(spec, dict) or spec.get("trusted") is not True
            or spec.get("source") not in {"user", "fixture"}
            or spec.get("kind") not in kinds):
        return None
    if spec["kind"] == "robot_goal" and (not isinstance(spec.get("goals"), list) or not spec["goals"]):
        return None
    if spec["kind"] == "robot_goal" and any(not isinstance(goal, dict) for goal in spec["goals"]):
        return None
    if spec["kind"] == "map_edit" and (not isinstance(spec.get("expected_operations"), list)
                                       or not spec["expected_operations"]):
        return None
    if spec["kind"] == "map_edit" and any(not isinstance(operation, dict) for operation in spec["expected_operations"]):
        return None
    return _clone(spec)


class EvaluationStore:
    """One immutable assessment per task and explicit evaluation run."""

    def __init__(self, root: str | Path, *, task_store=None):
        self.root = Path(root).resolve()
        if task_store is None:
            from embodied_agent.recording.store import TaskRecordStore
            task_store = TaskRecordStore(self.root)
        self.task_store = task_store

    def save(self, task_id: str, *, eval_run_id: str, evaluator_name: str,
             evaluator_version: str, components: dict, scalar_reward: float | None = None,
             evaluator_config: dict | None = None, evaluator_code_sha256: str | None = None,
             evidence_event_ids: list[str] | None = None, validity: dict | None = None,
             timing: str = "retrospective", trusted_task_spec: dict | None = None) -> dict:
        task_id, eval_run_id = _identifier(task_id), _identifier(eval_run_id)
        if (not isinstance(evaluator_name, str) or not evaluator_name.strip()
                or not isinstance(evaluator_version, str) or not evaluator_version.strip()):
            raise ValueError("Evaluator name and version are required")
        if not isinstance(components, dict) or timing not in {"online", "retrospective"}:
            raise ValueError("Assessment requires components and a valid timing source")
        if scalar_reward is not None and (isinstance(scalar_reward, bool)
                or not isinstance(scalar_reward, (int, float)) or not math.isfinite(scalar_reward)):
            raise ValueError("Scalar reward must be finite or null")
        task = self.task_store.load(task_id)
        check = self.task_store.verify(task_id)
        spec = trusted_spec(task)
        if trusted_task_spec is not None:
            supplied = trusted_spec({"metadata": {"trusted_task_spec": trusted_task_spec}})
            if supplied is None:
                raise ValueError("Explicit assessment task specification is not trusted and typed")
            spec = supplied
        events = self.task_store.read_events(task_id, resolve=True)
        by_id = {event["event_id"]: event for event in events}
        ids = evidence_event_ids if evidence_event_ids is not None else list(by_id)
        if not isinstance(ids, list) or any(identity not in by_id for identity in ids):
            raise ValueError("Assessment evidence must reference actual task events")
        usable = check.get("valid") is True and check.get("sealed") is True
        validity = _clone(validity or {"valid": usable, "reasons": [] if usable else ["task_not_sealed_or_integrity_failed"]})
        if not isinstance(validity, dict) or type(validity.get("valid")) is not bool:
            raise ValueError("validity requires a boolean valid field")
        if not usable:
            validity["valid"] = False
        if not ids:
            validity["valid"] = False
            validity.setdefault("reasons", []).append("assessment_evidence_unavailable")
        if spec is None or not validity["valid"]:
            scalar_reward = None
        config = _clone(evaluator_config or {})
        code_hash = evaluator_code_sha256 or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not isinstance(code_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", code_hash):
            raise ValueError("Evaluator code hash must be SHA-256")
        result = {"schema_version": 2, "evaluation_id": uuid.uuid4().hex,
            "eval_run_id": eval_run_id, "task_id": task_id, "scope": "episode",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "timing": timing,
            "evaluator": {"name": evaluator_name, "version": evaluator_version,
                          "code_sha256": code_hash,
                          "code_hash_scope": "provided_evaluator" if evaluator_code_sha256 else "assessment_module",
                          "config": config, "config_sha256": canonical_hash(config)},
            "source_task": {key: check.get(key) for key in
                ("manifest_sha256", "journal_sha256", "seal_sha256", "last_event_hash", "event_count")},
            "trusted_task_spec": spec, "trusted_task_spec_sha256": canonical_hash(spec) if spec is not None else None,
            "evidence": [{"event_id": identity, "resolved_sha256": canonical_hash(by_id[identity]),
                          "artifact_refs": _clone(by_id[identity].get("artifact_refs", []))} for identity in ids],
            "components": _clone(components), "scalar_reward": scalar_reward,
            "validity": validity}
        path = self.root / "evaluations" / eval_run_id / f"{task_id}.json"
        write_once(path, result)
        return {"evaluation_id": result["evaluation_id"], "eval_run_id": eval_run_id,
                "task_id": task_id, "path": path.relative_to(self.root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def load(self, reference: dict | str | Path) -> dict:
        requested = reference.get("path") if isinstance(reference, dict) else reference
        path = Path(requested)
        path = path if path.is_absolute() else self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.root / "evaluations"):
            raise ValueError("Evaluation reference must remain inside records/evaluations")
        raw = path.read_bytes()
        if isinstance(reference, dict) and reference.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("Evaluation hash does not match the frozen reference")
        result = decode(raw)
        if (not isinstance(result, dict) or result.get("schema_version") != 2
                or not isinstance(result.get("source_task"), dict)
                or not isinstance(result.get("components"), dict)
                or not isinstance(result.get("evaluator"), dict)
                or not isinstance(result.get("validity"), dict)
                or type(result["validity"].get("valid")) is not bool
                or not isinstance(result.get("evidence"), list)):
            raise ValueError("Unsupported assessment schema")
        if (result.get("trusted_task_spec_sha256") != (canonical_hash(result["trusted_task_spec"])
                if result.get("trusted_task_spec") is not None else None)
                or result["evaluator"].get("config_sha256") != canonical_hash(result["evaluator"].get("config"))):
            raise ValueError("Assessment specification or evaluator configuration hash does not match")
        reward = result.get("scalar_reward")
        if reward is not None and (isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward)):
            raise ValueError("Assessment scalar reward must be finite or null")
        if any(not isinstance(row, dict) or not isinstance(row.get("event_id"), str)
               or not isinstance(row.get("resolved_sha256"), str) for row in result["evidence"]):
            raise ValueError("Invalid assessment evidence")
        check = self.task_store.verify(result["task_id"])
        if any(result["source_task"].get(key) != check.get(key) for key in
               ("manifest_sha256", "journal_sha256", "seal_sha256")):
            raise ValueError("Assessment source task hashes no longer match")
        events = {event["event_id"]: event for event in self.task_store.read_events(result["task_id"], resolve=True)}
        if any(row["event_id"] not in events or row["resolved_sha256"] != canonical_hash(events[row["event_id"]])
               for row in result.get("evidence", [])):
            raise ValueError("Assessment evidence hashes no longer match")
        return result


def _find_map(detail: dict, names: tuple[str, ...]) -> dict | None:
    for name in names:
        value = detail.get(name)
        if isinstance(value, dict):
            return value
    return None


def _same_effect(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(expected, actual, rel_tol=0, abs_tol=1e-9)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(_same_effect(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(_same_effect(left, right) for left, right in zip(expected, actual))
    return expected == actual


def _expected_map(before: dict, operations: list[dict]) -> dict:
    """Pure expected-state transform; never changes a map file or runtime state."""
    candidate = copy.deepcopy(before)
    for operation in operations:
        op, identity = operation.get("op"), operation.get("object_id")
        if op == "set_name":
            candidate["name"] = operation["value"]
            continue
        if op == "set_description":
            target = candidate if identity == "world" else candidate["objects"][identity]
            target["description"] = operation["value"]
            continue
        if identity == "cube" and "cube_position_m" in candidate:
            target, field = candidate, "cube_position_m"
        else:
            target = (candidate["danger_zone"] if identity == "danger_zone" else
                      candidate["targets"][identity] if identity in candidate.get("targets", {}) else
                      candidate["objects"][identity])
            field = "position_m"
        if op == "set_position":
            target[field] = copy.deepcopy(operation["value"])
        elif op == "shift_axis":
            target[field]["xyz".index(operation["axis"])] += operation["delta_m"]
        elif op == "set_half_size":
            target["half_size_m"] = copy.deepcopy(operation["value"])
        elif op == "scale":
            target["half_size_m"] = [value * operation["factor"] for value in target["half_size_m"]]
        else:
            raise ValueError("Unsupported trusted map operation")
    return candidate


def assess_task(store, task_id: str, *, eval_run_id: str, scalar_reward: float | None = None,
                evaluator_config: dict | None = None, trusted_task_spec: dict | None = None) -> dict:
    """Evaluate supported explicit specifications; unknown intent remains null."""
    task = store.load(task_id)
    events = store.read_events(task_id, resolve=True)
    outcome = task.get("outcome") or {}
    spec = trusted_spec(task) if trusted_task_spec is None else trusted_spec(
        {"metadata": {"trusted_task_spec": trusted_task_spec}})
    if trusted_task_spec is not None and spec is None:
        raise ValueError("Assessment requires an explicitly trusted typed task specification")
    check = outcome.get("goal_check")
    verified = outcome.get("executor_verified", outcome.get("verified"))
    components = {"intent_correctness": None,
        "goal_verification": check.get("passed") if isinstance(check, dict) and type(check.get("passed")) is bool else None,
        "execution_success": None, "commit_status": outcome.get("commit_status", "unknown"),
        "reload_status": outcome.get("reload_status", "unknown"),
        "infrastructure_error": outcome.get("error_code") if outcome.get("error_code") in
            {"REFRESH_FAILED", "RECORDING_FAILED", "TIMEOUT", "CONNECTION_FAILED"} else None}
    details = [(event_name(event), event_detail(event)) for event in events]
    if spec is not None:
        kind = spec["kind"]
        if kind == "robot_goal":
            proposed = outcome.get("agent_interpreted_goals")
            components["intent_correctness"] = proposed == spec["goals"] if isinstance(proposed, list) else None
            if outcome.get("status") in {"FAILED", "ABORTED"}:
                components["execution_success"] = False
            elif outcome.get("status") == "SUCCESS" and type(verified) is bool and isinstance(check, dict) and type(check.get("passed")) is bool:
                components["execution_success"] = verified and check["passed"]
        elif kind == "map_edit":
            operations = None
            before = task.get("map", {}).get("definition")
            after = None
            for name, detail in details:
                proposal = detail.get("proposal", detail.get("plan", detail))
                if isinstance(proposal, dict) and isinstance(proposal.get("operations"), list):
                    operations = proposal["operations"]
                found = _find_map(detail, ("map_before", "stored_map_before", "before", "before_map"))
                if found is not None:
                    before = found
                if name in {"commit.finished", "environment_commit_finished"}:
                    after = _find_map(detail, ("map_after", "stored_map_after", "committed_map", "after", "saved_map"))
            components["intent_correctness"] = operations == spec["expected_operations"] if operations is not None else None
            components["operation_effect"] = None
            if isinstance(before, dict) and isinstance(after, dict):
                try:
                    expected = _expected_map(before, spec["expected_operations"])
                    actual = copy.deepcopy(after)
                    expected.pop("revision", None)
                    actual.pop("revision", None)
                    components["operation_effect"] = _same_effect(expected, actual)
                except (KeyError, TypeError, ValueError):
                    components["operation_effect"] = None
            commit_status = str(components["commit_status"]).lower()
            if commit_status in {"success", "committed"}:
                components["execution_success"] = components["operation_effect"]
            elif commit_status in {"failed", "not_started", "rejected"}:
                actual_failure = any(name in {"commit.rejected", "environment_commit_rejected",
                    "proposal.rejected", "environment_rejected"} for name, _ in details)
                if actual_failure:
                    components["execution_success"] = False
        elif kind in {"clarification", "rejection"}:
            expected = spec.get("expected_error_code", spec.get("error_code"))
            if expected is None:
                expected = "CLARIFICATION_REQUIRED" if kind == "clarification" else "CAPABILITY_GAP"
            correct = (outcome.get("error_code") == expected and
                       outcome.get("status") == spec.get("expected_status", "FAILED"))
            components["intent_correctness"] = components["execution_success"] = correct
        elif kind == "query":
            required = spec.get("query_names")
            successful = {detail.get("name") for name, detail in details
                if name in {"tool.finished", "agent_tool_result"} and
                isinstance(detail.get("result"), dict) and not detail["result"].get("error")}
            if isinstance(required, list) and required and all(isinstance(name, str) for name in required):
                correct = set(required) <= successful and outcome.get("status") == spec.get("expected_status", "SUCCESS")
                components["intent_correctness"] = components["execution_success"] = correct
        elif kind == "stop":
            completed = [(name, detail) for name, detail in details
                if name in {"action.finished", "after_tool", "post_skill_passed"}
                and isinstance(detail.get("action"), dict)]
            if completed:
                stopped = any(detail["action"].get("skill") == "stop" and
                    detail.get("status", (detail.get("skill_result") or {}).get("status")) == "SUCCESS"
                    for _, detail in completed)
                components["intent_correctness"] = components["execution_success"] = stopped and outcome.get("status") == "SUCCESS"
    validation = store.verify(task_id)
    known = spec is not None and components["intent_correctness"] is not None and components["execution_success"] is not None
    reasons = [] if known else ["trusted_intent_or_execution_evidence_unavailable"]
    complete = recording_evidence_complete(task)
    components["recording_evidence_complete"] = complete
    if not complete:
        reasons.append("recording_evidence_incomplete")
    validity = {"valid": known and complete and validation.get("valid") is True and validation.get("sealed") is True,
                "reasons": reasons}
    return EvaluationStore(store.root, task_store=store).save(task_id, eval_run_id=eval_run_id,
        evaluator_name="typed-task-spec", evaluator_version="typed-task-spec-v1",
        evaluator_config=evaluator_config, components=components, scalar_reward=scalar_reward,
        validity=validity, trusted_task_spec=trusted_task_spec)

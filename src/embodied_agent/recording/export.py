"""Freeze evidence views and purpose-specific qualification without training.

Exports retain actual assistant output, never executor-normalized proposals.
Dataset splits use connected provenance groups rather than individual rows.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from embodied_agent.recording.assessment import (
    EvaluationStore, canonical_hash, event_detail, event_name, recording_evidence_complete, trusted_spec, write_once,
)
from embodied_agent.recording.common import decode

EXPORT_KINDS = ("trajectory", "decision", "sft", "configuration", "task_pool", "preference")
EXPERIMENT_FIELDS = ("candidate_id", "rollout_group_id", "optimization_target", "branch_id")


def _experiment_fields(task: dict) -> dict:
    """Resolve public Session experiment metadata; conflicting duplicate fields reject export."""
    metadata = task.get("metadata", {})
    experiment = metadata.get("experiment")
    if experiment is None:
        experiment = {}
    if not isinstance(experiment, dict):
        raise ValueError("metadata.experiment must be an object or null")
    fields = {}
    for key in EXPERIMENT_FIELDS:
        top, nested = metadata.get(key), experiment.get(key)
        if top not in (None, "") and nested not in (None, "") and top != nested:
            raise ValueError(f"Conflicting {key} in metadata and metadata.experiment for {task['task_id']}")
        fields[key] = top if top not in (None, "") else nested
    return fields


def _tool_evidence(pair: dict, events: list[dict]) -> dict:
    """Link emitted native calls to validated local dispatch and completion facts."""
    calls = pair["raw_assistant"].get("tool_calls")
    empty = {"has_tool_calls": False, "requests_validated": True, "completed": True,
             "successful": True, "source_event_ids": [], "exclusion_reasons": []}
    if not calls:
        if pair["response_metadata"].get("finish_reason") == "tool_calls":
            return {**empty, "requests_validated": False, "completed": False, "successful": None,
                    "exclusion_reasons": ["tool_call_payload_unavailable"]}
        return empty
    answer = {**empty, "has_tool_calls": True}
    if not isinstance(calls, list) or pair["response_metadata"].get("finish_reason") != "tool_calls":
        return {**answer, "requests_validated": False, "completed": False, "successful": None,
                "exclusion_reasons": ["invalid_tool_call_protocol"]}
    related = [event for event in events if event.get("model_call_id") == pair["model_call_id"]]
    offered = {tool.get("function", {}).get("name") for tool in pair["request"].get("tools", [])
               if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
               and isinstance(tool["function"].get("name"), str)}
    native_ids = set()
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        native_id = call.get("id") if isinstance(call, dict) else None
        if (not isinstance(native_id, str) or not native_id or native_id in native_ids
                or call.get("type") != "function" or not isinstance(function, dict)
                or not isinstance(function.get("name"), str) or function.get("name") not in offered):
            answer["exclusion_reasons"].append("invalid_or_unregistered_tool_call")
            continue
        native_ids.add(native_id)
        try:
            arguments = decode(function.get("arguments"))
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
        except ValueError:
            answer["exclusion_reasons"].append("invalid_tool_arguments")
            continue
        requested = [event for event in related if event_name(event) in {"tool.requested", "agent_tool_call"}
            and (event_detail(event).get("native_tool_call_id", event.get("tool_call_id")) == native_id)
            and event_detail(event).get("name") == function["name"]
            and event_detail(event).get("arguments") == arguments]
        if len(requested) != 1 or not requested[0].get("tool_call_id"):
            answer["exclusion_reasons"].append("validated_tool_request_unavailable")
            continue
        issued = requested[0]
        answer["source_event_ids"].append(issued["event_id"])
        finished = [event for event in related if event_name(event) in {"tool.finished", "agent_tool_result"}
            and event.get("tool_call_id") == issued["tool_call_id"]
            and event_detail(event).get("native_tool_call_id", native_id) == native_id
            and event_detail(event).get("name") == function["name"]]
        if len(finished) != 1:
            answer["completed"] = False
            answer["successful"] = None
            continue
        answer["source_event_ids"].append(finished[0]["event_id"])
        detail = event_detail(finished[0])
        result = detail.get("result")
        success = (isinstance(result, dict) and not result.get("error") and result.get("ok") is not False
                   and result.get("status") not in ("FAILED", "ABORTED") and not detail.get("handler_error"))
        if not success and answer["successful"] is not None:
            answer["successful"] = False
    if answer["exclusion_reasons"]:
        answer.update(requests_validated=False, completed=False, successful=None)
    return answer


def _sft_pair_reasons(pair: dict) -> list[str]:
    reasons = []
    if not pair["actual_model_sample"]:
        reasons.append("actual_model_sample_unavailable")
    if not pair["response_consumed"]:
        reasons.append("response_consumption_unconfirmed")
    if pair["proposal_rejected"]:
        reasons.append("proposal_rejected")
    if pair["response_metadata"].get("finish_reason") not in {"stop", "tool_calls"}:
        reasons.append("assistant_completion_unavailable")
    if not (pair["raw_assistant"].get("content") or pair["raw_assistant"].get("tool_calls")):
        reasons.append("raw_assistant_output_unavailable")
    evidence = pair["tool_evidence"]
    reasons.extend(evidence["exclusion_reasons"])
    if not evidence["completed"]:
        reasons.append("tool_completion_unavailable")
    if evidence["successful"] is not True:
        reasons.append("tool_execution_success_unconfirmed")
    return sorted(set(reasons))


def _budget_evidence(task: dict, pairs: list[dict]) -> dict | None:
    """Use recorded limits only; provider defaults and missing sections stay unknown."""
    runtime = task.get("versions", {}).get("runtime_config")
    if not isinstance(runtime, dict) or not pairs:
        return None
    required = {"budgets": ("planner_timeout_s", "max_episode_steps"),
        "agent": ("max_rounds", "max_tool_calls", "max_output_tokens"),
        "task_budget": ("max_requests", "max_tokens", "max_actions", "max_replans", "timeout_s"),
        "execution": ("max_decision_rounds",)}
    for section, fields in required.items():
        limits = runtime.get(section)
        if not isinstance(limits, dict) or any(field not in limits for field in fields):
            return None
        if any(isinstance(limits[field], bool) or not isinstance(limits[field], (int, float))
               or not math.isfinite(limits[field]) or limits[field] < 0 for field in fields):
            return None
    def constraints(section):
        return {key: value for key, value in runtime[section].items()
                if key.startswith("max_") or "timeout" in key or "budget" in key or "limit" in key}
    request_limits = []
    for pair in pairs:
        request = pair["request"]
        token_keys = ("max_tokens", "max_completion_tokens", "max_output_tokens")
        if not any(type(request.get(key)) is int and request[key] > 0 for key in token_keys):
            return None
        request_limits.append({key: request[key] for key in (*token_keys, "timeout", "n") if key in request})
    task_limits = runtime["task_budget"]
    for attempt in task.get("attempts", []):
        feedback = attempt.get("feedback", {}).get("return_to_caller")
        snapshot = feedback.get("task_budget") if isinstance(feedback, dict) else None
        if isinstance(snapshot, dict) and isinstance(snapshot.get("limits"), dict):
            task_limits = snapshot["limits"]
    return {"runtime_budgets": runtime["budgets"], "agent_limits": constraints("agent"),
            "task_limits": task_limits, "execution_limits": constraints("execution"),
            "request_limits": request_limits}


def _model_pairs(events: list[dict]) -> list[dict]:
    requests, responses, rejected, delivered = {}, {}, set(), set()
    for event in events:
        identity = event.get("model_call_id")
        if not identity:
            continue
        name, detail = event_name(event), event_detail(event)
        if name in {"model.requested", "model_request", "model.call.started"}:
            request = detail.get("request")
            if isinstance(request, dict):
                requests[identity] = {"event": event, "request": request}
        elif name in {"model.responded", "model_response", "model.call.completed"}:
            if detail.get("consumed", True) is not False:
                responses[identity] = {"event": event, "response": detail}
        elif name in {"proposal.rejected", "proposal_rejected"}:
            rejected.add(identity)
        elif name in {"feedback.delivered", "model_response_delivered"} and detail.get("recipient") in {
                "dialogue_response_parser", "planner_response_parser"}:
            delivered.add(identity)
    pairs = []
    for identity, request in requests.items():
        if identity not in responses:
            continue
        response = responses[identity]
        detail = response["response"]
        message = detail.get("message") if isinstance(detail.get("message"), dict) else {}
        assistant = {"role": "assistant", "content": detail.get("response_text", detail.get("content", message.get("content")))}
        calls = detail.get("tool_calls", message.get("tool_calls"))
        if calls:
            assistant["tool_calls"] = calls
        source = response["event"].get("source", {})
        kind = source.get("kind") if isinstance(source, dict) else source
        real = (kind == "llm"
                and not (isinstance(source, dict) and source.get("synthetic") is True))
        pair = {"model_call_id": identity,
            "decision_id": request["event"].get("decision_id"),
            "request_event_id": request["event"]["event_id"],
            "response_event_id": response["event"]["event_id"],
            "request": request["request"], "raw_assistant": assistant,
            "response_metadata": detail, "actual_model_sample": real,
            "response_consumed": detail.get("consumed") is True or identity in delivered,
            "proposal_rejected": identity in rejected}
        pair["tool_evidence"] = _tool_evidence(pair, events)
        pairs.append(pair)
    return pairs


def _connections(task: dict, events: list[dict]) -> set[str]:
    metadata, identity = task.get("metadata", {}), task.get("identity", {})
    tokens = {"task:" + task["task_id"]}
    for key in ("run_id", "session_id", "task_family_id", "case_id", "parent_case_id"):
        value = metadata.get(key, identity.get(key, task.get(key)))
        if value is not None and value != "":
            tokens.add(("case_id" if key == "parent_case_id" else key) + ":" + str(value))
    for key, value in _experiment_fields(task).items():
        if key in {"rollout_group_id", "branch_id"} and value not in (None, ""):
            tokens.add(key + ":" + (value if isinstance(value, str) else canonical_hash(value)))
    initial = task.get("initial_state")
    if initial:
        tokens.add("map_start:" + canonical_hash({"map_id": task.get("map", {}).get("id"), "initial": initial}))
    for value in metadata.get("memory_connection_ids", []):
        tokens.add("memory_connection:" + str(value))
    for event in events:
        if event_name(event) in {"memory.read", "memory_read", "memory.write", "memory_write"}:
            detail = event_detail(event)
            context = detail.get("context", detail.get("entry", detail))
            namespace = context.get("namespace") if isinstance(context, dict) else None
            if namespace:
                tokens.add("memory_namespace:" + str(namespace))
    return tokens


def _split_groups(entries: list[dict], ratios: dict[str, float]) -> dict[str, dict]:
    if (not ratios or any(not isinstance(key, str) or not key for key in ratios)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in ratios.values())
            or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9)):
        raise ValueError("Split ratios must be nonnegative and sum to one")
    parent = {entry["task"]["task_id"]: entry["task"]["task_id"] for entry in entries}
    def find(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity
    owners = {}
    for entry in entries:
        task_id = entry["task"]["task_id"]
        for token in entry["connections"]:
            if token in owners:
                left, right = find(task_id), find(owners[token])
                parent[max(left, right)] = min(left, right)
            owners[token] = task_id
    groups = {}
    for entry in entries:
        groups.setdefault(find(entry["task"]["task_id"]), set()).update(entry["connections"])
    assigned = {}
    for entry in entries:
        task_id = entry["task"]["task_id"]
        group = canonical_hash(sorted(groups[find(task_id)]))
        selector, cumulative, split = int(group[:16], 16) / 2**64, 0, None
        for name, fraction in ratios.items():
            cumulative += fraction
            if selector < cumulative:
                split = name
                break
        assigned[task_id] = {"group_id": group, "split": split or next(reversed(ratios))}
    return assigned


def _availability(task: dict, pairs: list[dict]) -> dict:
    responses = [pair["response_metadata"] for pair in pairs]
    metadata = task.get("metadata", {})
    reproduction = metadata.get("reproducibility", task.get("reproducibility", {}))
    return {"model_input_output": bool(pairs),
        "recording_evidence_complete": recording_evidence_complete(task),
        "provider_logprobs": bool(responses) and all(response.get("logprobs") is not None for response in responses),
        "behavior_logprobs": bool(responses) and all(response.get("behavior_logprobs") is not None for response in responses),
        "completion_token_ids": bool(responses) and all(response.get("completion_token_ids", response.get("token_ids")) is not None for response in responses),
        "tokenizer_version": bool(metadata.get("tokenizer_version")),
        "behavior_policy": bool(metadata.get("behavior_policy_id")),
        "trusted_task_spec": trusted_spec(task) is not None,
        "reset_spec": isinstance(metadata.get("reset_spec"), dict) and metadata.get("reset_verified") is True,
        "checkpoint_restorable": isinstance(reproduction, dict) and reproduction.get("level") == "checkpoint_restorable"
            and reproduction.get("restore_verified") is True and bool(reproduction.get("checkpoint_ref"))}


def export_dataset(store, output_dir: str | Path, *, kind: str = "trajectory",
                   task_ids: list[str] | None = None, evaluation_refs: list[dict | str | Path] | None = None,
                   split_ratios: dict[str, float] | None = None, dataset_id: str | None = None) -> dict:
    """Write a new frozen dataset and explicit exclusions for the requested use."""
    if kind not in EXPORT_KINDS:
        raise ValueError(f"Unknown export kind: {kind}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Dataset output already exists: {output}")
    if output.is_relative_to(Path(store.root).resolve() / "tasks"):
        raise ValueError("Datasets cannot be written inside the canonical task directory")
    tasks = [store.load(identity) for identity in task_ids] if task_ids is not None else store.query()
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("A task may only appear once in a dataset")
    evaluations, frozen_evaluations = {}, []
    evaluator = EvaluationStore(store.root, task_store=store)
    for reference in evaluation_refs or []:
        value = evaluator.load(reference)
        if value["task_id"] in evaluations:
            raise ValueError("Choose one explicit evaluation per task per export")
        evaluations[value["task_id"]] = value
        path = Path(reference["path"] if isinstance(reference, dict) else reference)
        absolute = path if path.is_absolute() else evaluator.root / path
        frozen_evaluations.append({"task_id": value["task_id"], "evaluation_id": value["evaluation_id"],
            "path": absolute.resolve().relative_to(evaluator.root).as_posix(),
            "sha256": hashlib.sha256(absolute.read_bytes()).hexdigest(), "assessment": value})
    entries = []
    for task in tasks:
        verification = store.verify(task["task_id"])
        events = store.read_events(task["task_id"], resolve=True)
        pairs = _model_pairs(events)
        entry = {"task": task, "events": events, "pairs": pairs, "verification": verification,
                 "evaluation": evaluations.get(task["task_id"]), "connections": _connections(task, events)}
        entries.append(entry)
    assignments = _split_groups(entries, split_ratios or {"train": .8, "validation": .1, "test": .1})
    samples, qualification = [], []
    for entry in entries:
        task, events, pairs = entry["task"], entry["events"], entry["pairs"]
        task_id = task["task_id"]
        check, evaluation = entry["verification"], entry["evaluation"]
        available = _availability(task, pairs)
        entry["budget_evidence"] = _budget_evidence(task, pairs)
        available["comparable_budget"] = entry["budget_evidence"] is not None
        if evaluation and trusted_spec({"metadata": {"trusted_task_spec": evaluation.get("trusted_task_spec")}}):
            available["trusted_task_spec"] = True
        entry["effective_spec"] = (evaluation.get("trusted_task_spec") if evaluation else None) or trusted_spec(task)
        reasons = []
        if check.get("valid") is not True:
            reasons.append("record_integrity_failed")
        if check.get("sealed") is not True:
            reasons.append("task_not_sealed")
        if kind in {"sft", "preference", "task_pool"} and not available["recording_evidence_complete"]:
            reasons.append("recording_evidence_incomplete")
        valid_evaluation = bool(evaluation and evaluation.get("validity", {}).get("valid") is True)
        positive = valid_evaluation and evaluation.get("components", {}).get("intent_correctness") is True and evaluation.get("components", {}).get("execution_success") is True
        if kind in {"sft", "preference", "task_pool"} and not available["trusted_task_spec"]:
            reasons.append("trusted_task_spec_unavailable")
        if kind in {"sft", "preference"} and not valid_evaluation:
            reasons.append("trusted_evaluation_unavailable")
        if kind == "sft" and not positive:
            reasons.append("independent_intent_and_success_not_confirmed")
        if kind in {"sft", "preference", "decision"} and not pairs:
            reasons.append("paired_actual_model_input_output_unavailable")
        if kind == "task_pool" and not (available["reset_spec"] or available["checkpoint_restorable"]):
            reasons.append("verified_reset_or_checkpoint_unavailable")
        if kind == "preference" and (not evaluation or evaluation.get("scalar_reward") is None):
            reasons.append("explicit_comparable_scalar_reward_unavailable")
        if kind == "preference" and not available["comparable_budget"]:
            reasons.append("comparable_budget_unavailable")
        if kind == "preference" and not (available["reset_spec"] or available["checkpoint_restorable"] or
                task.get("task_kind") == "map_edit" and bool(task.get("map", {}).get("definition"))):
            reasons.append("comparable_start_not_verified")
        rl_reasons = [name + "_unavailable" for name in
            ("behavior_logprobs", "completion_token_ids", "tokenizer_version", "behavior_policy") if not available[name]]
        if not evaluation or evaluation.get("scalar_reward") is None:
            rl_reasons.append("trusted_scalar_reward_unavailable")
        if not pairs or not all(pair["actual_model_sample"] for pair in pairs):
            rl_reasons.append("actual_behavior_policy_samples_unavailable")
        if not available["recording_evidence_complete"]:
            rl_reasons.append("recording_evidence_incomplete")
        rl_reasons.extend(reasons[:])
        rl_reasons.append("algorithm_adapter_not_selected")
        report = {"task_id": task_id, "requested_use": kind, "eligible": not reasons,
            "exclusion_reasons": sorted(set(reasons)), "availability": available,
            "recording_evidence_complete": available["recording_evidence_complete"],
            "analysis_only": kind in {"trajectory", "configuration", "decision"} and not available["recording_evidence_complete"],
            "budget_evidence": entry["budget_evidence"],
            "model_calls": [{"model_call_id": pair["model_call_id"],
                             "sft_eligible": positive and available["trusted_task_spec"] and available["recording_evidence_complete"] and not _sft_pair_reasons(pair),
                             "exclusion_reasons": _sft_pair_reasons(pair) + ([] if available["recording_evidence_complete"] else ["recording_evidence_incomplete"]),
                             "tool_evidence": pair["tool_evidence"]}
                            for pair in pairs],
            "rl_update": {"eligible": False, "exclusion_reasons": sorted(set(rl_reasons))},
            **assignments[task_id]}
        qualification.append(report)
        entry["qualification"] = report
        if reasons:
            continue
        common = {"task_id": task_id, "source_event_ids": [event["event_id"] for event in events],
                  "evaluation_id": evaluation["evaluation_id"] if evaluation else None,
                  "recording_evidence_complete": available["recording_evidence_complete"], **assignments[task_id]}
        if kind == "trajectory":
            samples.append({**common, "task": task, "events": events})
        elif kind == "configuration":
            samples.append({**common, "behavior": task.get("versions", {}),
                "behavior_sha256": canonical_hash(task.get("versions", {})),
                **_experiment_fields(task), "experiment": task.get("metadata", {}).get("experiment"),
                "outcome": task.get("outcome"), "costs": task.get("costs", {}),
                "evaluation": evaluation})
        elif kind == "task_pool":
            samples.append({**common, "instruction": task.get("natural_language"),
                "trusted_task_spec": entry["effective_spec"], "map": task.get("map"),
                "initial_state": task.get("initial_state"),
                "reset_spec": task.get("metadata", {}).get("reset_spec"),
                "reproducibility": task.get("metadata", {}).get("reproducibility")})
        elif kind == "sft":
            selected = [pair for pair in pairs if not _sft_pair_reasons(pair)]
            if not selected:
                report["eligible"] = False
                report["exclusion_reasons"].append("trusted_raw_assistant_output_unavailable")
            for pair in selected:
                samples.append({**common, "model_call_id": pair["model_call_id"],
                    "source_event_ids": [pair["request_event_id"], pair["response_event_id"],
                                         *pair["tool_evidence"]["source_event_ids"]],
                    "messages": pair["request"].get("messages", []),
                    "tools": pair["request"].get("tools", []), "assistant": pair["raw_assistant"],
                    "tool_evidence": pair["tool_evidence"],
                    "loss_scope": "raw_assistant_generated_tokens_only"})
        elif kind == "decision":
            decision_ids = list(dict.fromkeys(pair["decision_id"] for pair in pairs if pair["decision_id"]))
            for decision_id in decision_ids:
                related = [event for event in events if event.get("decision_id") == decision_id]
                before, after, framework_action = None, None, None
                delivered = []
                for event in related:
                    payload, detail, name = event.get("payload", {}), event_detail(event), event_name(event)
                    observation = payload.get("observation", detail.get("observation")) if isinstance(payload, dict) else None
                    if name in {"decision.started", "decision_started"}:
                        before = observation
                    if observation is not None and name in {"action.finished", "after_tool", "post_skill_passed", "execution_failed", "goal.evaluated"}:
                        after = observation
                    if name in {"proposal.validated", "decision.finished", "decision_proposal", "plan_validated"}:
                        framework_action = detail.get("plan", detail.get("proposal", framework_action))
                    if name in {"feedback.delivered", "feedback_delivered"}:
                        delivered.append({"event_id": event["event_id"], "payload": detail})
                terminal = task.get("termination") or task.get("outcome") or {}
                is_last = decision_id == decision_ids[-1]
                samples.append({**common, "decision_id": decision_id,
                    "source_event_ids": [event["event_id"] for event in related],
                    "model_calls": [pair for pair in pairs if pair["decision_id"] == decision_id],
                    "observation_before": before, "next_observation": after,
                    "framework_action": framework_action, "feedback_delivered": delivered,
                    "evaluation": evaluation, "terminated": terminal.get("terminated") if is_last else False,
                    "truncated": terminal.get("truncated") if is_last else False,
                    "availability": {"observation_before": before is not None, "next_observation": after is not None,
                                     "framework_action": framework_action is not None,
                                     "recording_evidence_complete": available["recording_evidence_complete"]},
                    "events": related, "outcome": task.get("outcome"),
                    "transition_status": "requires_algorithm_specific_state_and_reward_adapter" if available["recording_evidence_complete"] else "incomplete_evidence_analysis_only"})
            if not any(pair["decision_id"] for pair in pairs):
                report["eligible"] = False
                report["exclusion_reasons"].append("decision_identity_unavailable")
    if kind == "preference":
        participating = set()
        for left, right in itertools.combinations(entries, 2):
            if not left["qualification"]["eligible"] or not right["qualification"]["eligible"]:
                continue
            lp, rp = left["pairs"], right["pairs"]
            if (len(lp) != 1 or len(rp) != 1 or not lp[0]["actual_model_sample"] or not rp[0]["actual_model_sample"]
                    or not lp[0]["response_consumed"] or not rp[0]["response_consumed"]
                    or lp[0]["proposal_rejected"] or rp[0]["proposal_rejected"]
                    or not lp[0]["tool_evidence"]["requests_validated"] or not rp[0]["tool_evidence"]["requests_validated"]
                    or not lp[0]["tool_evidence"]["completed"] or not rp[0]["tool_evidence"]["completed"]
                    or lp[0]["response_metadata"].get("finish_reason") not in {"stop", "tool_calls"}
                    or rp[0]["response_metadata"].get("finish_reason") not in {"stop", "tool_calls"}):
                continue
            lt, rt = left["task"], right["task"]
            input_left = {key: lp[0]["request"].get(key) for key in ("messages", "tools")}
            input_right = {key: rp[0]["request"].get(key) for key in ("messages", "tools")}
            if (input_left != input_right or lt.get("initial_state") != rt.get("initial_state")
                    or lt.get("map") != rt.get("map") or left["effective_spec"] != right["effective_spec"]
                    or left["budget_evidence"] != right["budget_evidence"]):
                continue
            if lp[0]["raw_assistant"] == rp[0]["raw_assistant"]:
                continue
            evaluator_fields = ("name", "version", "code_sha256", "config_sha256")
            if any(left["evaluation"].get("evaluator", {}).get(key) !=
                   right["evaluation"].get("evaluator", {}).get(key) for key in evaluator_fields):
                continue
            lreward, rreward = left["evaluation"]["scalar_reward"], right["evaluation"]["scalar_reward"]
            if lreward == rreward:
                continue
            winner, loser = (left, right) if lreward > rreward else (right, left)
            if winner["pairs"][0]["tool_evidence"]["successful"] is not True:
                continue
            task_id = winner["task"]["task_id"]
            # Identical map/start already connects both tasks into the same split.
            if assignments[task_id] != assignments[loser["task"]["task_id"]]:
                raise ValueError("Comparable preference candidates must share a provenance split")
            samples.append({"task_ids": [winner["task"]["task_id"], loser["task"]["task_id"]],
                "messages": winner["pairs"][0]["request"].get("messages", []),
                "tools": winner["pairs"][0]["request"].get("tools", []),
                "chosen": winner["pairs"][0]["raw_assistant"], "rejected": loser["pairs"][0]["raw_assistant"],
                "evaluation_ids": [winner["evaluation"]["evaluation_id"], loser["evaluation"]["evaluation_id"]],
                "budget_evidence": winner["budget_evidence"],
                "source_event_ids": [pair[key] for entry in (winner, loser) for pair in entry["pairs"]
                                     for key in ("request_event_id", "response_event_id")], **assignments[task_id]})
            participating.update((lt["task_id"], rt["task_id"]))
        for report in qualification:
            if report["eligible"] and report["task_id"] not in participating:
                report["eligible"] = False
                report["exclusion_reasons"].append("same_input_start_and_evaluation_pair_unavailable")
    manifest = {"schema_version": 2, "dataset_id": dataset_id or output.name,
        "exporter": {"version": "evidence-export-v1", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "kind": kind,
        "sample_count": len(samples), "task_count": len(tasks), "split_ratios": split_ratios or {"train": .8, "validation": .1, "test": .1},
        "experiment_metadata_policy": {"fields": list(EXPERIMENT_FIELDS), "duplicate_conflicts": "reject",
            "grouping_fields": ["rollout_group_id", "branch_id"],
            "configuration_control_fields": ["candidate_id", "optimization_target"]},
        "source_tasks": [{"task_id": entry["task"]["task_id"],
            "hashes": {key: entry["verification"].get(key) for key in
                ("manifest_sha256", "journal_sha256", "seal_sha256", "last_event_hash")},
            "split": assignments[entry["task"]["task_id"]]} for entry in entries],
        "evaluations": frozen_evaluations, "tokenization": "not_performed",
        "training": "no_trainer_or_policy_update", "qualification_ref": "qualification.json", "samples_ref": "samples.json"}
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "samples.json", {"kind": kind, "samples": samples})
    write_once(output / "qualification.json", {"tasks": qualification})
    manifest["file_hashes"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                              for name in ("samples.json", "qualification.json")}
    write_once(output / "manifest.json", manifest)
    write_once(output / "seal.json", {"schema_version": 2, "manifest_sha256": hashlib.sha256(
        (output / "manifest.json").read_bytes()).hexdigest(), "file_hashes": manifest["file_hashes"]})
    return manifest

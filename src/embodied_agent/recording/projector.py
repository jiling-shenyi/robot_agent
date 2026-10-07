"""Rebuild the convenient task view solely from manifest and validated events."""
from __future__ import annotations

import copy
import re

from embodied_agent.recording.common import SCHEMA_VERSION, TaskRecordError, clone

TERMINAL = frozenset({"COMPLETED", "FAILED", "ABORTED"})
IMMUTABLE_ATTEMPT_FIELDS = frozenset({"planning_observation", "model_input", "plan", "proposal",
                                     "actual_execution", "parent_attempt_id", "trigger_event_id"})
ATTEMPT_FIELDS = IMMUTABLE_ATTEMPT_FIELDS | {"phase", "metadata"}


def attempt(data: dict, attempt_id: int | None) -> dict:
    if attempt_id is None:
        attempt_id = len(data["attempts"])
    if type(attempt_id) is not int or not 1 <= attempt_id <= len(data["attempts"]):
        raise TaskRecordError("attempt_id does not identify an existing attempt")
    return data["attempts"][attempt_id - 1]


def initial(manifest: dict, get_artifact) -> dict:
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise TaskRecordError("Unsupported task manifest schema_version; v1 is not readable")
    required = {"task_id", "date", "sequence", "map", "created_at", "identities", "task_kind",
                "natural_language_ref", "initial_state_ref", "versions_ref", "metadata_ref"}
    if required - manifest.keys():
        raise TaskRecordError("Task manifest is missing required fields")
    if not isinstance(manifest["task_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", manifest["task_id"]):
        raise TaskRecordError("Invalid manifest task_id")
    if not isinstance(manifest["date"], str) or not re.fullmatch(r"\d{8}", manifest["date"]):
        raise TaskRecordError("Invalid manifest Shanghai start date")
    if type(manifest["sequence"]) is not int or manifest["sequence"] < 1:
        raise TaskRecordError("Invalid task sequence")
    map_definition = get_artifact(manifest["map"]["definition_ref"])
    metadata = get_artifact(manifest["metadata_ref"])
    versions = get_artifact(manifest["versions_ref"])
    language = get_artifact(manifest["natural_language_ref"])
    if not isinstance(metadata, dict) or not isinstance(versions, dict) or not isinstance(language, str):
        raise TaskRecordError("Invalid manifest metadata/versions/instruction artifacts")
    data = {"schema_version": SCHEMA_VERSION, "task_id": manifest["task_id"], "date": manifest["date"],
            "sequence": manifest["sequence"], "map": {"id": manifest["map"]["id"],
                "alias": manifest["map"]["alias"], "definition": map_definition,
                "definition_sha256": manifest["map"]["definition_ref"]["sha256"]},
            "initial_state": get_artifact(manifest["initial_state_ref"]), "natural_language": language,
            "versions": versions, "metadata": metadata, "attempts": [], "final_state": None,
            "outcome": None, "costs": {}, "record_status": "RUNNING", "created_at": manifest["created_at"],
            "updated_at": manifest["created_at"], "finished_at": None,
            "identity": copy.deepcopy(manifest["identities"]), "task_kind": manifest["task_kind"],
            "behavior_bundle_id": manifest.get("behavior_bundle_id"),
            "trusted_task_spec": get_artifact(manifest["trusted_task_spec_ref"]) if manifest.get("trusted_task_spec_ref") else None,
            "termination": None}
    data.update(copy.deepcopy(manifest["identities"]))
    return data


def apply(data: dict, event: dict) -> None:
    if data["record_status"] != "RUNNING":
        raise TaskRecordError("A terminal task cannot contain later fact events")
    payload, operation, timestamp = event["payload"], event["operation"], event["timestamp"]
    if not isinstance(payload, dict):
        raise TaskRecordError("Projected event payload must be an object")
    if operation == "append_attempt":
        if payload.get("attempt_id") != len(data["attempts"]) + 1:
            raise TaskRecordError("Attempt IDs must be continuous")
        parent = payload.get("parent_attempt_id")
        if parent is not None:
            attempt(data, parent)
        row = copy.deepcopy(payload)
        row.update(started_at=timestamp, updated_at=timestamp, phase="PLANNING",
                   actual_execution=None, events=[], checks=[], tool_events=[],
                   feedback={"return_to_caller": None, "sent_to_agent": []})
        data["attempts"].append(row)
    elif operation == "update_attempt":
        row = attempt(data, event["attempt_id"])
        fields = payload["fields"]
        if not isinstance(fields, dict) or set(fields) - ATTEMPT_FIELDS:
            raise TaskRecordError("Unsupported attempt update fields")
        for key, value in fields.items():
            if key in IMMUTABLE_ATTEMPT_FIELDS and row[key] is not None:
                raise TaskRecordError(f"Attempt {key} is immutable once set")
            if key == "parent_attempt_id" and value is not None:
                attempt(data, value)
                if value >= row["attempt_id"]:
                    raise TaskRecordError("Attempt parent must precede its child")
            if key == "metadata":
                if not isinstance(value, dict):
                    raise TaskRecordError("Attempt metadata must be an object")
                row[key].update(value)
            else:
                row[key] = copy.deepcopy(value)
        row["updated_at"] = timestamp
    elif operation == "append_event":
        row = attempt(data, event["attempt_id"])
        collection = event.get("collection", "events")
        if collection not in {"events", "checks", "tool_events"}:
            raise TaskRecordError("Unknown projected event collection")
        item = copy.deepcopy(payload)
        item.update(event_id=event["event_id"], sequence=len(row[collection]) + 1)
        item.setdefault("timestamp", timestamp)
        row[collection].append(item)
        row["updated_at"] = timestamp
    elif operation == "tool_view":
        row = attempt(data, event["attempt_id"])
        source_id = payload.get("source_event_id")
        source = next((item for item in row["events"] if item["event_id"] == source_id), None)
        if source is None:
            raise TaskRecordError("Tool view must reference an existing event of this attempt")
        if any(item.get("source_event_id") == source_id for item in row["tool_events"]):
            raise TaskRecordError("A tool event view cannot duplicate an existing source reference")
        item = copy.deepcopy(source)
        item.update(source_event_id=source_id, sequence=len(row["tool_events"]) + 1)
        state_field = "state_before" if item.get("event") in {"before_tool", "skill_started"} else "state_after"
        item[state_field] = copy.deepcopy(source.get("observation"))
        row["tool_events"].append(item)
        row["updated_at"] = timestamp
    elif operation == "feedback":
        row = attempt(data, event["attempt_id"])
        destination = payload["destination"]
        if destination == "sent_to_agent":
            row["feedback"][destination].append(copy.deepcopy(payload["feedback"]))
        elif destination == "return_to_caller" and row["feedback"][destination] is None:
            row["feedback"][destination] = copy.deepcopy(payload["feedback"])
        else:
            raise TaskRecordError("Invalid or repeated feedback destination")
        row["updated_at"] = timestamp
    elif operation == "update":
        for field in ("metadata", "costs"):
            if payload.get(field) is not None:
                if not isinstance(payload[field], dict):
                    raise TaskRecordError(f"{field} must be an object")
                data[field].update(copy.deepcopy(payload[field]))
    elif operation == "initial_state":
        if data["initial_state"] is not None or payload.get("state") is None:
            raise TaskRecordError("Initial observation is immutable once recorded")
        data["initial_state"] = copy.deepcopy(payload["state"])
    elif operation == "finish":
        outcome, status = payload["outcome"], payload["record_status"]
        if status not in TERMINAL or not isinstance(outcome, dict) or not isinstance(outcome.get("status"), str):
            raise TaskRecordError("Invalid terminal outcome")
        data.update(final_state=copy.deepcopy(payload["final_state"]), outcome=copy.deepcopy(outcome),
                    record_status=status, finished_at=timestamp, termination=copy.deepcopy(payload.get("termination")))
        if payload.get("costs") is not None:
            if not isinstance(payload["costs"], dict):
                raise TaskRecordError("Terminal costs must be an object")
            data["costs"].update(copy.deepcopy(payload["costs"]))
    else:
        raise TaskRecordError(f"Unknown journal projection operation: {operation}")
    data["updated_at"] = timestamp


def project(manifest: dict, events: list[dict], get_artifact) -> dict:
    data = initial(manifest, get_artifact)
    for event in events:
        apply(data, event)
    return clone(data)

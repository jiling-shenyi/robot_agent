"""Append-only canonical events with strict sequence and a manifest-anchored chain."""
from __future__ import annotations

import datetime as dt
import os
import math
from pathlib import Path
import re

from embodied_agent.recording.common import SCHEMA_VERSION, TaskRecordError, decode, digest, encode

IDENTITY_FIELDS = ("run_id", "session_id", "task_id", "agent_run_id", "agent_role",
                   "attempt_id", "decision_id", "model_call_id", "tool_call_id", "action_id")


def event_hash(event: dict) -> str:
    return digest(encode({key: value for key, value in event.items() if key != "event_hash"}))


def validate(event: dict, *, sequence: int, previous_hash: str, task_id: str) -> None:
    required = {"schema_version", "seq", "event_id", "event_type", "operation", "source", "visibility",
                "parent_event_ids", "timestamp", "monotonic_ns", "sim_time_s", "payload", "artifact_refs",
                "previous_hash", "event_hash", *IDENTITY_FIELDS}
    if not isinstance(event, dict) or required - event.keys():
        raise TaskRecordError("Journal event is missing canonical fields")
    if type(event["schema_version"]) is not int or event["schema_version"] != SCHEMA_VERSION:
        raise TaskRecordError("Unsupported journal schema_version; v1 is not readable")
    if type(event["seq"]) is not int or event["seq"] != sequence:
        raise TaskRecordError(f"Journal sequence mismatch: expected {sequence}")
    if event["event_id"] != f"{task_id}:{sequence}" or event["task_id"] != task_id:
        raise TaskRecordError("Journal event task identity mismatch")
    if event["previous_hash"] != previous_hash or event["event_hash"] != event_hash(event):
        raise TaskRecordError("Journal event hash-chain mismatch")
    for field in ("event_type", "operation", "source", "visibility"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise TaskRecordError(f"Invalid canonical {field}")
    if not isinstance(event["parent_event_ids"], list) or any(not isinstance(item, str) for item in event["parent_event_ids"]):
        raise TaskRecordError("parent_event_ids must be event ID strings")
    if not isinstance(event["artifact_refs"], list):
        raise TaskRecordError("artifact_refs must be an array")
    for field in IDENTITY_FIELDS:
        if field == "attempt_id":
            if event[field] is not None and (type(event[field]) is not int or event[field] < 1):
                raise TaskRecordError("attempt_id must be a positive integer or null")
        elif event[field] is not None and (not isinstance(event[field], str) or not event[field].strip()):
            raise TaskRecordError(f"{field} must be a nonempty string or null")
    try:
        parsed = dt.datetime.fromisoformat(event["timestamp"])
        if parsed.utcoffset() != dt.timedelta(hours=8):
            raise ValueError("timestamp must use Asia/Shanghai")
    except (ValueError, TypeError) as error:
        raise TaskRecordError(f"Invalid journal timestamp: {error}") from error
    if type(event["monotonic_ns"]) is not int or event["monotonic_ns"] < 0:
        raise TaskRecordError("monotonic_ns must be a nonnegative integer")
    if event.get("monotonic_elapsed_ns") is not None and (type(event["monotonic_elapsed_ns"]) is not int
            or event["monotonic_elapsed_ns"] < 0):
        raise TaskRecordError("monotonic_elapsed_ns must be a nonnegative integer or null")
    if event.get("sim_step") is not None and (type(event["sim_step"]) is not int or event["sim_step"] < 0):
        raise TaskRecordError("sim_step must be a nonnegative integer or null")
    if "caused_by_event_ids" in event and (not isinstance(event["caused_by_event_ids"], list)
            or any(not isinstance(identity, str) for identity in event["caused_by_event_ids"])):
        raise TaskRecordError("caused_by_event_ids must be an array of event IDs")
    if event["sim_time_s"] is not None and (type(event["sim_time_s"]) not in {int, float}
            or not math.isfinite(event["sim_time_s"]) or event["sim_time_s"] < 0):
        raise TaskRecordError("sim_time_s must be a finite nonnegative number or null")


def scan(path: Path, *, manifest_hash: str, task_id: str, recover: bool = False) -> tuple[list[dict], dict | None]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise TaskRecordError(f"Missing journal: {path}: {error}") from error
    events, previous, offset = [], manifest_hash, 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            if not line.endswith(b"\n"):
                raise TaskRecordError("Incomplete journal tail: missing newline")
            if not line.strip():
                raise TaskRecordError("Blank journal rows are not valid events")
            event = decode(line)
            validate(event, sequence=len(events) + 1, previous_hash=previous, task_id=task_id)
        except TaskRecordError as error:
            if not recover or index != len(lines) - 1:
                raise TaskRecordError(f"Journal corruption at line {index + 1}, byte {offset}: {error}") from error
            return events, {"kind": "corrupt_tail", "line": index + 1, "byte_offset": offset,
                            "error": str(error), "source_journal_sha256": digest(raw),
                            "valid_prefix_sha256": digest(raw[:offset]), "rejected_tail_sha256": digest(raw[offset:]),
                            "rejected_tail_bytes": len(raw) - offset, "valid_event_count": len(events),
                            "source_preserved": True, "training_eligible": False}
        events.append(event)
        previous, offset = event["event_hash"], offset + len(line)
    return events, None


def append(path: Path, event: dict) -> None:
    encoded = encode(event) + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "ab", buffering=0) as stream:
            view = memoryview(encoded)
            while view:
                written = stream.write(view)
                if not written:
                    raise OSError("Journal append wrote no bytes")
                view = view[written:]
            os.fsync(stream.fileno())
    except BaseException:
        # A partial tail remains original evidence. Never truncate or retry it.
        raise

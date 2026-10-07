"""Durable run audit; completed runs accept only unconsumed late replies."""
from __future__ import annotations

import copy
import datetime as dt
import math
import os
import re
import threading
import time
from embodied_agent.recording.common import (SCHEMA_VERSION, TaskRecordError, atomic_bytes,
    atomic_json, clone, decode, digest, encode, now, timestamp)
from embodied_agent.recording.journal import append, event_hash


ARTIFACT_FIELDS = frozenset({"sha256", "size_bytes", "media_type", "path"})
IDENTITY_FIELDS = ("agent_run_id", "agent_role", "attempt_id", "decision_id", "model_call_id",
                   "tool_call_id", "action_id", "parent_event_id", "environment_instance_id")
REQUIRED_FIELDS = {"schema_version", "event_id", "seq", "event_type", "operation", "run_id",
    "session_id", "task_id", "parent_event_ids", "caused_by_event_ids", "timestamp", "monotonic_ns",
    "monotonic_elapsed_ns", "monotonic_clock_scope", "monotonic_ns_source", "sim_step", "sim_time_s",
    "source", "visibility", "consumed", "payload", "artifact_refs", "previous_hash", "event_hash",
    "late_after_run_finished", *IDENTITY_FIELDS}


class RunRecorder:
    def __init__(self, store, run_id: str, *, session_id: str | None = None, manifest=None):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
            raise ValueError("run_id must be a safe directory identifier")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a nonempty string")
        if manifest is not None and not isinstance(manifest, dict):
            raise ValueError("run manifest must be an object")
        self.store, self.run_id = store, run_id
        self.session_id = session_id or run_id
        manifest = clone(manifest if manifest is not None else {})
        self.directory = store.root / "runs" / run_id
        self._lock = threading.RLock()
        self._sequence, self._started = 0, time.monotonic_ns()
        self._closed, self._poisoned = False, None
        data = {"schema_version": SCHEMA_VERSION, "kind": "agent_run", "run_id": run_id,
            "session_id": self.session_id, "created_at": timestamp(now()), "manifest": manifest,
            "collector_monotonic_started_ns": self._started, "collector_pid": os.getpid(),
            "monotonic_clock_scope": "single_process_collector"}
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.directory / "manifest.json"
        self.journal_path = self.directory / "events.jsonl"
        atomic_json(self.manifest_path, data, exclusive=True)
        atomic_bytes(self.journal_path, b"", exclusive=True)
        self._manifest_hash = self._previous = digest(self.manifest_path.read_bytes())
        self.append("run.started", {"manifest_ref": store.put_artifact(data)}, source="collector")

    def _available(self):
        if self._poisoned is not None:
            raise TaskRecordError(f"Run recording stopped after persistence/integrity failure: {self._poisoned}")

    def _artifacts(self, value, visited):
        if isinstance(value, dict):
            if ARTIFACT_FIELDS <= value.keys():
                key = digest(encode(value))
                if key not in visited:
                    visited.add(key)
                    self._artifacts(self.store.get_artifact(value), visited)
            else:
                for child in value.values():
                    self._artifacts(child, visited)
        elif isinstance(value, list):
            for child in value:
                self._artifacts(child, visited)

    def _validate(self, event, sequence, previous, *, finished):
        if not isinstance(event, dict) or REQUIRED_FIELDS - event.keys():
            raise TaskRecordError("Run event is missing canonical fields")
        if (type(event["schema_version"]) is not int or event["schema_version"] != SCHEMA_VERSION
                or type(event["seq"]) is not int or event["seq"] != sequence
                or event["event_id"] != f"{self.run_id}:{sequence}"
                or event["run_id"] != self.run_id or event["session_id"] != self.session_id
                or event["operation"] != "run_event" or event["previous_hash"] != previous
                or event["event_hash"] != event_hash(event)):
            raise TaskRecordError("Run journal identity/schema/hash-chain mismatch")
        for name in ("event_type", "source", "visibility"):
            if not isinstance(event[name], str) or not event[name].strip():
                raise TaskRecordError(f"Invalid run event {name}")
        for name in ("task_id", *IDENTITY_FIELDS):
            value = event[name]
            if name == "attempt_id":
                if value is not None and (type(value) is not int or value < 1):
                    raise TaskRecordError("Run attempt_id must be a positive integer or null")
            elif value is not None and (not isinstance(value, str) or not value.strip()):
                raise TaskRecordError(f"Run {name} must be a nonempty string or null")
        for name in ("parent_event_ids", "caused_by_event_ids"):
            if not isinstance(event[name], list) or any(not isinstance(item, str) or not item for item in event[name]):
                raise TaskRecordError(f"Run {name} must contain event ID strings")
        if event["parent_event_id"] is not None and event["parent_event_id"] not in event["parent_event_ids"]:
            raise TaskRecordError("Run parent_event_id is absent from parent_event_ids")
        try:
            parsed = dt.datetime.fromisoformat(event["timestamp"])
            if parsed.utcoffset() != dt.timedelta(hours=8):
                raise ValueError("timestamp must use Asia/Shanghai")
        except (ValueError, TypeError) as error:
            raise TaskRecordError(f"Invalid run timestamp: {error}") from error
        for name in ("monotonic_ns", "monotonic_elapsed_ns"):
            if type(event[name]) is not int or event[name] < 0:
                raise TaskRecordError(f"Run {name} must be a nonnegative integer")
        if (event["monotonic_clock_scope"] != "single_process_collector"
                or event["monotonic_ns_source"] != "collector"):
            raise TaskRecordError("Invalid run monotonic clock source/scope")
        if event["sim_step"] is not None and (type(event["sim_step"]) is not int or event["sim_step"] < 0):
            raise TaskRecordError("Run sim_step must be a nonnegative integer or null")
        if event["sim_time_s"] is not None and (type(event["sim_time_s"]) not in {int, float}
                or not math.isfinite(event["sim_time_s"]) or event["sim_time_s"] < 0):
            raise TaskRecordError("Run sim_time_s must be finite and nonnegative or null")
        if event["consumed"] is not None and type(event["consumed"]) is not bool:
            raise TaskRecordError("Run consumed must be boolean or null")
        if type(event["late_after_run_finished"]) is not bool:
            raise TaskRecordError("Run late_after_run_finished must be boolean")
        late = event["event_type"].startswith("late.") and event["consumed"] is False
        if event["event_type"].startswith("late.") and not late:
            raise TaskRecordError("Late audit events must be explicitly unconsumed")
        if finished and not (late and event["late_after_run_finished"]):
            raise TaskRecordError("Finished runs accept only unconsumed late audit events")
        if not finished and event["late_after_run_finished"]:
            raise TaskRecordError("Late event claims a run finished before its terminal event")
        if (sequence == 1) != (event["event_type"] == "run.started"):
            raise TaskRecordError("A run must contain exactly one initial run.started event")
        if not isinstance(event["artifact_refs"], list):
            raise TaskRecordError("Run artifact_refs must be an array")

    def append(self, event_type: str, payload=None, *, task_id=None, consumed=None,
               source="executor", visibility="executor_only", **identities):
        with self._lock:
            self._available()
            if self._closed and not (isinstance(event_type, str) and event_type.startswith("late.") and consumed is False):
                raise TaskRecordError("Finished runs accept only unconsumed late audit events")
            try:
                existing = self.read_events(resolve=False)
                last_hash = existing[-1]["event_hash"] if existing else self._manifest_hash
                if len(existing) != self._sequence or last_hash != self._previous:
                    raise TaskRecordError("Run journal changed outside its owning collector")
                sequence = self._sequence + 1
                body, refs = clone(payload if payload is not None else {}), []
                self._artifacts(body, set())
                if len(encode(body)) > 4096:
                    reference = self.store.put_artifact(body)
                    refs.append(reference)
                    body = {"$artifact": reference}
                monotonic = time.monotonic_ns()
                parent = identities.get("parent_event_id")
                parents = identities.get("parent_event_ids", [parent] if parent else [])
                event = {"schema_version": SCHEMA_VERSION, "event_id": f"{self.run_id}:{sequence}",
                    "seq": sequence, "event_type": event_type, "operation": "run_event",
                    "run_id": self.run_id, "session_id": self.session_id, "task_id": task_id,
                    **{name: identities.get(name) for name in IDENTITY_FIELDS},
                    "parent_event_ids": clone(parents),
                    "caused_by_event_ids": clone(identities.get("caused_by_event_ids", [])),
                    "timestamp": timestamp(now()), "monotonic_elapsed_ns": monotonic - self._started,
                    "monotonic_ns": monotonic, "monotonic_ns_source": "collector",
                    "monotonic_clock_scope": "single_process_collector",
                    "sim_step": identities.get("sim_step"), "sim_time_s": identities.get("sim_time_s"),
                    "source": source, "visibility": visibility, "consumed": consumed,
                    "late_after_run_finished": self._closed,
                    "payload": body, "artifact_refs": refs, "previous_hash": self._previous}
                event["event_hash"] = event_hash(event)
                self._validate(event, sequence, self._previous, finished=self._closed)
                append(self.journal_path, event)
                self._sequence, self._previous = sequence, event["event_hash"]
                if event_type == "run.finished":
                    self._closed = True
                return event["event_id"]
            except Exception as error:
                # Original bytes include any incomplete final row. Never retry.
                self._poisoned = f"{type(error).__name__}: {error}"
                raise

    def read_events(self, *, resolve=True):
        """Read and validate facts without creating, repairing or indexing files."""
        with self._lock:
            try:
                manifest_bytes = self.manifest_path.read_bytes()
                previous = digest(manifest_bytes)
                manifest = decode(manifest_bytes)
                if (not isinstance(manifest, dict) or previous != self._manifest_hash
                        or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION
                        or manifest.get("kind") != "agent_run" or manifest.get("run_id") != self.run_id
                        or manifest.get("session_id") != self.session_id):
                    raise TaskRecordError("Run manifest schema/identity/hash mismatch")
                visited = set()
                self._artifacts(manifest, visited)
                events, finished = [], False
                for sequence, line in enumerate(self.journal_path.read_bytes().splitlines(keepends=True), 1):
                    if not line.endswith(b"\n"):
                        raise TaskRecordError(f"Incomplete run journal tail at line {sequence}")
                    event = decode(line)
                    self._validate(event, sequence, previous, finished=finished)
                    for reference in event["artifact_refs"]:
                        self.store.get_artifact(reference)
                        self._artifacts(reference, visited)
                    payload = event["payload"]
                    if isinstance(payload, dict) and set(payload) == {"$artifact"}:
                        if payload["$artifact"] not in event["artifact_refs"]:
                            raise TaskRecordError("Run payload artifact is not declared in artifact_refs")
                        payload = self.store.get_artifact(payload["$artifact"])
                    self._artifacts(payload, visited)
                    previous = event["event_hash"]
                    finished = finished or event["event_type"] == "run.finished"
                    if resolve:
                        event = copy.deepcopy(event)
                        event["payload"] = payload
                    events.append(event)
                return events
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                if isinstance(error, TaskRecordError):
                    raise
                raise TaskRecordError(f"Cannot read run facts: {error}") from error

    def close(self):
        with self._lock:
            self._available()
            if not self._closed:
                self.append("run.finished", source="collector")

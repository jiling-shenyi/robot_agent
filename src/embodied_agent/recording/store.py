"""Task manifests, append-only facts, immutable seals and rebuildable views/index."""
from __future__ import annotations

import copy
from contextlib import contextmanager, closing
import datetime as dt
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import time
from typing import Any, Callable
import uuid

from embodied_agent.recording.artifacts import ArtifactStore
from embodied_agent.recording.common import (SCHEMA_VERSION, SHANGHAI, TaskRecordError, atomic_bytes,
    atomic_json, clone, decode, digest, encode, now, timestamp)
from embodied_agent.recording import journal, projector

MAP_ALIASES = {"classic": "cls", "alternate": "alt", "home_living_room": "hlr"}
PAYLOAD_INLINE_BYTES = 8192
ARTIFACT_FIELDS = frozenset({"sha256", "size_bytes", "media_type", "path"})


def map_alias(map_id: str) -> str:
    if map_id in MAP_ALIASES:
        return MAP_ALIASES[map_id]
    readable = re.sub(r"[^a-z0-9]+", "-", map_id.lower()).strip("-")[:16] or "map"
    return f"{readable}-{hashlib.sha256(map_id.encode('utf-8')).hexdigest()[:12]}"


def _validate_artifact_graph(value: Any, get_artifact, visited: set[str]) -> None:
    """Validate declared content-address references at any depth, not bare hashes."""
    if isinstance(value, dict):
        if ARTIFACT_FIELDS <= value.keys():
            identity = digest(encode(value))
            if identity not in visited:
                visited.add(identity)
                _validate_artifact_graph(get_artifact(value), get_artifact, visited)
        else:
            for child in value.values():
                _validate_artifact_graph(child, get_artifact, visited)
    elif isinstance(value, list):
        for child in value:
            _validate_artifact_graph(child, get_artifact, visited)


@contextmanager
def _task_lock(directory: Path):
    path = directory / ".writer.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise TaskRecordError("Task is owned by another writer or has an abandoned writer lock") from error
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


class TaskRecordStore:
    """Read-only construction/query; only explicit mutations create files.

    manifest.json and events.jsonl are facts. task.json and index.sqlite3 are
    expendable projections. seal.json binds immutable terminal facts. A damaged
    tail is never removed; recovery creates a separate explicitly ineligible view.
    """

    def __init__(self, root: str | Path, *, clock: Callable[[], dt.datetime] | None = None):
        self.root = Path(root).resolve()
        self._clock = clock or now
        self.artifacts = ArtifactStore(self.root)
        self.errors: list[dict] = []
        self.index_errors: list[str] = []

    def put_artifact(self, value: Any, *, media_type: str = "application/json") -> dict:
        return self.artifacts.put(value, media_type=media_type)

    def get_artifact(self, reference: dict) -> Any:
        return self.artifacts.get(reference)

    def _allocate(self, date: str) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.root / "index.sqlite3", timeout=30)) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS daily (date TEXT PRIMARY KEY, sequence INTEGER NOT NULL)")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT sequence FROM daily WHERE date=?", (date,)).fetchone()
            disk_max = 0
            for path in (self.root / "tasks" / date).glob("*/manifest.json"):
                try:
                    candidate = decode(path.read_bytes())
                    if type(candidate.get("sequence")) is int:
                        disk_max = max(disk_max, candidate["sequence"])
                except (OSError, TaskRecordError, AttributeError):
                    continue
            sequence = max(row[0] if row else 0, disk_max) + 1
            connection.execute("INSERT INTO daily VALUES (?,?) ON CONFLICT(date) DO UPDATE SET sequence=excluded.sequence", (date, sequence))
            connection.commit()
        return sequence

    def begin(self, *, map_id: str, map_definition: Any, initial_state: Any,
              natural_language: str, versions: dict | None = None, metadata: dict | None = None) -> "TaskRecord":
        if not isinstance(map_id, str) or not map_id.strip() or not isinstance(natural_language, str):
            raise TaskRecordError("map_id must be nonempty and natural_language must be a string")
        if versions is not None and not isinstance(versions, dict) or metadata is not None and not isinstance(metadata, dict):
            raise TaskRecordError("versions and metadata must be objects")
        # Validate all data and the clock before beginning any filesystem work.
        values = clone({"map": map_definition, "initial": initial_state, "language": natural_language,
                        "versions": versions or {}, "metadata": metadata or {}})
        started, task_id = timestamp(self._clock()), uuid.uuid4().hex
        date, info = started[:10].replace("-", ""), values["metadata"]
        role = info.get("agent_role", info.get("agent", "unknown"))
        if role == "robot":
            role = "instruction"
        identities = {"task_id": task_id, "run_id": info.get("run_id") or task_id,
                      "session_id": info.get("session_id") or info.get("run_id") or task_id,
                      "agent_run_id": info.get("agent_run_id") or uuid.uuid4().hex,
                      "agent_role": role}
        for name, identity in identities.items():
            if not isinstance(identity, str) or not identity.strip():
                raise TaskRecordError(f"{name} must be a nonempty string")
        kind = info.get("task_kind", "unknown")
        if not isinstance(kind, str) or not kind.strip():
            raise TaskRecordError("task_kind must be a nonempty string")
        sequence = self._allocate(date)
        directory = self.root / "tasks" / date / task_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": SCHEMA_VERSION, "kind": "agent_task", "task_id": task_id,
                    "date": date, "sequence": sequence, "created_at": started, "identities": identities,
                    "collector_monotonic_started_ns": time.monotonic_ns(), "collector_pid": os.getpid(),
                    "monotonic_clock_scope": "single_process_collector",
                    "task_kind": kind, "behavior_bundle_id": info.get("behavior_bundle_id"),
                    "map": {"id": map_id, "alias": map_alias(map_id),
                            "definition_ref": self.put_artifact(values["map"])},
                    "natural_language_ref": self.put_artifact(values["language"]),
                    "initial_state_ref": self.put_artifact(values["initial"]),
                    "versions_ref": self.put_artifact(values["versions"]),
                    "metadata_ref": self.put_artifact(values["metadata"]),
                    "trusted_task_spec_ref": self.put_artifact(info["trusted_task_spec"])
                    if info.get("trusted_task_spec") is not None else None}
        projector.initial(manifest, self.get_artifact)
        _validate_artifact_graph(manifest, self.get_artifact, set())
        atomic_json(directory / "manifest.json", manifest, exclusive=True)
        atomic_bytes(directory / "events.jsonl", b"", exclusive=True)
        data = projector.project(manifest, [], self.get_artifact)
        atomic_json(directory / "task.json", data)
        record = TaskRecord(directory / "task.json", data, self._clock, store=self)
        self._index_task(data)
        return record

    def list(self) -> list[Path]:
        def order(path):
            try:
                sequence = decode((path.parent / "manifest.json").read_bytes())["sequence"]
            except (OSError, TaskRecordError, KeyError, TypeError):
                sequence = 2**63
            return path.parent.parent.name, sequence, path.parent.name
        return sorted((directory / "task.json" for directory in (self.root / "tasks").glob("*/*")
                       if directory.is_dir() and re.fullmatch(r"[0-9a-f]{32}", directory.name)
                       and ((directory / "manifest.json").exists() or (directory / "events.jsonl").exists())), key=order)

    def resolve_task(self, path_or_task_id: str | Path) -> Path:
        requested = str(path_or_task_id)
        if re.fullmatch(r"[0-9a-f]{32}", requested):
            matches = list((self.root / "tasks").glob(f"*/{requested}"))
            if len(matches) != 1:
                raise TaskRecordError(f"Cannot uniquely find task_id {requested}")
            directory = matches[0].resolve()
        else:
            path = Path(path_or_task_id)
            path = path if path.is_absolute() else self.root / path
            directory = (path.parent if path.name in {"task.json", "manifest.json", "events.jsonl", "seal.json"} else path).resolve()
        if (not directory.is_relative_to(self.root / "tasks") or len(directory.relative_to(self.root / "tasks").parts) != 2
                or not re.fullmatch(r"\d{8}", directory.parent.name)
                or not re.fullmatch(r"[0-9a-f]{32}", directory.name)):
            raise TaskRecordError("Not a v2 task directory beneath this records root; v1 is not supported")
        return directory

    def _read(self, identifier, *, recover: bool = False, check_seal: bool = True):
        directory = self.resolve_task(identifier)
        try:
            manifest_bytes = (directory / "manifest.json").read_bytes()
            manifest = decode(manifest_bytes)
            if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
                raise TaskRecordError("Unsupported task manifest schema_version; v1 is not readable")
            artifact_cache = {}
            def checked_get(reference):
                cache_key = digest(encode(reference))
                if cache_key not in artifact_cache:
                    artifact_cache[cache_key] = self.get_artifact(reference)
                return copy.deepcopy(artifact_cache[cache_key])
            visited = set()
            _validate_artifact_graph(manifest, checked_get, visited)
            if manifest.get("task_id") != directory.name or manifest.get("date") != directory.parent.name:
                raise TaskRecordError("Manifest task/date does not match its directory")
            if (not isinstance(manifest.get("identities"), dict)
                    or manifest["identities"].get("task_id") != directory.name):
                raise TaskRecordError("Manifest identities do not match the task")
            for key in ("run_id", "session_id", "agent_run_id", "agent_role"):
                if not isinstance(manifest["identities"].get(key), str) or not manifest["identities"][key].strip():
                    raise TaskRecordError(f"Missing manifest identity {key}")
            if manifest["map"]["alias"] != map_alias(manifest["map"]["id"]):
                raise TaskRecordError("Map alias does not match map ID")
            if not isinstance(manifest.get("task_kind"), str) or not manifest["task_kind"].strip():
                raise TaskRecordError("Manifest task_kind must be a nonempty string")
            created = dt.datetime.fromisoformat(manifest["created_at"])
            if created.utcoffset() != dt.timedelta(hours=8):
                raise TaskRecordError("Manifest created_at must use Asia/Shanghai")
            if timestamp(created)[:10].replace("-", "") != manifest["date"]:
                raise TaskRecordError("Manifest date must be the Shanghai start date")
            events, report = journal.scan(directory / "events.jsonl", manifest_hash=digest(manifest_bytes),
                                          task_id=directory.name, recover=recover)
            resolved = []
            for event in events:
                if any(event[field] != manifest["identities"][field] for field in ("run_id", "session_id")):
                    raise TaskRecordError("Journal run/session identity disagrees with its task manifest")
                for reference in event["artifact_refs"]:
                    _validate_artifact_graph(reference, checked_get, visited)
                payload = event["payload"]
                if isinstance(payload, dict) and set(payload) == {"$artifact"}:
                    if payload["$artifact"] not in event["artifact_refs"]:
                        raise TaskRecordError("Payload artifact is not declared in artifact_refs")
                    payload = checked_get(payload["$artifact"])
                _validate_artifact_graph(payload, checked_get, visited)
                resolved.append({**event, "payload": payload})
            data = projector.project(manifest, resolved, checked_get)
            if check_seal and report is None and (directory / "seal.json").exists():
                self._check_seal(directory, manifest_bytes, events, data)
            return directory, manifest, manifest_bytes, events, resolved, data, report
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            if isinstance(error, TaskRecordError):
                raise
            raise TaskRecordError(f"Cannot read task facts {directory}: {error}") from error

    def _check_seal(self, directory, manifest_bytes, events, data):
        seal = decode((directory / "seal.json").read_bytes())
        expected = {"schema_version": SCHEMA_VERSION, "task_id": directory.name,
                    "manifest_sha256": digest(manifest_bytes),
                    "journal_sha256": digest((directory / "events.jsonl").read_bytes()),
                    "event_count": len(events),
                    "last_event_hash": events[-1]["event_hash"] if events else digest(manifest_bytes),
                    "projection_sha256": digest(encode(data, pretty=True)), "sealed_at": data["finished_at"]}
        if (not isinstance(seal, dict) or set(seal) != set(expected)
                or type(seal.get("event_count")) is not int or type(seal.get("schema_version")) is not int
                or any(seal.get(key) != value for key, value in expected.items())):
            raise TaskRecordError("Terminal seal hash/count/schema mismatch")
        if data["record_status"] not in projector.TERMINAL:
            raise TaskRecordError("A sealed task must have a terminal fact event")

    def load(self, path_or_task_id: str | Path) -> dict:
        directory, _, manifest_bytes, events, _, data, _ = self._read(path_or_task_id)
        data["integrity"] = {"sealed": (directory / "seal.json").exists(), "source_valid": True,
                             "event_count": len(events), "manifest_sha256": digest(manifest_bytes),
                             "last_event_hash": events[-1]["event_hash"] if events else digest(manifest_bytes)}
        data["source"] = {"task_directory": str(directory), "manifest": "manifest.json", "journal": "events.jsonl"}
        return clone(data)

    def open(self, path_or_task_id: str | Path) -> "TaskRecord":
        directory = self.resolve_task(path_or_task_id)
        return TaskRecord(directory / "task.json", self.load(directory), self._clock, store=self)

    def read_events(self, task_id: str | Path, resolve: bool = True) -> list[dict]:
        _, _, _, raw, resolved, _, _ = self._read(task_id)
        return clone(resolved if resolve else raw)

    def verify(self, task_id: str | Path) -> dict:
        result = {"valid": False, "sealed": False, "errors": [], "warnings": [], "event_count": 0,
                  "last_event_hash": None, "manifest_sha256": None, "journal_sha256": None,
                  "seal_sha256": None, "task_id": str(task_id), "path": None}
        try:
            directory, _, manifest_bytes, events, _, data, _ = self._read(task_id)
            result.update(path=str(directory), task_id=directory.name, event_count=len(events),
                manifest_sha256=digest(manifest_bytes), journal_sha256=digest((directory / "events.jsonl").read_bytes()),
                last_event_hash=events[-1]["event_hash"] if events else digest(manifest_bytes),
                sealed=(directory / "seal.json").exists())
            if result["sealed"]:
                result["seal_sha256"] = digest((directory / "seal.json").read_bytes())
            elif data["record_status"] in projector.TERMINAL:
                raise TaskRecordError("Missing terminal seal")
            projection = directory / "task.json"
            try:
                projection_current = projection.exists() and projection.read_bytes() == encode(data, pretty=True)
            except OSError as error:
                projection_current = False
                result["warnings"].append(f"Derived task.json is unreadable: {error}")
            if not projection_current:
                result["warnings"].append("Derived task.json is absent or stale; rebuild from facts")
            result["valid"] = True
        except (TaskRecordError, OSError) as error:
            result["errors"].append(str(error))
        return result

    def ref(self, task_id: str | Path) -> dict:
        directory = self.resolve_task(task_id)
        report = self.verify(directory)
        return {"task_id": directory.name, "path": (directory / "task.json").relative_to(self.root).as_posix(),
                "manifest_sha256": report["manifest_sha256"], "journal_sha256": report["journal_sha256"],
                "seal_sha256": report["seal_sha256"], "integrity_valid": report["valid"], "sealed": report["sealed"]}

    def rebuild(self, task_id: str | Path, *, recover: bool = False) -> dict:
        directory, _, _, _, _, data, report = self._read(task_id, recover=recover)
        if report is not None:
            destination = directory / "task.recovered.json"
            report.update(task_id=directory.name, created_at=timestamp(self._clock()),
                          recovered_projection="task.recovered.json")
            atomic_json(destination, {**data, "recovery": report})
            atomic_json(directory / "recovery.json", report)
            return {"task": data, "path": str(destination), "recovery_report": report}
        atomic_json(directory / "task.json", data)
        self._index_task(data)
        return {"task": data, "path": str(directory / "task.json"), "recovery_report": None}

    def query(self, *, map_id: str | None = None, status: str | None = None, date: str | None = None,
              record_status: str | None = None, training_only: bool = False,
              task_kind: str | None = None, agent_role: str | None = None, integrity: bool | None = None) -> list[dict]:
        self.errors, result = [], []
        for path in self.list():
            try:
                data = self.load(path)
                outcome = data["outcome"] or {}
                checks = [(map_id, data["map"]["id"]), (status, outcome.get("status")),
                          (date.replace("-", "") if date else None, data["date"]),
                          (record_status, data["record_status"]), (task_kind, data["task_kind"]),
                          (agent_role, data["agent_role"])]
                if any(requested is not None and requested != actual for requested, actual in checks):
                    continue
                verified = self.verify(path) if training_only or integrity is not None else None
                if integrity is not None and verified["valid"] != integrity:
                    continue
                if training_only and not (verified["valid"] and verified["sealed"]
                        and data["record_status"] == "COMPLETED" and outcome.get("status") == "SUCCESS"
                        and outcome.get("verified") is True):
                    continue
                result.append(data)
            except TaskRecordError as error:
                self.errors.append({"path": str(path), "error": str(error)})
        return result

    @staticmethod
    def _create_index(connection):
        connection.execute("CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, date TEXT NOT NULL, sequence INTEGER NOT NULL, map_id TEXT, status TEXT, record_status TEXT, task_kind TEXT, agent_role TEXT, relative_path TEXT)")
        connection.execute("CREATE TABLE IF NOT EXISTS daily (date TEXT PRIMARY KEY, sequence INTEGER NOT NULL)")

    def _index_task(self, data):
        try:
            with closing(sqlite3.connect(self.root / "index.sqlite3", timeout=30)) as connection:
                self._create_index(connection)
                connection.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?,?)", (
                    data["task_id"], data["date"], data["sequence"], data["map"]["id"],
                    (data["outcome"] or {}).get("status"), data["record_status"], data["task_kind"],
                    data["agent_role"], f"tasks/{data['date']}/{data['task_id']}/task.json"))
                connection.commit()
        except sqlite3.Error as error:
            # Index failure cannot cause a caller to replay a physical effect.
            self.index_errors.append(str(error))

    def rebuild_index(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".index-", suffix=".sqlite3", dir=self.root)
        os.close(descriptor)
        errors, count = [], 0
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                self._create_index(connection)
                for path in self.list():
                    try:
                        data = self.load(path)
                    except TaskRecordError as error:
                        errors.append({"path": str(path), "error": str(error)})
                        continue
                    connection.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)", (
                        data["task_id"], data["date"], data["sequence"], data["map"]["id"],
                        (data["outcome"] or {}).get("status"), data["record_status"], data["task_kind"],
                        data["agent_role"], str(path.relative_to(self.root))))
                    connection.execute("INSERT INTO daily VALUES (?,?) ON CONFLICT(date) DO UPDATE SET sequence=max(sequence,excluded.sequence)", (data["date"], data["sequence"]))
                    count += 1
                connection.commit()
            with Path(temporary).open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "index.sqlite3")
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {"task_count": count, "errors": errors, "path": str(self.root / "index.sqlite3")}


class TaskRecord:
    """Serialized durable updates for one task, preserving the execution-facing API."""

    def __init__(self, path: Path, data: dict | None = None, clock=None, *, store: TaskRecordStore | None = None):
        self.path = Path(path).resolve()
        self.directory = self.path.parent
        self.manifest_path, self.journal_path, self.seal_path = (
            self.directory / name for name in ("manifest.json", "events.jsonl", "seal.json"))
        self.store = store or TaskRecordStore(self.path.parents[3], clock=clock)
        self._clock, self._lock = clock or self.store._clock, threading.RLock()

    @property
    def task_id(self) -> str:
        return self.directory.name

    @property
    def data(self) -> dict:
        with self._lock:
            return self.store.load(self.path)

    def _mutate(self, operation: str, payload: dict, *, event_type: str,
                attempt_id: int | None = None, collection: str | None = None, row: dict | None = None) -> dict:
        payload, row = clone(payload), clone(row or {})
        with self._lock, _task_lock(self.directory):
            _, manifest, manifest_bytes, events, _, current, _ = self.store._read(self.directory)
            if current["record_status"] != "RUNNING" or self.seal_path.exists():
                raise TaskRecordError("Finished task records are immutable")
            if operation not in {"append_attempt", "update", "initial_state", "finish"}:
                attempt_id = projector.attempt(current, attempt_id)["attempt_id"]
            identities = copy.deepcopy(manifest["identities"])
            extra = row.get("identities", {})
            if not isinstance(extra, dict):
                raise TaskRecordError("Event identities must be an object")
            for field in journal.IDENTITY_FIELDS:
                value = extra.get(field, row.get(field, identities.get(field)))
                if field in {"task_id", "run_id", "session_id"} and value != identities[field]:
                    raise TaskRecordError(f"Event {field} disagrees with its task manifest")
                identities[field] = value
            if extra.get("attempt_id", row.get("attempt_id", attempt_id)) != attempt_id:
                raise TaskRecordError("Event attempt_id disagrees with its recording operation")
            identities["attempt_id"] = attempt_id
            sequence = len(events) + 1
            parents = row.get("parent_event_ids", extra.get("parent_event_ids", []))
            if not parents and row.get("parent_event_id"):
                parents = [row["parent_event_id"]]
            collector_monotonic_ns = time.monotonic_ns()
            event = {"schema_version": SCHEMA_VERSION, "seq": sequence,
                     "event_id": f"{self.task_id}:{sequence}", "event_type": event_type, "operation": operation,
                     **identities, "source": row.get("source", "runtime" if operation == "append_event" else "recording"),
                     "visibility": row.get("visibility", "internal"), "parent_event_ids": parents,
                     "timestamp": row.get("timestamp", timestamp(self._clock())),
                     "monotonic_ns": row.get("monotonic_ns", collector_monotonic_ns),
                     "monotonic_elapsed_ns": row.get("monotonic_elapsed_ns", (
                         collector_monotonic_ns - manifest["collector_monotonic_started_ns"]
                         if manifest.get("collector_pid") == os.getpid() else None)),
                     "monotonic_clock_scope": "single_process_collector",
                     "monotonic_ns_source": "provider" if "monotonic_ns" in row else "collector",
                     "sim_time_s": row.get("sim_time_s"), "payload": payload,
                     "artifact_refs": clone(row.get("artifact_refs", [])),
                     "previous_hash": events[-1]["event_hash"] if events else digest(manifest_bytes)}
            for field in ("environment_instance_id", "sim_step", "parent_event_id", "caused_by_event_ids", "consumed"):
                if field in row or field in extra:
                    event[field] = row.get(field, extra.get(field))
            if collection is not None:
                event["collection"] = collection
            candidate = clone(current)
            _validate_artifact_graph(payload, self.store.get_artifact, set())
            projector.apply(candidate, event)  # Validate lifecycle before committing artifacts or facts.
            if len(encode(payload)) > PAYLOAD_INLINE_BYTES:
                reference = self.store.put_artifact(payload)
                event["payload"] = {"$artifact": reference}
                event["artifact_refs"].append(reference)
            for reference in event["artifact_refs"]:
                _validate_artifact_graph(reference, self.store.get_artifact, set())
            event["event_hash"] = journal.event_hash(event)
            journal.validate(event, sequence=sequence, previous_hash=event["previous_hash"], task_id=self.task_id)
            journal.append(self.journal_path, event)
            # A projection is replaceable. A failed cache write does not invalidate
            # the already-fsynced fact or justify replaying a model/physical action.
            try:
                atomic_json(self.path, candidate)
            except OSError as error:
                self.store.index_errors.append(f"projection: {error}")
            if operation == "finish":
                seal = {"schema_version": SCHEMA_VERSION, "task_id": self.task_id,
                        "manifest_sha256": digest(manifest_bytes),
                        "journal_sha256": digest(self.journal_path.read_bytes()), "event_count": sequence,
                        "last_event_hash": event["event_hash"], "projection_sha256": digest(encode(candidate, pretty=True)),
                        "sealed_at": event["timestamp"]}
                atomic_json(self.seal_path, seal, exclusive=True)
            self.store._index_task(candidate)
            return clone(event)

    def append_attempt(self, *, planning_observation=None, model_input=None, plan=None, proposal=None,
                       parent_attempt_id: int | None = None, trigger_event_id: str | None = None,
                       metadata: dict | None = None) -> int:
        with self._lock:
            attempt_id = len(self.data["attempts"]) + 1
            self._mutate("append_attempt", {"attempt_id": attempt_id, "parent_attempt_id": parent_attempt_id,
                "trigger_event_id": trigger_event_id, "planning_observation": planning_observation,
                "model_input": model_input, "plan": plan, "proposal": proposal, "metadata": metadata or {}},
                event_type="attempt.started", attempt_id=attempt_id)
            return attempt_id

    def update_attempt(self, attempt_id: int | None = None, **fields) -> None:
        if set(fields) - projector.ATTEMPT_FIELDS:
            raise TaskRecordError("Unsupported attempt fields; use set_attempt_feedback for feedback")
        self._mutate("update_attempt", {"fields": fields}, event_type="attempt.updated", attempt_id=attempt_id)

    def append_event(self, event: dict, *, attempt_id: int | None = None, collection: str = "events") -> str:
        if not isinstance(event, dict) or collection not in {"events", "checks", "tool_events"}:
            raise TaskRecordError("event must be an object in events, checks, or tool_events")
        canonical = self._mutate("append_event", event, event_type=event.get("event_type", event.get("event", "runtime.event")),
                                  attempt_id=attempt_id, collection=collection, row=event)
        return canonical["event_id"]

    def append_check(self, check: dict, *, attempt_id: int | None = None) -> str:
        return self.append_event(check, attempt_id=attempt_id, collection="checks")

    def append_tool_event(self, event: dict, *, attempt_id: int | None = None) -> str:
        if isinstance(event, dict) and "source_event_id" in event:
            self._mutate("tool_view", {"source_event_id": event["source_event_id"]},
                         event_type="recording.tool_view_indexed", attempt_id=attempt_id)
            return event["source_event_id"]
        return self.append_event(event, attempt_id=attempt_id, collection="tool_events")

    def set_attempt_feedback(self, feedback: Any, *, attempt_id: int | None = None,
                             destination: str = "return_to_caller") -> None:
        self._mutate("feedback", {"feedback": feedback, "destination": destination},
                     event_type=f"feedback.{destination}", attempt_id=attempt_id)

    def update(self, *, costs: dict | None = None, metadata: dict | None = None) -> None:
        self._mutate("update", {"costs": costs, "metadata": metadata}, event_type="task.metadata_updated")

    def set_initial_state(self, state: Any) -> None:
        self._mutate("initial_state", {"state": state}, event_type="observation.initial_recorded")

    def finish(self, *, final_state: Any, outcome: dict, costs: dict | None = None,
               record_status: str = "COMPLETED", termination: dict | None = None) -> dict:
        self._mutate("finish", {"final_state": final_state, "outcome": outcome, "costs": costs,
                     "record_status": record_status, "termination": termination or outcome.get("termination")},
                     event_type="task.finished")
        return self.data

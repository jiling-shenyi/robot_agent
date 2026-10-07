"""Fact durability, chain/artifact integrity, recovery and derived-cache boundaries."""
from __future__ import annotations

import datetime as dt
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.recording import TaskRecordError, TaskRecordStore
from embodied_agent.recording.common import atomic_json
from embodied_agent.recording.journal import event_hash

START = dt.datetime(2026, 10, 7, 16, 30, tzinfo=dt.timezone.utc)


class RecordingV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "records"
        self.store = TaskRecordStore(self.root, clock=lambda: START)

    def begin(self, **overrides):
        arguments = {"map_id": "classic", "map_definition": {"objects": ["cube"]},
            "initial_state": {"position": [0, 1, 2]}, "natural_language": "摆放方块",
            "metadata": {"run_id": "run-a", "session_id": "session-a", "agent_role": "instruction",
                         "task_kind": "physical_execution", "trusted_task_spec": {"target": "table"}}}
        arguments.update(overrides)
        return self.store.begin(**arguments)

    def finish(self, task):
        task.finish(final_state={"position": [1, 2, 3]}, outcome={"status": "SUCCESS", "verified": True})

    def test_artifacts_are_atomic_deduplicated_and_checked(self):
        first = self.store.put_artifact({"中文": "内容", "number": 1})
        second = self.store.put_artifact({"number": 1, "中文": "内容"})
        self.assertEqual(first, second)
        self.assertEqual(self.store.get_artifact(first), {"中文": "内容", "number": 1})
        binary = self.store.put_artifact(b"physics-checkpoint")
        self.assertEqual(self.store.get_artifact(binary), b"physics-checkpoint")
        (self.root / first["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(TaskRecordError, "integrity"):
            self.store.get_artifact(first)
        with self.assertRaises(TaskRecordError):
            self.store.get_artifact({**binary, "path": "../outside.bin"})
        with patch("embodied_agent.recording.common.os.fsync", side_effect=OSError("power loss")):
            with self.assertRaises(OSError):
                self.store.put_artifact({"never": "published"})
        self.assertEqual(len(list((self.root / "artifacts").rglob("*.json"))), 1)

    def test_manifest_identity_canonical_fields_and_large_payload(self):
        task = self.begin()
        self.assertEqual(task.path.parent.parent.name, "20261008")
        attempt = task.append_attempt()
        event_id = task.append_event({"event": "model_request", "event_type": "model.request",
            "source": "model_transport", "visibility": "agent_input", "sim_time_s": 1.25, "sim_step": 125,
            "caused_by_event_ids": [f"{task.task_id}:1"],
            "identities": {"decision_id": "decision-1", "model_call_id": "model-1"},
            "detail": {"messages": [{"role": "user", "content": "context" * 3000}]}}, attempt_id=attempt)
        raw = self.store.read_events(task.task_id, resolve=False)[-1]
        self.assertEqual(raw["event_id"], event_id)
        self.assertEqual(raw["event_type"], "model.request")
        self.assertEqual(raw["run_id"], "run-a")
        self.assertEqual(raw["decision_id"], "decision-1")
        self.assertEqual(raw["model_call_id"], "model-1")
        self.assertEqual(raw["visibility"], "agent_input")
        self.assertEqual(raw["sim_step"], 125)
        self.assertEqual(raw["caused_by_event_ids"], [f"{task.task_id}:1"])
        self.assertGreaterEqual(raw["monotonic_elapsed_ns"], 0)
        self.assertIn("$artifact", raw["payload"])
        self.assertEqual(len(raw["artifact_refs"]), 1)
        resolved = self.store.read_events(task.task_id)[-1]
        self.assertEqual(len(resolved["payload"]["detail"]["messages"][0]["content"]), 21000)
        self.assertEqual(task.data["trusted_task_spec"], {"target": "table"})
        self.assertEqual(task.data["attempts"][0]["events"][0]["event"], "model_request")

    def test_terminal_facts_are_immutable_and_projection_is_rebuildable(self):
        task = self.begin()
        task.append_attempt()
        self.finish(task)
        original_seal = task.seal_path.read_bytes()
        task.path.write_text("{stale projection", encoding="utf-8")
        self.assertEqual(self.store.load(task.task_id)["outcome"]["status"], "SUCCESS")
        self.assertTrue(self.store.verify(task.task_id)["valid"])
        self.assertTrue(self.store.verify(task.task_id)["warnings"])
        self.store.rebuild(task.task_id)
        self.assertEqual(task.seal_path.read_bytes(), original_seal)
        self.assertFalse(self.store.verify(task.task_id)["warnings"])
        reopened = self.store.open(task.task_id)
        for mutate in (lambda: reopened.append_attempt(), lambda: reopened.append_event({"event": "late"}),
                       lambda: reopened.update(metadata={"rewrite": True}),
                       lambda: reopened.finish(final_state={}, outcome={"status": "FAILED"})):
            with self.assertRaises(TaskRecordError):
                mutate()

    def test_tool_view_points_to_same_fact_and_cannot_drift(self):
        task = self.begin()
        attempt = task.append_attempt()
        source = task.append_event({"event": "before_tool", "observation": {"held": None},
            "detail": {"skill": "pick"}}, attempt_id=attempt)
        tool_id = task.append_tool_event({"source_event_id": source, "detail": {"skill": "fake"}}, attempt_id=attempt)
        tool = task.data["attempts"][0]["tool_events"][0]
        self.assertEqual(tool_id, source)
        self.assertEqual(tool["event_id"], source)
        self.assertEqual(tool["detail"]["skill"], "pick")
        self.assertEqual(tool["state_before"], {"held": None})
        self.assertEqual(self.store.read_events(task.task_id)[-1]["payload"], {"source_event_id": source})
        with self.assertRaises(TaskRecordError):
            task.append_tool_event({"source_event_id": source})

    def test_corrupt_tail_recovery_preserves_original_evidence(self):
        task = self.begin()
        task.append_attempt()
        self.finish(task)
        with task.journal_path.open("ab") as stream:
            stream.write(b'{"torn":')
        original = task.journal_path.read_bytes()
        with self.assertRaisesRegex(TaskRecordError, "tail"):
            self.store.load(task.task_id)
        recovered = self.store.rebuild(task.task_id, recover=True)
        self.assertEqual(task.journal_path.read_bytes(), original)
        self.assertTrue(recovered["recovery_report"]["source_preserved"])
        self.assertFalse(recovered["recovery_report"]["training_eligible"])
        self.assertTrue((task.directory / "recovery.json").exists())
        self.assertEqual(recovered["task"]["outcome"]["status"], "SUCCESS")
        self.assertFalse(self.store.verify(task.task_id)["valid"])
        self.assertEqual(self.store.query(training_only=True), [])

    def test_hash_sequence_missing_artifact_and_seal_damage_are_detected(self):
        for failure in ("sequence", "artifact", "seal"):
            with self.subTest(failure=failure):
                task = self.begin()
                task.append_attempt()
                task.append_event({"event": "large", "detail": {"data": failure * 10000}})
                self.finish(task)
                if failure == "sequence":
                    lines = task.journal_path.read_bytes().splitlines()
                    row = json.loads(lines[0]); row["seq"] = 9
                    lines[0] = json.dumps(row).encode("utf-8")
                    task.journal_path.write_bytes(b"\n".join(lines) + b"\n")
                elif failure == "artifact":
                    reference = self.store.read_events(task.task_id, resolve=False)[1]["artifact_refs"][0]
                    (self.root / reference["path"]).unlink()
                else:
                    seal = json.loads(task.seal_path.read_text(encoding="utf-8"))
                    seal["event_count"] += 1
                    task.seal_path.write_text(json.dumps(seal), encoding="utf-8")
                report = self.store.verify(task.task_id)
                self.assertFalse(report["valid"])
                self.assertTrue(report["errors"])

    def test_artifact_or_seal_failure_does_not_replay_or_reopen_facts(self):
        task = self.begin()
        task.append_attempt()
        before = task.journal_path.read_bytes()
        with patch.object(self.store, "put_artifact", side_effect=OSError("artifact disk full")):
            with self.assertRaises(OSError):
                task.append_event({"event": "large", "detail": "x" * 10000})
        self.assertEqual(task.journal_path.read_bytes(), before)
        def write(path, value, **kwargs):
            if path.name == "seal.json":
                raise OSError("seal disk full")
            return atomic_json(path, value, **kwargs)
        with patch("embodied_agent.recording.store.atomic_json", side_effect=write):
            with self.assertRaises(OSError):
                self.finish(task)
        self.assertEqual(self.store.load(task.task_id)["record_status"], "COMPLETED")
        self.assertFalse(self.store.verify(task.task_id)["valid"])
        with self.assertRaises(TaskRecordError):
            task.append_event({"event": "retry physical action"})

    def test_read_only_queries_and_rebuilt_index_have_no_authority(self):
        self.assertFalse(self.root.exists())
        self.assertEqual(self.store.query(), [])
        self.assertFalse(self.root.exists())
        with self.assertRaises(TaskRecordError):
            self.store.load("0" * 32)
        self.assertFalse(self.root.exists())
        task = self.begin(); task.append_attempt(); self.finish(task)
        (self.root / "index.sqlite3").unlink()
        self.assertEqual(len(self.store.query(training_only=True)), 1)
        self.assertFalse((self.root / "index.sqlite3").exists())
        rebuilt = self.store.rebuild_index()
        self.assertEqual(rebuilt["task_count"], 1)
        with closing(sqlite3.connect(rebuilt["path"])) as connection:
            self.assertEqual(connection.execute("SELECT task_id FROM tasks").fetchone()[0], task.task_id)
        self.assertEqual(self.begin().data["sequence"], 2)
        legacy = self.root / "old.json"
        legacy.write_text('{"schema_version":1}', encoding="utf-8")
        with self.assertRaisesRegex(TaskRecordError, "v1"):
            self.store.load(legacy)

    def test_nested_source_artifact_integrity_and_equal_content_do_not_alias(self):
        source = self.store.put_artifact({"files": {"agent.py": "real source"}})
        task = self.begin(versions={"source_snapshot_ref": source})
        task.append_attempt(); self.finish(task)
        (self.root / source["path"]).unlink()
        self.assertFalse(self.store.verify(task.task_id)["valid"])
        other = self.store.begin(map_id="fixture", map_definition={}, initial_state={},
                                 natural_language="copy isolation", versions={}, metadata={})
        other.update(metadata={"changed": True})
        self.assertEqual(other.data["versions"], {})
        self.assertEqual(other.data["metadata"], {"changed": True})
        self.assertEqual(other.data["map"]["definition"], {})
        self.assertEqual(other.data["initial_state"], {})

    def test_unreadable_projection_does_not_invalidate_sealed_facts(self):
        task = self.begin()
        task.append_attempt(); self.finish(task)
        original_read = Path.read_bytes
        def read(path):
            if path == task.path:
                raise PermissionError("projection cache unavailable")
            return original_read(path)
        with patch.object(Path, "read_bytes", read):
            report = self.store.verify(task.task_id)
            self.assertTrue(report["valid"])
            self.assertTrue(report["sealed"])
            self.assertTrue(any("unreadable" in warning for warning in report["warnings"]))

    def test_rehashed_event_cannot_claim_another_run(self):
        task = self.begin()
        task.append_attempt()
        row = json.loads(task.journal_path.read_bytes())
        row["run_id"] = "different-run"
        row["event_hash"] = event_hash(row)
        task.journal_path.write_bytes(json.dumps(row).encode("utf-8") + b"\n")
        report = self.store.verify(task.task_id)
        self.assertFalse(report["valid"])
        self.assertTrue(any("identity" in error for error in report["errors"]))

    def test_unavailable_nested_attachment_fails_before_journal_append(self):
        missing = self.store.put_artifact({"checkpoint": "binary reference placeholder"})
        attachment = self.store.put_artifact({"nested_checkpoint_ref": missing})
        task = self.begin(); task.append_attempt()
        original = task.journal_path.read_bytes()
        (self.root / missing["path"]).unlink()
        with self.assertRaisesRegex(TaskRecordError, "artifact"):
            task.append_event({"event": "attachment", "artifact_refs": [attachment]})
        self.assertEqual(task.journal_path.read_bytes(), original)
        self.assertTrue(self.store.verify(task.task_id)["valid"])


if __name__ == "__main__":
    unittest.main()

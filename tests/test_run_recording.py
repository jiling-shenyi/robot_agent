"""Run audit durability, bounded late replies and read-only integrity checks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.recording import TaskRecordError, TaskRecordStore
from embodied_agent.recording.common import decode, encode
from embodied_agent.recording.journal import event_hash
from embodied_agent.recording.run import RunRecorder


class RunRecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = TaskRecordStore(Path(temporary.name) / "records")

    def run_record(self, name="test-run", **kwargs):
        return RunRecorder(self.store, name, session_id="test-session", **kwargs)

    def snapshot(self):
        return {path.relative_to(self.store.root).as_posix(): path.read_bytes()
                for path in self.store.root.rglob("*") if path.is_file()}

    def test_canonical_artifacts_preserve_payload_and_detach_data(self):
        run = self.run_record(manifest={"purpose": "test audit"})
        reference = self.store.put_artifact({"actual_source": "source code"})
        body = {"source_ref": reference, "messages": ["context" * 2000]}
        event_id = run.append("model.response", body, task_id="task-one", consumed=False,
            decision_id="decision-one", model_call_id="model-one", sim_step=12, sim_time_s=0.12,
            parent_event_ids=["test-run:1"], caused_by_event_ids=["test-run:1"])
        body["messages"].clear()
        raw = run.read_events(resolve=False)[-1]
        self.assertEqual(raw["event_id"], event_id)
        self.assertEqual(raw["run_id"], "test-run")
        self.assertEqual(raw["session_id"], "test-session")
        self.assertEqual(raw["decision_id"], "decision-one")
        self.assertEqual(raw["sim_step"], 12)
        self.assertIn("$artifact", raw["payload"])
        self.assertIn(raw["payload"]["$artifact"], raw["artifact_refs"])
        self.assertEqual(run.read_events()[-1]["payload"]["messages"], ["context" * 2000])
        run.append("audit.scalar", False)
        self.assertIs(run.read_events()[-1]["payload"], False)

    def test_concurrent_append_sequence_is_continuous(self):
        run = self.run_record()
        with ThreadPoolExecutor(max_workers=4) as pool:
            identifiers = list(pool.map(lambda index: run.append("audit.concurrent", {"index": index}), range(20)))
        run.close()
        events = run.read_events()
        self.assertEqual(len(set(identifiers)), 20)
        self.assertEqual([row["seq"] for row in events], list(range(1, 23)))
        self.assertEqual(events[-1]["event_type"], "run.finished")

    def test_partial_append_poison_preserves_tail_and_blocks_close(self):
        run = self.run_record()
        before = run.journal_path.read_bytes()
        def torn_write(path, event):
            with path.open("ab") as stream:
                stream.write(b'{"partial":')
            raise OSError("disk full during append")
        with patch("embodied_agent.recording.run.append", side_effect=torn_write):
            with self.assertRaisesRegex(OSError, "disk full"):
                run.append("audit.failed", {"effect": "already happened"})
        evidence = run.journal_path.read_bytes()
        self.assertEqual(evidence, before + b'{"partial":')
        for mutation in (lambda: run.append("audit.retry"), run.close):
            with self.assertRaisesRegex(TaskRecordError, "stopped"):
                mutation()
        self.assertEqual(run.journal_path.read_bytes(), evidence)
        with self.assertRaisesRegex(TaskRecordError, "tail"):
            run.read_events()

    def test_fsync_failure_cannot_be_retried_even_when_row_is_complete(self):
        run = self.run_record()
        with patch("embodied_agent.recording.journal.os.fsync", side_effect=OSError("fsync failed")):
            with self.assertRaisesRegex(OSError, "fsync"):
                run.append("audit.unconfirmed", {"effect": "actual"})
        evidence = run.journal_path.read_bytes()
        self.assertEqual(run.read_events()[-1]["event_type"], "audit.unconfirmed")
        with self.assertRaisesRegex(TaskRecordError, "stopped"):
            run.close()
        self.assertEqual(run.journal_path.read_bytes(), evidence)

    def test_external_corrupt_tail_is_detected_before_another_append(self):
        run = self.run_record()
        with run.journal_path.open("ab") as stream:
            stream.write(b'{"external":')
        evidence = run.journal_path.read_bytes()
        with self.assertRaisesRegex(TaskRecordError, "tail"):
            run.append("audit.must_not_append")
        with self.assertRaisesRegex(TaskRecordError, "stopped"):
            run.close()
        self.assertEqual(run.journal_path.read_bytes(), evidence)

    def test_rehashed_invalid_schema_identity_and_artifact_declaration_are_rejected(self):
        mutations = (
            lambda row: row.pop("source"),
            lambda row: row.update(session_id="other-session"),
            lambda row: row.update(sim_step=True),
            lambda row: row.update(payload={"$artifact": row["artifact_refs"][0]}, artifact_refs=[]),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                run = self.run_record(f"run-{index}")
                run.append("audit.large", {"text": "x" * 5000})
                rows = [decode(line) for line in run.journal_path.read_bytes().splitlines()]
                mutate(rows[-1])
                rows[-1]["event_hash"] = event_hash(rows[-1])
                run.journal_path.write_bytes(b"".join(encode(row) + b"\n" for row in rows))
                with self.assertRaises(TaskRecordError):
                    run.read_events()

    def test_missing_nested_artifact_is_detected_and_prevents_append(self):
        source = self.store.put_artifact({"source": "actual file text"})
        bundle = self.store.put_artifact({"source_snapshot_ref": source})
        run = self.run_record(manifest={"behavior_bundle_ref": bundle})
        run.append("audit.bundle", {"bundle_ref": bundle})
        (self.store.root / source["path"]).unlink()
        evidence = run.journal_path.read_bytes()
        with self.assertRaisesRegex(TaskRecordError, "artifact"):
            run.read_events()
        with self.assertRaisesRegex(TaskRecordError, "artifact"):
            run.append("audit.next")
        self.assertEqual(run.journal_path.read_bytes(), evidence)

    def test_closed_run_accepts_only_unconsumed_late_audit_without_changing_tasks(self):
        run = self.run_record()
        task = self.store.begin(map_id="fixture", map_definition={}, initial_state={},
            natural_language="cancelled request", metadata={"run_id": run.run_id})
        task.finish(final_state={}, outcome={"status": "ABORTED"}, record_status="ABORTED")
        task_facts = {path: path.read_bytes() for path in task.directory.iterdir() if path.is_file()}
        run.close()
        before_late = run.journal_path.read_bytes()
        run.close()
        self.assertEqual(run.journal_path.read_bytes(), before_late)
        for event_type, consumed in (("task.started", None), ("late.model.response", True),
                                     ("late.model.response", None)):
            with self.assertRaisesRegex(TaskRecordError, "Finished"):
                run.append(event_type, consumed=consumed)
        run.append("late.model.response", {"content": "arrived after cancellation"},
                   consumed=False, task_id=task.task_id, model_call_id="late-model")
        event = run.read_events()[-1]
        self.assertFalse(event["consumed"])
        self.assertTrue(event["late_after_run_finished"])
        self.assertEqual(event["task_id"], task.task_id)
        self.assertEqual({path: path.read_bytes() for path in task_facts}, task_facts)
        self.assertEqual(sum(row["event_type"] == "run.finished" for row in run.read_events()), 1)

    def test_read_events_is_read_only_and_returns_isolated_copies(self):
        run = self.run_record()
        run.append("audit.record", {"nested": {"actual": True}})
        run.close()
        before = self.snapshot()
        events = run.read_events()
        events[1]["payload"]["nested"]["actual"] = False
        self.assertTrue(run.read_events()[1]["payload"]["nested"]["actual"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.store.root / "index.sqlite3").exists())

    def test_artifact_failure_preserves_journal_and_stops_collector(self):
        run = self.run_record()
        before = run.journal_path.read_bytes()
        with patch.object(self.store, "put_artifact", side_effect=OSError("artifact disk full")):
            with self.assertRaisesRegex(OSError, "artifact"):
                run.append("audit.large", {"data": "x" * 5000})
        self.assertEqual(run.journal_path.read_bytes(), before)
        with self.assertRaisesRegex(TaskRecordError, "stopped"):
            run.append("audit.retry")


if __name__ == "__main__":
    unittest.main()

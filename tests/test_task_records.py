"""Task persistence guarantees, independent of models or physics runtimes."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.recording import TaskRecordError, TaskRecordStore, map_alias


START = dt.datetime(2026, 10, 5, 16, 30, tzinfo=dt.timezone.utc)


def _allocate_in_process(root: str, count: int, alias: str, result_queue) -> None:
    try:
        store = TaskRecordStore(root, clock=lambda: START)
        paths = [store.begin(map_id=alias, map_definition={"id": alias}, initial_state={},
                             natural_language="并发任务").task_id for _ in range(count)]
        result_queue.put({"paths": paths})
    except BaseException as exc:
        result_queue.put({"error": repr(exc)})


class TaskRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "records"
        self.store = TaskRecordStore(self.root, clock=lambda: START)

    def begin(self, map_id="classic", **overrides):
        arguments = {"map_id": map_id, "map_definition": {"map": map_id, "objects": ["cube"]},
                     "initial_state": {"position": [1, 2, 3]}, "natural_language": "移动方块",
                     "versions": {"map": "abc", "model": "test"}}
        arguments.update(overrides)
        return self.store.begin(**arguments)

    def test_shanghai_start_date_and_global_counter_survive_restart(self):
        first = self.begin()
        self.assertEqual(first.path.name, "task.json")
        self.assertEqual(first.path.parent.name, first.task_id)
        self.assertEqual(first.path.parent.parent.name, "20261006")
        self.assertEqual(first.data["sequence"], 1)
        self.assertEqual(first.data["created_at"], "2026-10-06T00:30:00.000000+08:00")
        self.assertEqual(first.data["record_status"], "RUNNING")
        restarted = TaskRecordStore(self.root, clock=lambda: START)
        second = restarted.begin(map_id="home_living_room", map_definition={}, initial_state={},
                                 natural_language="摆放遥控器")
        self.assertEqual(second.data["sequence"], 2)
        self.assertEqual(second.data["map"]["alias"], "hlr")
        # The task's date remains its start date even after midnight.
        first._clock = lambda: START + dt.timedelta(days=1)
        first.finish(final_state={}, outcome={"status": "FAILED"})
        self.assertEqual(first.data["date"], "20261006")
        self.assertEqual(len(restarted.query(date="2026-10-06")), 2)

    def test_unknown_map_aliases_are_readable_distinct_and_safe(self):
        self.assertEqual(map_alias("alternate"), "alt")
        self.assertNotEqual(map_alias("office upstairs!"), map_alias("office upstairs?"))
        self.assertTrue(map_alias("office upstairs!").startswith("office-upstairs-"))
        self.assertNotIn("/", map_alias("../../楼上"))
        task = self.begin(map_id="../../楼上")
        self.assertTrue(task.path.is_relative_to(self.root / "tasks" / "20261006"))
        self.assertEqual(task.path.parent.name, task.task_id)
        self.assertEqual(self.store.load(task.task_id)["map"]["id"], "../../楼上")

    def test_inputs_returned_data_and_execution_slices_are_isolated(self):
        state = {"position": [1, 2]}
        map_definition = {"objects": [{"id": "remote"}]}
        task = self.begin(initial_state=state, map_definition=map_definition)
        state["position"][0] = 99
        map_definition["objects"].clear()
        plan = {"actions": [{"skill": "pick"}]}
        attempt = task.append_attempt(plan=plan)
        plan["actions"][0]["skill"] = "delete"
        execution = {"physics_events": [{"event": "grasped"}], "trajectory": [[1, 2]]}
        task.update_attempt(attempt, actual_execution=execution)
        execution["physics_events"].clear()
        execution["trajectory"][0][0] = 99
        view = task.data
        view["attempts"].clear()
        persisted = self.store.load(task.path)
        self.assertEqual(persisted["initial_state"], {"position": [1, 2]})
        self.assertEqual(persisted["map"]["definition"], {"objects": [{"id": "remote"}]})
        self.assertEqual(persisted["attempts"][0]["plan"]["actions"][0]["skill"], "pick")
        self.assertEqual(persisted["attempts"][0]["actual_execution"]["trajectory"], [[1, 2]])

    def test_attempt_chain_keeps_rejected_plan_prediction_and_actual_events_separate(self):
        task = self.begin()
        first = task.append_attempt(planning_observation={"held": None}, model_input={"messages": []})
        task.update_attempt(first, plan={"actions": [{"skill": "pick"}]}, proposal={"accepted": False})
        check = task.append_check({"prediction": {"danger": "collision"}, "executed": False})
        task.set_attempt_feedback({"status": "BLOCKED"})
        second = task.append_attempt(parent_attempt_id=first, trigger_event_id=check,
                                     planning_observation={"held": None}, plan={"actions": [{"skill": "stop"}]})
        task.append_tool_event({"skill": "stop", "phase": "completed", "actual": True}, attempt_id=second)
        task.set_attempt_feedback({"status": "SUCCESS"}, attempt_id=second)
        task.finish(final_state={"held": None}, outcome={"status": "SUCCESS", "verified": True})
        attempts = self.store.load(task.path)["attempts"]
        self.assertEqual([row["attempt_id"] for row in attempts], [1, 2])
        self.assertEqual(attempts[0]["tool_events"], [])
        self.assertEqual(attempts[0]["checks"][0]["prediction"], {"danger": "collision"})
        self.assertEqual(attempts[0]["feedback"], {"return_to_caller": {"status": "BLOCKED"}, "sent_to_agent": []})
        self.assertEqual(attempts[1]["parent_attempt_id"], 1)
        self.assertEqual(attempts[1]["trigger_event_id"], check)
        self.assertEqual(attempts[1]["tool_events"][0]["skill"], "stop")

    def test_immutable_fields_and_finished_task_cannot_be_overwritten(self):
        task = self.begin()
        first = task.append_attempt(plan={"actions": ["stop"]})
        before = task.path.read_bytes()
        with self.assertRaisesRegex(TaskRecordError, "immutable"):
            task.update_attempt(first, plan={"actions": ["move"]})
        self.assertEqual(task.path.read_bytes(), before)
        task.update_attempt(first, actual_execution={"trajectory": []})
        with self.assertRaises(TaskRecordError):
            task.update_attempt(first, actual_execution={"trajectory": [1]})
        task.finish(final_state={}, outcome={"status": "FAILED"})
        for mutation in (lambda: task.append_attempt(), lambda: task.append_event({}),
                         lambda: task.update(costs={"requests": 1}),
                         lambda: task.set_attempt_feedback({}),
                         lambda: task.finish(final_state={}, outcome={"status": "SUCCESS"})):
            with self.subTest(mutation=mutation), self.assertRaisesRegex(TaskRecordError, "immutable"):
                mutation()

    def test_deferred_initial_state_is_observed_only_once(self):
        task = self.begin(initial_state=None)
        with self.assertRaises(TaskRecordError):
            task.set_initial_state(None)
        state = {"held": "remote"}
        task.set_initial_state(state)
        state["held"] = None
        self.assertEqual(task.data["initial_state"], {"held": "remote"})
        with self.assertRaisesRegex(TaskRecordError, "immutable"):
            task.set_initial_state({"held": None})

    def test_strict_json_invalid_data_never_changes_last_committed_record(self):
        task = self.begin()
        task.append_attempt()
        before = task.path.read_bytes()
        for value in (float("nan"), float("inf"), float("-inf"), object()):
            with self.subTest(value=value), self.assertRaises(TaskRecordError):
                task.append_event({"state": value})
            self.assertEqual(task.path.read_bytes(), before)
        with self.assertRaises(TaskRecordError):
            self.begin(versions={"calibration": float("nan")})
        self.assertEqual(len(self.store.list()), 1)
        with self.assertRaises(TaskRecordError):
            task.append_event({1: "non-string key"})

    def test_projection_replace_failure_preserves_source_fact_then_rebuilds(self):
        task = self.begin()
        task.append_attempt()
        before_bytes, before_data = task.path.read_bytes(), task.data
        with patch("embodied_agent.recording.common.os.replace", side_effect=OSError("disk unavailable")):
            task.append_event({"event": "durable despite stale cache"})
        self.assertEqual(task.path.read_bytes(), before_bytes)
        self.assertEqual(task.data["attempts"][0]["events"][0]["event"], "durable despite stale cache")
        self.assertNotEqual(task.data, before_data)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])
        self.store.rebuild(task.task_id)
        self.assertNotEqual(task.path.read_bytes(), before_bytes)
        task.append_event({"event": "committed"})
        self.assertEqual(task.data["attempts"][0]["events"][1]["sequence"], 2)

    def test_journal_append_failure_keeps_last_committed_fact(self):
        task = self.begin(); task.append_attempt()
        before, view = task.journal_path.read_bytes(), task.data
        with patch("embodied_agent.recording.journal.append", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                task.append_event({"event": "never issued"})
        self.assertEqual(task.journal_path.read_bytes(), before)
        self.assertEqual(task.data, view)
        task.append_event({"event": "committed"})
        self.assertEqual(task.data["attempts"][0]["events"][0]["sequence"], 1)

    def test_failed_begin_reports_error_and_does_not_leave_claimed_task(self):
        with patch("embodied_agent.recording.common.os.link", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                self.begin()
        self.assertEqual(self.store.list(), [])
        next_task = self.begin()
        self.assertEqual(next_task.data["sequence"], 2)  # Persistent gaps are safe.

    def test_query_uses_actual_fields_reports_corrupt_files_and_requires_verification(self):
        running = self.begin()
        failed = self.begin(map_id="alternate")
        failed.finish(final_state={}, outcome={"status": "FAILED", "verified": True})
        unverified = self.begin()
        unverified.finish(final_state={}, outcome={"status": "SUCCESS"})
        verified = self.begin(map_id="home_living_room")
        verified.finish(final_state={}, outcome={"status": "SUCCESS", "verified": True})
        malformed = self.root / "tasks" / "20261006" / ("f" * 32)
        malformed.mkdir(parents=True)
        (malformed / "manifest.json").write_text('{"schema_version":2,"broken":', encoding="utf-8")
        self.assertEqual([row["task_id"] for row in self.store.query(training_only=True)], [verified.task_id])
        self.assertEqual(len(self.store.errors), 1)
        self.assertEqual(self.store.errors[0]["path"], str(malformed / "task.json"))
        self.assertEqual([row["task_id"] for row in self.store.query(record_status="RUNNING")], [running.task_id])
        self.assertEqual([row["task_id"] for row in self.store.query(map_id="alternate", status="FAILED")], [failed.task_id])
        self.assertEqual(len(self.store.query(date="20261006")), 4)
        self.assertEqual(self.store.query(date="20261005"), [])

    def test_load_rejects_duplicate_keys_nonfinite_numbers_and_inconsistent_identity(self):
        task = self.begin()
        original = task.manifest_path.read_text(encoding="utf-8")
        invalids = [original.replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2'),
                    original.replace('"sequence": 1', '"sequence": NaN'),
                    original.replace('"sequence": 1', '"sequence": 1e999'),
                    original.replace('"date": "20261006"', '"date": "20261005"'),
                    original.replace(task.task_id, "0" * 32)]
        for invalid in invalids:
            with self.subTest(invalid=invalid[:50]):
                task.manifest_path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(TaskRecordError):
                    self.store.load(task.path)

    def test_counter_recovers_from_existing_records_without_overwrite(self):
        task = self.begin()
        before = task.path.read_bytes()
        (self.root / "index.sqlite3").unlink()
        self.assertEqual(self.begin(map_id="alternate").data["sequence"], 2)
        self.assertEqual(task.path.read_bytes(), before)

    def test_relative_store_root_and_path_loading(self):
        original_cwd = Path.cwd()
        try:
            os.chdir(self.temporary.name)
            store = TaskRecordStore("relative", clock=lambda: START)
            task = store.begin(map_id="classic", map_definition={}, initial_state={}, natural_language="task")
            self.assertEqual(store.load(task.task_id), store.load(task.path))
            self.assertEqual(store.load(task.path.relative_to(store.root)), task.data)
            self.assertEqual(len(store.query()), 1)
        finally:
            os.chdir(original_cwd)

    def test_uuid_identity_is_unique_across_roots_with_identical_filenames(self):
        first = self.begin()
        other_store = TaskRecordStore(Path(self.temporary.name) / "other", clock=lambda: START)
        second = other_store.begin(map_id="classic", map_definition={}, initial_state={},
                                   natural_language="another run")
        self.assertEqual(first.path.name, second.path.name)
        self.assertNotEqual(first.task_id, second.task_id)
        self.assertNotEqual(first.task_id, first.path.stem)
        self.assertEqual(self.store.load(first.task_id), first.data)
        with self.assertRaises(TaskRecordError):
            self.store.load(second.task_id)

    def test_map_definition_digest_is_canonical_and_tampering_is_rejected(self):
        definition = {"地图": "客厅", "objects": [{"name": "遥控器", "position": [1.0, 2.0]}]}
        first = self.begin(map_definition=definition)
        second = self.begin(map_definition={"objects": definition["objects"], "地图": "客厅"})
        canonical = json.dumps(definition, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        expected = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(first.data["map"]["definition_sha256"], expected)
        self.assertEqual(second.data["map"]["definition_sha256"], expected)
        manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
        reference = manifest["map"]["definition_ref"]
        invalid = dict(definition)
        invalid["地图"] = "修改后的场景"
        (self.root / reference["path"]).write_text(json.dumps(invalid, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(TaskRecordError, "integrity"):
            self.store.load(first.path)
        # Both tasks reference the identical content address, not independent copies.
        self.assertEqual(self.store.query(training_only=True), [])
        self.assertEqual(len(self.store.errors), 2)

    def test_query_reports_invalid_field_types_instead_of_crashing(self):
        task = self.begin()
        original = json.loads(task.manifest_path.read_text(encoding="utf-8"))
        for field, value in (("map", []), ("schema_version", True),
                             ("sequence", True), ("identities", ["SUCCESS"]),
                             ("versions_ref", []), ("created_at", "2026-10-06T00:30:00")):
            with self.subTest(field=field):
                invalid = dict(original)
                invalid[field] = value
                task.manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
                self.assertEqual(self.store.query(training_only=True), [])
                self.assertEqual(len(self.store.errors), 1)

    def test_concurrent_processes_share_one_global_daily_sequence(self):
        context = mp.get_context("spawn")
        queue = context.Queue()
        processes = [context.Process(target=_allocate_in_process,
                                     args=(str(self.root), 5, map_id, queue))
                     for map_id in ("classic", "alternate", "home_living_room", "custom map")]
        try:
            for process in processes:
                process.start()
            outputs = [queue.get(timeout=40) for _ in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertFalse([output for output in outputs if "error" in output], outputs)
            names = [name for output in outputs for name in output["paths"]]
            self.assertEqual(len(names), len(set(names)))
            records = self.store.query()
            self.assertEqual(len(records), 20)
            self.assertEqual(sorted(row["sequence"] for row in records), list(range(1, 21)))
            self.assertEqual(self.store.errors, [])
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            queue.close()


if __name__ == "__main__":
    unittest.main()

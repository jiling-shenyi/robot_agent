"""Task JSON boundaries backed by the unchanged live simulation/control chain."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.apps.demo.session import create_batch_session, run_batch
from embodied_agent.evaluation.evidence import jsonable
from embodied_agent.recording import TaskRecordStore


class TaskRecordIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "run"
        self.records = Path(self.temporary.name) / "records"
        self.session = create_batch_session(output_dir=self.output, records_dir=self.records,
                                            planner_kind="stub", environment_mode="rules")
        self.addCleanup(self.session.close)

    def read_task(self, result):
        path = Path(result["task_record_path"])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.name, "task.json")
        self.assertEqual(path.parent.name, result["task_id"])
        self.assertRegex(path.parent.parent.name, r"^\d{8}$")
        self.assertEqual(path.parent.parent.parent, self.records / "tasks")
        record = self.session.writer.store.load(result["task_id"])
        self.assertEqual(record["schema_version"], 2)
        self.assertTrue(self.session.writer.store.verify(result["task_id"])["valid"])
        self.assertEqual(self.records / result["task_record"]["uri"], path)
        for source in ("manifest.json", "events.jsonl", "seal.json"):
            self.assertTrue((path.parent / source).is_file())
        self.assertEqual(record["task_id"], result["task_id"])
        self.assertEqual(record["natural_language"], result["instruction"])
        self.assertEqual(record["outcome"]["status"], result["status"])
        self.assertEqual(record["outcome"].get("error_code"), result.get("error_code"))
        self.assertTrue(record["attempts"])
        for attempt in record["attempts"]:
            self.assertEqual(attempt["checks"], [])
            self.assertEqual(attempt["feedback"]["sent_to_agent"], [])
        for name in ("actions.jsonl", "events.jsonl", "episodes.jsonl"):
            self.assertFalse((self.output / name).exists())
        return record

    def test_continuous_tasks_capture_request_state_and_own_only_their_evidence(self):
        episode = self.session.select_map("home_living_room")
        before_first = jsonable(episode.snapshot())
        first = self.session.run_agent("停止", "robot")
        first_record = self.read_task(first)
        first_bytes = Path(first["task_record_path"]).read_bytes()
        self.assertEqual(first_record["initial_state"], before_first)
        self.assertEqual(first_record["final_state"], jsonable(episode.snapshot()))
        self.assertGreater(first["physics_steps"], 0)

        before_second = jsonable(episode.snapshot())
        second = self.session.run_agent("观察", "robot", record=False)
        second_record = self.read_task(second)
        self.assertIs(self.session.episode, episode)
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.assertEqual(second_record["initial_state"], before_second)
        self.assertEqual(second_record["initial_state"], first_record["final_state"])
        self.assertNotEqual(second_record["initial_state"], before_first)
        self.assertEqual(second["physics_steps"], 0)
        first_physics = first_record["attempts"][0]["actual_execution"]
        second_physics = second_record["attempts"][0]["actual_execution"]
        self.assertTrue(first_physics["physics_events"])
        self.assertEqual(second_physics["physics_events"], [])
        self.assertEqual(second_physics["trajectory"], [])
        self.assertFalse(any(event.get("type") == "safe_stop"
                             for event in second_physics["episode_events"]))
        self.assertEqual(Path(first["task_record_path"]).read_bytes(), first_bytes)
        self.session.reset()
        self.assertEqual(Path(first["task_record_path"]).read_bytes(), first_bytes)
        self.assertEqual(self.session.finish()["episode_count"], 1)

    def test_environment_success_and_rejection_keep_truthful_before_after(self):
        self.session.select_map("classic")
        before = jsonable(self.session.episode.snapshot())
        edited = self.session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        edit_record = self.read_task(edited)
        self.assertEqual(edited["status"], "SUCCESS", edited)
        self.assertEqual(edit_record["initial_state"], before)
        self.assertEqual(edit_record["final_state"], jsonable(self.session.episode.snapshot()))
        self.assertEqual(edit_record["map"]["definition"]["cube_position_m"],
                         edited["before"]["cube_position_m"])
        self.assertEqual(edited["after"], self.session.store.load("classic").to_dict())
        edit_bytes = Path(edited["task_record_path"]).read_bytes()
        before_rejected = jsonable(self.session.episode.snapshot())
        rejected = self.session.edit_environment("删除所有文件")
        rejected_record = self.read_task(rejected)
        self.assertEqual(rejected["status"], "FAILED")
        self.assertEqual(rejected_record["initial_state"], before_rejected)
        self.assertEqual(rejected_record["final_state"], before_rejected)
        self.assertEqual(rejected_record["attempts"][0]["tool_events"], [])
        self.assertEqual(Path(edited["task_record_path"]).read_bytes(), edit_bytes)

    def test_saved_environment_edit_and_refresh_failure_are_both_recorded(self):
        self.session.select_map("classic")
        with patch.object(self.session, "reset", side_effect=RuntimeError("renderer stopped")):
            result = self.session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        record = self.read_task(result)
        self.assertEqual(result["error_code"], "REFRESH_FAILED")
        feedback = record["attempts"][0]["feedback"]["return_to_caller"]
        self.assertTrue(feedback["persisted"])
        self.assertEqual(feedback["after"], self.session.store.load("classic").to_dict())
        self.assertEqual(feedback["refresh_error"]["message"], "renderer stopped")

    def test_panda_goal_rejection_does_not_reuse_the_previous_tasks_physical_events(self):
        episode = self.session.select_map("classic")
        self.session.runtime["budgets"]["max_episode_steps"] = episode.thresholds["scene_settle_steps"] + 1
        self.session.runtime["task_budget"]["max_replans"] = 0
        first = self.session.run_agent("把方块放到 A 区。", "robot")
        first_record = self.read_task(first)
        self.assertEqual(first["status"], "FAILED")
        self.assertEqual(first["first_error"]["error_code"], "BUDGET_EXHAUSTED")
        self.assertTrue(first_record["attempts"][0]["actual_execution"]["episode_events"])
        first_bytes = Path(first["task_record_path"]).read_bytes()
        before = jsonable(episode.snapshot())
        rejected = self.session.run_agent("把方块扔到 target_a", "robot")
        record = self.read_task(rejected)
        self.assertEqual(rejected["error_code"], "CAPABILITY_GAP")
        self.assertEqual(rejected["actions"], [])
        self.assertEqual(record["initial_state"], before)
        self.assertGreaterEqual(record["final_state"]["step"], before["step"])
        self.assertEqual(record["attempts"][0]["tool_events"], [])
        physical = record["attempts"][0]["actual_execution"]
        self.assertFalse(any(row.get("type") in {"pick_failed", "grasp_verified"} for row in physical["episode_events"]))
        self.assertEqual(physical["range"]["start_step"], before["step"])
        self.assertEqual(Path(first["task_record_path"]).read_bytes(), first_bytes)

    def test_log_failure_after_saved_environment_edit_keeps_truth_and_releases_writer(self):
        self.session.select_map("home_living_room")
        record_event = self.session.writer.record_event

        def fail_environment_plan(event, *arguments, **keywords):
            if event == "environment_plan":
                raise OSError("task log disk unavailable")
            return record_event(event, *arguments, **keywords)

        with patch.object(self.session.writer, "record_event", side_effect=fail_environment_plan):
            edited = self.session.edit_environment("将房间名称设为日志故障测试客厅")
        self.assertEqual(edited["status"], "SUCCESS", edited)
        self.assertTrue(edited["persisted"])
        self.assertEqual(edited["after"], self.session.store.load("home_living_room").to_dict())
        self.assertEqual(self.session.world.name, "日志故障测试客厅")
        self.assertEqual(edited["recording_error"],
                         {"type": "OSError", "message": "task log disk unavailable"})
        self.assertIsNone(self.session.writer.active_record)
        self.assertIsNone(self.session.writer.active_attempt)
        path = Path(edited["task_record_path"])
        incomplete_bytes = path.read_bytes()
        incomplete = json.loads(incomplete_bytes)
        self.assertEqual(incomplete["task_id"], edited["task_id"])
        self.assertEqual(incomplete["record_status"], "RUNNING")
        self.assertIsNone(incomplete["outcome"])
        self.assertIsNone(incomplete["finished_at"])

        before_observe = jsonable(self.session.episode.snapshot())
        observed = self.session.run_agent("观察", "robot")
        record = self.read_task(observed)
        self.assertEqual(observed["status"], "SUCCESS", observed)
        self.assertNotEqual(observed["task_id"], edited["task_id"])
        self.assertEqual(record["initial_state"], before_observe)
        self.assertEqual(record["map"]["definition"]["name"], "日志故障测试客厅")
        self.assertEqual(record["record_status"], "COMPLETED")
        self.assertEqual(path.read_bytes(), incomplete_bytes)
        self.assertIsNone(self.session.writer.active_record)

    def test_batch_keeps_each_language_unit_reference_and_expected_rejections(self):
        cases = [{"case_id": "short-life", "map_id": "home_living_room", "steps": [
            {"agent": "robot", "instruction": "stop", "robot_plan": {
                "schema_version": 1, "actions": [{"skill": "stop"}]}},
            {"agent": "robot", "instruction": "query", "robot_plan": {
                "schema_version": 1, "actions": [{"skill": "observe"}]}},
        ]}, {"case_id": "restricted", "map_id": "home_living_room", "agent": "robot",
             "instruction": "registered restricted pick", "robot_plan": {
                 "schema_version": 1, "actions": [{"skill": "pick", "object_id": "medicine"}]},
             "expected": {"status": "FAILED", "error_code": "UNSUPPORTED_OPERATION"}}]
        output = Path(self.temporary.name) / "batch"
        records = Path(self.temporary.name) / "batch-records"
        summary = run_batch(cases, output_dir=output, records_dir=records,
                            planner_kind="stub", environment_mode="rules")
        self.assertTrue(summary["all_pass"], summary)
        self.assertEqual(summary["expected_rejection_count"], 1)
        self.assertEqual(summary["results"][1]["status"], "FAILED")
        actions = [action for case in summary["results"] for action in case["actions"]]
        self.assertEqual(len(actions), 3)
        self.assertEqual(len({action["task_id"] for action in actions}), 3)
        store = TaskRecordStore(records)
        self.assertEqual(len(store.list()), 3)
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(saved["task_records"], summary["task_records"])
        saved_actions = [action for case in saved["results"] for action in case["actions"]]
        self.assertEqual([action["task_id"] for action in saved_actions], [action["task_id"] for action in actions])
        for action in saved_actions:
            self.assertNotIn("actions", action)
            self.assertNotIn("episode_events", action)
            self.assertNotIn("initial_snapshot", action)
            self.assertNotIn("final_snapshot", action)
        for action in actions:
            record = store.load(action["task_id"])
            self.assertTrue(store.verify(action["task_id"])["valid"])
            self.assertEqual(record["task_id"], action["task_id"])
            self.assertEqual(record["outcome"]["status"], action["status"])
            self.assertEqual(record["natural_language"], action["instruction"])


if __name__ == "__main__":
    unittest.main()

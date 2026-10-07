"""Automated batch checks for component isolation and real task-result memory."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.apps.demo.session import create_batch_session
from embodied_agent.knowledge import KnowledgeBase
from embodied_agent.memory import MemoryStore
from embodied_agent.recording import TaskRecordStore


class SessionComponentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = json.loads((ROOT / "configs" / "agent_runtime.json").read_text(encoding="utf-8"))
        self.config["components"]["memory"]["enabled"] = True
        self.config["components"]["knowledge"]["enabled"] = True
        self.memory = MemoryStore()
        self.knowledge = KnowledgeBase([{"id": "units", "text": "坐标使用米"}], version="session-test-v1")

    def session(self, name, *, memory=None):
        session = create_batch_session(output_dir=self.directory / name, runtime_config=self.config,
            records_dir=self.directory / "records",
            planner_kind="stub", environment_mode="rules",
            memory_store=self.memory if memory is None else memory, knowledge_base=self.knowledge)
        self.addCleanup(session.close)
        return session

    def task(self, result):
        return TaskRecordStore(self.directory / "records").load(result["task_id"])

    def test_injected_candidates_are_identified_and_role_run_memory_is_isolated(self):
        text = "独立候选的完整计划提示词"
        self.config["components"]["prompts"]["overrides"]["instruction.plan"] = {
            "text": text, "version": "candidate-v3"}
        original = copy.deepcopy(self.config)
        first, second = self.session("one"), self.session("two")
        self.assertEqual(self.config, original)
        version = first.record_versions()["agent_components"]
        self.assertEqual(version["prompts"]["instruction.plan"]["text"], text)
        self.assertEqual(version["prompts"]["instruction.plan"]["version"], "candidate-v3")
        self.assertEqual(version["prompts"]["instruction.plan"]["sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertIn("src/embodied_agent/prompts/resources/instruction.plan.txt", first.record_versions()["source_hashes"])
        self.assertEqual(first.instruction_agent.context_provider, first.contexts["instruction"])
        self.assertEqual(first.environment_agent.context_provider, first.contexts["environment"])
        context = first.contexts["instruction"]
        context.remember("坐标查询", {"status": "FAILED", "error_code": "fixture"}, task_id="example")
        self.assertTrue(context.build("坐标查询")["memory"]["entries"])
        self.assertEqual(first.contexts["environment"].build("坐标查询")["memory"]["entries"], [])
        self.assertEqual(second.contexts["instruction"].build("坐标查询")["memory"]["entries"], [])
        self.config["components"]["prompts"]["overrides"]["instruction.plan"]["text"] = "caller mutated"
        self.assertEqual(first.prompt_catalog.get("instruction.plan").text, text)

    def test_session_records_real_failed_persisted_edit_and_separate_query_outcome(self):
        session = self.session("outcomes")
        session.select_map("classic")
        with patch.object(session, "reset", side_effect=RuntimeError("display unavailable")):
            edited = session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertEqual(edited["status"], "FAILED")
        self.assertTrue(edited["persisted"])
        entry = self.memory.read(session.contexts["environment"].namespace, edited["task_id"])
        self.assertEqual(entry["value"]["status"], "FAILED")
        self.assertEqual(entry["value"]["error_code"], "REFRESH_FAILED")
        self.assertTrue(entry["value"]["persisted"])
        events = [event for attempt in self.task(edited)["attempts"] for event in attempt["events"]]
        writes = [event for event in events if event["event"] == "memory_write"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["detail"]["entry"], entry)
        observed = session.run_agent("观察", "robot")
        self.assertEqual(observed["status"], "SUCCESS", observed)
        robot_entry = self.memory.read(session.contexts["instruction"].namespace, observed["task_id"])
        self.assertEqual(robot_entry["value"]["status"], "SUCCESS")
        self.assertEqual(robot_entry["value"]["verified"], observed.get("verified"))
        self.assertIsNone(self.memory.read(session.contexts["instruction"].namespace, edited["task_id"]))

    def test_task_versions_refresh_a_changed_file_candidate(self):
        prompt_file = self.directory / "candidate.txt"
        prompt_file.write_text("candidate at session creation", encoding="utf-8")
        self.config["components"]["prompts"]["overrides"]["instruction.plan"] = {
            "path": str(prompt_file), "version": "file-candidate-v1"}
        session = self.session("reload")
        session.select_map("classic")
        prompt_file.write_text("candidate at task execution", encoding="utf-8")
        result = session.run_agent("观察", "robot")
        self.assertEqual(result["status"], "SUCCESS", result)
        prompt = self.task(result)["versions"]["agent_components"]["prompts"]["instruction.plan"]
        self.assertEqual(prompt["text"], "candidate at task execution")
        self.assertEqual(prompt["sha256"], hashlib.sha256(prompt_file.read_bytes()).hexdigest())

    def test_memory_failure_preserves_committed_result_without_repeating_edit(self):
        class UnavailableMemory(MemoryStore):
            def write(self, *args, **kwargs):
                raise OSError("memory backend unavailable")

        session = self.session("failure", memory=UnavailableMemory())
        session.select_map("classic")
        with patch.object(session.store, "save", wraps=session.store.save) as save:
            result = session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertTrue(result["persisted"])
        save.assert_called_once()
        self.assertEqual(result["component_errors"][0]["component"], "memory_write")
        record = self.task(result)
        self.assertEqual(record["record_status"], "COMPLETED")
        self.assertEqual(record["outcome"]["status"], "SUCCESS")
        self.assertEqual(record["attempts"][0]["feedback"]["return_to_caller"]["component_errors"], result["component_errors"])


if __name__ == "__main__":
    unittest.main()

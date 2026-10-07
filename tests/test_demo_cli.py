"""Verify the original demo entry's visible default and explicit automation path."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.recording import TaskRecordStore


class DemoCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import native extensions before sys.modules overlays are restored.
        # Removing a first NumPy import would make the next test reload its DLL.
        import embodied_agent.apps.demo.session

    def test_batch_default_uses_original_visible_application_and_home_map(self):
        from embodied_agent.apps.demo.cli import main
        session = Mock()
        app = Mock()
        app.run.return_value = 0
        gui = types.ModuleType("embodied_agent.visualization.demo_ui")
        gui.DemoApp = Mock(return_value=app)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict(sys.modules, {gui.__name__: gui}), \
             patch("dotenv.load_dotenv"), \
             patch("embodied_agent.evaluation.cases.load_cases", return_value=[{"case_id": "room", "instruction": "观察。"}]), \
             patch("embodied_agent.apps.demo.session.create_batch_session", return_value=session) as create, \
             patch("embodied_agent.apps.demo.session.run_batch") as headless:
            code = main(["--mode", "batch", "--output", str(Path(temporary) / "run"),
                         "--records-dir", str(Path(temporary) / "records"),
                         "--planner", "stub", "--environment-planner", "rules"])
        self.assertEqual(code, 0)
        create.assert_called_once()
        headless.assert_not_called()
        gui.DemoApp.assert_called_once_with(session,
            [{"case_id": "room", "instruction": "观察。", "map_id": "home_living_room"}],
            initial_map="home_living_room", batch=True)
        app.run.assert_called_once()
        session.close.assert_called_once()

    def test_explicit_headless_never_imports_the_original_gui(self):
        from embodied_agent.apps.demo.cli import main
        cases = [{"case_id": "room", "map_id": "home_living_room", "instruction": "观察。"}]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.dict(sys.modules, {"embodied_agent.visualization.demo_ui": None}), \
             patch("dotenv.load_dotenv"), \
             patch("embodied_agent.evaluation.cases.load_cases", return_value=cases), \
             patch("embodied_agent.apps.demo.session.create_batch_session") as visible, \
             patch("embodied_agent.apps.demo.session.run_batch", return_value={"all_pass": True, "results": []}) as headless, \
             contextlib.redirect_stdout(io.StringIO()):
            code = main(["--mode", "batch", "--headless", "--output", str(Path(temporary) / "run"),
                         "--records-dir", str(Path(temporary) / "records"),
                         "--planner", "stub", "--environment-planner", "rules"])
        self.assertEqual(code, 0)
        visible.assert_not_called()
        headless.assert_called_once()
        self.assertEqual(headless.call_args.args[0], cases)

    def test_free_mode_requires_the_original_visible_window(self):
        from embodied_agent.apps.demo.cli import main
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--headless"])
        self.assertEqual(caught.exception.code, 2)

    def test_headless_short_case_persists_task_reference_in_cli_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            records = Path(temporary) / "records"
            process = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "demo.py"), "--mode", "batch", "--headless",
                 "--planner", "stub", "--environment-planner", "rules", "--case", "home_room_only",
                 "--output", str(output), "--records-dir", str(records)],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["all_pass"])
            action = summary["results"][0]["actions"][0]
            path = Path(action["task_record_path"])
            self.assertEqual(path.name, "task.json")
            self.assertEqual(path.parent.name, action["task_id"])
            self.assertEqual(path.parent.parent.parent, records / "tasks")
            store = TaskRecordStore(records)
            record = store.load(action["task_id"])
            self.assertTrue(store.verify(action["task_id"])["valid"])
            self.assertEqual(record["schema_version"], 2)
            self.assertNotIn("episode_events", action)
            self.assertNotIn("actions", action)
            self.assertEqual(record["task_id"], action["task_id"])
            self.assertEqual(record["outcome"]["status"], "SUCCESS")
            self.assertFalse(record["outcome"].get("transport_success", False))
            for name in ("actions.jsonl", "episodes.jsonl", "events.jsonl"):
                self.assertFalse((output / name).exists())


if __name__ == "__main__":
    unittest.main()

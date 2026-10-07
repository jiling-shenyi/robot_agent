"""Query public task JSON through the CLI without simulators or model clients."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.evaluation.task_query import main
from embodied_agent.recording.store import TaskRecordStore

START = dt.datetime(2026, 10, 5, 13, 0, tzinfo=dt.timezone.utc)


class TaskQueryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "records"
        self.verified = self.make_task("home_living_room", "SUCCESS", verified=True)
        self.failed = self.make_task("home_living_room", "FAILED", verified=False)
        self.unverified = self.make_task("classic", "SUCCESS", verified=False)
        self.earlier = self.make_task("home_living_room", "SUCCESS", verified=True,
                                      started=START - dt.timedelta(days=1))

    def make_task(self, map_id, status, *, verified, started=START):
        store = TaskRecordStore(self.root, clock=lambda: started)
        task = store.begin(map_id=map_id, map_definition={"name": map_id}, initial_state={},
                           natural_language="查询测试数据",
                           metadata={"agent_role": "instruction", "task_kind": "robot_task"})
        task.append_attempt(plan={"schema_version": 1, "actions": [{"skill": "observe"}]})
        task.finish(final_state={}, outcome={"status": status, "verified": verified})
        return task

    def query(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--records-dir", str(self.root), *arguments])
        return code, json.loads(output.getvalue())

    def test_combined_date_map_status_filters_use_shanghai_start_date(self):
        code, payload = self.query("--date", "2026-10-05", "--map", "home_living_room", "--status", "FAILED")
        self.assertEqual(code, 0)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["tasks"][0]["task_id"], self.failed.task_id)
        self.assertEqual(payload["errors"], [])
        self.assertEqual(self.query("--date", "20261005")[1]["count"], 3)

    def test_verified_only_excludes_unverified_success_and_failed_tasks(self):
        code, payload = self.query("--verified-only", "--date", "20261005")
        self.assertEqual(code, 0)
        self.assertEqual([row["task_id"] for row in payload["tasks"]], [self.verified.task_id])
        self.assertEqual(payload["tasks"][0]["attempt_count"], 1)
        self.assertEqual(self.query("--record-status", "RUNNING")[1]["count"], 0)

    def test_uuid_relative_path_directory_and_absolute_path_load_full_record(self):
        for identity in (self.verified.task_id, str(self.verified.path.relative_to(self.root)),
                         str(self.verified.path.parent), str(self.verified.path)):
            with self.subTest(identity=identity):
                code, payload = self.query("--task", identity)
                self.assertEqual(code, 0)
                self.assertEqual(payload, self.verified.data)

    def test_invalid_json_is_reported_without_discarding_valid_query_results(self):
        damaged = self.make_task("classic", "FAILED", verified=False)
        broken = damaged.path.parent / "manifest.json"
        broken.write_text("{ broken", encoding="utf-8")
        code, payload = self.query()
        self.assertEqual(code, 1)
        self.assertEqual(payload["count"], 4)
        self.assertEqual(len(payload["errors"]), 1)
        self.assertIn(str(damaged.path.parent), payload["errors"][0]["path"])
        self.assertIn("JSON", payload["errors"][0]["error"])

    def test_explicit_invalid_or_missing_task_reports_failure(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(["--records-dir", str(self.root), "--task", "0" * 32])
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("find task_id", errors.getvalue())

    def test_kind_role_and_complete_integrity_filters(self):
        code, payload = self.query("--task-kind", "robot_task", "--agent-role", "instruction", "--integrity", "complete")
        self.assertEqual(code, 0)
        self.assertEqual(payload["count"], 4)
        self.assertEqual(self.query("--task-kind", "map_edit")[1]["count"], 0)

    def test_verify_rebuild_index_and_rebuild_missing_projection(self):
        code, result = self.query("--task", self.verified.task_id, "--verify")
        self.assertEqual(code, 0)
        self.assertTrue(result["valid"])
        self.assertTrue(result["sealed"])
        self.verified.path.unlink()
        self.assertEqual(self.query("--task", self.verified.task_id, "--rebuild")[0], 0)
        self.assertTrue(self.verified.path.exists())
        self.assertEqual(self.query("--rebuild-index")[0], 0)

    def test_v1_path_is_explicitly_rejected(self):
        legacy = self.root / "20261005_cls_000001.json"
        legacy.write_text('{"schema_version":1}', encoding="utf-8")
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(["--records-dir", str(self.root), "--task", str(legacy)])
        self.assertEqual(code, 1)
        self.assertIn("v1", errors.getvalue())

    def test_missing_directory_is_not_created_by_the_query_command(self):
        missing = self.root / "does-not-exist"
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            main(["--records-dir", str(missing)])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("directory does not exist", errors.getvalue())
        self.assertFalse(missing.exists())

    def test_cli_trajectory_export_freezes_manifest_and_does_not_overwrite(self):
        output = Path(self.temporary.name) / "dataset"
        code, result = self.query("--export", "trajectory", "--output", str(output))
        self.assertEqual(code, 0)
        self.assertEqual(result["sample_count"], 4)
        self.assertTrue((output / "seal.json").is_file())
        stream, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
            repeated = main(["--records-dir", str(self.root), "--export", "trajectory", "--output", str(output)])
        self.assertEqual(repeated, 1)
        self.assertIn("already exists", errors.getvalue())

    def test_script_runs_with_site_packages_disabled_and_no_model_credentials(self):
        # -S excludes site packages, including MuJoCo, NumPy and model SDKs.
        # The read-only entry point must work with the standard library alone.
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "-S", str(ROOT / "scripts" / "task_records.py"),
             "--records-dir", str(self.root), "--task", self.verified.task_id],
            cwd=self.temporary.name,
            env={key: value for key, value in os.environ.items()
                 if key not in {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"}},
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(json.loads(process.stdout), self.verified.data)


if __name__ == "__main__":
    unittest.main()

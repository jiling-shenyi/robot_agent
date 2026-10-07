"""Behavioral regression tests for the shared demo session and case contract."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.apps.demo.session import DemoSession, create_batch_session, run_batch
from embodied_agent.evaluation.cases import check_expected, load_cases


class DemoSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "run"
        self.session = create_batch_session(output_dir=self.output, records_dir=Path(self.temp.name) / "records",
                                            planner_kind="stub", environment_mode="rules")

    def tearDown(self):
        self.session.close()
        self.temp.cleanup()

    def test_selected_map_reset_and_persisted_edit(self):
        self.assertIsNone(self.session.episode)
        episode = self.session.select_map("alternate")
        original = self.session.world.to_dict()
        np.testing.assert_allclose(episode.data.xpos[episode.cube_body_id], original["cube_position_m"])
        episode.data.qpos[episode.cube_qpos_adr] += 0.05
        mujoco.mj_forward(episode.model, episode.data)
        reset = self.session.reset()
        np.testing.assert_allclose(reset.data.xpos[reset.cube_body_id], original["cube_position_m"])
        result = self.session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertEqual(self.session.world.revision, original["revision"] + 1)
        np.testing.assert_allclose(self.session.episode.data.xpos[self.session.episode.cube_body_id], [0.43, -0.27, 0.425])
        saved = self.session.store.load("alternate")
        self.assertEqual(saved.to_dict(), self.session.world.to_dict())

    def test_free_tasks_keep_live_state_and_use_frame_callbacks(self):
        frames = []
        self.session.select_map("classic")
        # Callbacks may be installed by the UI after session/scene construction.
        self.session.on_frame = lambda episode: frames.append((id(episode.data), episode.total_steps))
        episode = self.session.episode
        first = self.session.run_agent("把方块放到 A 区。", "robot")
        self.assertEqual(first["status"], "SUCCESS", first)
        position = episode.data.xpos[episode.cube_body_id].copy()
        second = self.session.run_agent("把方块放到 A 区。", "robot")
        self.assertEqual(second["status"], "SUCCESS", second)
        self.assertTrue(second["verified"])
        self.assertEqual(second["actions"], [])
        self.assertIs(episode, self.session.episode)
        np.testing.assert_allclose(episode.data.xpos[episode.cube_body_id], position, atol=1e-4)
        self.assertGreater(len(frames), 10)
        self.assertEqual({identity for identity, _ in frames}, {id(episode.data)})
        self.assertTrue(all(a[1] <= b[1] for a, b in zip(frames, frames[1:])))
        restored = self.session.reset()
        np.testing.assert_allclose(restored.data.xpos[restored.cube_body_id], self.session.world.cube_position_m)

    def test_unsafe_map_can_be_shown_but_execution_retains_guard(self):
        self.session.select_map("classic")
        result = self.session.edit_environment("将危险区中心移动到 (0.4, -0.29, 0.535)")
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertEqual(self.session.episode.total_steps, 0)
        action = self.session.run_agent("把方块放到 A 区。", "robot")
        self.assertEqual(action["status"], "FAILED", action)
        self.assertEqual(action["error_code"], "DANGER_ZONE_VIOLATION", action)
        self.assertIsNotNone(self.session.episode.danger_zone_violation)

    def test_case_reset_extensible_agent_and_negative_expectation(self):
        self.session.select_map("classic")
        episode = self.session.episode
        episode.data.qpos[episode.cube_qpos_adr] += 0.1
        mujoco.mj_forward(episode.model, episode.data)
        starts = []

        def future_agent(session, instruction):
            starts.append(session.episode.data.xpos[session.episode.cube_body_id].copy())
            return {"status": "FAILED", "error_code": "EXPECTED_REJECTION"}

        self.session.register_agent("m4", future_agent)
        result = self.session.run_case({"case_id": "m4_rejection", "map_id": "classic", "agent": "m4",
                                        "instruction": "reject", "expected": {"status": "FAILED", "error_code": "EXPECTED_REJECTION"}})
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "FAILED")
        self.assertIsNot(episode, self.session.episode)
        np.testing.assert_allclose(starts[0], self.session.world.cube_position_m)
        action = result["actions"][0]
        record = json.loads(Path(action["task_record_path"]).read_text(encoding="utf-8"))
        self.assertEqual(record["task_id"], action["task_id"])
        self.assertEqual(record["outcome"]["status"], "FAILED")
        self.assertIsNone(record["attempts"][0]["plan"])
        self.assertEqual(record["attempts"][0]["tool_events"], [])
        self.assertTrue(self.session.finish()["all_pass"])

    def test_rejected_instruction_does_not_mutate_map(self):
        self.session.select_map("classic")
        before = self.session.world.to_dict()
        result = self.session.edit_environment("删除所有文件")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(before, self.session.world.to_dict())
        self.assertEqual(before, self.session.store.load("classic").to_dict())

    def test_external_map_change_requires_reload_before_edit(self):
        self.session.select_map("classic")
        self.session.environment_agent.apply("classic", "将危险区扩大20%")
        stored = self.session.store.load("classic").to_dict()
        result = self.session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertEqual(result["error_code"], "REVISION_CONFLICT", result)
        self.assertEqual(stored, self.session.store.load("classic").to_dict())

    def test_environment_edit_after_case_override_uses_persisted_map(self):
        self.session.select_map("classic")
        original = self.session.world.to_dict()
        self.session.register_agent("inspect", lambda session, instruction: {"status": "SUCCESS"})
        case = {"case_id": "temporary_layout", "map_id": "classic",
                "initial_overrides": {"cube_position_m": [0.48, -0.29, 0.425]},
                "agent": "inspect", "instruction": "inspect"}
        self.assertTrue(self.session.run_case(case)["passed"])
        self.assertEqual(self.session.world.cube_position_m, (0.48, -0.29, 0.425))
        np.testing.assert_allclose(self.session.episode.data.xpos[self.session.episode.cube_body_id],
                                   [0.48, -0.29, 0.425])
        self.assertEqual(self.session.store.load("classic").to_dict(), original)

        result = self.session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertEqual(result["status"], "SUCCESS", result)
        self.assertTrue(result["auto_reset_from_case"])
        self.assertEqual(result["before"], original)
        self.assertEqual(result["after"]["cube_position_m"], [0.43, -0.27, 0.425])
        self.assertEqual(self.session.world.to_dict(), self.session.store.load("classic").to_dict())
        self.assertIsNone(self.session._case_override_base)
        np.testing.assert_allclose(self.session.episode.data.xpos[self.session.episode.cube_body_id],
                                   [0.43, -0.27, 0.425])

    def test_external_map_change_after_case_override_still_conflicts(self):
        self.session.register_agent("inspect", lambda session, instruction: {"status": "SUCCESS"})
        case = {"case_id": "temporary_layout", "map_id": "classic",
                "initial_overrides": {"cube_position_m": [0.48, -0.29, 0.425]},
                "agent": "inspect", "instruction": "inspect"}
        self.assertTrue(self.session.run_case(case)["passed"])
        self.session.environment_agent.apply("classic", "将危险区扩大20%")
        externally_saved = self.session.store.load("classic").to_dict()

        result = self.session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertEqual(result["error_code"], "REVISION_CONFLICT", result)
        self.assertEqual(self.session.store.load("classic").to_dict(), externally_saved)
        self.assertEqual(self.session.world.cube_position_m, (0.48, -0.29, 0.425))
        self.session.reset()
        self.assertIsNone(self.session._case_override_base)
        self.assertEqual(self.session.world.to_dict(), externally_saved)

    def test_environment_step_in_overridden_case_restores_base_before_following_step(self):
        self.session.select_map("classic")
        original = self.session.world.to_dict()
        observed = []

        def inspect(session, instruction):
            observed.append(session.world.to_dict())
            return {"status": "SUCCESS"}

        self.session.register_agent("inspect", inspect)
        case = {"case_id": "override_then_edit", "map_id": "classic",
                "initial_overrides": {"cube_position_m": [0.48, -0.29, 0.425]},
                "steps": [
                    {"agent": "environment", "instruction": "将危险区扩大20%"},
                    {"agent": "inspect", "instruction": "inspect"},
                ]}
        result = self.session.run_case(case)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["actions"][0]["auto_reset_from_case"])
        self.assertEqual(result["actions"][0]["before"], original)
        self.assertEqual(observed[0], self.session.store.load("classic").to_dict())
        self.assertEqual(observed[0]["cube_position_m"], original["cube_position_m"])
        self.assertGreater(observed[0]["danger_zone"]["half_size_m"][0],
                           original["danger_zone"]["half_size_m"][0])

    def test_saved_edit_evidence_survives_display_refresh_failure(self):
        self.session.select_map("classic")
        with patch.object(self.session, "reset", side_effect=RuntimeError("renderer stopped")):
            result = self.session.edit_environment("将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_code"], "REFRESH_FAILED")
        self.assertTrue(result["persisted"])
        self.assertEqual(result["after"]["cube_position_m"], [0.43, -0.27, 0.425])
        self.assertEqual(result["refresh_error"]["message"], "renderer stopped")
        self.assertEqual(result["after"], self.session.store.load("classic").to_dict())
        record = json.loads(Path(result["task_record_path"]).read_text(encoding="utf-8"))
        self.assertEqual(record["outcome"]["error_code"], "REFRESH_FAILED")
        self.assertTrue(record["attempts"][0]["feedback"]["return_to_caller"]["persisted"])

    def test_batch_case_baseline_is_independent_of_earlier_edits(self):
        self.session.select_map("classic")
        baseline = self.session.world.to_dict()
        case = next(case for case in load_cases() if case["case_id"] == "environment_cube_position")
        self.assertTrue(self.session.run_case(case)["passed"])

        def inspect_agent(session, instruction):
            self.assertEqual(session.world.to_dict(), baseline)
            return {"status": "SUCCESS"}

        self.session.register_agent("inspect", inspect_agent)
        result = self.session.run_case({"case_id": "baseline", "map_id": "classic", "agent": "inspect", "instruction": "check"})
        self.assertTrue(result["passed"], result)

    def test_viewer_close_is_aborted_and_retains_action_evidence(self):
        class ViewerClosed(RuntimeError):
            pass

        self.session.select_map("classic")
        self.session.register_agent("inspect", lambda session, instruction: {"status": "SUCCESS"})
        frames = 0

        def close_after_reset(episode):
            nonlocal frames
            frames += 1
            if frames >= 2:
                raise ViewerClosed("user closed")

        self.session.on_frame = close_after_reset
        result = self.session.run_case({"case_id": "close", "map_id": "classic", "agent": "inspect", "instruction": "check"})
        self.assertEqual(result["status"], "ABORTED", result)
        self.assertEqual(result["error_code"], "VIEWER_CLOSED")
        self.assertEqual(len(result["actions"]), 1)

    def test_scene_loading_from_external_cwd_preserves_cwd(self):
        previous = Path.cwd()
        try:
            os.chdir(self.temp.name)
            self.session.select_map("alternate")
            self.assertEqual(Path.cwd(), Path(self.temp.name))
        finally:
            os.chdir(previous)

    def test_existing_output_is_never_overwritten(self):
        occupied = Path(self.temp.name) / "occupied"
        occupied.mkdir()
        evidence = occupied / "manifest.json"
        evidence.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            DemoSession(output_dir=occupied, records_dir=Path(self.temp.name) / "occupied-records",
                        planner_kind="stub", environment_mode="rules")
        self.assertEqual(evidence.read_text(encoding="utf-8"), "original")


class DemoCaseTests(unittest.TestCase):
    def test_default_and_legacy_catalog_preserve_twenty_frozen_scenes(self):
        cases = load_cases()
        self.assertEqual(len(cases), 34)
        legacy = load_cases(ROOT / "configs" / "m3_cases.json")
        self.assertEqual(len(legacy), 20)
        self.assertEqual(cases[:20], legacy)
        self.assertEqual(len({case["case_id"] for case in cases}), 34)
        self.assertTrue(all("cube_position_m" in case["initial_overrides"] for case in legacy))

    def test_invalid_catalogs_fail_fast(self):
        base = {"case_id": "test", "instruction": "move", "expected": {"status": "SUCCESS"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            for cases in ([base, base], [dict(base, agent="typo")], [dict(base, expected={"typo": True})],
                          [dict(base, expected={"status": "PASSED"})], [dict(base, steps=[])], []):
                with self.subTest(cases=cases):
                    path.write_text(json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_cases(path)

    def test_expectation_failure_is_distinct_from_execution_success(self):
        passed, failures = check_expected({"status": "SUCCESS", "target_id": "target_a"},
                                           {"status": "SUCCESS", "target_id": "target_b"})
        self.assertFalse(passed)
        self.assertIn("target_id", failures[0])
        self.assertFalse(check_expected({"status": "FAILED", "target_id": "target_a"},
                                        {"target_id": "target_a"})[0])

    def test_catalog_schema_and_include_flag_reject_boolean_coercion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            for payload in ({"schema_version": True, "cases": []},
                            {"schema_version": 1, "include_m3_cases": "false", "cases": []}):
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_batch_edits_are_isolated_with_revision_evidence(self):
        source = {p.name: p.read_bytes() for p in (ROOT / "configs" / "maps").glob("*.json")}
        case = next(case for case in load_cases() if case["case_id"] == "environment_cube_position")
        with tempfile.TemporaryDirectory() as temp:
            summary = run_batch([case], output_dir=Path(temp) / "run", records_dir=Path(temp) / "records",
                                planner_kind="stub", environment_mode="rules")
            self.assertTrue(summary["all_pass"], summary)
            self.assertTrue(summary["batch_isolated"])
            self.assertGreater(summary["results"][0]["map_revision"], summary["results"][0]["initial_map"]["map_revision"])
        self.assertEqual(source, {p.name: p.read_bytes() for p in (ROOT / "configs" / "maps").glob("*.json")})

    def test_interrupted_batch_records_remaining_cases(self):
        cases = [{"case_id": str(index), "instruction": "move"} for index in range(3)]
        with tempfile.TemporaryDirectory() as temp, patch.object(DemoSession, "run_case", side_effect=KeyboardInterrupt):
            summary = run_batch(cases, output_dir=Path(temp) / "run", records_dir=Path(temp) / "records",
                                planner_kind="stub", environment_mode="rules")
            self.assertFalse(summary["all_pass"])
            self.assertEqual([row["status"] for row in summary["results"]], ["ABORTED", "NOT_RUN", "NOT_RUN"])
            self.assertEqual(summary["not_run_count"], 2)

    def test_batch_output_cannot_recurse_into_source_maps(self):
        with tempfile.TemporaryDirectory() as temp:
            maps = Path(temp) / "maps"
            maps.mkdir()
            with self.assertRaises(ValueError):
                create_batch_session(map_dir=maps, output_dir=maps / "output", records_dir=Path(temp) / "records")


if __name__ == "__main__":
    unittest.main()

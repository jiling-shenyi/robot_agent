"""Real desktop integration checks; opt in with ROBOT_AGENT_GUI_TESTS=1."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(os.getenv("ROBOT_AGENT_GUI_TESTS") == "1", "requires an interactive Tk/OpenGL desktop")
class DemoGuiTests(unittest.TestCase):
    def setUp(self):
        from embodied_agent.apps.demo.session import DemoSession
        from embodied_agent.evaluation.cases import load_cases
        from embodied_agent.visualization.demo_ui import DemoApp

        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        shutil.copytree(ROOT / "configs" / "maps", base / "maps")
        self.session = DemoSession(map_dir=base / "maps", output_dir=base / "run",
                                   records_dir=base / "records",
                                   planner_kind="stub", environment_mode="rules")
        self.app = DemoApp(self.session, load_cases(), initial_map="classic")
        self.app.window.update()

    def tearDown(self):
        if not self.app.closed:
            self.app.request_close()
        self.temporary.cleanup()

    def test_map_selection_edit_reset_and_visible_instruction_execution(self):
        self.assertIsNone(self.session.episode)
        self.assertIsNone(self.app.view._renderer)
        self.app._execute(self.app._select_map)
        self.assertIsNotNone(self.app.view._renderer)
        self.assertIs(self.app.view._model, self.session.episode.model)
        self.assertIsNone(self.app.view.last_error)
        self.app._execute(self.app._edit_environment)
        self.assertEqual(self.session.results[-1]["status"], "SUCCESS")
        self.assertAlmostEqual(self.session.store.load("classic").cube_position_m[1], -0.26)
        self.app._execute(self.app._reset)
        np.testing.assert_allclose(self.session.episode.data.xpos[self.session.episode.cube_body_id], [0.42, -0.26, 0.425])
        self.app.case_choice.current(0)
        self.app._execute(self.app._run_selected_case)
        result = self.session.results[-1]
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIs(self.app.view._model, self.session.episode.model)
        self.assertGreater(self.session.episode.total_steps, 1000)
        self.assertIsNone(self.app.view.last_error)
        # Evidence is the real rendered RGB frame, saved by Tk without conversion dependencies.
        evidence = ROOT / "results" / "tmp" / "demo_gui_final.png"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        self.app.view._photo.write(str(evidence), format="png")
        # The same visible scene also exercises the new shared Panda path,
        # continuously from the previous task's actual final state.
        self.app.robot_input.delete("1.0", "end")
        self.app.robot_input.insert("1.0", "把方块放到 B 区。")
        self.app._execute(self.app._run_robot)
        shared = self.session.results[-1]
        self.assertEqual(shared["status"], "SUCCESS", shared)
        self.assertTrue(shared["verified"])
        self.assertEqual(shared["instruction_agent"], "universal-instruction-v1")
        self.assertIs(self.app.view._model, self.session.episode.model)
        self.app.view._photo.write(str(ROOT / "results" / "tmp" / "demo_unified_panda_final.png"), format="png")

    def test_close_interrupts_visual_batch_and_accounts_for_pending_cases(self):
        self.app._execute(self.app._select_map)
        self.app.cases = self.app.cases[:2]
        self.app.batch = True
        self.app.window.after(180, self.app.request_close)
        self.app._execute(self.app._run_batch)
        self.assertTrue(self.app.closed)
        summary = self.session.finish()
        self.assertFalse(summary["all_pass"])
        self.assertEqual(summary["aborted_count"], 1)
        self.assertEqual(summary["not_run_count"], 1)

    def test_home_selection_shared_renderer_read_only_frame_and_desktop_switch(self):
        self.assertEqual(set(self.app.map_names.values()), {"classic", "alternate", "home_living_room"})
        label = next(label for label, map_id in self.app.map_names.items() if map_id == "home_living_room")
        self.app.map_choice.set(label)
        self.app._execute(self.app._select_map)
        episode = self.session.episode
        self.assertEqual(episode.robot_kind, "stretch")
        self.assertIs(self.app.view._model, episode.model)
        self.assertIs(self.app.view._episode, episode)
        self.assertEqual(self.app.view._camera_kind, "stretch")
        self.assertIn("Stretch", self.app.world_info.get())
        self.assertIn("机器人 Agent", self.app.agent_choice.cget("values"))
        self.assertTrue(self.app.case_names)
        self.assertTrue(all(case["map_id"] == "home_living_room" for case in self.app.case_names.values()))
        qpos, qvel, steps = episode.data.qpos.copy(), episode.data.qvel.copy(), episode.total_steps
        self.app.view.render(episode)
        np.testing.assert_array_equal(episode.data.qpos, qpos)
        np.testing.assert_array_equal(episode.data.qvel, qvel)
        self.assertEqual(episode.total_steps, steps)
        self.app.robot_input.delete("1.0", "end")
        self.app.robot_input.insert("1.0", "观察，然后查询杯子。")
        self.app._execute(self.app._run_robot)
        self.assertEqual(self.session.results[-1]["status"], "SUCCESS")
        self.assertEqual(self.session.results[-1]["agent"], "robot")
        self.assertIs(self.app.view._model, self.session.episode.model)
        self.assertFalse(self.session.results[-1]["transport_success"])
        classic = next(label for label, map_id in self.app.map_names.items() if map_id == "classic")
        self.app.map_choice.set(classic)
        self.app._execute(self.app._select_map)
        self.assertEqual(self.app.view._camera_kind, "panda")
        self.assertIs(self.app.view._model, self.session.episode.model)
        self.assertLess(self.app.view.camera.distance, 3)
        self.assertIsNone(self.app.view.last_error)

    def test_home_close_aborts_the_same_original_window_action(self):
        label = next(label for label, map_id in self.app.map_names.items() if map_id == "home_living_room")
        self.app.map_choice.set(label)
        self.app._execute(self.app._select_map)
        self.app.robot_input.delete("1.0", "end")
        self.app.robot_input.insert("1.0", "导航到茶几。")
        # Close after real execution begins, independently of planning speed.
        original_frame = self.app.on_frame
        def close_on_movement(episode, **kwargs):
            original_frame(episode, **kwargs)
            if episode.total_steps >= 80 and episode.active_skill == "navigate":
                self.app.request_close()
        self.session.on_frame = close_on_movement
        self.app._execute(self.app._run_robot)
        self.assertTrue(self.app.closed)
        self.assertEqual(self.session.results[-1]["status"], "ABORTED")
        self.assertEqual(self.session.results[-1]["error_code"], "VIEWER_CLOSED")
        self.assertFalse(self.session.results[-1]["transport_success"])
        self.assertGreater(self.session.results[-1]["physical_evidence"]["trajectory_samples"], 0)


if __name__ == "__main__":
    unittest.main()

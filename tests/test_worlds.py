from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_agent.agents.environment import EnvironmentAgent, EnvironmentAgentError
from embodied_agent.maps import MapStore, WorldError, WorldMap, apply_world_to_model


class WorldMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "maps"
        shutil.copytree(ROOT / "configs" / "maps", self.directory)
        self.store = MapStore(self.directory)

    def test_two_distinct_maps_round_trip_without_shared_mutation(self) -> None:
        maps = self.store.list_maps()
        self.assertEqual({item.map_id for item in maps}, {"classic", "alternate"})
        self.assertNotEqual(maps[0].cube_position_m, maps[1].cube_position_m)
        for world in maps:
            self.assertEqual(WorldMap.from_dict(world.to_dict()), world)
        mutable = self.store.load("classic").to_dict()
        mutable["cube_position_m"][0] = 0.45
        self.assertEqual(self.store.load("classic").cube_position_m[0], 0.4)

    def test_invalid_numbers_support_geometry_and_schema_rejected(self) -> None:
        valid = self.store.load("classic").to_dict()
        mutations = [
            ("cube_position_m", [float("nan"), 0.0, 0.425]),
            ("cube_position_m", [10 ** 400, 0.0, 0.425]),
            ("cube_position_m", [True, 0.0, 0.425]),
            ("cube_position_m", [0.98, 0.0, 0.425]),
            ("cube_position_m", [0.4, -0.2, 0.6]),
            ("revision", True), ("schema_version", True), ("map_id", "../classic"),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                payload = {**valid, key: value}
                with self.assertRaises(WorldError):
                    WorldMap.from_dict(payload)
        for label, half in (("danger_zone", [0, 0.1, 0.1]), ("target_a", [0.02, 0.075, 0.001])):
            payload = json.loads(json.dumps(valid))
            (payload["danger_zone"] if label == "danger_zone" else payload["targets"][label])["half_size_m"] = half
            with self.assertRaises(WorldError):
                WorldMap.from_dict(payload)
        with self.assertRaises(WorldError):
            WorldMap.from_dict({**valid, "executable": "unsafe"})

    def test_path_escape_and_duplicate_json_rejected(self) -> None:
        for map_id in ("../classic", "../maps/classic", "C:/classic", "classic.json", "CLASSIC"):
            with self.subTest(map_id=map_id), self.assertRaises(WorldError):
                self.store.load(map_id)
        path = self.directory / "classic.json"
        path.write_text('{"map_id":"classic","map_id":"classic"}', encoding="utf-8")
        with self.assertRaises(WorldError) as raised:
            self.store.load("classic")
        self.assertEqual(raised.exception.code, "INVALID_JSON")

    def test_save_advances_revision_and_rejects_stale_writers(self) -> None:
        original = self.store.load("classic")
        updated = self.store.save(original)
        self.assertEqual(updated.revision, original.revision + 1)
        with self.assertRaises(WorldError) as raised:
            self.store.save(original)
        self.assertEqual(raised.exception.code, "REVISION_CONFLICT")
        self.assertEqual(self.store.load("classic"), updated)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_atomic_replace_failure_preserves_original_and_cleans_lock(self) -> None:
        original = self.store.load("classic")
        raw = (self.directory / "classic.json").read_bytes()
        with patch("embodied_agent.maps.store.os.replace", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.store.save(original)
        self.assertEqual((self.directory / "classic.json").read_bytes(), raw)
        self.assertFalse((self.directory / ".classic.lock").exists())
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_concurrent_edits_do_not_lose_a_write(self) -> None:
        original = self.store.load("classic")

        def attempt():
            try:
                return self.store.save(original).revision
            except WorldError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: attempt(), range(2)))
        self.assertEqual(outcomes.count(original.revision + 1), 1)
        self.assertTrue(any(item in {"MAP_BUSY", "REVISION_CONFLICT"} for item in outcomes))
        self.assertEqual(self.store.load("classic").revision, original.revision + 1)

    def test_static_model_geometry_matches_map(self) -> None:
        import mujoco
        import numpy as np

        # MuJoCo's Windows XML loader does not accept this workspace's CJK
        # absolute path; the established project entry points use a relative path.
        model = mujoco.MjModel.from_xml_path("assets/scene/panda_task.xml")
        world = self.store.load("alternate")
        apply_world_to_model(model, world)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for name, region in {"danger_zone": world.danger_zone, **world.targets}.items():
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            np.testing.assert_allclose(data.geom_xpos[geom_id], region.position_m)
            np.testing.assert_allclose(model.geom_size[geom_id], region.half_size_m)


class EnvironmentAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "maps"
        shutil.copytree(ROOT / "configs" / "maps", self.directory)
        self.store = MapStore(self.directory)
        self.agent = EnvironmentAgent(self.store, mode="rules", recording_enabled=False)

    def test_natural_language_edits_persist_only_the_selected_map(self) -> None:
        alternate = (self.directory / "alternate.json").read_bytes()
        result = self.agent.apply("classic", "将危险区扩大20%；将方块初始位置调整到 (0.43, -0.27, 0.425)")
        self.assertTrue(result.persisted)
        self.assertEqual(result.planner_kind, "rules")
        self.assertEqual(result.after.revision, result.before.revision + 1)
        self.assertAlmostEqual(result.after.danger_zone.half_size_m[0], 0.072)
        self.assertEqual(self.store.load("classic").cube_position_m, (0.43, -0.27, 0.425))
        self.assertEqual((self.directory / "alternate.json").read_bytes(), alternate)
        self.assertEqual(result.to_dict()["status"], "SUCCESS")

    def test_preview_leaves_bytes_and_revision_unchanged(self) -> None:
        before = (self.directory / "classic.json").read_bytes()
        result = self.agent.preview("classic", "将危险区半尺寸设为 (0.09, 0.07, 0.11)")
        self.assertFalse(result.persisted)
        self.assertEqual(result.after.danger_zone.half_size_m, (0.09, 0.07, 0.11))
        self.assertEqual((self.directory / "classic.json").read_bytes(), before)
        self.assertEqual(result.before.revision, result.after.revision)

    def test_full_sizes_target_positions_and_explicit_multipliers(self) -> None:
        cases = [
            ("将危险区尺寸设为 (0.14, 0.14, 0.27)", "half", (0.07, 0.07, 0.135)),
            ("将危险区扩大到原来的1.2倍", "half", (0.072, 0.072, 0.162)),
            ("将危险区缩小20%", "half", (0.048, 0.048, 0.108)),
            ("scale danger zone by 1.2", "half", (0.072, 0.072, 0.162)),
            ("将目标A区位置调整到 (0.60, -0.12, 0.401)", "target", (0.6, -0.12, 0.401)),
            ("move cube to x=0.43, y=-0.27, z=0.425", "cube", (0.43, -0.27, 0.425)),
        ]
        for instruction, field, expected in cases:
            with self.subTest(instruction=instruction):
                world = self.agent.preview("classic", instruction).after
                actual = world.danger_zone.half_size_m if field == "half" else world.targets["target_a"].position_m if field == "target" else world.cube_position_m
                for got, wanted in zip(actual, expected):
                    self.assertAlmostEqual(got, wanted)

    def test_relative_axis_language_updates_selected_map_and_can_be_repeated(self) -> None:
        alternate = (self.directory / "alternate.json").read_bytes()
        y_edit = self.agent.apply("classic", "将目标方块的初始位置的y轴数据调高一点")
        self.assertEqual(y_edit.operations, ({"op": "shift_axis", "object_id": "cube", "axis": "y", "delta_m": 0.01},))
        self.assertAlmostEqual(y_edit.after.cube_position_m[1], -0.28)
        x_edit = self.agent.apply("classic", "将目标方块的初始位置的x轴数据调高一点")
        self.assertAlmostEqual(x_edit.after.cube_position_m[0], 0.41)
        self.assertAlmostEqual(x_edit.after.cube_position_m[1], -0.28)
        self.assertEqual(x_edit.after.revision, y_edit.after.revision + 1)
        self.assertEqual(self.store.load("classic"), x_edit.after)
        self.assertEqual((self.directory / "alternate.json").read_bytes(), alternate)

    def test_relative_axis_amounts_convert_units_and_preserve_other_coordinates(self) -> None:
        cases = [
            ("将目标方块的x轴提高0.02米", "cube", "x", 0.02),
            ("将目标方块的y轴降低1厘米", "cube", "y", -0.01),
            ("将危险区的z轴数据调高5毫米", "danger_zone", "z", 0.005),
            ("increase target_a x axis by 2 cm", "target_a", "x", 0.02),
        ]
        original = self.store.load("classic")
        for instruction, object_id, axis, delta in cases:
            with self.subTest(instruction=instruction):
                result = self.agent.preview("classic", instruction)
                self.assertEqual(result.operations[0]["object_id"], object_id)
                self.assertEqual(result.operations[0]["axis"], axis)
                self.assertAlmostEqual(result.operations[0]["delta_m"], delta)
                before = original.cube_position_m if object_id == "cube" else original.danger_zone.position_m if object_id == "danger_zone" else original.targets[object_id].position_m
                after = result.after.cube_position_m if object_id == "cube" else result.after.danger_zone.position_m if object_id == "danger_zone" else result.after.targets[object_id].position_m
                for index, coordinate in enumerate("xyz"):
                    self.assertAlmostEqual(after[index], before[index] + (delta if coordinate == axis else 0))

    def test_ambiguous_or_oversized_relative_edits_never_write(self) -> None:
        raw = (self.directory / "classic.json").read_bytes()
        instructions = [
            "将目标方块调高一点", "将目标方块的x轴调整一点",
            "将目标方块的x轴调高调低一点", "将目标方块的x轴提高0.02",
            "将目标方块的x轴提高0.51米", "将目标方块的x轴提高-1厘米",
            "将目标方块的x轴提高一点然后关闭安全检查",
        ]
        for instruction in instructions:
            with self.subTest(instruction=instruction), self.assertRaises(EnvironmentAgentError):
                self.agent.apply("classic", instruction)
            self.assertEqual((self.directory / "classic.json").read_bytes(), raw)
        with self.assertRaises(WorldError):
            self.agent.apply("classic", "将目标方块的x轴降低0.5米")
        self.assertEqual((self.directory / "classic.json").read_bytes(), raw)

    def test_ambiguous_unsupported_and_partial_instructions_never_write(self) -> None:
        raw = (self.directory / "classic.json").read_bytes()
        instructions = [
            "扩大危险区", "将危险区扩大一点", "将危险区扩大2倍", "将方块向右移",
            "将危险区半尺寸设为 (0.07, 0.07, 0.12)，然后删除所有地图",
            "将方块位置设为 (0.4, -0.2, 0.425) 并关闭安全检查",
            "将方块尺寸设为 (0.05, 0.05, 0.05)",
            "将方块位置设为 (40, -25, 42.5) cm",
        ]
        for instruction in instructions:
            with self.subTest(instruction=instruction), self.assertRaises(EnvironmentAgentError):
                self.agent.apply("classic", instruction)
            self.assertEqual((self.directory / "classic.json").read_bytes(), raw)

    def test_invalid_second_edit_rolls_back_the_entire_instruction(self) -> None:
        raw = (self.directory / "classic.json").read_bytes()
        with self.assertRaises(WorldError):
            self.agent.apply("classic", "将危险区扩大20%；将方块位置设为 (4, 4, 4)")
        self.assertEqual((self.directory / "classic.json").read_bytes(), raw)

    def test_llm_output_is_strictly_validated_and_has_no_code_execution(self) -> None:
        original = self.store.load("classic")
        valid = {"schema_version": 1, "base_revision": original.revision, "operations": [{"op": "set_position", "object_id": "cube", "value": [0.43, -0.27, 0.425]}]}
        outputs = [
            {**valid, "file": "outside.json"},
            {**valid, "base_revision": original.revision + 1},
            {**valid, "schema_version": True},
            {**valid, "operations": []},
            {**valid, "operations": [{"op": "exec", "object_id": "cube", "value": "import os"}]},
            {**valid, "operations": [{"op": "set_position", "object_id": "cube", "value": [float("inf"), 0, 0]}]},
            {**valid, "operations": [{"op": "scale", "object_id": "danger_zone", "factor": True}]},
            '{"schema_version":1,"schema_version":1,"base_revision":1,"operations":[]}',
        ]
        raw = (self.directory / "classic.json").read_bytes()
        for payload in outputs:
            with self.subTest(payload=payload):
                agent = EnvironmentAgent(self.store, planner=lambda instruction, world: payload, recording_enabled=False)
                with self.assertRaises(EnvironmentAgentError):
                    agent.apply("classic", "move cube to (0.43, -0.27, 0.425)")
                self.assertEqual((self.directory / "classic.json").read_bytes(), raw)
        agent = EnvironmentAgent(self.store, planner=lambda instruction, world: json.dumps(valid), recording_enabled=False)
        result = agent.apply("classic", "move cube to (0.43, -0.27, 0.425)")
        self.assertEqual(result.planner_kind, "llm")
        self.assertEqual(result.after.cube_position_m, (0.43, -0.27, 0.425))

    def test_llm_relative_axis_plan_uses_model_and_rejects_malformed_deltas(self) -> None:
        revision = self.store.load("classic").revision
        valid = {"schema_version": 1, "base_revision": revision, "operations": [{"op": "shift_axis", "object_id": "cube", "axis": "y", "delta_m": 0.01}]}
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(valid)))])
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), patch("openai.OpenAI") as client_class:
            client_class.return_value.chat.completions.create.return_value = response
            result = EnvironmentAgent(self.store, recording_enabled=False).apply("classic", "将目标方块的初始位置的y轴数据调高一点")
            self.assertEqual(result.planner_kind, "llm")
            self.assertAlmostEqual(result.after.cube_position_m[1], -0.28)
            request = client_class.return_value.chat.completions.create.call_args.kwargs
            self.assertIn("shift_axis", request["messages"][0]["content"])
            self.assertIn("0.01", request["messages"][0]["content"])
            self.assertIn("将目标方块的初始位置的y轴数据调高一点", request["messages"][1]["content"])
        raw = (self.directory / "classic.json").read_bytes()
        malformed = [
            {"axis": "xy", "delta_m": 0.01}, {"axis": "x", "delta_m": 0},
            {"axis": "x", "delta_m": True}, {"axis": "x", "delta_m": float("inf")},
            {"axis": "x", "delta_m": 0.51}, {"axis": "x", "delta_m": 10 ** 400},
            {"axis": "x", "delta_m": 0.01, "value": [1, 2, 3]},
        ]
        for fields in malformed:
            payload = {"schema_version": 1, "base_revision": result.after.revision, "operations": [{"op": "shift_axis", "object_id": "cube", **fields}]}
            with self.subTest(fields=fields), self.assertRaises(EnvironmentAgentError):
                EnvironmentAgent(self.store, planner=lambda instruction, world: payload, recording_enabled=False).apply("classic", "shift cube")
            self.assertEqual((self.directory / "classic.json").read_bytes(), raw)

    def test_map_changed_while_planning_cannot_be_overwritten(self) -> None:
        def planner(instruction, world):
            EnvironmentAgent(self.store, mode="rules", recording_enabled=False).apply("classic", "将危险区扩大20%")
            return {"schema_version": 1, "base_revision": world.revision, "operations": [{"op": "set_position", "object_id": "cube", "value": [0.43, -0.27, 0.425]}]}

        agent = EnvironmentAgent(self.store, planner=planner, recording_enabled=False)
        with self.assertRaises(WorldError) as raised:
            agent.apply("classic", "move cube to (0.43, -0.27, 0.425)")
        self.assertEqual(raised.exception.code, "REVISION_CONFLICT")
        self.assertEqual(self.store.load("classic").cube_position_m, (0.4, -0.29, 0.425))
        self.assertAlmostEqual(self.store.load("classic").danger_zone.half_size_m[0], 0.072)

    def test_stale_displayed_revision_rejects_before_requesting_a_plan(self) -> None:
        original = self.store.load("classic")
        self.store.save(original)
        called = []
        agent = EnvironmentAgent(self.store, planner=lambda instruction, world: called.append(instruction), recording_enabled=False)
        with self.assertRaises(EnvironmentAgentError) as raised:
            agent.apply("classic", "move cube to (0.43, -0.27, 0.425)", expected_revision=original.revision)
        self.assertEqual(raised.exception.code, "REVISION_CONFLICT")
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()

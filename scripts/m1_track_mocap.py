"""Run the M1 Panda mocap-following, gripper, and scene smoke checks."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import mujoco
import mujoco_menagerie as menagerie
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "scene" / "panda_task.xml"
MODEL_SOURCE = ROOT / "assets" / "third_party" / "franka_emika_panda" / "panda.xml"
MODEL_DERIVATIVE = ROOT / "assets" / "third_party" / "franka_emika_panda" / "panda_mocap.xml"
ERROR_LIMIT_M = 0.02
PENETRATION_LIMIT_M = 0.005
SAMPLE_EVERY_STEPS = 10
DWELL_STEPS = 150
MOTION_STEPS = 300


def _name_id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, obj, name)
    if result < 0:
        raise RuntimeError(f"Required MuJoCo name not found: {name}")
    return int(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _site_quat(data: mujoco.MjData, site_id: int) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return quat


def _warning_counts(data: mujoco.MjData) -> dict[str, int]:
    warning_types = {
        int(getattr(mujoco.mjtWarning, name)): name
        for name in dir(mujoco.mjtWarning)
        if name.startswith("mjWARN_")
    }
    return {
        warning_types[index]: int(count)
        for index, count in enumerate(data.warning.number)
        if count and index in warning_types
    }


def _write_png(path: Path, pixels: np.ndarray) -> None:
    """Write an RGB uint8 image using only the standard library."""
    image = np.asarray(pixels, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _contact_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pair_min_distance: dict[str, float],
) -> None:
    for index in range(data.ncon):
        contact = data.contact[index]
        def describe_geom(geom_id: int) -> str:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom{geom_id}"
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
            return f"{geom_name}@{body_name}"

        first = describe_geom(int(contact.geom1))
        second = describe_geom(int(contact.geom2))
        pair = " <-> ".join(sorted((first, second)))
        pair_min_distance[pair] = min(pair_min_distance.get(pair, math.inf), float(contact.dist))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "m1",
        help="directory for CSV, JSON summary, and screenshot (default: results/m1)",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    screenshots = output / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)

    # Use an ASCII relative path for MuJoCo's Windows XML loader, which accepts
    # Unicode paths poorly on some Python builds.
    os.chdir(ROOT)
    scene_relative_path = str(SCENE.relative_to(ROOT)).replace("\\", "/")
    model = mujoco.MjModel.from_xml_path(scene_relative_path)
    data = mujoco.MjData(model)

    key_id = _name_id(model, mujoco.mjtObj.mjOBJ_KEY, "home_scene")
    ee_site_id = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    mocap_site_id = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "mocap_site")
    mocap_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")
    cube_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    table_geom_id = _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    target_geom_ids = [
        _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_a"),
        _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_b"),
    ]
    gripper_actuator_id = _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    arm_actuator_ids = [
        _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)
    ]
    arm_joint_ids = [
        _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)
    ]
    finger_joint_ids = [
        _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
        _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
    ]
    cube_joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    mocap_id = int(model.body_mocapid[mocap_body_id])
    if mocap_id < 0:
        raise RuntimeError("mocap_target is not a MuJoCo mocap body")
    hand_body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if int(model.site_bodyid[ee_site_id]) != hand_body_id:
        raise RuntimeError("ee_site must be attached to the Panda hand body")
    if int(model.site_bodyid[mocap_site_id]) != mocap_body_id:
        raise RuntimeError("mocap_site must be attached to mocap_target")
    if not np.all(model.actuator_group[arm_actuator_ids] == 0):
        raise RuntimeError("Panda arm actuators must be in disabled group 0")
    if int(model.actuator_group[gripper_actuator_id]) != 1 or int(model.opt.disableactuator) != 1:
        raise RuntimeError("Only actuator group 0 should be disabled; gripper group 1 must remain enabled")
    if not np.allclose(model.actuator_ctrlrange[gripper_actuator_id], [0.0, 255.0]):
        raise RuntimeError("Unexpected gripper actuator control range")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    home_pos = data.site_xpos[ee_site_id].copy()
    home_quat = _site_quat(data, ee_site_id)
    data.mocap_pos[mocap_id] = home_pos
    data.mocap_quat[mocap_id] = home_quat
    mujoco.mj_forward(model, data)

    arm_qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in arm_joint_ids]
    finger_qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in finger_joint_ids]
    home_arm_qpos = data.qpos[arm_qpos_addresses].copy()
    table_top_z = float(data.geom_xpos[table_geom_id][2] + model.geom_size[table_geom_id][2])
    cube_half_height = float(model.geom_size[_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")][2])

    samples: list[dict[str, Any]] = []
    pair_min_distance: dict[str, float] = {}
    motion_errors: dict[str, list[float]] = {}
    gripper_end_states: dict[str, list[float]] = {"closed": [], "open": []}
    gripper_site_positions: list[np.ndarray] = []
    bad_state_seen = False
    step_count = 0

    def sample(trial_id: str, phase: str, target: np.ndarray) -> float:
        nonlocal bad_state_seen
        actual = data.site_xpos[ee_site_id].copy()
        error = float(np.linalg.norm(actual - target))
        arm_qpos = data.qpos[arm_qpos_addresses].copy()
        ee_quat = _site_quat(data, ee_site_id)
        cube_xyz = data.xpos[cube_body_id].copy()
        cube_linear_speed = float(np.linalg.norm(data.cvel[cube_body_id][3:]))
        finite = bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(actual).all()
            and np.isfinite(cube_xyz).all()
        )
        bad_state_seen |= not finite
        samples.append(
            {
                "trial_id": trial_id,
                "phase": phase,
                "target_xyz": json.dumps(target.tolist(), separators=(",", ":")),
                "actual_ee_xyz": json.dumps(actual.tolist(), separators=(",", ":")),
                "position_error": f"{error:.9f}",
                "ee_quaternion": json.dumps(ee_quat.tolist(), separators=(",", ":")),
                "arm_qpos": json.dumps(arm_qpos.tolist(), separators=(",", ":")),
                "gripper_ctrl": f"{float(data.ctrl[gripper_actuator_id]):.6f}",
                "finger_qpos": json.dumps(data.qpos[finger_qpos_addresses].tolist(), separators=(",", ":")),
                "cube_xyz": json.dumps(cube_xyz.tolist(), separators=(",", ":")),
                "cube_speed": f"{cube_linear_speed:.9f}",
                "sim_time": f"{float(data.time):.6f}",
                "warning_or_nan": json.dumps({"warnings": _warning_counts(data), "non_finite": not finite}, separators=(",", ":")),
            }
        )
        _contact_snapshot(model, data, pair_min_distance)
        return error

    def step_recorded(trial_id: str, phase: str, target: np.ndarray) -> float:
        nonlocal step_count
        mujoco.mj_step(model, data)
        step_count += 1
        if step_count % SAMPLE_EVERY_STEPS == 0:
            return sample(trial_id, phase, target)
        _contact_snapshot(model, data, pair_min_distance)
        return float(np.linalg.norm(data.site_xpos[ee_site_id] - target))

    def move_to(trial_id: str, goal: np.ndarray) -> list[float]:
        start = data.mocap_pos[mocap_id].copy()
        errors: list[float] = []
        for step in range(1, MOTION_STEPS + 1):
            fraction = step / MOTION_STEPS
            target = start * (1.0 - fraction) + goal * fraction
            data.mocap_pos[mocap_id] = target
            data.mocap_quat[mocap_id] = home_quat
            errors.append(step_recorded(trial_id, "move", target))
        for _ in range(DWELL_STEPS):
            data.mocap_pos[mocap_id] = goal
            data.mocap_quat[mocap_id] = home_quat
            errors.append(step_recorded(trial_id, "hold", goal))
        return errors[-max(1, DWELL_STEPS // SAMPLE_EVERY_STEPS) :]

    # Let the free cube settle before moving the arm; the target zones are visual-only geoms.
    idle_target = home_pos.copy()
    for _ in range(500):
        step_recorded("scene_settle", "idle", idle_target)

    goal_offsets = [
        np.array([0.025, 0.0, 0.0]),
        np.array([0.025, 0.025, 0.0]),
        np.array([0.0, 0.025, 0.0]),
    ]
    goal_positions = [home_pos + offset for offset in goal_offsets]
    for index, goal in enumerate(goal_positions, start=1):
        trial_id = f"tracking_goal_{index}"
        dwell_errors = move_to(trial_id, goal)
        motion_errors[trial_id] = dwell_errors
    return_errors = move_to("return_home", home_pos.copy())
    motion_errors["return_home"] = return_errors

    # Repeated free-space opening/closing checks: actuator8 controls the gripper only.
    data.mocap_pos[mocap_id] = home_pos
    data.mocap_quat[mocap_id] = home_quat
    gripper_site_positions.append(data.site_xpos[ee_site_id].copy())
    for cycle in range(1, 4):
        for state, command in (("closed", 0.0), ("open", 255.0)):
            data.ctrl[gripper_actuator_id] = command
            for step in range(175):
                step_recorded(f"gripper_cycle_{cycle}_{state}", state, home_pos)
            finger_values = data.qpos[finger_qpos_addresses].copy()
            gripper_end_states[state].extend(finger_values.tolist())
            gripper_site_positions.append(data.site_xpos[ee_site_id].copy())

    # Check that the cube remains supported and comes to rest in an otherwise idle scene.
    for _ in range(1000):
        step_recorded("scene_idle", "idle", home_pos)

    cube_position = data.xpos[cube_body_id].copy()
    cube_speed = float(np.linalg.norm(data.cvel[cube_body_id][3:]))
    cube_bottom_z = float(cube_position[2] - cube_half_height)
    target_collision_disabled = all(
        int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0
        for geom_id in target_geom_ids
    )
    target_max_errors = {name: float(max(errors)) for name, errors in motion_errors.items()}
    tracking_pass = all(
        max(motion_errors[f"tracking_goal_{index}"]) <= ERROR_LIMIT_M for index in range(1, 4)
    ) and max(return_errors) <= ERROR_LIMIT_M
    closed_avg = float(np.mean(gripper_end_states["closed"]))
    open_avg = float(np.mean(gripper_end_states["open"]))
    gripper_span = open_avg - closed_avg
    ee_motion_during_gripper = max(
        float(np.linalg.norm(position - home_pos)) for position in gripper_site_positions
    )
    gripper_pass = (
        closed_avg < 0.01
        and open_avg > 0.03
        and gripper_span > 0.025
        and ee_motion_during_gripper <= ERROR_LIMIT_M
    )
    scene_pass = (
        abs(cube_bottom_z - table_top_z) <= 0.003
        and cube_speed <= 0.01
        and target_collision_disabled
    )
    warnings = _warning_counts(data)
    min_contact_distance = min(pair_min_distance.values(), default=0.0)
    penetration_pass = min_contact_distance >= -PENETRATION_LIMIT_M
    tracking_pass = tracking_pass and not bad_state_seen and not warnings and penetration_pass

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "tracking.csv"
    fields = [
        "trial_id",
        "phase",
        "target_xyz",
        "actual_ee_xyz",
        "position_error",
        "ee_quaternion",
        "arm_qpos",
        "gripper_ctrl",
        "finger_qpos",
        "cube_xyz",
        "cube_speed",
        "sim_time",
        "warning_or_nan",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)

    # Render a reproducible offscreen view of the full task scene.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = np.array([0.48, 0.0, 0.39])
    camera.distance = 1.6
    camera.azimuth = 135.0
    camera.elevation = -24.0
    screenshot_path = screenshots / "panda_task.png"
    with mujoco.Renderer(model, height=900, width=1200) as renderer:
        renderer.update_scene(data, camera=camera)
        _write_png(screenshot_path, renderer.render())

    package_model = menagerie.get("franka_emika_panda")
    model_info = {
        "date": dt.date.today().isoformat(),
        "phase": "M1",
        "mujoco_version": mujoco.__version__,
        "menagerie_package_version": menagerie.__version__,
        "menagerie_repository_commit": menagerie.commit(),
        "menagerie_model_oid": package_model.oid,
        "license": package_model.license,
        "upstream_panda_xml_sha256": _sha256(MODEL_SOURCE),
        "project_panda_mocap_xml_sha256": _sha256(MODEL_DERIVATIVE),
        "scene_xml": str(SCENE.relative_to(ROOT)).replace("\\", "/"),
        "world_frame": {
            "origin": "MuJoCo world origin at the Panda base frame",
            "vertical_axis": "+Z",
            "length_unit": "m",
            "angle_unit": "rad",
            "gravity_m_s2": model.opt.gravity.tolist(),
        },
        "home": {
            "keyframe_qpos": model.key_qpos[key_id].tolist(),
            "arm_qpos": home_arm_qpos.tolist(),
            "ee_site_name": "ee_site",
            "ee_site_local_pos_in_hand_m": model.site_pos[ee_site_id].tolist(),
            "ee_site_local_quat_in_hand_wxyz": model.site_quat[ee_site_id].tolist(),
            "ee_site_world_pos_m": home_pos.tolist(),
            "ee_site_world_quat_wxyz": home_quat.tolist(),
            "hand_body_id": _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
            "panda_base_body": "link0",
            "panda_base_body_pos_m": model.body_pos[_name_id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")].tolist(),
        },
        "actuators": {
            "arm_names": [f"actuator{i}" for i in range(1, 8)],
            "arm_groups": model.actuator_group[arm_actuator_ids].tolist(),
            "gripper_name": "actuator8",
            "gripper_group": int(model.actuator_group[gripper_actuator_id]),
            "gripper_ctrlrange": model.actuator_ctrlrange[gripper_actuator_id].tolist(),
            "disabled_actuator_group_mask": int(model.opt.disableactuator),
        },
        "scene": {
            "table_top_z_m": table_top_z,
            "cube_half_size_m": model.geom_size[_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")].tolist(),
            "cube_final_position_m": cube_position.tolist(),
            "cube_bottom_z_m": cube_bottom_z,
            "cube_linear_speed_m_s": cube_speed,
            "target_markers_collision_disabled": target_collision_disabled,
            "target_regions_world_centers_m": [model.geom_pos[i].tolist() for i in target_geom_ids],
        },
        "results": {
            "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
            "screenshot": str(screenshot_path.relative_to(ROOT)).replace("\\", "/"),
            "sample_count": len(samples),
            "simulation_time_s": float(data.time),
            "tracking_dwell_max_error_m": target_max_errors,
            "tracking_limit_m": ERROR_LIMIT_M,
            "model_gate_pass": True,
            "tracking_gate_pass": tracking_pass,
            "gripper_closed_mean_joint_qpos_m": closed_avg,
            "gripper_open_mean_joint_qpos_m": open_avg,
            "gripper_span_m": gripper_span,
            "ee_max_drift_during_gripper_m": ee_motion_during_gripper,
            "gripper_gate_pass": gripper_pass,
            "scene_idle_gate_pass": scene_pass,
            "warnings": warnings,
            "contact_pair_min_distance_m": pair_min_distance,
            "minimum_contact_distance_m": min_contact_distance,
            "penetration_limit_m": PENETRATION_LIMIT_M,
            "penetration_gate_pass": penetration_pass,
            "non_finite_state_seen": bad_state_seen,
            "overall_pass": tracking_pass and gripper_pass and scene_pass and penetration_pass,
        },
    }
    info_path = output / "model_info.json"
    info_path.write_text(json.dumps(model_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Scene loaded: {model.nbody} bodies, {model.ngeom} geoms, {model.nu} actuators, {model.neq} equalities")
    print(f"Home ee_site (m): {np.array2string(home_pos, precision=6)}")
    print(f"Tracking max dwell errors (m): {target_max_errors}")
    print(f"Tracking gate: {'PASS' if tracking_pass else 'FAIL'}")
    print(f"Gripper gate: {'PASS' if gripper_pass else 'FAIL'}; closed={closed_avg:.5f}, open={open_avg:.5f}, ee drift={ee_motion_during_gripper:.5f} m")
    print(f"Scene idle gate: {'PASS' if scene_pass else 'FAIL'}; cube bottom={cube_bottom_z:.5f} m, speed={cube_speed:.6f} m/s")
    print(f"Warnings: {warnings or 'none'}; minimum contact distance={min_contact_distance:.6f} m")
    print(f"Evidence: {csv_path}, {info_path}, {screenshot_path}")
    return 0 if model_info["results"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

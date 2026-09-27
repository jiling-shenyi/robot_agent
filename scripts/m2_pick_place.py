"""Run deterministic MuJoCo Panda pick-and-place episodes for M2."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "scene" / "panda_task.xml"
SCENARIOS_PATH = ROOT / "configs" / "m2_scenarios.json"
THRESHOLDS_PATH = ROOT / "configs" / "m2_thresholds.json"
SAMPLE_EVERY_STEPS = 25
WARNING_TYPES = {
    int(getattr(mujoco.mjtWarning, name)): name
    for name in dir(mujoco.mjtWarning)
    if name.startswith("mjWARN_")
}


class M2Failure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def name_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Required MuJoCo name not found: {name}")
    return int(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warning_counts(data: mujoco.MjData) -> dict[str, int]:
    return {
        WARNING_TYPES[index]: int(count)
        for index, count in enumerate(data.warning.number)
        if count and index in WARNING_TYPES
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_scenario(seed: int, target: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    for item in entries:
        if int(item["seed"]) == seed and item["target"] == target:
            return dict(item)
    rng = np.random.default_rng(seed)
    return {
        "scenario_id": f"adhoc_seed_{seed}_{target}",
        "seed": seed,
        "target": target,
        "cube_position_m": [
            round(float(rng.uniform(0.400, 0.440)), 6),
            round(float(rng.uniform(-0.290, -0.250)), 6),
            0.425,
        ],
    }


class Episode:
    def __init__(
        self,
        scenario: dict[str, Any],
        thresholds: dict[str, Any],
        output_dir: Path,
    ) -> None:
        self.scenario = scenario
        self.thresholds = thresholds
        self.model = mujoco.MjModel.from_xml_path("assets/scene/panda_task.xml")
        self.data = mujoco.MjData(self.model)
        self.output_dir = output_dir
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.total_steps = 0
        self.max_ee_error = 0.0
        self.minimum_contact_distance = math.inf
        self.pair_min_distance: dict[str, float] = {}
        self.penetration_exceptions_seen: dict[str, float] = {}
        self.max_unilateral_contact_steps = 0
        self.last_contacts: list[dict[str, Any]] = []
        self.stage = "initialize"
        self.result: dict[str, Any] = {}

        self.key_id = name_id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home_scene")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)
        self.ee_site_id = name_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.mocap_site_id = name_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mocap_site")
        self.mocap_body_id = name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")
        self.cube_body_id = name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = name_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.table_geom_id = name_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.target_geom_id = name_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "target_a" if scenario["target"] == "a" else "target_b",
        )
        self.gripper_actuator_id = name_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.cube_joint_id = name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.mocap_id = int(self.model.body_mocapid[self.mocap_body_id])
        self.table_top_z = float(
            self.data.geom_xpos[self.table_geom_id][2]
            + self.model.geom_size[self.table_geom_id][2]
        )
        self.cube_half_size = self.model.geom_size[self.cube_geom_id].copy()
        self.finger_joint_ids = [
            name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
            name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
        ]
        self.finger_qpos_addresses = [
            int(self.model.jnt_qposadr[joint_id]) for joint_id in self.finger_joint_ids
        ]
        self.finger_body_ids = {
            name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
            name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        }
        self.left_finger_body_id = name_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_body_id = name_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        self.target_center = np.zeros(3)
        self.ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.initial_cube_position = np.asarray(scenario["cube_position_m"], dtype=float)
        self.grasp_relative_position: np.ndarray | None = None
        self.unilateral_contact_steps = 0
        self.geom_labels: list[str] = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom{geom_id}"
            self.geom_labels.append(f"{body_name}:{geom_name}")

        self._validate_model()

    def _validate_model(self) -> None:
        if self.mocap_id < 0:
            raise RuntimeError("mocap_target must be a mocap body")
        if int(self.model.site_bodyid[self.ee_site_id]) != name_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        ):
            raise RuntimeError("ee_site must be attached to the Panda hand")
        if int(self.model.site_bodyid[self.mocap_site_id]) != self.mocap_body_id:
            raise RuntimeError("mocap_site must be attached to mocap_target")
        if int(self.model.jnt_type[self.cube_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError("cube_free must remain a free joint")
        if not np.allclose(self.model.actuator_ctrlrange[self.gripper_actuator_id], [0, 255]):
            raise RuntimeError("Unexpected gripper actuator control range")
        if len(self.initial_cube_position) != 3 or not np.isfinite(
            self.initial_cube_position
        ).all():
            raise ValueError("cube_position_m must contain three finite coordinates")
        table_center = self.data.geom_xpos[self.table_geom_id]
        table_half = self.model.geom_size[self.table_geom_id]
        if np.any(
            np.abs(self.initial_cube_position[:2] - table_center[:2])
            + self.cube_half_size[:2]
            > table_half[:2]
        ):
            raise ValueError("Scenario cube is outside the table support area")
        if abs(self.initial_cube_position[2] - (self.table_top_z + self.cube_half_size[2])) > 0.01:
            raise ValueError("Scenario cube must start on the tabletop")

    def _body_name(self, body_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom{geom_id}"

    def _current_contacts(self) -> list[dict[str, Any]]:
        by_body_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            distance = float(contact.dist)
            name1, name2 = self._body_name(body1), self._body_name(body2)
            pair = tuple(sorted((name1, name2)))
            if pair not in by_body_pair:
                by_body_pair[pair] = {
                    "body1": pair[0],
                    "body2": pair[1],
                    "contact_count": 0,
                    "minimum_distance_m": distance,
                }
            entry = by_body_pair[pair]
            entry["contact_count"] += 1
            entry["minimum_distance_m"] = min(entry["minimum_distance_m"], distance)
        contacts = list(by_body_pair.values())
        self.last_contacts = contacts
        return contacts

    def _update_contact_metrics(self) -> None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            distance = float(contact.dist)
            pair = " <-> ".join(sorted((self.geom_labels[geom1], self.geom_labels[geom2])))
            self.minimum_contact_distance = min(self.minimum_contact_distance, distance)
            previous = self.pair_min_distance.get(pair)
            if previous is None or distance < previous:
                self.pair_min_distance[pair] = distance

    def _cube_corners(self) -> np.ndarray:
        half = self.cube_half_size
        local = np.array(
            [
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        )
        rotation = self.data.xmat[self.cube_body_id].reshape(3, 3)
        center = self.data.xpos[self.cube_body_id]
        return local @ rotation.T + center

    def _cube_speed(self) -> tuple[float, float]:
        velocity = self.data.cvel[self.cube_body_id]
        return float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3]))

    def _snapshot(self) -> dict[str, Any]:
        contacts = self._current_contacts()
        cube_position = self.data.xpos[self.cube_body_id].copy()
        cube_linear_speed, cube_angular_speed = self._cube_speed()
        ee_actual = self.data.site_xpos[self.ee_site_id].copy()
        ee_target = self.data.mocap_pos[self.mocap_id].copy()
        fingers = self.data.qpos[self.finger_qpos_addresses].copy()
        finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(cube_position).all()
            and np.isfinite(ee_actual).all()
        )
        return {
            "phase": self.stage,
            "step": self.total_steps,
            "sim_time_s": float(self.data.time),
            "ee_target_xyz_m": ee_target.tolist(),
            "ee_actual_xyz_m": ee_actual.tolist(),
            "ee_position_error_m": float(np.linalg.norm(ee_actual - ee_target)),
            "cube_xyz_m": cube_position.tolist(),
            "cube_linear_speed_m_s": cube_linear_speed,
            "cube_angular_speed_rad_s": cube_angular_speed,
            "gripper_ctrl": float(self.data.ctrl[self.gripper_actuator_id]),
            "finger_qpos_m": fingers.tolist(),
            "contacts": contacts,
            "minimum_contact_distance_so_far_m": (
                self.minimum_contact_distance if math.isfinite(self.minimum_contact_distance) else None
            ),
            "warnings": warning_counts(self.data),
            "finite_state": finite,
        }

    def _record_event(self, event: str, detail: dict[str, Any] | None = None) -> None:
        entry = {"event": event, "state": self._snapshot()}
        if detail:
            entry["detail"] = jsonable(detail)
        self.events.append(entry)

    def _record_sample(self) -> None:
        self.samples.append(self._snapshot())

    def _step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.total_steps += 1
        if self.total_steps > int(self.thresholds["max_episode_steps"]):
            raise M2Failure("BUDGET_EXHAUSTED", "Episode step budget exhausted")
        self._update_contact_metrics()
        if not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.site_xpos[self.ee_site_id]).all()
            and np.isfinite(self.data.xpos[self.cube_body_id]).all()
        ):
            raise M2Failure("SIMULATION_ERROR", "Non-finite MuJoCo state observed")
        warnings = warning_counts(self.data)
        if warnings:
            raise M2Failure("SIMULATION_ERROR", f"MuJoCo warning observed: {warnings}")
        global_penetration_limit = float(self.thresholds["penetration_limit_m"])
        pair_exceptions = self.thresholds.get("contact_penetration_exceptions_m", {})
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pair = " <-> ".join(sorted((self.geom_labels[geom1], self.geom_labels[geom2])))
            distance = float(contact.dist)
            allowed_penetration = float(pair_exceptions.get(pair, global_penetration_limit))
            if distance < -allowed_penetration:
                raise M2Failure(
                    "SIMULATION_ERROR",
                    f"Contact penetration exceeded {allowed_penetration * 1000:.1f} mm "
                    f"for {pair}: {distance * 1000:.3f} mm",
                )
            if pair in pair_exceptions and distance < -global_penetration_limit:
                previous = self.penetration_exceptions_seen.get(pair)
                if previous is None or distance < previous:
                    self.penetration_exceptions_seen[pair] = distance
        ee_error = float(
            np.linalg.norm(self.data.site_xpos[self.ee_site_id] - self.data.mocap_pos[self.mocap_id])
        )
        self.max_ee_error = max(self.max_ee_error, ee_error)
        if ee_error > float(self.thresholds["ee_error_limit_m"]):
            raise M2Failure(
                "EE_TRACKING_ERROR",
                f"Actual ee_site exceeded position error limit: {ee_error:.6f} m",
            )
        if self.total_steps % SAMPLE_EVERY_STEPS == 0:
            self._record_sample()

    def _hold(self, stage: str, steps: int, control: float | None = None) -> None:
        self.stage = stage
        if control is not None:
            self.data.ctrl[self.gripper_actuator_id] = control
        self._record_event("phase_start")
        for _ in range(steps):
            self._step()
        self._record_event("phase_end")

    def _move_ee(
        self,
        stage: str,
        target_xyz: np.ndarray,
        held: bool = False,
        require_transport_clearance: bool = False,
    ) -> None:
        self.stage = stage
        self._record_event("phase_start", {"target_xyz_m": target_xyz.tolist()})
        start = self.data.mocap_pos[self.mocap_id].copy()
        distance = float(np.linalg.norm(target_xyz - start))
        max_increment = float(self.thresholds["motion_max_step_m"])
        steps = max(1, int(math.ceil(distance / max_increment)))
        if steps > 12000:
            raise M2Failure("BUDGET_EXHAUSTED", f"Motion segment too long: {steps} steps")
        for index in range(1, steps + 1):
            fraction = index / steps
            self.data.mocap_pos[self.mocap_id] = start * (1.0 - fraction) + target_xyz * fraction
            self.data.mocap_quat[self.mocap_id] = self.ee_quat
            self._step()
            if held:
                self._assert_held_integrity(require_transport_clearance)
        self.data.mocap_pos[self.mocap_id] = target_xyz
        for _ in range(int(self.thresholds["terminal_dwell_steps"])):
            self._step()
            if held:
                self._assert_held_integrity(require_transport_clearance)
        if held:
            self._assert_held_integrity(
                require_transport_clearance, require_bilateral_contact=True
            )
        self._record_event("phase_end", {"segment_steps": steps})

    def _finger_contact_sides(self) -> set[int]:
        sides: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            if body1 == self.cube_body_id and body2 in {
                self.left_finger_body_id,
                self.right_finger_body_id,
            }:
                sides.add(body2)
            elif body2 == self.cube_body_id and body1 in {
                self.left_finger_body_id,
                self.right_finger_body_id,
            }:
                sides.add(body1)
        return sides

    def _assert_held_integrity(
        self,
        require_transport_clearance: bool = False,
        require_bilateral_contact: bool = False,
    ) -> None:
        sides = self._finger_contact_sides()
        if len(sides) < 2:
            self.unilateral_contact_steps += 1
            self.max_unilateral_contact_steps = max(
                self.max_unilateral_contact_steps, self.unilateral_contact_steps
            )
            if require_bilateral_contact or self.unilateral_contact_steps > int(
                self.thresholds["gripper_unilateral_contact_grace_steps"]
            ):
                raise M2Failure(
                    "OBJECT_SLIPPED",
                    "Bilateral finger contact was not restored within the allowed window",
                )
        else:
            self.unilateral_contact_steps = 0
        cube_xyz = self.data.xpos[self.cube_body_id]
        if require_transport_clearance and cube_xyz[2] < (
            self.table_top_z + self.cube_half_size[2] + 0.03
        ):
            raise M2Failure("OBJECT_DROPPED", "Cube fell below the transport clearance")
        if self.grasp_relative_position is not None:
            ee_xyz = self.data.site_xpos[self.ee_site_id]
            drift = float(np.linalg.norm((cube_xyz - ee_xyz) - self.grasp_relative_position))
            if drift > float(self.thresholds["grasp_relative_drift_limit_m"]):
                raise M2Failure(
                    "OBJECT_SLIPPED",
                    f"Cube-to-ee relative position drift exceeded limit: {drift:.6f} m",
                )

    def _postconditions(self) -> dict[str, Any]:
        self._update_contact_metrics()
        self._current_contacts()
        corners = self._cube_corners()
        lower = self.data.geom_xpos[self.target_geom_id][:2] + (
            -self.model.geom_size[self.target_geom_id][:2]
            + float(self.thresholds["target_margin_m"])
        )
        upper = self.data.geom_xpos[self.target_geom_id][:2] + (
            self.model.geom_size[self.target_geom_id][:2]
            - float(self.thresholds["target_margin_m"])
        )
        footprint_inside = bool(np.all(corners[:, :2] >= lower) and np.all(corners[:, :2] <= upper))
        cube_bottom_z = float(np.min(corners[:, 2]))
        bottom_error = abs(cube_bottom_z - self.table_top_z)
        table_supported = any(
            {int(self.data.contact[index].geom1), int(self.data.contact[index].geom2)}
            == {self.table_geom_id, self.cube_geom_id}
            for index in range(self.data.ncon)
        )
        gripper_contact = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            if (body1 == self.cube_body_id and body2 in self.finger_body_ids) or (
                body2 == self.cube_body_id and body1 in self.finger_body_ids
            ):
                gripper_contact = True
        finger_qpos = self.data.qpos[self.finger_qpos_addresses].copy()
        linear_speed, angular_speed = self._cube_speed()
        check = {
            "gripper_open": bool(
                np.all(finger_qpos >= float(self.thresholds["finger_open_qpos_min_m"]))
            ),
            "no_gripper_contact": not gripper_contact,
            "table_supported": table_supported,
            "bottom_height_within_tolerance": bool(
                bottom_error <= float(self.thresholds["table_support_tolerance_m"])
            ),
            "full_projected_footprint_inside_target_with_margin": footprint_inside,
            "linear_speed_below_limit": bool(
                linear_speed <= float(self.thresholds["linear_speed_limit_m_s"])
            ),
            "angular_speed_below_limit": bool(
                angular_speed <= float(self.thresholds["angular_speed_limit_rad_s"])
            ),
            "warnings_absent": not bool(warning_counts(self.data)),
            "finite_state": bool(
                np.isfinite(self.data.qpos).all()
                and np.isfinite(self.data.qvel).all()
                and np.isfinite(corners).all()
            ),
            "cube_position_m": self.data.xpos[self.cube_body_id].copy().tolist(),
            "cube_bottom_z_m": cube_bottom_z,
            "table_top_z_m": self.table_top_z,
            "bottom_height_error_m": bottom_error,
            "cube_linear_speed_m_s": linear_speed,
            "cube_angular_speed_rad_s": angular_speed,
            "target_inset_lower_xy_m": lower.tolist(),
            "target_inset_upper_xy_m": upper.tolist(),
            "projected_cube_xy_min_m": corners[:, :2].min(axis=0).tolist(),
            "projected_cube_xy_max_m": corners[:, :2].max(axis=0).tolist(),
            "finger_qpos_m": finger_qpos.tolist(),
            "gripper_contact": gripper_contact,
        }
        condition_keys = (
            "gripper_open",
            "no_gripper_contact",
            "table_supported",
            "bottom_height_within_tolerance",
            "full_projected_footprint_inside_target_with_margin",
            "linear_speed_below_limit",
            "angular_speed_below_limit",
            "warnings_absent",
            "finite_state",
        )
        check["all_instantaneous_conditions"] = all(check[key] for key in condition_keys)
        return check

    def run(self) -> dict[str, Any]:
        status = "FAILED"
        error_code: str | None = None
        error_message: str | None = None
        stable_steps_observed = 0
        final_check: dict[str, Any] | None = None
        try:
            self.stage = "reset_and_settle"
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
            self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 7] = np.array(
                [*self.initial_cube_position, 1.0, 0.0, 0.0, 0.0]
            )
            self.data.ctrl[self.gripper_actuator_id] = 255.0
            mujoco.mj_forward(self.model, self.data)
            self.data.mocap_pos[self.mocap_id] = self.data.site_xpos[self.ee_site_id].copy()
            self.ee_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(self.ee_quat, self.data.site_xmat[self.ee_site_id])
            self.data.mocap_quat[self.mocap_id] = self.ee_quat
            mujoco.mj_forward(self.model, self.data)
            self._record_event("reset_complete", {"cube_start_xyz_m": self.initial_cube_position})
            self._hold("scene_settle", int(self.thresholds["scene_settle_steps"]))
            self.target_center = self.data.geom_xpos[self.target_geom_id].copy()
            initial_check = self._postconditions()
            start_cube = self.data.xpos[self.cube_body_id].copy()
            if (
                abs(start_cube[2] - (self.table_top_z + self.cube_half_size[2])) > 0.003
                or not initial_check["table_supported"]
                or initial_check["cube_linear_speed_m_s"] > 0.01
            ):
                raise M2Failure("PRECONDITION_FAILED", "Cube did not settle on the table")
            if initial_check["full_projected_footprint_inside_target_with_margin"]:
                raise M2Failure("PRECONDITION_FAILED", "Cube starts inside the selected target")

            self._hold("open_before_pick", int(self.thresholds["gripper_open_steps"]), 255.0)
            pregrasp = start_cube.copy()
            pregrasp[2] += float(self.thresholds["pregrasp_clearance_m"])
            self._move_ee("approach_above_cube", pregrasp)
            grasp_pose = start_cube.copy()
            self._move_ee("descend_to_grasp", grasp_pose)
            self._hold("close_on_cube", int(self.thresholds["gripper_close_steps"]), 0.0)
            sides = self._finger_contact_sides()
            finger_qpos = self.data.qpos[self.finger_qpos_addresses].copy()
            if len(sides) < 2 or np.min(finger_qpos) < float(
                self.thresholds["finger_closed_contact_qpos_min_m"]
            ):
                raise M2Failure(
                    "GRASP_EMPTY",
                    f"No bilateral grasp detected; sides={len(sides)}, finger_qpos={finger_qpos.tolist()}",
                )
            self.grasp_relative_position = (
                self.data.xpos[self.cube_body_id] - self.data.site_xpos[self.ee_site_id]
            ).copy()
            self._record_event(
                "grasp_verified",
                {
                    "finger_contact_side_count": len(sides),
                    "finger_qpos_m": finger_qpos,
                    "cube_to_ee_offset_m": self.grasp_relative_position,
                },
            )

            lifted_pose = grasp_pose.copy()
            lifted_pose[2] += float(self.thresholds["lift_height_m"])
            self._move_ee("vertical_lift", lifted_pose, held=True)
            lifted_cube = self.data.xpos[self.cube_body_id].copy()
            achieved_lift = float(lifted_cube[2] - start_cube[2])
            if achieved_lift < float(self.thresholds["minimum_lift_m"]):
                raise M2Failure(
                    "GRASP_EMPTY",
                    f"Cube did not rise enough after grasp: {achieved_lift:.6f} m",
                )
            self._record_event("lift_verified", {"achieved_lift_m": achieved_lift})

            transit_pose = lifted_pose.copy()
            transit_pose[:2] = self.target_center[:2]
            self._move_ee(
                "transport_to_target", transit_pose, held=True, require_transport_clearance=True
            )
            place_pose = transit_pose.copy()
            place_pose[2] = (
                self.table_top_z
                + float(self.cube_half_size[2])
                + float(self.thresholds["place_clearance_m"])
            )
            self._move_ee("lower_to_place", place_pose)
            if len(self._finger_contact_sides()) < 2:
                placement_contact = self._postconditions()
                if not (
                    placement_contact["table_supported"]
                    and placement_contact["bottom_height_within_tolerance"]
                    and placement_contact["full_projected_footprint_inside_target_with_margin"]
                ):
                    raise M2Failure(
                        "OBJECT_SLIPPED",
                        "Bilateral grasp ended before a supported in-target placement",
                    )
            self._hold("release_cube", int(self.thresholds["gripper_open_steps"]), 255.0)
            retreat_pose = place_pose.copy()
            retreat_pose[2] += float(self.thresholds["pregrasp_clearance_m"])
            self._move_ee("retreat_after_release", retreat_pose)

            self.stage = "post_place_stability"
            self._record_event("phase_start", {"required_stable_steps": self.thresholds["stable_steps"]})
            stable_required = int(self.thresholds["stable_steps"])
            for _ in range(int(self.thresholds["placement_settle_max_steps"])):
                self._step()
                final_check = self._postconditions()
                if final_check["all_instantaneous_conditions"]:
                    stable_steps_observed += 1
                    if stable_steps_observed >= stable_required:
                        break
                else:
                    stable_steps_observed = 0
            final_check = self._postconditions()
            self._record_event(
                "phase_end",
                {"stable_steps_observed": stable_steps_observed, "postconditions": final_check},
            )
            if stable_steps_observed < stable_required:
                false_conditions = [
                    key
                    for key in (
                        "gripper_open",
                        "no_gripper_contact",
                        "table_supported",
                        "bottom_height_within_tolerance",
                        "full_projected_footprint_inside_target_with_margin",
                        "linear_speed_below_limit",
                        "angular_speed_below_limit",
                        "warnings_absent",
                        "finite_state",
                    )
                    if not final_check[key]
                ]
                raise M2Failure(
                    "GOAL_NOT_MET",
                    "Post-place conditions did not remain true for the required window: "
                    + ", ".join(false_conditions),
                )
            status = "SUCCESS"
        except M2Failure as error:
            error_code, error_message = error.code, str(error)
            self._record_event("episode_failed", {"error_code": error_code, "message": error_message})
        except Exception as error:
            error_code, error_message = "SIMULATION_ERROR", f"{type(error).__name__}: {error}"
            self._record_event(
                "episode_failed",
                {
                    "error_code": error_code,
                    "message": error_message,
                    "traceback": traceback.format_exc(),
                },
            )

        if final_check is None:
            try:
                final_check = self._postconditions()
            except Exception:
                final_check = {"all_instantaneous_conditions": False}
        self.result = {
            "scenario_id": self.scenario["scenario_id"],
            "seed": int(self.scenario["seed"]),
            "target": self.scenario["target"],
            "cube_start_xyz_m": self.initial_cube_position.tolist(),
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            "step_count": self.total_steps,
            "simulated_time_s": float(self.data.time),
            "max_ee_position_error_m": self.max_ee_error,
            "minimum_contact_distance_m": (
                self.minimum_contact_distance if math.isfinite(self.minimum_contact_distance) else None
            ),
            "contact_pair_minimum_distances_m": self.pair_min_distance,
            "penetration_exceptions_seen_m": self.penetration_exceptions_seen,
            "max_unilateral_contact_steps": self.max_unilateral_contact_steps,
            "stable_steps_observed": stable_steps_observed,
            "postconditions": final_check,
            "events": self.events,
        }
        return self.result


def run_one(scenario: dict[str, Any], thresholds: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode = Episode(scenario, thresholds, output_dir)
    result = episode.run()
    return result, episode.samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--batch", action="store_true", help="run the frozen 10-scene list")
    selection.add_argument("--target", choices=("a", "b"), help="run one episode for target a or b")
    parser.add_argument("--seed", type=int, default=0, help="scenario seed for a single episode")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "m2",
        help="output directory (existing evidence files are never overwritten by default)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace episodes.jsonl, trajectory.csv, and summary.json in the output directory",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    scenario_data = load_json(SCENARIOS_PATH)
    thresholds = load_json(THRESHOLDS_PATH)
    if args.batch:
        if scenario_data.get("status") != "frozen":
            parser.error("--batch requires configs/m2_scenarios.json status=frozen")
        if thresholds.get("status") != "frozen":
            parser.error("--batch requires configs/m2_thresholds.json status=frozen")
        scenarios = list(scenario_data["scenarios"])
        if len(scenarios) != 10:
            parser.error(f"M2 acceptance requires exactly 10 scenarios, found {len(scenarios)}")
        if sum(item["target"] == "a" for item in scenarios) < 5 or sum(
            item["target"] == "b" for item in scenarios
        ) < 5:
            parser.error("The frozen scenario list must include at least five episodes per target")
    else:
        scenarios = [make_scenario(args.seed, args.target, scenario_data.get("scenarios", []))]

    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / "episodes.jsonl", output_dir / "trajectory.csv", output_dir / "summary.json"]
    if not args.overwrite and any(path.exists() for path in output_paths):
        parser.error(
            f"Output evidence already exists in {output_dir}; choose another --output or pass --overwrite"
        )

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    episode_results: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(
            f"[{scenario['scenario_id']}] target={scenario['target']} seed={scenario['seed']} "
            f"cube={scenario['cube_position_m']}"
        )
        result, samples = run_one(scenario, thresholds, output_dir)
        result["run_id"] = run_id
        episode_results.append(result)
        all_samples.extend(
            {"run_id": run_id, "scenario_id": scenario["scenario_id"], **sample}
            for sample in samples
        )
        print(
            f"  {result['status']} {result['error_code'] or ''}; "
            f"steps={result['step_count']} max_ee_error={result['max_ee_position_error_m']:.6f} m"
        )

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for result in episode_results:
            stream.write(json.dumps(jsonable(result), ensure_ascii=False, separators=(",", ":")) + "\n")
    trajectory_fields = [
        "run_id",
        "scenario_id",
        "phase",
        "step",
        "sim_time_s",
        "ee_target_xyz_m",
        "ee_actual_xyz_m",
        "ee_position_error_m",
        "cube_xyz_m",
        "cube_linear_speed_m_s",
        "cube_angular_speed_rad_s",
        "gripper_ctrl",
        "finger_qpos_m",
        "contacts",
        "minimum_contact_distance_so_far_m",
        "warnings",
        "finite_state",
    ]
    with (output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=trajectory_fields)
        writer.writeheader()
        for sample in all_samples:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in sample.items()
                }
            )

    model = mujoco.MjModel.from_xml_path("assets/scene/panda_task.xml")
    summary = {
        "run_id": run_id,
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": "M2",
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "scene_xml": "assets/scene/panda_task.xml",
        "scene_xml_sha256": sha256(SCENE),
        "panda_mocap_xml_sha256": sha256(
            ROOT / "assets" / "third_party" / "franka_emika_panda" / "panda_mocap.xml"
        ),
        "scenario_config": "configs/m2_scenarios.json",
        "scenario_config_sha256": sha256(SCENARIOS_PATH),
        "threshold_config": "configs/m2_thresholds.json",
        "threshold_config_sha256": sha256(THRESHOLDS_PATH),
        "scenario_count": len(episode_results),
        "success_count": sum(result["status"] == "SUCCESS" for result in episode_results),
        "failure_count": sum(result["status"] != "SUCCESS" for result in episode_results),
        "all_pass": all(result["status"] == "SUCCESS" for result in episode_results),
        "episodes": [
            {
                "scenario_id": result["scenario_id"],
                "seed": result["seed"],
                "target": result["target"],
                "status": result["status"],
                "error_code": result["error_code"],
                "step_count": result["step_count"],
                "max_ee_position_error_m": result["max_ee_position_error_m"],
                "minimum_contact_distance_m": result["minimum_contact_distance_m"],
                "penetration_exceptions_seen_m": result["penetration_exceptions_seen_m"],
                "max_unilateral_contact_steps": result["max_unilateral_contact_steps"],
                "stable_steps_observed": result["stable_steps_observed"],
                "postconditions": result["postconditions"],
            }
            for result in episode_results
        ],
        "model_dimensions": {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nu": int(model.nu),
            "neq": int(model.neq),
            "timestep_s": float(model.opt.timestep),
        },
        "thresholds": thresholds,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"M2 result: {summary['success_count']}/{summary['scenario_count']} passed. "
        f"Evidence: {output_dir}"
    )
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

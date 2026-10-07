"""Live MuJoCo episode state shared by execution, skills and visualization."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..evaluation.evidence import jsonable
from ..evaluation.placement import PlacementEvaluationMixin
from ..execution.task import TaskExecutionMixin
from ..safety.guards import EpisodeSafetyMixin
from ..skills.panda import PandaSkillsMixin
from ..visualization.episode_viewer import EpisodeViewerMixin
from .errors import M2Failure, ViewerClosed
from .model import SAMPLE_EVERY_STEPS, load_scene_model, name_id, warning_counts


class Episode(TaskExecutionMixin, PandaSkillsMixin, PlacementEvaluationMixin,
              EpisodeSafetyMixin, EpisodeViewerMixin):
    """One real model/data pair with monitored deterministic Panda execution."""

    def __init__(
        self,
        scenario: dict[str, Any],
        thresholds: dict[str, Any],
        output_dir: Path,
        enable_viewer: bool = False,
        show_ui: bool = False,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        self.scenario = scenario
        self.thresholds = thresholds
        self.model = model if model is not None else load_scene_model()
        self.data = mujoco.MjData(self.model)
        self.output_dir = output_dir
        self.enable_viewer = enable_viewer
        self.show_ui = show_ui
        self.viewer = None
        self.next_viewer_deadline = time.perf_counter()
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.total_steps = 0
        self.step_budget_limit = int(thresholds["max_episode_steps"])
        self.max_ee_error = 0.0
        self.minimum_contact_distance = math.inf
        self.pair_min_distance: dict[str, float] = {}
        self.penetration_exceptions_seen: dict[str, float] = {}
        self.max_unilateral_contact_steps = 0
        self.last_contacts: list[dict[str, Any]] = []
        self.stage = "initialize"
        self.result: dict[str, Any] = {}
        self.prepared = False

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
        self.danger_zone_geom_id = name_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "danger_zone"
        )
        self.danger_zone_clearance_m = float(thresholds["danger_zone_clearance_m"])
        self.danger_zone_fromto = np.zeros(6, dtype=np.float64)
        self.robot_collision_geom_ids: list[int] = []
        self.minimum_danger_zone_separation = math.inf
        self.danger_zone_violation: dict[str, Any] | None = None
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
            if (
                body_name not in {"world", "table", "cube", "mocap_target"}
                and geom_id != self.danger_zone_geom_id
                and (int(self.model.geom_contype[geom_id]) or int(self.model.geom_conaffinity[geom_id]))
            ):
                self.robot_collision_geom_ids.append(geom_id)

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
        if self.danger_zone_clearance_m <= 0 or not math.isfinite(self.danger_zone_clearance_m):
            raise ValueError("danger_zone_clearance_m must be a positive finite value")
        if int(self.model.geom_contype[self.danger_zone_geom_id]) != 0 or int(
            self.model.geom_conaffinity[self.danger_zone_geom_id]
        ) != 0:
            raise RuntimeError("danger_zone must be visual-only; M2 enforces it in software")
        if int(self.model.geom_type[self.danger_zone_geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise RuntimeError("danger_zone must remain a 3D box so checks match its visible volume")
        danger_half_size = self.model.geom_size[self.danger_zone_geom_id]
        if not np.isfinite(danger_half_size).all() or np.any(danger_half_size <= 0):
            raise RuntimeError("danger_zone must have finite, positive half-sizes")
        if not self.robot_collision_geom_ids:
            raise RuntimeError("No Panda collision geoms found for danger-zone validation")
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
            "minimum_danger_zone_separation_so_far_m": (
                self.minimum_danger_zone_separation
                if math.isfinite(self.minimum_danger_zone_separation)
                else None
            ),
            "danger_zone_clear": self.danger_zone_violation is None,
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

    def prepare(self, *, check_safety: bool = True) -> dict[str, Any]:
        """Reset the episode and align the mocap target with the real Panda hand."""
        if self.prepared:
            raise RuntimeError("Episode is already prepared")
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
        if check_safety:
            self._check_danger_zone()
        self._open_viewer()
        self._record_event("reset_complete", {"cube_start_xyz_m": self.initial_cube_position})
        self.prepared = True
        return self._snapshot()

    def _step(self) -> None:
        from embodied_agent.models.budget import current_budget
        task_budget = current_budget()
        if task_budget is not None and self.stage != "instruction_hold":
            task_budget.check()
        if self.total_steps >= self.step_budget_limit:
            raise M2Failure("BUDGET_EXHAUSTED", "Episode step budget exhausted")
        mujoco.mj_step(self.model, self.data)
        self.total_steps += 1
        self._update_contact_metrics()
        self.check_step_safety()
        if self.total_steps % SAMPLE_EVERY_STEPS == 0:
            self._record_sample()
            self._sync_viewer()

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

    def snapshot(self) -> dict[str, Any]:
        """Return measured telemetry from the current model/data pair."""
        return self._snapshot()

    def cube_speed(self) -> tuple[float, float]:
        return self._cube_speed()

    def finger_contact_sides(self) -> set[int]:
        return self._finger_contact_sides()

    def check_safety(self) -> None:
        """Check the current danger-zone clearance without stepping physics."""
        self._check_danger_zone()

    def step(self) -> None:
        """Advance one budgeted, fully monitored physics step."""
        self._step()

    def hold(self, stage: str, steps: int, control: float | None = None) -> None:
        self._hold(stage, steps, control)

    def postconditions(self) -> dict[str, Any]:
        return self._postconditions()

    def close(self) -> None:
        """Close the episode's optional display."""
        self._close_viewer()


    def record_event(self, event: str, detail: dict[str, Any] | None = None) -> None:
        """Append an event with the measured current state."""
        self._record_event(event, detail)

    def current_contacts(self) -> list[dict[str, Any]]:
        return self._current_contacts()

    def update_contact_metrics(self) -> None:
        self._update_contact_metrics()

    def cube_corners(self) -> np.ndarray:
        return self._cube_corners()

    def assert_held_integrity(
        self, require_transport_clearance: bool = False,
        require_bilateral_contact: bool = False,
    ) -> None:
        self._assert_held_integrity(require_transport_clearance, require_bilateral_contact)

    def body_name(self, body_id: int) -> str:
        return self._body_name(body_id)

    def viewer_closed(self) -> bool:
        """Whether an attached display has been closed by the user."""
        return self.viewer is not None and not self.viewer.is_running()

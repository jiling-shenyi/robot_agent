"""Deterministic monitored Panda motion and pick/place skills."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..simulation.errors import M2Failure


class PandaSkillsMixin:
    """Skills use the episode's real controls and safety-monitored physics steps."""

    def move_ee(
        self,
        stage: str,
        target_xyz: np.ndarray,
        held: bool = False,
        require_transport_clearance: bool = False,
    ) -> None:
        """Move the actual Panda hand through M2's monitored mocap control path."""
        if not self.prepared:
            raise RuntimeError("Call prepare() before moving the Panda hand")
        target = np.asarray(target_xyz, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_xyz must contain three finite coordinates")
        self._move_ee(stage, target, held, require_transport_clearance)

    def _hold(self, stage: str, steps: int, control: float | None = None) -> None:
        self.stage = stage
        if control is not None:
            self.data.ctrl[self.gripper_actuator_id] = control
        self.record_event("phase_start")
        for _ in range(steps):
            self.step()
        self.record_event("phase_end")

    def _move_ee(
        self,
        stage: str,
        target_xyz: np.ndarray,
        held: bool = False,
        require_transport_clearance: bool = False,
    ) -> None:
        self.stage = stage
        self.record_event("phase_start", {"target_xyz_m": target_xyz.tolist()})
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
            self.step()
            if held:
                self.assert_held_integrity(require_transport_clearance)
        self.data.mocap_pos[self.mocap_id] = target_xyz
        for _ in range(int(self.thresholds["terminal_dwell_steps"])):
            self.step()
            if held:
                self.assert_held_integrity(require_transport_clearance)
        if held:
            self.assert_held_integrity(
                require_transport_clearance, require_bilateral_contact=True
            )
        self.record_event("phase_end", {"segment_steps": steps})

    def pick_skill(self) -> dict[str, Any]:
        """Pick the cube using M2's verified top grasp and return measured lift state."""
        if not self.prepared or self.total_steps < int(self.thresholds["scene_settle_steps"]):
            raise M2Failure("PRECONDITION_FAILED", "Episode has not passed scene initialization")
        initial_check = self.postconditions()
        start_cube = self.data.xpos[self.cube_body_id].copy()
        self.hold("open_before_pick", int(self.thresholds["gripper_open_steps"]), 255.0)
        pregrasp = start_cube.copy()
        pregrasp[2] += float(self.thresholds["pregrasp_clearance_m"])
        self.move_ee("approach_above_cube", pregrasp)
        grasp_pose = start_cube.copy()
        self.move_ee("descend_to_grasp", grasp_pose)
        self.hold("close_on_cube", int(self.thresholds["gripper_close_steps"]), 0.0)
        sides = self.finger_contact_sides()
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
        self.record_event(
            "grasp_verified",
            {
                "finger_contact_side_count": len(sides),
                "finger_qpos_m": finger_qpos,
                "cube_to_ee_offset_m": self.grasp_relative_position,
            },
        )

        lifted_pose = grasp_pose.copy()
        lifted_pose[2] += float(self.thresholds["lift_height_m"])
        self.move_ee("vertical_lift", lifted_pose, held=True)
        lifted_cube = self.data.xpos[self.cube_body_id].copy()
        achieved_lift = float(lifted_cube[2] - start_cube[2])
        if achieved_lift < float(self.thresholds["minimum_lift_m"]):
            raise M2Failure(
                "GRASP_EMPTY",
                f"Cube did not rise enough after grasp: {achieved_lift:.6f} m",
            )
        self.record_event("lift_verified", {"achieved_lift_m": achieved_lift})
        return {
            "achieved_lift_m": achieved_lift,
            "finger_contact_side_count": len(sides),
            "cube_to_ee_offset_m": self.grasp_relative_position.copy(),
        }

    def place_skill(self) -> dict[str, Any]:
        """Transport and release the verified grasp at the selected M2 target."""
        if self.grasp_relative_position is None or len(self.finger_contact_sides()) < 2:
            raise M2Failure("PRECONDITION_FAILED", "Place requires a verified bilateral grasp")
        cube_position = self.data.xpos[self.cube_body_id].copy()
        if cube_position[2] < self.table_top_z + self.cube_half_size[2] + float(
            self.thresholds["minimum_lift_m"]
        ):
            raise M2Failure("PRECONDITION_FAILED", "Place requires the cube to be lifted")

        lifted_pose = self.data.mocap_pos[self.mocap_id].copy()
        transit_pose = lifted_pose.copy()
        transit_pose[:2] = self.target_center[:2]
        self.move_ee(
            "transport_to_target", transit_pose, held=True, require_transport_clearance=True
        )
        place_pose = transit_pose.copy()
        place_pose[2] = (
            self.table_top_z
            + float(self.cube_half_size[2])
            + float(self.thresholds["place_clearance_m"])
        )
        self.move_ee("lower_to_place", place_pose)
        if len(self.finger_contact_sides()) < 2:
            placement_contact = self.postconditions()
            if not (
                placement_contact["table_supported"]
                and placement_contact["bottom_height_within_tolerance"]
                and placement_contact["full_projected_footprint_inside_target_with_margin"]
            ):
                raise M2Failure(
                    "OBJECT_SLIPPED",
                    "Bilateral grasp ended before a supported in-target placement",
                )
        self.hold("release_cube", int(self.thresholds["gripper_open_steps"]), 255.0)
        retreat_pose = place_pose.copy()
        retreat_pose[2] += float(self.thresholds["pregrasp_clearance_m"])
        self.move_ee("retreat_after_release", retreat_pose)
        return {"target_center_m": self.target_center.copy()}

"""Measured per-step safety guards and grasp integrity checks."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from ..simulation.errors import M2Failure
from ..simulation.model import warning_counts


class EpisodeSafetyMixin:
    """Safety rules operating on the live episode, without a second world state."""

    def _check_danger_zone(self) -> None:
        """Enforce clearance for every Panda collision geom and the manipulated cube."""
        for geom_id in (*self.robot_collision_geom_ids, self.cube_geom_id):
            distance = float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom_id,
                    self.danger_zone_geom_id,
                    10.0,
                    self.danger_zone_fromto,
                )
            )
            if not math.isfinite(distance):
                raise M2Failure(
                    "SIMULATION_ERROR",
                    f"Non-finite separation from danger_zone for {self.geom_labels[geom_id]}",
                )
            self.minimum_danger_zone_separation = min(
                self.minimum_danger_zone_separation, distance
            )
            if distance < self.danger_zone_clearance_m:
                self.danger_zone_violation = {
                    "phase": self.stage,
                    "step": self.total_steps,
                    "geom": self.geom_labels[geom_id],
                    "distance_m": distance,
                    "required_clearance_m": self.danger_zone_clearance_m,
                    "closest_points_m": self.danger_zone_fromto.reshape(2, 3).copy(),
                }
                self.record_event("danger_zone_violation", self.danger_zone_violation)
                self.set_viewer_status(
                    "DANGER_ZONE_VIOLATION",
                    f"{self.geom_labels[geom_id]} entered the {self.danger_zone_clearance_m * 1000:.1f} mm safety margin",
                )
                raise M2Failure(
                    "DANGER_ZONE_VIOLATION",
                    f"{self.geom_labels[geom_id]} came within "
                    f"{self.danger_zone_clearance_m * 1000:.1f} mm of danger_zone "
                    f"(separation={distance * 1000:.3f} mm)",
                )

    def check_step_safety(self) -> None:
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
        self.check_safety()
        global_penetration_limit = float(self.thresholds["penetration_limit_m"])
        pair_exceptions = self.thresholds.get("contact_penetration_exceptions_m", {})
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pair = " <-> ".join(sorted((self.geom_labels[geom1], self.geom_labels[geom2])))
            body1, body2 = int(self.model.geom_bodyid[geom1]), int(self.model.geom_bodyid[geom2])
            body_pair = " <-> ".join(
                sorted((self.body_name(body1), self.body_name(body2)))
            )
            distance = float(contact.dist)
            exception_key = pair if pair in pair_exceptions else body_pair
            allowed_penetration = float(
                pair_exceptions.get(exception_key, global_penetration_limit)
            )
            if distance < -allowed_penetration:
                raise M2Failure(
                    "SIMULATION_ERROR",
                    f"Contact penetration exceeded {allowed_penetration * 1000:.1f} mm "
                    f"for {pair}: {distance * 1000:.3f} mm",
                )
            if exception_key in pair_exceptions and distance < -global_penetration_limit:
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

    def _assert_held_integrity(
        self,
        require_transport_clearance: bool = False,
        require_bilateral_contact: bool = False,
    ) -> None:
        sides = self.finger_contact_sides()
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

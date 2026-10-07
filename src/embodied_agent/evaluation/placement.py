"""Independent placement scoring from measured simulator state."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..simulation.errors import M2Failure
from ..simulation.model import warning_counts


class PlacementEvaluationMixin:
    """Score the measured footprint, support, contact and stability window."""

    def _postconditions(self) -> dict[str, Any]:
        self.update_contact_metrics()
        self.current_contacts()
        corners = self.cube_corners()
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
        linear_speed, angular_speed = self.cube_speed()
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
            "danger_zone_clear": self.danger_zone_violation is None,
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
            "danger_zone_clear",
            "finite_state",
        )
        check["all_instantaneous_conditions"] = all(check[key] for key in condition_keys)
        return check

    def verify_goal(self) -> tuple[int, dict[str, Any]]:
        """Independently verify stable, supported placement against the selected target."""
        self.stage = "post_place_stability"
        self.record_event("phase_start", {"required_stable_steps": self.thresholds["stable_steps"]})
        stable_required = int(self.thresholds["stable_steps"])
        stable_steps_observed = 0
        final_check: dict[str, Any] = {}
        for _ in range(int(self.thresholds["placement_settle_max_steps"])):
            self.step()
            final_check = self.postconditions()
            if final_check["all_instantaneous_conditions"]:
                stable_steps_observed += 1
                if stable_steps_observed >= stable_required:
                    break
            else:
                stable_steps_observed = 0
        final_check = self.postconditions()
        self.record_event(
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
                    "danger_zone_clear",
                    "finite_state",
                )
                if not final_check[key]
            ]
            raise M2Failure(
                "GOAL_NOT_MET",
                "Post-place conditions did not remain true for the required window: "
                + ", ".join(false_conditions),
            )
        return stable_steps_observed, final_check

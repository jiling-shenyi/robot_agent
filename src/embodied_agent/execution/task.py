"""The existing bounded deterministic M2 task lifecycle."""

from __future__ import annotations

import math
import traceback
from typing import Any

from ..simulation.errors import M2Failure, ViewerClosed


class TaskExecutionMixin:
    """Coordinate existing skills, safety checks and independent goal scoring."""

    def initialize_task(self) -> dict[str, Any]:
        """Prepare, settle, and validate the selected target before any skill runs."""
        self.prepare()
        self.hold("scene_settle", int(self.thresholds["scene_settle_steps"]))
        self.target_center = self.data.geom_xpos[self.target_geom_id].copy()
        initial_check = self.postconditions()
        start_cube = self.data.xpos[self.cube_body_id].copy()
        if (
            abs(start_cube[2] - (self.table_top_z + self.cube_half_size[2])) > 0.003
            or not initial_check["table_supported"]
            or initial_check["cube_linear_speed_m_s"] > 0.01
        ):
            raise M2Failure("PRECONDITION_FAILED", "Cube did not settle on the table")
        if initial_check["full_projected_footprint_inside_target_with_margin"]:
            raise M2Failure("PRECONDITION_FAILED", "Cube starts inside the selected target")
        return self.snapshot()

    def record_failure(self, error_code: str, message: str) -> None:
        """Record an episode-level failure after a monitored action raises."""
        self.record_event("episode_failed", {"error_code": error_code, "message": message})

    def build_result(
        self,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
        stable_steps_observed: int = 0,
        postconditions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create the common M2/M3 episode record from measured simulator state."""
        final_check = postconditions
        if final_check is None:
            try:
                final_check = self.postconditions()
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
            "minimum_danger_zone_separation_m": (
                self.minimum_danger_zone_separation
                if math.isfinite(self.minimum_danger_zone_separation)
                else None
            ),
            "danger_zone_violation": self.danger_zone_violation,
            "stable_steps_observed": stable_steps_observed,
            "postconditions": final_check,
            "events": self.events,
        }
        return self.result

    def run(self) -> dict[str, Any]:
        status = "FAILED"
        error_code: str | None = None
        error_message: str | None = None
        stable_steps_observed = 0
        final_check: dict[str, Any] | None = None
        try:
            self.initialize_task()
            self.pick_skill()
            self.place_skill()
            stable_steps_observed, final_check = self.verify_goal()
            status = "SUCCESS"
        except ViewerClosed:
            self.close()
            raise
        except M2Failure as error:
            error_code, error_message = error.code, str(error)
            self.record_failure(error_code, error_message)
        except Exception as error:
            error_code, error_message = "SIMULATION_ERROR", f"{type(error).__name__}: {error}"
            self.record_event(
                "episode_failed",
                {
                    "error_code": error_code,
                    "message": error_message,
                    "traceback": traceback.format_exc(),
                },
            )
        self.build_result(status, error_code, error_message, stable_steps_observed, final_check)
        title = "M2 TASK SUCCESS" if status == "SUCCESS" else f"M2 TASK FAILED: {error_code}"
        self.set_viewer_status(title, error_message or "All postconditions remained satisfied")
        return self.result

"""Measured Panda observations and shared physical validation errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Observation:
    obs_id: int
    sim_time_s: float
    ee_pose_actual: dict[str, list[float]]
    gripper_opening_m: list[float]
    cube: dict[str, Any]
    target_regions: dict[str, dict[str, Any]]
    danger_zone: dict[str, Any]
    held_estimate: bool | None
    completed_steps: list[str]
    previous_error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

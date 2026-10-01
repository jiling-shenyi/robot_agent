"""Small, strict contracts for the M3 language-to-skill boundary."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np


class ContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TaskGoal:
    instruction: str
    object_id: str
    target_id: str
    resolver_version: str


@dataclass(frozen=True)
class PlanStep:
    skill: str
    object_id: str | None = None
    target_id: str | None = None
    approach_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class Plan:
    schema_version: int
    base_obs_id: int
    steps: tuple[PlanStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_obs_id": self.base_obs_id,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class SkillResult:
    skill: str
    status: str
    error_code: str | None
    obs_before_id: int
    obs_after_id: int
    sim_steps_used: int
    measured_metrics: dict[str, Any]


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


def resolve_task(instruction: str, config: dict[str, Any]) -> TaskGoal:
    text = " ".join(instruction.strip().split())
    if not text or len(text) > int(config["language"]["max_instruction_chars"]):
        raise ContractError("INVALID_TASK", "Instruction is empty or exceeds the configured length limit")
    normalized = text.casefold()
    language = config["language"]
    for phrase in language["unsupported_phrases"]:
        phrase = str(phrase).casefold()
        found_unsupported = (
            bool(re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", normalized))
            if phrase.isascii() and phrase.isalpha()
            else phrase in normalized
        )
        if found_unsupported:
            raise ContractError("INVALID_TASK", f"Unsupported task action: {phrase}")

    has_supported_action = False
    for raw_verb in language["supported_action_phrases"]:
        verb = str(raw_verb).casefold()
        if (
            re.search(rf"(?<![a-z]){re.escape(verb)}(?![a-z])", normalized)
            if verb.isascii() and verb.isalpha()
            else verb in normalized
        ):
            has_supported_action = True
            break
    if not has_supported_action:
        raise ContractError("INVALID_TASK", "Instruction must explicitly request placing or moving the cube")

    found: list[str] = []
    for target_id, aliases in language["target_aliases"].items():
        for alias in aliases:
            if re.search(str(alias), text, flags=re.IGNORECASE):
                found.append(target_id)
                break
    found = sorted(set(found))
    if not found:
        raise ContractError("INVALID_TASK", "Instruction must name exactly one supported target area")
    if len(found) != 1:
        raise ContractError("INVALID_TASK", "Instruction mentions conflicting target areas")
    return TaskGoal(text, "cube", found[0], str(language["resolver_version"]))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_plan(raw: str, goal: TaskGoal, expected_obs_id: int, config: dict[str, Any]) -> Plan:
    max_chars = int(config["planner"]["max_plan_chars"])
    if not raw or len(raw) > max_chars:
        raise ContractError("INVALID_PLAN", "Planner response is empty or exceeds the plan size limit")
    try:
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ContractError("INVALID_PLAN", f"Planner response is not strict JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "base_obs_id", "steps"}:
        raise ContractError("INVALID_PLAN", "Plan must contain exactly schema_version, base_obs_id, and steps")
    if type(data["schema_version"]) is not int or data["schema_version"] != int(
        config["planner"]["schema_version"]
    ):
        raise ContractError("INVALID_PLAN", "Unsupported plan schema_version")
    if type(data["base_obs_id"]) is not int or data["base_obs_id"] != expected_obs_id:
        raise ContractError("STALE_OBSERVATION", "Plan base_obs_id does not match the planning observation")
    steps_data = data["steps"]
    if not isinstance(steps_data, list) or len(steps_data) != 2:
        raise ContractError("INVALID_PLAN", "M3 requires exactly one pick followed by one place")

    steps: list[PlanStep] = []
    expected_shapes = (
        {"skill", "object_id", "approach_mode"},
        {"skill", "target_id"},
    )
    for index, (raw_step, expected_shape) in enumerate(zip(steps_data, expected_shapes, strict=True)):
        if not isinstance(raw_step, dict) or set(raw_step) != expected_shape:
            raise ContractError("INVALID_PLAN", f"Step {index} has unknown, missing, or invalid fields")
        if index == 0:
            if (
                raw_step["skill"] != "pick"
                or raw_step["object_id"] != "cube"
                or raw_step["approach_mode"] != "top"
            ):
                raise ContractError("INVALID_PLAN", "The first step must be pick(cube, approach_mode=top)")
            steps.append(PlanStep("pick", object_id="cube", approach_mode="top"))
        else:
            if raw_step["skill"] != "place":
                raise ContractError("INVALID_PLAN", "The second step must be place(target)")
            if raw_step["target_id"] != goal.target_id:
                raise ContractError("GOAL_MISMATCH", "Plan target differs from the immutable TaskGoal")
            steps.append(PlanStep("place", target_id=goal.target_id))
    return Plan(int(data["schema_version"]), int(data["base_obs_id"]), tuple(steps))


def make_observation(
    episode: Any,
    obs_id: int,
    completed_steps: list[str],
    previous_error: dict[str, str] | None = None,
) -> Observation:
    data, model = episode.data, episode.model
    ee_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(ee_quaternion, data.site_xmat[episode.ee_site_id])
    cube_velocity, cube_angular_velocity = episode._cube_speed()
    cube_position = data.xpos[episode.cube_body_id].copy()
    target_regions: dict[str, dict[str, Any]] = {}
    for name in ("target_a", "target_b"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"Missing target region: {name}")
        target_regions[name] = {
            "center_xyz_m": data.geom_xpos[geom_id].copy().tolist(),
            "half_size_xyz_m": model.geom_size[geom_id].copy().tolist(),
            "aliases": ["green"] if name == "target_a" else ["blue"],
        }
    zone_id = episode.danger_zone_geom_id
    side_count = len(episode._finger_contact_sides())
    lifted = cube_position[2] >= episode.table_top_z + episode.cube_half_size[2] + 0.03
    held: bool | None
    if side_count >= 2 and lifted and episode.grasp_relative_position is not None:
        held = True
    elif not lifted:
        held = False
    else:
        held = None
    return Observation(
        obs_id=obs_id,
        sim_time_s=float(data.time),
        ee_pose_actual={
            "position_xyz_m": data.site_xpos[episode.ee_site_id].copy().tolist(),
            "quaternion_wxyz": ee_quaternion.tolist(),
        },
        gripper_opening_m=data.qpos[episode.finger_qpos_addresses].copy().tolist(),
        cube={
            "object_id": "cube",
            "position_xyz_m": cube_position.tolist(),
            "linear_speed_m_s": cube_velocity,
            "angular_speed_rad_s": cube_angular_velocity,
            "held_estimate": held,
            "finger_contact_side_count": side_count,
        },
        target_regions=target_regions,
        danger_zone={
            "center_xyz_m": data.geom_xpos[zone_id].copy().tolist(),
            "half_size_xyz_m": model.geom_size[zone_id].copy().tolist(),
            "required_clearance_m": episode.danger_zone_clearance_m,
        },
        held_estimate=held,
        completed_steps=list(completed_steps),
        previous_error=previous_error,
    )

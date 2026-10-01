"""Bounded M3 state machine, deterministic motion precheck, and run evidence."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import ContractError, Observation, Plan, TaskGoal, make_observation, parse_plan, resolve_task
from .planner import Planner, PlannerError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class M3RunWriter:
    """Append-only event, episode, and trajectory evidence for one run."""

    TRAJECTORY_FIELDS = (
        "run_id",
        "episode_id",
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
        "minimum_danger_zone_separation_so_far_m",
        "danger_zone_clear",
        "warnings",
        "finite_state",
    )

    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = output_dir
        self.run_id = run_id
        self._event_sequence = 0
        self._trajectory_new = not (output_dir / "trajectory.csv").exists()

    def emit(
        self,
        episode_id: str,
        event: str,
        state: str,
        detail: dict[str, Any] | None = None,
        observation: Observation | None = None,
    ) -> None:
        self._event_sequence += 1
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "sequence": self._event_sequence,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            "state": state,
            "remaining_observation_id": observation.obs_id if observation else None,
        }
        if observation is not None:
            row["observation"] = observation.to_dict()
        if detail:
            row["detail"] = detail
        path = self.output_dir / "events.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":")) + "\n")

    def append_episode(self, result: dict[str, Any]) -> None:
        path = self.output_dir / "episodes.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_jsonable(result), ensure_ascii=False, separators=(",", ":")) + "\n")

    def append_trajectory(self, episode_id: str, samples: list[dict[str, Any]]) -> None:
        path = self.output_dir / "trajectory.csv"
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.TRAJECTORY_FIELDS)
            if self._trajectory_new:
                writer.writeheader()
                self._trajectory_new = False
            for sample in samples:
                row = {"run_id": self.run_id, "episode_id": episode_id, **sample}
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    }
                )

    def finish(self, results: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
        statuses = {name: sum(row["status"] == name for row in results) for name in ("SUCCESS", "FAILED", "ABORTED")}
        summary = {
            **manifest,
            "run_id": self.run_id,
            "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "episode_count": len(results),
            "success_count": statuses["SUCCESS"],
            "failure_count": statuses["FAILED"],
            "aborted_count": statuses["ABORTED"],
            "not_run_count": sum(row["status"] == "NOT_RUN" for row in results),
            "all_pass": bool(results) and statuses["SUCCESS"] == len(results),
            "episodes": [
                {
                    "episode_id": row["episode_id"],
                    "case_id": row.get("case_id"),
                    "status": row["status"],
                    "error_code": row.get("error_code"),
                    "target_id": row.get("target_id"),
                    "expected_target_id": row.get("expected_target_id"),
                    "step_count": row.get("step_count", 0),
                    "planner_kind": row.get("planner_kind"),
                    "llm_request_count": row.get("llm_request_count", 0),
                    "token_usage": row.get("token_usage"),
                    "wall_time_s": row.get("wall_time_s"),
                    "postconditions": row.get("postconditions"),
                }
                for row in results
            ],
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary


def _point_to_box_distance(point: np.ndarray, center: np.ndarray, half_size: np.ndarray) -> float:
    outside = np.maximum(np.abs(point - center) - half_size, 0.0)
    return float(np.linalg.norm(outside))


def check_skill_path(episode: Any, skill: str, config: dict[str, Any]) -> dict[str, Any]:
    """Conservatively sample the mocap path against an expanded visible danger box."""
    start = episode.data.mocap_pos[episode.mocap_id].copy()
    cube = episode.data.xpos[episode.cube_body_id].copy()
    center = episode.data.geom_xpos[episode.danger_zone_geom_id].copy()
    half = episode.model.geom_size[episode.danger_zone_geom_id].copy()
    path_config = config["path_precheck"]
    expansion = float(path_config["danger_zone_clearance_m"]) + float(
        path_config["ee_payload_envelope_radius_m"]
    )
    expanded_half = half + expansion
    spacing = float(path_config["sample_spacing_m"])
    if skill == "pick":
        pregrasp = cube.copy()
        pregrasp[2] += float(episode.thresholds["pregrasp_clearance_m"])
        grasp = cube.copy()
        lift = grasp.copy()
        lift[2] += float(episode.thresholds["lift_height_m"])
        waypoints = [start, pregrasp, grasp, lift]
    elif skill == "place":
        transit = start.copy()
        transit[:2] = episode.target_center[:2]
        place = transit.copy()
        place[2] = (
            episode.table_top_z
            + float(episode.cube_half_size[2])
            + float(episode.thresholds["place_clearance_m"])
        )
        retreat = place.copy()
        retreat[2] += float(episode.thresholds["pregrasp_clearance_m"])
        waypoints = [start, transit, place, retreat]
    else:
        raise ContractError("INVALID_PLAN", f"No path generator exists for skill {skill}")

    minimum_distance = math.inf
    checked_samples = 0
    for first, second in zip(waypoints, waypoints[1:]):
        length = float(np.linalg.norm(second - first))
        count = max(1, int(math.ceil(length / spacing)))
        for index in range(count + 1):
            point = first + (second - first) * (index / count)
            distance = _point_to_box_distance(point, center, expanded_half)
            minimum_distance = min(minimum_distance, distance)
            checked_samples += 1
            if distance <= 0.0:
                raise ContractError(
                    "PATH_REJECTED",
                    f"{skill} mocap path enters danger-zone envelope "
                    f"(centerline clearance={distance:.6f} m at {point.tolist()})",
                )
    return {
        "skill": skill,
        "waypoints_m": [point.tolist() for point in waypoints],
        "danger_zone_expansion_m": expansion,
        "minimum_sampled_centerline_clearance_m": minimum_distance,
        "samples_checked": checked_samples,
    }


class M3EpisodeRunner:
    def __init__(
        self,
        episode_factory: Callable[..., Any],
        config: dict[str, Any],
        writer: M3RunWriter,
        planner: Planner,
        run_id: str,
    ):
        self.episode_factory = episode_factory
        self.config = config
        self.writer = writer
        self.planner = planner
        self.run_id = run_id

    def run_episode(
        self,
        instruction: str,
        scenario: dict[str, Any] | None,
        thresholds: dict[str, Any],
        output_dir: Path,
        enable_viewer: bool,
        show_ui: bool,
        episode_id: str,
        case_id: str | None = None,
        expected_target_id: str | None = None,
    ) -> dict[str, Any]:
        started = dt.datetime.now(dt.timezone.utc)
        goal: TaskGoal | None = None
        episode: Any | None = None
        observation: Observation | None = None
        plan: Plan | None = None
        raw_plan: str | None = None
        completed_steps: list[str] = []
        planner_requests = 0
        token_usage: dict[str, Any] | None = None
        planner_model: str | None = None
        error_code: str | None = None
        error_message: str | None = None
        status = "FAILED"
        stable_steps = 0
        final_check: dict[str, Any] | None = None
        sim_steps = 0
        samples: list[dict[str, Any]] = []
        last_observation_id = 0
        previous_error: dict[str, str] | None = None

        self.writer.emit(episode_id, "state_transition", "INIT", {"case_id": case_id})
        try:
            goal = resolve_task(instruction, self.config)
            if expected_target_id is not None and expected_target_id != goal.target_id:
                raise ContractError(
                    "GOAL_MISMATCH", "Parsed target differs from the preregistered evaluation label"
                )
            if scenario is None:
                raise ContractError("INVALID_TASK", "No scenario was selected for this task")
            scenario_target_id = f"target_{scenario['target']}"
            if scenario_target_id != goal.target_id:
                raise ContractError("GOAL_MISMATCH", "Scenario target differs from the parsed TaskGoal")

            goal_dict = {
                "instruction": goal.instruction,
                "object_id": goal.object_id,
                "target_id": goal.target_id,
                "resolver_version": goal.resolver_version,
            }
            episode = self.episode_factory(scenario, thresholds, output_dir, enable_viewer, show_ui)
            episode.step_budget_limit = min(
                int(self.config["budgets"]["max_episode_steps"]),
                int(thresholds["max_episode_steps"]),
            )
            self.writer.emit(episode_id, "state_transition", "OBSERVE", {"goal": goal_dict})
            episode.initialize_task()
            last_observation_id += 1
            observation = make_observation(episode, last_observation_id, completed_steps)
            self.writer.emit(episode_id, "observation_created", "OBSERVE", observation=observation)

            if planner_requests >= int(self.config["budgets"]["planner_requests_per_episode"]):
                raise ContractError("BUDGET_EXHAUSTED", "Planner request budget is exhausted")
            planner_requests += 1
            self.writer.emit(episode_id, "state_transition", "PLAN", observation=observation)
            episode.set_viewer_status("M3 PLANNING", f"Planner: {self.planner.kind}; target: {goal.target_id}")
            response = self.planner.plan(goal_dict, observation.to_dict())
            planner_model = response.response_model or response.requested_model
            token_usage = response.usage
            raw_plan = response.content
            if response.finish_reason == "length":
                raise PlannerError("TRUNCATED_RESPONSE", "Planner output reached its token limit", {"usage": token_usage})
            if episode.viewer is not None and not episode.viewer.is_running():
                class M3ViewerClosed(RuntimeError):
                    pass

                raise M3ViewerClosed("Viewer closed while the planner was running")
            self.writer.emit(
                episode_id,
                "planner_response",
                "VALIDATE",
                {
                    "planner_kind": response.planner_kind,
                    "requested_model": response.requested_model,
                    "response_model": response.response_model,
                    "usage": token_usage,
                    "latency_s": response.latency_s,
                    "finish_reason": response.finish_reason,
                    "request_id": response.request_id,
                    "response_text": raw_plan,
                },
            )
            plan = parse_plan(raw_plan, goal, observation.obs_id, self.config)
            self.writer.emit(episode_id, "plan_validated", "VALIDATE", {"plan": plan.to_dict()})

            if len(plan.steps) > int(self.config["budgets"]["max_skill_calls"]):
                raise ContractError("BUDGET_EXHAUSTED", "Plan exceeds the skill-call budget")
            episode.set_viewer_status("M3 PLAN VALID", f"Target: {goal.target_id}; skills: pick then place")
            self.writer.emit(episode_id, "state_transition", "PRE_SKILL", observation=observation)
            for step in plan.steps:
                last_observation_id += 1
                observation = make_observation(
                    episode, last_observation_id, completed_steps, previous_error
                )
                if step.skill == "pick":
                    if observation.held_estimate is not False:
                        raise ContractError("PRECONDITION_FAILED", "Pick requires the cube not to be held")
                elif step.skill == "place":
                    if observation.held_estimate is not True:
                        raise ContractError("PRECONDITION_FAILED", "Place requires current bilateral held evidence")
                else:
                    raise ContractError("INVALID_PLAN", f"Skill is not allow-listed: {step.skill}")

                path_check = check_skill_path(episode, step.skill, self.config)
                self.writer.emit(
                    episode_id,
                    "pre_skill_passed",
                    "PRE_SKILL",
                    {"skill": step.skill, "path_check": path_check},
                    observation,
                )
                if len(completed_steps) >= int(self.config["budgets"]["max_skill_calls"]):
                    raise ContractError("BUDGET_EXHAUSTED", "Skill-call budget is exhausted")

                call_started = episode.total_steps
                skill_budget = int(
                    self.config["budgets"][
                        "pick_skill_steps" if step.skill == "pick" else "place_skill_steps"
                    ]
                )
                global_limit = min(
                    int(self.config["budgets"]["max_episode_steps"]),
                    int(thresholds["max_episode_steps"]),
                )
                episode.step_budget_limit = min(global_limit, call_started + skill_budget)
                self.writer.emit(episode_id, "state_transition", "EXECUTE", {"skill": step.skill})
                episode.set_viewer_status("M3 EXECUTING", f"Skill: {step.skill}; target: {goal.target_id}")
                try:
                    if step.skill == "pick":
                        metrics = episode.pick_skill()
                    else:
                        metrics = episode.place_skill()
                finally:
                    episode.step_budget_limit = global_limit
                steps_used = episode.total_steps - call_started
                completed_steps.append(step.skill)
                last_observation_id += 1
                after_observation = make_observation(
                    episode, last_observation_id, completed_steps
                )
                if step.skill == "pick" and after_observation.held_estimate is not True:
                    raise ContractError("GRASP_EMPTY", "Pick postcondition lacks measured held evidence")
                self.writer.emit(
                    episode_id,
                    "post_skill_passed",
                    "VERIFY",
                    {
                        "skill_result": {
                            "skill": step.skill,
                            "status": "success",
                            "error_code": None,
                            "obs_before_id": observation.obs_id,
                            "obs_after_id": after_observation.obs_id,
                            "sim_steps_used": steps_used,
                            "measured_metrics": metrics,
                        }
                    },
                    after_observation,
                )
                observation = after_observation
                self.writer.emit(
                    episode_id,
                    "state_transition",
                    "PRE_SKILL" if len(completed_steps) < len(plan.steps) else "GOAL_CHECK",
                    observation=observation,
                )

            self.writer.emit(episode_id, "state_transition", "GOAL_CHECK", observation=observation)
            episode.set_viewer_status("M3 VERIFYING", "Checking independent placement postconditions")
            stable_steps, final_check = episode.verify_goal()
            if not final_check.get("all_instantaneous_conditions"):
                raise ContractError("GOAL_NOT_MET", "Independent task postconditions are not all satisfied")
            status = "SUCCESS"
            episode.set_viewer_status("M3 TASK SUCCESS", f"Independently verified target: {goal.target_id}")
            self.writer.emit(
                episode_id,
                "state_transition",
                "SUCCESS",
                {"stable_steps_observed": stable_steps, "postconditions": final_check},
            )
        except ContractError as exc:
            error_code, error_message = exc.code, str(exc)
            previous_error = {"code": error_code, "message": error_message}
            self.writer.emit(episode_id, "runtime_rejected", "FAILED", {**previous_error, "executed_steps": completed_steps})
        except PlannerError as exc:
            error_code, error_message = exc.code, str(exc)
            previous_error = {"code": error_code, "message": error_message}
            self.writer.emit(
                episode_id,
                "planner_failed",
                "FAILED",
                {**previous_error, "details": exc.details, "planner_requests": planner_requests},
            )
        except Exception as exc:
            if type(exc).__name__ in {"ViewerClosed", "M3ViewerClosed"}:
                status = "ABORTED"
                error_code, error_message = "VIEWER_CLOSED", "Viewer closed before the episode completed"
                self.writer.emit(episode_id, "episode_aborted", "ABORTED", {"code": error_code})
            else:
                error_code = getattr(exc, "code", "SIMULATION_ERROR")
                error_message = str(exc)
                previous_error = {"code": error_code, "message": error_message}
                self.writer.emit(
                    episode_id,
                    "episode_failed",
                    "FAILED",
                    {"code": error_code, "message": error_message, "executed_steps": completed_steps},
                )

        if episode is not None:
            sim_steps = int(episode.total_steps)
            samples = episode.samples
            if status != "SUCCESS":
                episode.record_failure(error_code or status, error_message or status)
                episode.set_viewer_status(
                    "M3 EPISODE " + status,
                    f"{error_code or status}: {error_message or 'No additional detail'}",
                )
            m2_result = episode.build_result(
                status,
                error_code,
                error_message,
                stable_steps,
                final_check,
            )
            if status == "SUCCESS" and m2_result["postconditions"].get(
                "full_projected_footprint_inside_target_with_margin"
            ) is not True:
                status = "FAILED"
                error_code = "GOAL_MISMATCH"
                error_message = "Measured cube footprint is outside the selected target"
                episode.record_failure(error_code, error_message)
                episode.set_viewer_status("M3 TASK FAILED", error_message)
                m2_result = episode.build_result(status, error_code, error_message, stable_steps)
            self.writer.emit(
                episode_id,
                "episode_finalized",
                status,
                {"error_code": error_code, "step_count": sim_steps, "target_id": goal.target_id if goal else None},
            )
        else:
            m2_result = {}

        ended = dt.datetime.now(dt.timezone.utc)
        result = {
            "run_id": self.run_id,
            "episode_id": episode_id,
            "case_id": case_id,
            "instruction": instruction,
            "goal": {
                "object_id": goal.object_id,
                "target_id": goal.target_id,
                "resolver_version": goal.resolver_version,
            }
            if goal
            else None,
            "target_id": goal.target_id if goal else None,
            "expected_target_id": expected_target_id,
            "scenario_id": scenario["scenario_id"] if scenario else None,
            "seed": scenario["seed"] if scenario else None,
            "planner_kind": self.planner.kind,
            "planner_model": planner_model,
            "planner_requests": planner_requests,
            "llm_request_count": planner_requests if self.planner.kind == "llm" else 0,
            "token_usage": token_usage,
            "planner_response_text": raw_plan,
            "plan": plan.to_dict() if plan else None,
            "completed_steps": completed_steps,
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            "step_count": sim_steps,
            "stable_steps_observed": stable_steps,
            "postconditions": final_check or m2_result.get("postconditions"),
            "m2_metrics": {
                key: m2_result.get(key)
                for key in (
                    "max_ee_position_error_m",
                    "minimum_contact_distance_m",
                    "minimum_danger_zone_separation_m",
                    "danger_zone_violation",
                    "penetration_exceptions_seen_m",
                )
            }
            if m2_result
            else {},
            "started_at_utc": started.isoformat(),
            "finished_at_utc": ended.isoformat(),
            "wall_time_s": (ended - started).total_seconds(),
            "episode_events": m2_result.get("events", []),
        }
        self.writer.append_episode(result)
        self.writer.append_trajectory(episode_id, samples)
        if episode is not None and enable_viewer:
            episode.wait_for_viewer_close()
        return result

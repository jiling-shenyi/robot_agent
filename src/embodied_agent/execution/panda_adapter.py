"""Panda mechanics behind the shared instruction/runtime interface."""
from __future__ import annotations

import copy
import hashlib
import math
import uuid
from typing import Any

import mujoco

from embodied_agent.contracts import ContractError
from embodied_agent.models.budget import current_budget
from embodied_agent.safety.path_precheck import check_skill_path
from embodied_agent.simulation.observations import make_observation


class PandaInstructionAdapter:
    robot_kind = "panda"

    def __init__(self, episode: Any, config: dict[str, Any]):
        self.episode, self.world, self.config = episode, episode.world, config
        self._obs_id = 0
        self.action_results: list[dict] = []
        self.placements: list[dict] = []
        self._permits: set[str] = set()
        self.episode.step_budget_limit = episode.total_steps + int(config["budgets"]["max_episode_steps"])
        episode.prepare()
        episode.hold("scene_settle", int(episode.thresholds["scene_settle_steps"]))
        initial = episode.postconditions()
        held = make_observation(episode, 0, []).held_estimate is True
        if not held and (not initial["table_supported"] or initial["cube_linear_speed_m_s"] > .01):
            raise ContractError("PRECONDITION_FAILED", "Object must be supported and settled before planning")

    def __getattr__(self, name):
        return getattr(self.episode, name)

    def observe(self) -> dict:
        ep = self.episode
        self._obs_id += 1
        measured = make_observation(ep, self._obs_id, [r["action"]["skill"] for r in self.action_results])
        position = ep.data.xpos[ep.cube_body_id].copy().tolist()
        targets = {}
        for key in self.world.targets:
            geom = ep.model.geom(key).id
            targets[key] = {"position_m": ep.data.geom_xpos[geom].tolist(),
                            "half_size_m": ep.model.geom_size[geom].tolist()}
        held = "cube" if measured.held_estimate is True else None
        return {"schema_kind": "desktop_snapshot", "map_id": self.world.map_id,
                "map_revision": self.world.revision, "world_version": ep.total_steps,
                "obs_id": self._obs_id, "sim_time_s": float(ep.data.time),
                "cube_xyz_m": position, "cube_position_m": position,
                "objects": {"cube": {"category": "cube", "position_m": position,
                    "half_size_m": ep.cube_half_size.tolist(), "mass_kg": float(ep.model.body_mass[ep.cube_body_id]),
                    "support": "gripper" if held else "table", "states": {},
                    "operations": ["inspect", "pick", "place"]}},
                "robot": {"held_object": held, "held_verified": measured.held_estimate,
                          "position_m": ep.data.site_xpos[ep.ee_site_id].tolist()},
                "targets": targets, "target_regions": targets,
                "danger_zone": self.world.danger_zone.to_dict(),
                "grasp_evidence": measured.cube,
                "evidence": self.evaluate_goal_evidence([])}

    def evaluate_goal_evidence(self, goals) -> dict:
        ep = self.episode
        placements = []
        previous_target = ep.target_geom_id
        try:
            for old in self.placements:
                ep.target_geom_id = ep.model.geom(old["target_id"]).id
                check = ep.postconditions()
                placements.append({**old, "passed": bool(old["passed"] and check["all_instantaneous_conditions"]),
                                   "predicates": check})
            # Verify already satisfied destinations with the same guarded,
            # per-step stable-window evaluator as a newly performed place.
            for goal in goals or []:
                target = goal.get("target_id")
                if (goal.get("predicate") not in {"supported_on", "inside"}
                        or goal.get("object_id") != "cube" or target not in self.world.targets
                        or any(p["target_id"] == target and p["passed"] for p in placements)):
                    continue
                ep.target_geom_id = ep.model.geom(target).id
                if ep.postconditions()["all_instantaneous_conditions"]:
                    stable_steps, measured = ep.verify_goal()
                    verified = {"object_id": "cube", "support_id": "table", "target_id": target,
                                "passed": True, "stable_window_passed": True,
                                "stable_steps": stable_steps, "predicates": measured,
                                "source": "independent_existing_state_verification"}
                    self.placements.append(verified)
                    placements.append(copy.deepcopy(verified))
        finally:
            ep.target_geom_id = previous_target
        check = ep.postconditions()
        side_count = len(ep.finger_contact_sides())
        return {"placements": placements, "action_results": copy.deepcopy(self.action_results),
                "objects": {"cube": {"released": bool(check["no_gripper_contact"]),
                                      "stable": bool(placements and any(p["passed"] for p in placements)),
                                      "finger_contact_side_count": side_count,
                                      "lift_verified": bool(ep.data.xpos[ep.cube_body_id][2] > ep.table_top_z + ep.cube_half_size[2] + .03),
                                      "robot_contact": not check["no_gripper_contact"],
                                      "supports": ["table"] if check["table_supported"] else []}},
                "performed_methods": ["controlled_place"] if placements else []}

    def action_satisfied(self, action: dict, previous: dict) -> bool:
        if action.get("skill") == "pick":
            return self.observe()["robot"]["held_verified"] is True
        if action.get("skill") == "place":
            return any(row["target_id"] == action.get("target_id") and row["passed"]
                       for row in self.evaluate_goal_evidence([])["placements"])
        return False

    def _fingerprint(self) -> str:
        ep = self.episode
        return hashlib.sha256(ep.data.qpos.tobytes() + ep.data.qvel.tobytes() +
                              ep.data.geom_xpos.tobytes() + ep.model.geom_size.tobytes() +
                              ep.model.geom_pos.tobytes() + ep.data.ctrl.tobytes() +
                              ep.data.mocap_pos.tobytes() + ep.data.mocap_quat.tobytes()).hexdigest()

    def review_next_action(self, action: dict) -> dict:
        if not isinstance(action, dict):
            raise ContractError("INVALID_PLAN", "Action must be an object")
        ep = self.episode
        skill = action.get("skill")
        shapes = {"pick": [{"skill", "object_id"}, {"skill", "object_id", "approach_mode"}],
                  "place": [{"skill", "target_id"}], "stop": [{"skill"}], "wait": [{"skill", "seconds"}]}
        if skill not in shapes or set(action) not in shapes[skill]:
            raise ContractError("INVALID_PLAN", "Unsupported Panda action parameters")
        observation = self.observe()
        path = None
        if skill == "pick":
            if action["object_id"] not in observation["objects"] or action.get("approach_mode", "top") != "top":
                raise ContractError("UNSUPPORTED_SKILL", "Panda top grasp requires a currently observed manipulable object")
            if observation["robot"]["held_verified"] is not False:
                raise ContractError("PRECONDITION_FAILED", "Pick requires measured empty-hand state")
        elif skill == "place":
            target = action["target_id"]
            if target not in self.world.targets:
                raise ContractError("UNKNOWN_TARGET", "Destination is not in the current region catalogue")
            if observation["robot"]["held_verified"] is not True:
                raise ContractError("PRECONDITION_FAILED", "Place requires measured bilateral holding")
            ep.target_geom_id = ep.model.geom(target).id
            ep.target_center = ep.data.geom_xpos[ep.target_geom_id].copy()
        elif skill == "wait":
            seconds = action["seconds"]
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 10:
                raise ContractError("INVALID_PLAN", "Wait requires finite 0–10 seconds")
        if skill in {"pick", "place"}:
            path = check_skill_path(ep, skill, self.config)
        permit = uuid.uuid4().hex
        self._permits.add(permit)
        return {"action": copy.deepcopy(action), "state": self._fingerprint(), "path": path,
                "map_revision": self.world.revision, "permit": permit}

    def run_action(self, action: dict, *, record_event=None) -> dict:
        ep = self.episode
        started = ep.total_steps
        invoked = False
        try:
            review = self.review_next_action(action)
            if review["state"] != self._fingerprint() or review["map_revision"] != self.world.revision:
                raise ContractError("STALE_ACTION", "State changed after execution review")
            if record_event:
                record_event("action_review", {"action": action, "review": review}, self.observe())
                record_event("skill_started", {"action": action}, self.observe())
            budget = current_budget()
            if budget is not None:
                budget.check()
            if (review["permit"] not in self._permits or review["state"] != self._fingerprint()
                    or review["map_revision"] != self.world.revision):
                raise ContractError("STALE_ACTION", "Reviewed action state changed or permit was consumed")
            self._permits.remove(review["permit"])
            invoked = True
            if action["skill"] == "pick":
                start_z = float(ep.data.xpos[ep.cube_body_id][2])
                metrics = ep.pick_skill()
                grasp = self.evaluate_goal_evidence([])["objects"]["cube"]
                grasp.update(held_verified=self.observe()["robot"]["held_verified"],
                             achieved_lift_m=float(ep.data.xpos[ep.cube_body_id][2]) - start_z)
                evidence = {"skill": metrics, "grasp": grasp}
            elif action["skill"] == "place":
                evidence = ep.place_skill()
                stable_steps, check = ep.verify_goal()
                if not check["all_instantaneous_conditions"]:
                    raise ContractError("GOAL_NOT_MET", "Measured stable placement did not meet region goal")
                self.placements.append({"object_id": "cube", "support_id": "table", "target_id": action["target_id"],
                                        "passed": True, "stable_window_passed": True,
                                        "stable_steps": stable_steps, "predicates": check})
                evidence = {"skill": evidence, "placement": self.placements[-1]}
            else:
                evidence = self.hold(seconds=float(action.get("seconds", .1)))
            if action["skill"] == "pick" and self.observe()["robot"]["held_verified"] is not True:
                raise ContractError("GRASP_EMPTY", "Grasp lacks actual bilateral contact and lift")
            self.action_results.append({"action": copy.deepcopy(action), "success": True, "result": evidence})
            result = {"status": "SUCCESS", "action": action, "evidence": evidence,
                      "physics_steps": ep.total_steps - started, "executed": True, "observation": self.observe()}
            if record_event:
                record_event("post_skill_passed", {"action": action, "skill_result": result}, result["observation"])
            return result
        except Exception as exc:
            cleanup = []
            try:
                self.safe_stop()
            except Exception as stop_exc:
                cleanup.append({"stage": "safe_stop", "message": str(stop_exc)})
            try:
                observation = self.observe()
            except Exception as observation_exc:
                observation = None
                cleanup.append({"stage": "observation", "message": str(observation_exc)})
            result = {"status": "ABORTED" if type(exc).__name__ in {"ViewerClosed"} else "FAILED",
                      "action": action, "error_code": getattr(exc, "code", type(exc).__name__),
                      "error_message": str(exc), "error_evidence": getattr(exc, "details", {}),
                      "physics_steps": ep.total_steps - started, "executed": invoked,
                      "observation": observation, "cleanup_errors": cleanup,
                      "stop_failed": any(e["stage"] == "safe_stop" for e in cleanup)}
            if record_event:
                record_event("execution_failed", result, observation)
            return result

    def hold(self, seconds=.02) -> dict:
        ep = self.episode
        steps = max(1, math.ceil(seconds / ep.model.opt.timestep))
        ep.hold("instruction_hold", steps)
        return {"sim_time_s": float(ep.data.time), "held_side_count": len(ep.finger_contact_sides())}

    def safe_stop(self) -> dict:
        ep = self.episode
        callback = getattr(ep, "on_frame", None)
        limit = ep.step_budget_limit
        ep.on_frame = None
        try:
            ep.step_budget_limit = max(limit, ep.total_steps + math.ceil(.1 / ep.model.opt.timestep))
            ep.data.mocap_pos[ep.mocap_id] = ep.data.site_xpos[ep.ee_site_id]
            mujoco.mju_mat2Quat(ep.data.mocap_quat[ep.mocap_id], ep.data.site_xmat[ep.ee_site_id])
            return self.hold(seconds=.1)
        finally:
            ep.step_budget_limit = limit
            ep.on_frame = callback

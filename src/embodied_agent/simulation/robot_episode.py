"""Mobile robot implementation of the original Demo's live episode interface.

DemoSession owns maps, requests, evidence and lifecycle; DemoApp owns display.
This episode supplies Stretch physics, bounded skills and measured state only.
"""
from __future__ import annotations

import json
import math
import copy
import hashlib
import uuid
from pathlib import Path
from typing import Any, Callable

from embodied_agent.execution.home_contracts import HomeExecutionError, validate_home_plan
from embodied_agent.evaluation.home import placement_evidence
from embodied_agent.maps.home_schema import project_damage_event
from embodied_agent.paths import PROJECT_ROOT
from embodied_agent.safety.home import HomeSafety


class StretchDemoEpisode:
    robot_kind = "stretch"

    def __init__(self, world: Any, thresholds: dict[str, Any] | None = None,
                 output_dir: Path | None = None, *, root: Path = PROJECT_ROOT,
                 on_frame=None, on_status=None, step_budget: int = 120000):
        if type(step_budget) is not int or not 1 <= step_budget <= 500000:
            raise HomeExecutionError("INVALID_BUDGET", "Physics step budget requires an integer in 1..500000")
        from embodied_agent.simulation.stretch import StretchSimulation
        from embodied_agent.skills.stretch import StretchSkills
        self.world = world
        self.on_frame, self.on_status = on_frame, on_status
        self.events, self.samples = [], []
        self.object_states = {name: dict(obj.states) for name, obj in world.objects.items()}
        self.version, self.step_budget = world.world_version, step_budget
        self._budget_start_steps = 0
        self.active_skill = "idle"
        self._motion_radius = None
        self._fault_protocol = None
        self._last_positions, self._falling = {}, {}
        self._last_base_pose = None
        self._last_risk_signature = None
        self.goals = []
        self._goal_stability: dict[int, int] = {}
        self._object_stable_since: dict[str, float] = {}
        self._transported_objects: set[str] = set()
        self._consumed_permits: set[str] = set()
        self._permits: dict[str, dict] = {}
        self.episode_id = uuid.uuid4().hex
        self._task_budget = None
        self._task_running = False
        self._holding = False
        self._navigation_route = None
        self._navigation_replans = 0
        self._navigation_motion = {}
        self.safety = HomeSafety(world)
        self.sim = StretchSimulation(world)
        self.skills = StretchSkills(self.sim)
        self.sim.on_step = self._after_step
        self.initial = self.observe()
        self.scenario = {"scenario_id": f"map_{world.map_id}", "seed": 0}
        self.prepared = True

    @property
    def model(self):
        return self.sim.model

    @property
    def data(self):
        return self.sim.data

    @property
    def total_steps(self):
        return self.sim.total_steps

    def snapshot(self) -> dict[str, Any]:
        """Measured observation, not a restorable MuJoCo checkpoint."""
        return self.observe()

    def prepare(self, *, check_safety: bool = True) -> dict[str, Any]:
        if check_safety:
            self.sim.check_physics()
            self.safety.during_geometry(self.sim, self.positions(), self.object_states)
        return self.snapshot()

    def _sync_viewer(self) -> None:
        if self.on_frame is not None:
            self.on_frame(self)

    def set_viewer_status(self, title: str, detail: str) -> None:
        if self.on_status is not None:
            self.on_status(title, detail)

    def wait_for_viewer_close(self) -> None:
        # The original DemoApp keeps the last measured world visible.
        pass

    def _event(self, kind: str, **fields) -> None:
        self.events.append({"type": kind, "step": self.sim.total_steps,
                            "time_s": float(self.sim.data.time), "world_version": self.version, **fields})

    def positions(self) -> dict[str, list[float]]:
        return {name: self.sim.data.xpos[body].tolist() for name, body in self.sim.object_body_ids.items()}

    def observe(self) -> dict:
        x, y, yaw = self.sim.base_pose()
        mode = ("navigate" if self.active_skill == "navigate" else "carry" if self.active_skill == "carry"
                else "manipulate" if self.active_skill in {"pick", "place"} else "stopped" if self.active_skill == "stop" else "idle")
        result = self.world.snapshot(self.positions(), self.object_states, world_version=self.version,
            robot_position_m=(x, y, 0), robot_mode=mode,
            held_object=getattr(self.sim, "held_object_id", None),
            completed_goals=tuple(g["object_id"] + "@" + g["support_id"] for g in self.goals if g.get("valid")),
            events=tuple(self.events[-10:]))
        result["robot"]["yaw_rad"] = yaw
        result["robot"]["stage"] = self.sim.stage
        result["robot"]["base_speed"] = list(self.sim.base_speed())
        held = getattr(self.sim, "held_object_id", None)
        result["robot"]["held_evidence"] = self.sim.object_contact_evidence(held) if held else None
        result["observation_id"] = f"{self.episode_id}:{self.sim.total_steps}:{self.version}"
        result["observed_at_simulation_s"] = float(self.sim.data.time)
        result["episode_id"] = self.episode_id
        return result

    def begin_task_execution(self, budget=None) -> None:
        self._task_budget = budget
        self._task_running = True
        self._budget_start_steps = self.sim.total_steps

    def end_task_execution(self) -> None:
        self._task_budget = None
        self._task_running = False

    def hold(self, seconds: float = 0.02) -> dict:
        """Keep real braking and grasp guards alive while a language worker waits."""
        if not math.isfinite(seconds) or not 0 <= seconds <= 0.25:
            raise HomeExecutionError("INVALID_WAIT", "One monitored HOLD tick is bounded to 0.25 seconds")
        previous_skill, previous_stage = self.active_skill, self.sim.stage
        self._holding = True
        self.active_skill, self.sim.stage = "stop", "hold"
        try:
            self.sim.command_base(0, 0)
            for _ in range(math.ceil(seconds / self.sim.model.opt.timestep)):
                # Unlike emergency braking, all physical and risk guards remain.
                self.sim.step()
            return {"held_object": self.sim.held_object_id, "base_speed": list(self.sim.base_speed()),
                    "held_evidence": self.sim.object_contact_evidence(self.sim.held_object_id) if self.sim.held_object_id else None}
        finally:
            self._holding = False
            self.active_skill, self.sim.stage = previous_skill, previous_stage

    def safe_stop(self) -> dict:
        """Bounded emergency braking independent of exhausted task/model budgets."""
        previous_skill, callback = self.active_skill, self.sim.on_step
        self.active_skill = "stop"
        self._navigation_route = None
        # Emergency braking deliberately bypasses already-latched callbacks.
        # That gap cannot count toward continuous temporal proof afterwards.
        self._object_stable_since.clear()
        for goal in self.goals:
            self._goal_stability[id(goal)] = 0
            goal["valid"] = False
        self.sim.on_step = None
        try:
            return self.skills.safe_stop(emergency=True)
        finally:
            self.sim.on_step, self.active_skill = callback, previous_skill

    def _state_fingerprint(self) -> str:
        current = {"episode": self.episode_id, "map_revision": self.world.revision,
            "rule_version": self.world.rule_version, "positions": self.positions(),
            "states": self.object_states, "base_pose": self.sim.base_pose(),
            "base_speed": self.sim.base_speed(), "held": self.sim.held_object_id,
            "qpos": self.sim.data.qpos.tolist(), "qvel": self.sim.data.qvel.tolist(),
            "controls": self.sim.data.ctrl.tolist(), "simulation_time_s": float(self.sim.data.time),
            "geom_positions": self.sim.data.geom_xpos.tolist(),
            "geom_sizes": self.sim.model.geom_size.tolist(), "geom_local_positions": self.sim.model.geom_pos.tolist()}
        return hashlib.sha256(json.dumps(current, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def review_next_action(self, action: dict) -> dict:
        """Compile the next action against current state; never grant model ALLOW."""
        action = validate_home_plan({"schema_version": 1, "actions": [action]})[0]
        if self._task_budget is not None:
            self._task_budget.check()
        positions, held = self.positions(), self.sim.held_object_id
        skill = action["skill"]
        compiled = {"decision": "ALLOW", "permit_id": uuid.uuid4().hex, "episode_id": self.episode_id,
            "map_revision": self.world.revision, "rule_version": self.world.rule_version,
            "state_fingerprint": self._state_fingerprint(), "action": copy.deepcopy(action),
            "reviewed_at_step": self.sim.total_steps}
        if skill in {"navigate", "carry"}:
            if (skill == "carry") != (held is not None):
                raise HomeExecutionError("HOLD_PRECONDITION", "Navigate empty-handed; carry only with measured grasp")
            if held:
                self.safety.object_action("carry", held, positions, self.object_states)
                if self.sim.object_contact_evidence(held)["finger_contact_side_count"] != 2:
                    raise HomeExecutionError("GRASP_UNVERIFIED", "Carry requires current bilateral contact")
            if float(self.sim.data.ctrl[self.sim.actuator_ids["arm_extend"]]) > .035:
                raise HomeExecutionError("TRANSPORT_POSTURE", "Retract the telescope before navigating")
            from embodied_agent.maps.manipulation import resolve_navigation_target, candidate_operation_points
            target = action["target"]
            point = resolve_navigation_target(self.world, self.observe(), target)
            if target in self.world.operation_points or target not in {*self.world.objects, *self.world.furniture, "floor"}:
                candidates = [point]
            else:
                candidates = candidate_operation_points(self.world,
                    **({"object_id": target} if target in self.world.objects else {"support_id": target}),
                    object_positions=positions)
            nav = self._navigator(positions)
            from embodied_agent.skills.navigation import NavigationError
            reachable = []
            for candidate in candidates:
                try:
                    route = nav.plan(self.sim.base_pose()[:2], candidate.position_m[:2])
                except NavigationError:
                    continue
                length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                reachable.append((length, candidate.point_id, candidate, route))
            if not reachable:
                raise HomeExecutionError("PATH_BLOCKED", "No current candidate operation pose has a safe route")
            _, _, point, path = min(reachable, key=lambda row: row[:2])
            compiled.update(point=point, path=path, radius=nav.radius)
        elif skill == "pick":
            name = action["object_id"]
            if held:
                raise HomeExecutionError("GRIPPER_OCCUPIED", "Place the held object before another pick")
            if name not in self.world.objects:
                raise HomeExecutionError("UNKNOWN_OBJECT", name)
            self.safety.object_action(skill, name, positions, self.object_states)
            if self.sim.base_speed()[0] > .01 or self.sim.base_speed()[1] > .03:
                raise HomeExecutionError("BASE_MOVING", "Stop before manipulating")
            from embodied_agent.maps.manipulation import compute_pick_candidates
            docks = compute_pick_candidates(self.world, self.observe(), name)
            if not any(math.dist(self.sim.base_pose()[:2], p.position_m[:2]) <= .08 and
                    abs(math.atan2(math.sin(self.sim.base_pose()[2] - p.yaw_rad), math.cos(self.sim.base_pose()[2] - p.yaw_rad))) <= .10 for p in docks):
                raise HomeExecutionError("NOT_DOCKED", "Pick requires a current reachable operation pose")
        elif skill == "place":
            if not held:
                raise HomeExecutionError("HOLD_PRECONDITION", "Place requires a verified held object")
            from embodied_agent.maps.manipulation import lookup_surface
            support = lookup_surface(self.world, action["support_id"])
            if self.sim.base_speed()[0] > .01 or self.sim.base_speed()[1] > .03:
                raise HomeExecutionError("BASE_MOVING", "Stop before manipulating")
            if self.sim.object_contact_evidence(held)["finger_contact_side_count"] != 2:
                raise HomeExecutionError("GRASP_UNVERIFIED", "Place requires current bilateral contact")
            target = (*action["target_xy"], support.top_z + self.world.objects[held].half_size_m[2])
            self.safety.object_action(skill, held, positions, self.object_states, target)
        elif skill in {"inspect", "query_world"}:
            if action["object_id"] not in self.world.objects:
                raise HomeExecutionError("UNKNOWN_OBJECT", action["object_id"])
            if skill == "inspect":
                self.safety.object_action(skill, action["object_id"], positions, self.object_states)
        self._permits[compiled["permit_id"]] = copy.deepcopy(compiled)
        return compiled

    def execute_reviewed_action(self, compiled: dict) -> dict:
        self._consume_review(compiled)
        return self._execute_action(compiled["action"], compiled)

    def _consume_review(self, compiled: dict) -> None:
        permit = compiled.get("permit_id")
        if compiled.get("decision") != "ALLOW" or not permit or permit in self._consumed_permits:
            raise HomeExecutionError("INVALID_PERMIT", "An unused trusted next-action permit is required")
        trusted = self._permits.pop(permit, None)
        if trusted is None or trusted != compiled:
            raise HomeExecutionError("INVALID_PERMIT", "Reviewed action or compiled geometry was replaced")
        if compiled.get("episode_id") != self.episode_id or compiled.get("state_fingerprint") != self._state_fingerprint():
            raise HomeExecutionError("STALE_ACTION", "The reviewed action state changed before execution")
        if self._task_budget is not None:
            self._task_budget.check()
        self._consumed_permits.add(permit)

    def _navigator(self, positions, velocities=None, *, footprint_radius=None):
        held = self.sim.held_object_id
        return self.safety.navigator(positions, self.object_states, carrying=held is not None,
            carried_half_size=self.world.objects[held].half_size_m if held else None,
            footprint_radius=(self.sim.transport_footprint_radius() + (.035 if held else 0))
                if footprint_radius is None else footprint_radius,
            held_object_id=held, object_velocities=velocities)

    def _check_navigation_route(self, positions) -> None:
        if not self._navigation_route or self._fault_protocol or self.sim.stage in {"stop", "safe_stop", "hold"}:
            return
        # Reaction horizon includes the measured speed and conservative swept
        # object motion. Collision/contact guards still execute on every step.
        now = float(self.sim.data.time)
        velocities = {}
        for name, position in positions.items():
            previous = self._navigation_motion.get(name)
            if previous and now > previous[0]:
                velocities[name] = [(position[i] - previous[1][i]) / (now - previous[0]) for i in (0, 1)]
            self._navigation_motion[name] = (now, list(position))
        nav = self._navigator(positions, velocities, footprint_radius=self._motion_radius)
        if not nav.route_free(self.sim.base_pose()[:2], self._navigation_route):
            self._event("navigation_interrupted", reason="updated_obstacles", remaining_path=nav.remaining_path(
                self.sim.base_pose()[:2], self._navigation_route))
            raise HomeExecutionError("ROUTE_CHANGED", "Updated obstacle motion invalidated the remaining path")

    def _navigate_supervised(self, skill, path, point, target=None) -> dict:
        from embodied_agent.skills.navigation import NavigationError
        self._navigation_motion = {}
        self._navigation_replans = 0
        try:
            while True:
                self._navigation_route = copy.deepcopy(path)
                try:
                    result = getattr(self.skills, skill)(path, final_yaw=point.yaw_rad)
                    result = {**result, "resolved_target": {"point_id": point.point_id, "target_id": target,
                        "support_id": point.support, "position_m": list(point.position_m), "yaw_rad": point.yaw_rad}}
                    if self._navigation_replans:
                        result = {**result, "local_replans": self._navigation_replans}
                    return result
                except HomeExecutionError as exc:
                    if exc.code != "ROUTE_CHANGED":
                        raise
                    self._navigation_route = None
                    self.sim.stop()  # checked real braking, never a teleported pause
                    if self._navigation_replans >= 3:
                        raise HomeExecutionError("LOCAL_REPLAN_BUDGET", "Navigation exhausted its local reroute budget") from exc
                    self._navigation_replans += 1
                    try:
                        # A local repair passes the same trusted review as a new
                        # task action. It cannot execute the previous stale path.
                        compiled = self.review_next_action({"skill": skill, "target": target or point.point_id})
                        self._consume_review(compiled)
                        point, path = compiled["point"], compiled["path"]
                        self._motion_radius = compiled["radius"]
                        self._event("navigation_replanned", attempt=self._navigation_replans, path=path,
                                    target=target, source="local_trusted_planner")
                    except NavigationError as blocked:
                        raise HomeExecutionError("PATH_BLOCKED", str(blocked)) from blocked
        finally:
            self._navigation_route = None

    def action_satisfied(self, action: dict, prior_result: dict | None = None) -> bool:
        """A recovery may skip a completed effect only after actual remeasurement."""
        skill = action.get("skill")
        if skill == "pick":
            name = action.get("object_id")
            return self.sim.held_object_id == name and self.sim.object_contact_evidence(name)["finger_contact_side_count"] == 2
        if skill in {"navigate", "carry"}:
            from embodied_agent.maps.manipulation import resolve_navigation_target
            try:
                point = resolve_navigation_target(self.world, self.observe(), action["target"])
                return math.dist(self.sim.base_pose()[:2], point.position_m[:2]) < .025 and abs(math.atan2(
                    math.sin(self.sim.base_pose()[2] - point.yaw_rad), math.cos(self.sim.base_pose()[2] - point.yaw_rad))) < .05
            except Exception:
                return False
        if skill == "place" and prior_result:
            name = prior_result.get("evidence", {}).get("independent_placement", {}).get("object_id")
            return bool(name and placement_evidence(self.sim, self.world, name, action["support_id"], action["target_xy"])["passed"])
        # Explicit waits and stops may be process requirements; do not erase them.
        return False

    def evaluate_goal_evidence(self, goals=None) -> dict:
        placements, objects = [], {}
        for goal in self.goals:
            score = placement_evidence(self.sim, self.world, goal["object_id"], goal["support_id"], goal["target_xy"])
            stable = bool(goal.get("valid") and score["passed"])
            placements.append({**score, "passed": stable, "stable_window_passed": stable})
        pending_stability = False
        for goal in goals or []:
            if not isinstance(goal, dict) or goal.get("predicate") not in {"supported_on", "inside", "stable"}:
                continue
            name, support = goal.get("object_id"), goal.get("target_id")
            if name in self.world.objects and support is None:
                support = self.world.support_at(name, self.positions()[name])
            if name not in self.world.objects or any(row["object_id"] == name and row["support_id"] == support for row in placements):
                continue
            try:
                xy = self.positions()[name][:2]
                score = placement_evidence(self.sim, self.world, name, support, xy)
            except (KeyError, ValueError):
                continue
            stable = name in self._object_stable_since and float(self.sim.data.time) - self._object_stable_since[name] >= .5
            instantaneous = all(value for key, value in score["predicates"].items() if key != "physical_support_contact")
            # mj_forward alone has not established loaded contact forces. A
            # geometrically supported, released, still object may enter guarded
            # temporal verification; this is never itself a passing predicate.
            pending_stability = pending_stability or (instantaneous and not stable)
            placements.append({**score, "passed": bool(score["passed"] and stable), "stable_window_passed": stable})
        for name in self.world.objects:
            contact = self.sim.object_contact_evidence(name)
            # Contact telemetry is factual. Continuous stable placement is
            # supplied by the independent scorer, not inferred from a flag.
            objects[name] = {**contact, "released": not contact["robot_contact"] and self.sim.held_object_id != name,
                "stable": any(row["object_id"] == name and row["passed"] for row in placements) or (
                    name in self._object_stable_since and float(self.sim.data.time) - self._object_stable_since[name] >= .5)}
        return {"placements": placements, "objects": objects, "source": "measured_execution",
                "pending_stability": pending_stability}

    def _monitor_object_stability(self) -> None:
        import mujoco
        import numpy as np
        now = float(self.sim.data.time)
        pairs = self.sim.contacts()
        support_geoms = set().union(*self.sim.furniture_geom_ids.values(), {self.sim.floor_geom_id})
        for name, body in self.sim.object_body_ids.items():
            geom = self.sim.object_geom_ids[name]
            contacts = [row for row in pairs if geom in row["geom_ids"]]
            robot_contact = any(any(g in self.sim.robot_geom_ids for g in row["geom_ids"] if g != geom) for row in contacts)
            supported = any(row["normal_force_n"] > .05 and any(g in support_geoms
                for g in row["geom_ids"] if g != geom) for row in contacts)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            stable = bool(supported and not robot_contact and self.sim.held_object_id != name and
                float(np.linalg.norm(velocity[3:])) <= .025 and float(np.linalg.norm(velocity[:3])) <= .2)
            if stable:
                self._object_stable_since.setdefault(name, now)
            else:
                self._object_stable_since.pop(name, None)

    def run_action(self, action: dict, *, record_event=None) -> dict:
        from embodied_agent.execution.supervisor import ExecutionResult
        start, executed, evidence, observation = self.sim.total_steps, False, {}, None
        cleanup_errors = []
        status, code, message, error_evidence = "SUCCESS", None, None, {}
        def emit(name, detail, measured=None, required=True):
            self._event(name, **detail)
            if record_event is not None:
                try:
                    record_event(name, {"observation_only": True, "state_restorable": False,
                        "step": self.sim.total_steps, "world_version": self.version, **detail}, measured)
                except Exception as exc:
                    if required:
                        raise HomeExecutionError("RECORDING_FAILED", f"Cannot record {name}: {exc}") from exc
                    cleanup_errors.append({"phase": "recording", "event": name, "message": str(exc)})
        try:
            compiled = self.review_next_action(action)
            observation = self.observe()
            # Keep the live point object internal; records receive only geometry.
            check = {key: value for key, value in compiled.items() if key != "point"}
            if "point" in compiled:
                check["point"] = compiled["point"].to_dict()
            emit("action_reviewed", {"check": check}, observation)
            emit("before_tool", {"action": action}, observation)
            self._consume_review(compiled)
            executed = True
            evidence = self._execute_action(action, compiled)
            self.version += 1
            observation = self.observe()
            emit("after_tool", {"action": action, "status": "SUCCESS", "evidence": evidence,
                "physics_steps_used": self.sim.total_steps - start}, observation)
        except Exception as exc:
            status, code, message = "FAILED", getattr(exc, "code", type(exc).__name__), str(exc)
            if code == "TASK_CANCELLED" or type(exc).__name__ in {"ViewerClosed"}:
                status = "ABORTED"
                if type(exc).__name__ in {"ViewerClosed"}:
                    code = "VIEWER_CLOSED"
            error_evidence = getattr(exc, "evidence", {})
            try:
                evidence["safe_stop"] = self.safe_stop()
            except Exception as stop_exc:
                cleanup_errors.append({"phase": "stop", "code": getattr(stop_exc, "code", type(stop_exc).__name__), "message": str(stop_exc)})
            try:
                observation = self.observe()
            except Exception as obs_exc:
                observation = None
                cleanup_errors.append({"phase": "observation", "message": str(obs_exc)})
            emit("execution_failed", {"action": action, "error_code": code, "message": message,
                "evidence": error_evidence, "physics_steps_used": self.sim.total_steps - start,
                "tool_invoked": executed}, observation, required=False)
        finally:
            self._navigation_route = None
            self.active_skill = "idle"
        result = ExecutionResult(status=status, action=copy.deepcopy(action), evidence=evidence,
            observation=observation, physics_steps=self.sim.total_steps - start,
            error_code=code, error_message=message, error_evidence=error_evidence,
            cleanup_errors=cleanup_errors, executed=executed).to_dict()
        result["stop_failed"] = any(row["phase"] == "stop" for row in cleanup_errors)
        released = evidence.get("safe_stop", {}).get("released_evidence") or {}
        released_id = released.get("object_id")
        result["recoverable"] = bool(code == "OBJECT_SLIPPED" and not result["stop_failed"] and
            evidence.get("safe_stop", {}).get("released_after_slip") and self.sim.held_object_id is None and
            released_id in self.world.objects and "fragile" not in self.world.objects[released_id].risk_tags and
            self.object_states[released_id].get("damage") != "damaged")
        return result

    def _damage_observation(self, positions: dict) -> None:
        import mujoco
        import numpy as np
        held = getattr(self.sim, "held_object_id", None)
        for name, obj in self.world.objects.items():
            if "fragile" not in obj.risk_tags or self.object_states[name].get("damage") == "damaged":
                continue
            geom = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, obj.geom_name)
            contacts = [c for c in self.sim.data.contact[:self.sim.data.ncon] if geom in (c.geom1, c.geom2)]
            robot_contact = any(c.geom1 in self.sim.robot_geom_ids or c.geom2 in self.sim.robot_geom_ids for c in contacts)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY,
                                    self.sim.object_body_ids[name], velocity, 0)
            if name == held or robot_contact:
                self._falling.pop(name, None)
                continue
            if not contacts and velocity[5] < -0.15:
                self._falling.setdefault(name, {"height": positions[name][2], "speed": 0.0})
                self._falling[name]["speed"] = max(self._falling[name]["speed"], float(np.linalg.norm(velocity[3:])))
            elif contacts and name in self._falling:
                fall = self._falling.pop(name)
                event = project_damage_event(name, drop_height_m=max(0, fall["height"] - positions[name][2]),
                                             impact_speed_m_s=fall["speed"])
                if event:
                    self.object_states[name]["damage"] = "damaged"
                    self.version += 1
                    self._event("project_damage", evidence=event)
                    raise HomeExecutionError("OBJECT_DAMAGED", f"Measured free fall damaged {name} under the project rule")

    def _after_step(self, sim: Any) -> None:
        if self._task_budget is not None and not self._holding:
            self._task_budget.check()
        if sim.total_steps - self._budget_start_steps > self.step_budget:
            raise HomeExecutionError("STEP_BUDGET", "Home task exhausted its physics step budget")
        positions = self.positions()
        base_pose = sim.base_pose()
        previous_base = self._last_base_pose or base_pose
        yaw_delta = abs(math.atan2(math.sin(base_pose[2] - previous_base[2]), math.cos(base_pose[2] - previous_base[2])))
        if math.dist(base_pose[:2], previous_base[:2]) > 0.05 or yaw_delta > 0.1:
            self.version += 1
            self._last_base_pose = base_pose
            self._event("robot_motion", base_pose=list(base_pose), source="physics")
        elif self._last_base_pose is None:
            self._last_base_pose = base_pose
        for name, position in positions.items():
            previous = self._last_positions.get(name, position)
            if math.dist(position, previous) > 0.01:
                self.version += 1
                self._last_positions[name] = position
                self._event("object_motion", object_id=name, position_m=position, source="physics")
            else:
                self._last_positions.setdefault(name, position)
        self._damage_observation(positions)
        risks = self.world.risks(positions, self.object_states)
        risk_signature = tuple((r.kind, r.object_ids) for r in risks)
        if risk_signature != self._last_risk_signature:
            self._last_risk_signature = risk_signature
            self._event("risk_relations", risks=[r.to_dict() for r in risks], source="trusted_state_and_physics")
        self.safety.during_geometry(sim, positions, self.object_states)
        if self.active_skill in {"navigate", "carry"}:
            self.safety.during_motion(sim, positions, self.object_states, self._motion_radius)
            if sim.total_steps % 50 == 0:
                self._check_navigation_route(positions)
        for goal in self.goals:
            passed = placement_evidence(sim, self.world, goal["object_id"], goal["support_id"], goal["target_xy"])["passed"]
            required = math.ceil(.5 / sim.model.opt.timestep)
            previous_stable = self._goal_stability.get(id(goal), required if goal.get("valid") else 0)
            stable = min(required, previous_stable + 1) if passed else 0
            self._goal_stability[id(goal)] = stable
            valid = stable >= required
            if goal.get("valid") and not valid:
                self.version += 1
                self._event("goal_revoked", object_id=goal["object_id"], support_id=goal["support_id"])
            goal["valid"] = valid
        if self._task_running or self._holding:
            self._monitor_object_stability()
        if sim.total_steps % 50 == 0:
            x, y, yaw = sim.base_pose()
            self.samples.append({"step": sim.total_steps, "time_s": float(sim.data.time),
                "base_x": x, "base_y": y, "base_yaw": yaw, "stage": sim.stage,
                "held_object": getattr(sim, "held_object_id", None),
                "object_positions": json.dumps(positions, ensure_ascii=False), "world_version": self.version})
            if not self._holding:
                self._sync_viewer()

    def _tool(self, action: dict) -> Any:
        return self.execute_reviewed_action(self.review_next_action(action))

    def _execute_action(self, action: dict, compiled: dict) -> Any:
        skill = action["skill"]
        self.active_skill = skill
        self.set_viewer_status(skill.upper(), str(action))
        positions = self.positions()
        held = getattr(self.sim, "held_object_id", None)
        if skill == "observe":
            return self.observe()
        if skill in {"query_world", "inspect"}:
            name = action["object_id"]
            if name not in self.world.objects:
                raise HomeExecutionError("UNKNOWN_OBJECT", name)
            if skill == "inspect":
                self.safety.object_action("inspect", name, positions, self.object_states)
            return {"source": "declared_current_simulation_state", "object_id": name,
                    "object": self.observe()["objects"][name], "world_version": self.version}
        if skill in {"navigate", "carry"}:
            target = action["target"]
            if (skill == "carry") != (held is not None):
                raise HomeExecutionError("HOLD_PRECONDITION", "Use carry with a grasped object and navigate with an empty gripper")
            point = compiled["point"]
            initial_carried_position = positions[held] if held else None
            if held:
                self.safety.object_action("carry", held, positions, self.object_states)
            self._motion_radius = compiled["radius"]
            path = compiled["path"]
            self._event("navigation_path", target=target, path=path, footprint_radius_m=self._motion_radius,
                        footprint_evidence=self.sim.footprint_evidence())
            if self._fault_protocol == "navigation_waypoint_into_tea_table_v1":
                # Trusted, explicitly labelled tester corrupts the controller's
                # waypoint after normal planning. Same motor/step/guard chain;
                # no geometry/state teleport and no guard suppression.
                if skill != "navigate" or target != "tea_table":
                    raise HomeExecutionError("INVALID_FAULT_PROTOCOL", "Collision fixture requires empty-hand tea_table navigation")
                faulty_point = (point.position_m[0], self.world.furniture["tea_table"].position_m[1])
                path = [*path, faulty_point]
                self._event("test_fault_injection", protocol=self._fault_protocol,
                            source="trusted_test_controller", injected_path=path,
                            expected_guard="ROBOT_FURNITURE_COLLISION")
            evidence = self._navigate_supervised(skill, path, point, target)
            if held:
                evidence = {**evidence, "object_displacement_m": math.dist(initial_carried_position, self.positions()[held])}
            if skill == "carry" and evidence.get("object_displacement_m", 0) > 0.10:
                self._transported_objects.add(held)
            return evidence
        if skill == "pick":
            name = action["object_id"]
            if held:
                raise HomeExecutionError("GRIPPER_OCCUPIED", "Place the held object before another pick")
            if name not in self.world.objects:
                raise HomeExecutionError("UNKNOWN_OBJECT", name)
            self.safety.object_action(skill, name, positions, self.object_states)
            # Registered poses constrain the controller, while the live
            # object/support decides which pose applies after an earlier task.
            # Initial object_ids are map hints, not persistent attachments.
            position = positions[name]
            support = self.world.support_at(name, position)
            docks = []
            from embodied_agent.maps.manipulation import compute_pick_candidates
            for point in compute_pick_candidates(self.world, self.observe(), name):
                dx, dy = (position[i] - point.position_m[i] for i in (0, 1))
                local_x = math.cos(point.yaw_rad) * dx + math.sin(point.yaw_rad) * dy
                local_y = -math.sin(point.yaw_rad) * dx + math.cos(point.yaw_rad) * dy
                extension = -local_y - .34036 - .008
                if (point.support == support and abs(local_x + .021385) <= .018
                        and .06 <= extension <= .49):
                    docks.append(point)
            if not any(math.dist(self.sim.base_pose()[:2], p.position_m[:2]) <= 0.08 and
                       abs(math.atan2(math.sin(self.sim.base_pose()[2] - p.yaw_rad), math.cos(self.sim.base_pose()[2] - p.yaw_rad))) <= 0.10 for p in docks):
                raise HomeExecutionError("NOT_DOCKED", "Pick requires a registered validated operation pose")
            evidence = self.skills.pick(name)
            self._transported_objects.discard(name)
            return evidence
        if skill == "place":
            if not held:
                raise HomeExecutionError("HOLD_PRECONDITION", "Place requires a real grasped object")
            support = action["support_id"]
            from embodied_agent.maps.manipulation import lookup_surface
            surface = lookup_surface(self.world, support)
            xy = action["target_xy"]
            target = (*xy, surface.top_z + self.world.objects[held].half_size_m[2])
            self.safety.object_action(skill, held, positions, self.object_states, target)
            evidence = self.skills.place(support, xy)
            # Score a continuous stable window, rather than trusting skill return.
            stable = 0
            required = math.ceil(0.5 / self.sim.model.opt.timestep)
            for _ in range(math.ceil(1.5 / self.sim.model.opt.timestep)):
                self.sim.step()
                score = placement_evidence(self.sim, self.world, held, support, xy)
                stable = stable + 1 if score["passed"] else 0
                if stable >= required:
                    break
            if stable < required:
                raise HomeExecutionError("PLACEMENT_UNSTABLE", json.dumps(score["predicates"]))
            self.goals.append({"object_id": held, "support_id": support, "target_xy": list(xy), "valid": True,
                               "transported": held in self._transported_objects})
            self._transported_objects.discard(held)
            return {"skill_evidence": evidence, "independent_placement": score, "stable_steps": stable}
        if skill == "wait":
            return self.sim.wait(action["seconds"])
        if skill == "stop":
            return self.sim.stop()
        raise HomeExecutionError("UNREGISTERED_SKILL", skill)

    def run_plan(self, plan: dict, *, case_id: str = "custom", fault_protocol: str | None = None,
                 record_event: Callable[[str, dict[str, Any], dict[str, Any] | None], None] | None = None) -> dict:
        """Run existing skills and synchronously record their actual boundaries.

        Recorded observations describe measured state only. They are not complete
        MuJoCo checkpoints and cannot be used to restore a physics episode.
        """
        if fault_protocol not in {None, "navigation_waypoint_into_tea_table_v1"}:
            raise HomeExecutionError("INVALID_FAULT_PROTOCOL", "Unregistered trusted test protocol")
        self._fault_protocol = fault_protocol
        status, error, message = "SUCCESS", None, None
        error_evidence = {}
        action_results = []
        recording_errors = []
        action_index = None
        current_action = None
        action_started = self.sim.total_steps
        tool_invocations = 0
        tool_invoked = False
        start_steps = self.sim.total_steps
        self._budget_start_steps = start_steps
        initial_goal_count = len(self.goals)
        # Invalid direct Python payloads can contain NaN or arbitrary objects;
        # preserve their representation without breaking the evidence writer.
        try:
            json.dumps(plan, allow_nan=False)
            recorded_plan = plan
        except (ValueError, TypeError):
            try:
                recorded_plan = {"invalid_payload_repr": repr(plan)[:4000]}
            except Exception:
                recorded_plan = {"invalid_payload_type": type(plan).__name__, "representation": "unrepresentable payload"}
        self._event("plan_received", case_id=case_id, plan=recorded_plan)

        def record(name: str, detail: dict, observation: dict | None = None, *, required: bool = True) -> None:
            if record_event is None:
                return
            try:
                record_event(name, {"case_id": case_id, "observation_only": True,
                    "state_restorable": False, "step": self.sim.total_steps,
                    "time_s": float(self.sim.data.time), "world_version": self.version, **detail}, observation)
            except Exception as exc:
                recording_errors.append({"event": name, "error_type": type(exc).__name__, "message": str(exc)})
                if required:
                    raise HomeExecutionError("RECORDING_FAILED", f"Cannot record {name}: {exc}") from exc

        try:
            record("plan_received", {"plan": recorded_plan, "test_fault_protocol": fault_protocol}, self.observe())
            actions = validate_home_plan(plan)
            for index, action in enumerate(actions):
                action_index, current_action = index, action
                action_started = self.sim.total_steps
                tool_invoked = False
                self._event("before_tool", action_index=index, action=action)
                record("before_tool", {"action_index": index, "action": action,
                    "completed_actions": action_results}, self.observe())
                tool_invocations += 1
                tool_invoked = True
                evidence = self._tool(action)
                action_results.append({"action_index": index, "action": action, "evidence": evidence,
                    "physics_steps_used": self.sim.total_steps - action_started})
                if action["skill"] not in {"observe", "query_world", "inspect"}:
                    self.version += 1
                self._event("after_tool", action_index=index, action=action)
                record("after_tool", {"action_index": index, "action": action,
                    "evidence": evidence, "status": "SUCCESS",
                    "physics_steps_used": self.sim.total_steps - action_started,
                    "completed_actions": action_results}, self.observe())
        except Exception as exc:
            status, error, message = "FAILED", getattr(exc, "code", type(exc).__name__), str(exc)
            if type(exc).__name__ in {"ViewerClosed"}:
                status, error = "ABORTED", "VIEWER_CLOSED"
            error_evidence = getattr(exc, "evidence", {})
            failed_steps_used = self.sim.total_steps - action_started
            failure_step, failure_time, failure_version = self.sim.total_steps, float(self.sim.data.time), self.version
            try:
                failure_observation = self.observe()
            except Exception as observation_exc:
                failure_observation = None
                error_evidence = {**error_evidence, "failure_observation_error": str(observation_exc)}
            self._event("execution_failed", error_code=error, message=message, evidence=error_evidence)
            # Reserve a bounded safe holding stop after a budget/guard failure.
            self.active_skill = "stop"
            callback = self.sim.on_step
            self.sim.on_step = None
            try:
                stop_evidence = self.skills.safe_stop(emergency=True)
                self._event("safe_stop", evidence=stop_evidence)
                stop_detail = {"evidence": stop_evidence, "status": "SUCCESS"}
            except Exception as stop_exc:
                self._event("safe_stop_failed", message=str(stop_exc))
                stop_detail = {"status": "FAILED", "message": str(stop_exc)}
            finally:
                self.sim.on_step = callback
            # Keeping the robot safe precedes feedback persistence. A repeated
            # write failure must not skip the stop or conceal the physical result.
            record("execution_failed", {"error_code": error, "message": message,
                "step": failure_step, "time_s": failure_time, "world_version": failure_version,
                "evidence": error_evidence, "action_index": action_index,
                "action": current_action, "physics_steps_used": failed_steps_used,
                "tool_invoked": tool_invoked,
                "completed_actions": action_results}, failure_observation, required=False)
            try:
                stopped_observation = self.observe()
            except Exception as observation_exc:
                stopped_observation = None
                error_evidence = {**error_evidence, "stopped_observation_error": str(observation_exc)}
            record("safe_stop" if stop_detail["status"] == "SUCCESS" else "safe_stop_failed",
                   stop_detail, stopped_observation, required=False)
        finally:
            self.active_skill = "idle"
            self._fault_protocol = None
        try:
            final = self.observe()
        except Exception as observation_exc:
            final = None
            error_evidence = {**error_evidence, "final_observation_error": str(observation_exc)}
            if status == "SUCCESS":
                status, error, message = "FAILED", "OBSERVATION_FAILED", str(observation_exc)
        new_goals = self.goals[initial_goal_count:]
        result = {"case_id": case_id, "status": status, "error_code": error, "message": message, "error_message": message,
            "execution_status": ("INTERRUPTED" if tool_invocations else "NOT_EXECUTED")
                if error == "RECORDING_FAILED" else status,
            "test_fault_protocol": fault_protocol,
            "error_evidence": error_evidence,
            "actions": action_results, "physics_steps": self.sim.total_steps - start_steps,
            "simulation_time_s": float(self.sim.data.time), "final_snapshot": final,
            "completed_goals": [dict(goal) for goal in self.goals],
            "transport_success": status == "SUCCESS" and bool(new_goals) and
                all(goal.get("valid") and goal.get("transported") for goal in new_goals)}
        record("run_plan_finished", {"result": result}, final, required=False)
        if recording_errors:
            result["recording_errors"] = recording_errors
        self.set_viewer_status(status, f"{case_id}: {error or 'completed'}")
        return result

    def evidence_bundle(self) -> dict[str, Any]:
        """Return cumulative physical evidence to the owning DemoSession writer."""
        return {"initial_snapshot": self.initial, "step_budget": self.step_budget,
                "events": list(self.events), "physics_events": list(self.sim.events),
                "trajectory": list(self.samples)}

    def close(self) -> None:
        self.sim.on_step = None
        self.on_frame = None
        self.on_status = None

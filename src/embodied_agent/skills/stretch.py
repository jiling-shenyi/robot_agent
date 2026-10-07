"""Finite-pose Stretch skills with actual contact grasp and free-body release."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ..simulation.stretch import HomeSkillError, StretchSimulation, wrap_angle
from ..maps.manipulation import MAX_PLACE_SURFACE_HEIGHT_M, grasp_supported, lookup_surface


class StretchSkills:
    def __init__(self, sim: StretchSimulation):
        self.sim = sim

    def _move(self, name: str, value: float, stage: str, rate: float = .07) -> None:
        sim = self.sim
        sim.stage = stage
        sim.command_base(0, 0)
        initial = float(sim.data.ctrl[sim.actuator_ids[name]])
        steps = max(1, math.ceil(abs(value-initial)/rate/sim.model.opt.timestep))
        sim.event("phase_start", actuator=name, target=value)
        for step in range(steps):
            sim.set_control(name, initial+(value-initial)*(step+1)/steps)
            sim.step()
        for _ in range(400):
            sim.step()
        sim.event("phase_end", actuator=name, target=value)

    def _rotate(self, yaw: float) -> None:
        sim = self.sim
        origin = np.array(sim.base_pose()[:2])
        for _ in range(12000):
            x, y, current_yaw = sim.base_pose()
            error = wrap_angle(yaw - current_yaw)
            if abs(error) < .006 and sim.base_speed()[1] < .015:
                break
            omega = float(np.clip(error*1.5, -.4, .4))
            direction = np.array([math.cos(current_yaw), math.sin(current_yaw)])
            # Asymmetric mast/payload loading shifts Stretch's physical turn
            # centre. Wheel feedback keeps the base origin at the docking point.
            forward = float(np.clip(1.8*np.dot(origin-np.array([x, y]), direction)+.025*omega, -.035, .035))
            sim.command_base(forward, omega)
            sim.step()
        else:
            raise HomeSkillError("NAVIGATION_TIMEOUT", "Could not reach required base yaw")
        sim.command_base(0, 0)
        for _ in range(150):
            sim.step()

    def navigate(self, path: Iterable[Iterable[float]], final_yaw: float | None = None) -> dict[str, Any]:
        sim = self.sim
        points = [np.asarray(tuple(point)[:2], dtype=float) for point in path]
        if not points or any(point.shape != (2,) or not np.isfinite(point).all() for point in points):
            raise HomeSkillError("INVALID_PATH", "Navigation needs finite xy waypoints")
        if float(sim.data.ctrl[sim.actuator_ids["arm_extend"]]) > .035:
            raise HomeSkillError("TRANSPORT_POSTURE", "Retract telescope before navigating")
        if sim.held_object_id:
            raise HomeSkillError("USE_CARRY", "A held object requires the carry skill")
        return self._navigate(points, final_yaw, "navigate")

    def _navigate(self, points: list[np.ndarray], final_yaw: float | None, stage: str) -> dict[str, Any]:
        sim = self.sim
        sim.stage = stage
        initial = np.array(sim.base_pose()[:2])
        travelled = 0.0
        previous = initial.copy()
        for index, target in enumerate(points):
            delta = target - np.array(sim.base_pose()[:2])
            if np.linalg.norm(delta) < .010:
                continue
            self._rotate(math.atan2(delta[1], delta[0]))
            for _ in range(40000):
                x, y, yaw = sim.base_pose()
                delta = target - np.array([x, y])
                distance = float(np.linalg.norm(delta))
                tolerance = .004 if index == len(points)-1 else .012
                if distance < tolerance:
                    break
                heading = wrap_angle(math.atan2(delta[1], delta[0]) - yaw)
                # Stop advancing while turning toward the segment; this keeps
                # the registered whole-body footprint inside the planned route.
                linear = min(.16, distance*.8) * max(0.0, math.cos(heading)) if abs(heading) < .35 else 0
                angular = float(np.clip(heading*2.0, -.4, .4))
                sim.command_base(linear, angular)
                sim.step()
                current = np.array(sim.base_pose()[:2])
                travelled += float(np.linalg.norm(current-previous))
                previous = current
            else:
                raise HomeSkillError("NAVIGATION_TIMEOUT", "Waypoint tracking exceeded 80 simulated seconds")
            sim.command_base(0, 0)
            for _ in range(100):
                sim.step()
        if final_yaw is not None:
            if not math.isfinite(final_yaw):
                raise HomeSkillError("INVALID_TARGET", "Final yaw must be finite")
            self._rotate(final_yaw)
        sim.stop()
        sim.stage = stage
        error = float(np.linalg.norm(np.array(sim.base_pose()[:2])-points[-1]))
        if error > .02:
            raise HomeSkillError("DOCKING_FAILED", "Final xy error exceeds 20 mm", {"xy_error_m": error})
        return sim.event("arrived", base_pose=list(sim.base_pose()), xy_error_m=error,
                         physical_distance_m=travelled, initial_xy=initial.tolist())

    def _stationary(self) -> None:
        if self.sim.base_speed()[0] > .01 or self.sim.base_speed()[1] > .03:
            raise HomeSkillError("BASE_MOVING", "Stop the base before manipulating")

    def _aim(self, object_position: np.ndarray) -> float:
        sim = self.sim
        wrist = sim.data.body("base_link")
        local = wrist.xmat.reshape(3, 3).T @ (object_position-wrist.xpos)
        # Default wrist yaw=0, side approach along the base's -y axis.
        if abs(local[0] - (-.021385)) > .018:
            raise HomeSkillError("UNREACHABLE_GRASP", "Object lateral error exceeds restricted grasp envelope", {"object_in_base_m": local.tolist()})
        # Grip on the forward half of the rubber discs, clear of the proximal
        # metal finger links as their hinges close.
        extension = float(-local[1]-.34036-.008)
        if not .06 <= extension <= .49:
            raise HomeSkillError("UNREACHABLE_GRASP", "Object outside validated telescope reach", {"extension_m": extension})
        return extension

    def pick(self, object_id: str) -> dict[str, Any]:
        sim = self.sim
        self._stationary()
        if object_id not in sim.world.objects or sim.held_object_id:
            raise HomeSkillError("PICK_PRECONDITION", "Pick requires a known object and an empty hand")
        obj = sim.world.objects[object_id]
        if "pick" not in obj.operations:
            raise HomeSkillError("OBJECT_PERMISSION", f"Object {object_id} does not permit pick")
        if not grasp_supported(obj):
            raise HomeSkillError("UNSUPPORTED_GRASP", "Validated box grasp requires 38-60 mm width, 38-70 mm height, mass <= 0.15 kg")
        start = sim.data.xpos[sim.object_body_ids[object_id]].copy()
        extension = self._aim(start)
        initial_supports = sim.object_contact_evidence(object_id)["supports"]
        if len(initial_supports) != 1:
            raise HomeSkillError("PICK_PRECONDITION", "Object must initially be independently supported")
        sim.target_object_id = object_id
        sim.contact_support_id = initial_supports[0]
        lift_target = float(start[2] - sim.data.body("base_link").xpos[2] - .532 + .02)
        lo, hi = sim.model.actuator_ctrlrange[sim.actuator_ids["lift"]]
        if not lo <= lift_target <= hi - .15:
            raise HomeSkillError("UNREACHABLE_HEIGHT", "Grasp height needs 100 mm verified lift clearance", {"requested_lift_m": lift_target})
        self._move("grip", .035, "pick_open", .025)
        self._move("lift", min(float(hi), max(.30, lift_target + .15)), "pick_pregrasp")
        self._move("arm_extend", extension, "pick_approach")
        self._move("lift", lift_target, "pick_descend")
        self._move("grip", -.005, "pick_close", .015)
        evidence = sim.object_contact_evidence(object_id)
        if evidence["finger_contact_side_count"] != 2:
            raise HomeSkillError("GRASP_EMPTY", "No measured bilateral rubber contact", evidence)
        sim.held_object_id = object_id
        sim.grasp_offset = sim.object_in_wrist(object_id).copy()
        sim.grasp_start_z = float(start[2])
        sim.event("grasp_verified", object_id=object_id, **evidence)
        # Ground grasps must clear the chassis before retracting. Merely adding
        # 15 cm to a floor-height grasp leaves the hand inside the base envelope.
        self._move("lift", max(.20, lift_target+.15), "pick_lift")
        lifted = sim.data.xpos[sim.object_body_ids[object_id]].copy()
        achieved = float(lifted[2]-start[2])
        evidence = sim.object_contact_evidence(object_id)
        if achieved < .10 or evidence["supports"] or evidence["finger_contact_side_count"] != 2:
            raise HomeSkillError("LIFT_FAILED", "Lift needs >100 mm rise, bilateral contact and no support", {"lift_m": achieved, **evidence})
        self._move("arm_extend", .02, "pick_retract")
        sim.target_object_id = None
        sim.contact_support_id = None
        return sim.event("pick_completed", object_id=object_id, achieved_lift_m=achieved,
                         object_position_m=sim.data.xpos[sim.object_body_ids[object_id]].tolist(),
                         wrist_relative_m=sim.grasp_offset.tolist(), **sim.object_contact_evidence(object_id))

    def carry(self, path: Iterable[Iterable[float]], final_yaw: float | None = None) -> dict[str, Any]:
        sim = self.sim
        if not sim.held_object_id or sim.object_contact_evidence(sim.held_object_id)["finger_contact_side_count"] != 2:
            raise HomeSkillError("CARRY_EMPTY", "Carry requires a verified bilateral grasp")
        if float(sim.data.ctrl[sim.actuator_ids["arm_extend"]]) > .035:
            raise HomeSkillError("TRANSPORT_POSTURE", "Carry requires a retracted telescope")
        points = [np.asarray(tuple(point)[:2], dtype=float) for point in path]
        if not points or any(point.shape != (2,) or not np.isfinite(point).all() for point in points):
            raise HomeSkillError("INVALID_PATH", "Carry needs finite xy waypoints")
        initial = sim.data.xpos[sim.object_body_ids[sim.held_object_id]].copy()
        arrival = self._navigate(points, final_yaw, "carry")
        displacement = float(np.linalg.norm(sim.data.xpos[sim.object_body_ids[sim.held_object_id]]-initial))
        evidence = sim.object_contact_evidence(sim.held_object_id)
        if evidence["finger_contact_side_count"] != 2 or evidence["supports"]:
            raise HomeSkillError("OBJECT_SLIPPED", "Carry ended without an unsupported bilateral grasp", evidence)
        return sim.event("carry_completed", object_id=sim.held_object_id,
                         object_displacement_m=displacement, arrival=arrival, **evidence)

    def place(self, surface_id: str, target_xy: Iterable[float]) -> dict[str, Any]:
        sim = self.sim
        self._stationary()
        if not sim.held_object_id or (surface_id != "floor" and surface_id not in sim.world.furniture):
            raise HomeSkillError("PLACE_PRECONDITION", "Place requires a held object and a known support")
        object_id = sim.held_object_id
        obj = sim.world.objects[object_id]
        support = lookup_surface(sim.world, surface_id)
        if "place" not in obj.operations:
            raise HomeSkillError("OBJECT_PERMISSION", f"Object {object_id} does not permit place")
        if not 0 <= support.top_z <= MAX_PLACE_SURFACE_HEIGHT_M:
            raise HomeSkillError("UNREACHABLE_HEIGHT", "Support height needs room for approach and release clearance", {"support_height_m": support.top_z, "maximum_m": MAX_PLACE_SURFACE_HEIGHT_M})
        xy = np.asarray(tuple(target_xy), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise HomeSkillError("INVALID_TARGET", "Placement target requires finite xy")
        if any(abs(xy[axis]-support.position_m[axis])+obj.half_size_m[axis]+.025 > support.half_size_m[axis] for axis in (0, 1)):
            raise HomeSkillError("PLACE_EDGE", "Full object footprint needs 25 mm table edge margin")
        current = sim.data.xpos[sim.object_body_ids[object_id]].copy()
        desired = np.array([*xy, current[2]])
        extension = self._aim(desired)
        sim.contact_support_id = surface_id
        clearance_z = support.top_z + obj.half_size_m[2] + .08
        if current[2] < clearance_z:
            lift = float(sim.data.ctrl[sim.actuator_ids["lift"]]) + clearance_z - float(current[2])
            if lift > sim.model.actuator_ctrlrange[sim.actuator_ids["lift"]][1]:
                raise HomeSkillError("UNREACHABLE_HEIGHT", "Loaded grasp cannot clear the support before extension")
            self._move("lift", lift, "place_preapproach")
        self._move("arm_extend", extension, "place_approach")
        # Calibrated open-hand reach does not fully describe a closed, loaded
        # grasp (especially after a ground pick). Align the actual held body
        # through the same telescope actuator before touching the support.
        for _ in range(3):
            current = sim.data.xpos[sim.object_body_ids[object_id]].copy()
            delta = np.array([xy[0] - current[0], xy[1] - current[1], 0.])
            local_delta = sim.data.body("base_link").xmat.reshape(3, 3).T @ delta
            if np.linalg.norm(delta[:2]) <= .006:
                break
            if abs(local_delta[0]) > .020:
                raise HomeSkillError("UNREACHABLE_ALIGNMENT", "Dock yaw/lateral error exceeds closed-grasp correction", {"delta_in_base_m": local_delta.tolist()})
            correction = float(sim.data.ctrl[sim.actuator_ids["arm_extend"]] - local_delta[1])
            if not .06 <= correction <= .49:
                raise HomeSkillError("UNREACHABLE_ALIGNMENT", "Loaded grasp needs extension beyond the physical envelope", {"extension_m": correction})
            self._move("arm_extend", correction, "place_approach_align", .035)
        # Relative closed-hand grasp offsets are measured, never guessed.
        current_z = float(sim.data.xpos[sim.object_body_ids[object_id]][2])
        target_z = support.top_z+obj.half_size_m[2]+.003
        lift = float(sim.data.ctrl[sim.actuator_ids["lift"]])+target_z-current_z
        lower, upper = sim.model.actuator_ctrlrange[sim.actuator_ids["lift"]]
        # Ground support can occur at the lift's lower limit. A small measured
        # calibration offset may request beyond it; never extend the joint or
        # teleport the object to satisfy the target.
        if lift < lower - .012 or lift > upper + .005:
            raise HomeSkillError("UNREACHABLE_HEIGHT", "Support height lies outside the current lift reach", {"requested_lift_m": lift, "range_m": [float(lower), float(upper)]})
        lift = float(np.clip(lift, lower, upper))
        self._move("lift", lift, "place_lower", .045)
        # Opening fully in place sweeps the long hinged metal links past the
        # object. First release with a partial aperture, then withdraw along
        # the approach axis before completing the finger-opening arc.
        self._move("grip", .020, "place_release_partial", .015)
        release_extension = max(.02, float(sim.data.ctrl[sim.actuator_ids["arm_extend"]])-.08)
        self._move("arm_extend", release_extension, "place_release_clear", .035)
        release_evidence = sim.object_contact_evidence(object_id)
        if release_evidence["robot_contact"] or surface_id not in release_evidence["supports"]:
            raise HomeSkillError("RELEASE_FAILED", "Withdrawn hand must separate from the independently supported object", release_evidence)
        sim.event("release_verified", object_id=object_id, **release_evidence)
        sim.held_object_id = None
        sim.grasp_offset = None
        sim.target_object_id = None
        if surface_id == "floor":
            # Clear the ground under the declared release transition before
            # sweeping the fully open fingers or retracting toward the chassis.
            self._move("lift", .30, "place_release_retreat")
        sim.contact_support_id = None
        self._move("grip", .035, "place_open_clear", .020)
        self._move("lift", .30, "place_retreat")
        self._move("arm_extend", .02, "place_retract")
        sim.stage = "place_settle"
        sim.wait(1.0)
        evidence = sim.object_contact_evidence(object_id)
        position = sim.data.xpos[sim.object_body_ids[object_id]].copy()
        joint = sim.model.joint(obj.joint_name)
        speed = float(np.linalg.norm(sim.data.qvel[int(joint.dofadr[0]):int(joint.dofadr[0])+3]))
        xy_error = float(np.linalg.norm(position[:2]-xy))
        result = {"object_id": object_id, "surface_id": surface_id, "target_xy_m": xy.tolist(),
                  "position_m": position.tolist(), "xy_error_m": xy_error,
                  "linear_speed_m_s": speed, "released": not evidence["robot_contact"],
                  "stable_supported": surface_id in evidence["supports"], **evidence}
        if not result["released"] or not result["stable_supported"] or speed > .01 or xy_error > .025:
            raise HomeSkillError("PLACE_FAILED", "Object must be released, independently supported, settled and within 25 mm", result)
        return sim.event("place_completed", **result)

    def safe_stop(self, *, emergency: bool = False) -> dict[str, Any]:
        return self.sim.stop(emergency=emergency)

    def wait(self, seconds: float = .5) -> dict[str, Any]:
        self.sim.stage = "wait"
        return self.sim.wait(seconds)

"""Stretch 2 physics, actuator adaptation and measured safety telemetry.

Only construction initializes qpos. Every task motion uses the official wheel,
lift, telescope and finger actuators; objects remain independent free bodies.
"""
from __future__ import annotations

import math
import os
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from ..paths import PROJECT_ROOT

STRETCH_ASSET = PROJECT_ROOT / "assets" / "third_party" / "hello_robot_stretch"
COLLISION_NORMAL_FORCE_THRESHOLD_N = .02
_LOAD_LOCK = threading.RLock()


class HomeSkillError(RuntimeError):
    def __init__(self, code: str, message: str, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.evidence = evidence or {}


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


class StretchSimulation:
    """One live MjModel/MjData pair shared with the viewer and execution guards."""

    def __init__(self, world: Any, on_step: Callable[["StretchSimulation"], None] | None = None,
                 initial_pose: tuple[float, float, float] | None = None):
        from ..maps.home_scene import build_home_scene, home_geometry_names

        self.world = world
        xml = build_home_scene(world, robot_xml=STRETCH_ASSET / "stretch.xml")
        root = ET.fromstring(xml)
        root.find("compiler").set("assetdir", "assets")
        root.find("compiler").attrib.pop("meshdir", None)
        root.find("compiler").attrib.pop("texturedir", None)
        root.find("option").set("timestep", ".002")
        # Naming by subtree makes the collision registry independent of Panda.
        base_xml = root.find("worldbody/body[@name='base_link']")
        for index, geom in enumerate(base_xml.iter("geom")):
            if not geom.get("name"):
                geom.set("name", f"stretch_geom_{index}")
        # The official XML declares some mesh masses only on visual geoms.
        # The complete original model is retained, including collision shapes.
        with _LOAD_LOCK:
            previous = Path.cwd()
            try:
                os.chdir(STRETCH_ASSET)
                self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
            finally:
                os.chdir(previous)
        self.data = mujoco.MjData(self.model)
        self.on_step = on_step
        self.stage = "initialize"
        self.total_steps = 0
        self.held_object_id: str | None = None
        self.target_object_id: str | None = None
        self.contact_support_id: str | None = None
        self.grasp_offset: np.ndarray | None = None
        self.grasp_start_z: float | None = None
        self._missing_contact_steps = 0
        self._base_command = (0.0, 0.0)
        self._wrist_target = 0.0
        self.events: list[dict[str, Any]] = []
        self.actuator_ids = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.base_body_id = self.model.body("base_link").id
        self.object_body_ids = {key: self.model.body(obj.body_name).id for key, obj in world.objects.items()}
        self.object_geom_ids = {key: self.model.geom(obj.geom_name).id for key, obj in world.objects.items()}
        robot_bodies: set[int] = {self.base_body_id}
        for body in range(1, self.model.nbody):
            if int(self.model.body_parentid[body]) in robot_bodies:
                robot_bodies.add(body)
        self.robot_body_ids = frozenset(robot_bodies)
        self.robot_geom_ids = frozenset(i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) in robot_bodies)
        self.furniture_geom_ids = {key: frozenset(self.model.geom(name).id for name in item.geom_names)
                                   for key, item in world.furniture.items()}
        geometry_registry = home_geometry_names(world)
        self.floor_geom_id = self.model.geom("home_floor").id
        self.furniture_geom_ids["floor"] = frozenset({self.floor_geom_id})
        self.room_geom_ids = frozenset(self.model.geom(name).id for name in geometry_registry["room"] if name != "home_floor")
        self.obstacle_geom_ids = frozenset(self.room_geom_ids.union(
            *(geoms for key, geoms in self.furniture_geom_ids.items() if key != "floor")))
        self.finger_geom_ids: dict[str, frozenset[int]] = {}
        self.grasp_geom_ids: set[int] = set()
        for side in ("left", "right"):
            body_id = self.model.body(f"rubber_tip_{side}").id
            self.finger_geom_ids[side] = frozenset(i for i in self.robot_geom_ids
                                                  if int(self.model.geom_bodyid[i]) == body_id)
            finger_body = self.model.body(f"link_gripper_finger_{side}").id
            self.grasp_geom_ids.update(i for i in self.robot_geom_ids if int(self.model.geom_bodyid[i]) in (body_id, finger_body))
        pose = initial_pose or (*world.navigation.initial_position_m[:2], world.navigation.initial_yaw_rad)
        # Initialization is the sole permitted free-joint assignment.
        self.data.qpos[:7] = [pose[0], pose[1], .001, math.cos(pose[2]/2), 0, 0, math.sin(pose[2]/2)]
        for joint, value in (("joint_lift", .30), ("joint_gripper_slide", .035),
                             ("joint_gripper_finger_left_open", .35), ("joint_gripper_finger_right_open", .35)):
            self.data.qpos[self.model.joint(joint).qposadr] = value
        self.set_control("lift", .30)
        self.set_control("grip", .035)
        mujoco.mj_forward(self.model, self.data)

    def set_control(self, name: str, value: float) -> None:
        index = self.actuator_ids[name]
        lo, hi = self.model.actuator_ctrlrange[index]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise HomeSkillError("CONTROL_RANGE", f"{name} target {value} outside [{lo}, {hi}]")
        self.data.ctrl[index] = value
        if name == "wrist_yaw":
            self._wrist_target = value

    def command_base(self, linear_m_s: float, angular_rad_s: float) -> None:
        if not all(math.isfinite(v) for v in (linear_m_s, angular_rad_s)):
            raise HomeSkillError("INVALID_TARGET", "Base command must be finite")
        self._base_command = (float(np.clip(linear_m_s, -.20, .20)), float(np.clip(angular_rad_s, -.45, .45)))

    def _wheel_control(self) -> None:
        # Both tendons use coefficients +/- .5 with gear=3. The tire radius
        # is .05 m and the wheel centres are 0.3407 m apart.
        right = float(self.data.qvel[self.model.joint("joint_right_wheel").dofadr][0])
        left = float(self.data.qvel[self.model.joint("joint_left_wheel").dofadr][0])
        # Official wheel axes rotate positive rates toward base -x. The turn
        # tendon similarly produces negative world yaw for positive control.
        target_forward = -self._base_command[0] / .05
        target_turn = -self._base_command[1] * .3407 / (.05 * 2)
        forward = .20 * target_forward + .32 * (target_forward - (right + left)/2)
        turn = (.20 * target_turn + .32 * (target_turn - (right - left)/2)
                -.65*(self._base_command[1]-float(self.data.qvel[5])))
        self.data.ctrl[self.actuator_ids["forward"]] = np.clip(forward, -1, 1)
        self.data.ctrl[self.actuator_ids["turn"]] = np.clip(turn, -1, 1)

    def _wrist_control(self) -> None:
        # The supplied general yaw actuator has only 1 Nm/rad position gain.
        # External feedback through that same actuator holds the validated side
        # grasp orientation during finger closure/release without editing MJCF.
        joint = self.model.joint("joint_wrist_yaw")
        angle = float(self.data.qpos[joint.qposadr][0])
        velocity = float(self.data.qvel[joint.dofadr][0])
        control = self._wrist_target+18*(self._wrist_target-angle)-.9*velocity
        actuator = self.actuator_ids["wrist_yaw"]
        self.data.ctrl[actuator] = np.clip(control, *self.model.actuator_ctrlrange[actuator])

    def step(self, *, check: bool = True, notify: bool = True) -> None:
        self._wheel_control()
        self._wrist_control()
        mujoco.mj_step(self.model, self.data)
        self.total_steps += 1
        if check:
            self.check_physics()
        if notify and self.on_step:
            self.on_step(self)

    def event(self, event: str, **evidence: Any) -> dict[str, Any]:
        item = {"event": event, "stage": self.stage, "step": self.total_steps,
                "simulation_time_s": float(self.data.time), **evidence}
        self.events.append(item)
        return item

    def base_pose(self) -> tuple[float, float, float]:
        pos = self.data.xpos[self.base_body_id]
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        return float(pos[0]), float(pos[1]), math.atan2(rotation[1, 0], rotation[0, 0])

    def base_speed(self) -> tuple[float, float]:
        return float(np.linalg.norm(self.data.qvel[:2])), float(abs(self.data.qvel[5]))

    def object_positions(self) -> dict[str, list[float]]:
        return {key: self.data.xpos[body].tolist() for key, body in self.object_body_ids.items()}

    def contacts(self) -> list[dict[str, Any]]:
        result = []
        for index, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            result.append({"geom_ids": [int(contact.geom1), int(contact.geom2)],
                           "geom_names": [self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name],
                           "distance_m": float(contact.dist), "normal_force_n": float(force[0]),
                           "normal_force_threshold_n": COLLISION_NORMAL_FORCE_THRESHOLD_N})
        return result

    def object_contact_evidence(self, object_id: str) -> dict[str, Any]:
        obj_geom = self.object_geom_ids[object_id]
        pairs = [item for item in self.contacts() if obj_geom in item["geom_ids"]]
        sides = {side for side, geoms in self.finger_geom_ids.items()
                 if any(any(geom in geoms for geom in item["geom_ids"]) and item["normal_force_n"] > .05 for item in pairs)}
        supports = [support for support, geoms in self.furniture_geom_ids.items()
                    if any(any(geom in geoms for geom in item["geom_ids"]) and item["normal_force_n"] > .05 for item in pairs)]
        robot_contact = any(any(geom in self.robot_geom_ids for geom in item["geom_ids"]) for item in pairs)
        return {"finger_contact_sides": sorted(sides), "finger_contact_side_count": len(sides),
                "supports": supports, "robot_contact": robot_contact, "contact_pairs": pairs}

    def gripper_center(self) -> np.ndarray:
        return (self.data.body("rubber_tip_left").xpos + self.data.body("rubber_tip_right").xpos)/2

    def object_in_wrist(self, object_id: str) -> np.ndarray:
        wrist = self.data.body("link_wrist_yaw")
        return wrist.xmat.reshape(3, 3).T @ (self.data.xpos[self.object_body_ids[object_id]] - wrist.xpos)

    def observe(self) -> dict[str, Any]:
        return {"stage": self.stage, "step": self.total_steps, "time_s": float(self.data.time),
                "base_pose": list(self.base_pose()), "base_speed": list(self.base_speed()),
                "object_positions": self.object_positions(), "held_object_id": self.held_object_id,
                "contact_pairs": self.contacts(), "controls": dict(zip(self.actuator_ids, self.data.ctrl.tolist()))}

    def footprint_evidence(self, clearance_m: float = .015) -> dict[str, Any]:
        """Conservative radial projection of registered collision geometry.

        This includes the actual side-offset hand and held object's rotation,
        rather than adding only object half-size to a nominal base radius.
        """
        origin = self.data.xpos[self.base_body_id, :2]
        radii: dict[str, float] = {}
        geoms = set(self.robot_geom_ids)
        if self.held_object_id:
            geoms.add(self.object_geom_ids[self.held_object_id])
        signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for geom in geoms:
            if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                continue
            kind = self.model.geom_type[geom]
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            position = self.data.geom_xpos[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[geom]
                start = self.model.mesh_vertadr[mesh]
                vertices = self.model.mesh_vert[start:start+self.model.mesh_vertnum[mesh]]
                local = vertices
            else:
                size = self.model.geom_size[geom].copy()
                if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    size = np.array([size[0], size[0], size[1]])
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    size[:] = size[0]
                local = signs*size
            projected = (local @ rotation.T + position)[:, :2]-origin
            radii[self.model.geom(geom).name] = float(np.max(np.linalg.norm(projected, axis=1)))
        physical = max(radii.values())
        return {"radius_m": physical+clearance_m, "physical_radius_m": physical,
                "clearance_m": clearance_m, "held_object_id": self.held_object_id,
                "maximum_geom": max(radii, key=radii.get)}

    def transport_footprint_radius(self) -> float:
        return float(self.footprint_evidence()["radius_m"])

    def check_physics(self) -> None:
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise HomeSkillError("PHYSICS_NONFINITE", "Non-finite simulator state")
        if any(self.data.warning.number):
            raise HomeSkillError("PHYSICS_WARNING", "MuJoCo emitted a physics warning")
        if self.data.xmat[self.base_body_id].reshape(3, 3)[2, 2] < math.cos(.12):
            raise HomeSkillError("BASE_UNSTABLE", "Stretch base tilt exceeds 0.12 rad")
        for pair in self.contacts():
            a, b = pair["geom_ids"]
            if pair["normal_force_n"] <= COLLISION_NORMAL_FORCE_THRESHOLD_N:
                continue
            if self.floor_geom_id in (a, b):
                other = b if a == self.floor_geom_id else a
                if other in self.robot_geom_ids:
                    body = self.model.body(int(self.model.geom_bodyid[other])).name
                    ground_running = body in {"base_link", "link_right_wheel", "link_left_wheel"}
                    controlled_tip = (other in self.grasp_geom_ids and self.contact_support_id == "floor"
                        and self.stage.startswith(("pick_descend", "pick_close", "place_lower", "place_release"))
                        and pair["normal_force_n"] <= 2.)
                    if not ground_running and not controlled_tip:
                        raise HomeSkillError("ROBOT_FLOOR_COLLISION", "Only chassis/wheels or a low-force declared grasp transition may touch the floor", pair)
                if self.held_object_id and other == self.object_geom_ids[self.held_object_id]:
                    allowed_floor = self.contact_support_id == "floor" and self.stage.startswith(("pick_lift", "place_lower", "place_release"))
                    if not allowed_floor:
                        raise HomeSkillError("HELD_OBJECT_COLLISION", "Held object touched the floor outside its support transition", pair)
            if ((a in self.robot_geom_ids and b in self.obstacle_geom_ids) or
                    (b in self.robot_geom_ids and a in self.obstacle_geom_ids)):
                raise HomeSkillError("ROBOT_FURNITURE_COLLISION", "Robot touched furniture or a wall", pair)
            if self.held_object_id is not None:
                held_geom = self.object_geom_ids[self.held_object_id]
                if held_geom in (a, b):
                    other = b if a == held_geom else a
                    if other in self.object_geom_ids.values():
                        raise HomeSkillError("HELD_OBJECT_COLLISION", "Held object contacted another object", pair)
                    if other in self.obstacle_geom_ids:
                        allowed_support = (self.contact_support_id is not None
                            and other in self.furniture_geom_ids[self.contact_support_id]
                            and self.stage.startswith(("pick_lift", "place_lower", "place_release")))
                        if not allowed_support:
                            raise HomeSkillError("HELD_OBJECT_COLLISION", "Held object contacted furniture outside its declared support transition", pair)
            for object_id, obj_geom in self.object_geom_ids.items():
                if obj_geom not in (a, b):
                    continue
                other = b if a == obj_geom else a
                if other in self.robot_geom_ids:
                    allowed = object_id in (self.target_object_id, self.held_object_id) and other in self.grasp_geom_ids
                    if not allowed:
                        raise HomeSkillError("UNEXPECTED_OBJECT_CONTACT", f"Robot contacted {object_id} outside registered gripper fingers", pair)
        if self.held_object_id and not self.stage.startswith(("place_", "safe_stop")):
            evidence = self.object_contact_evidence(self.held_object_id)
            self._missing_contact_steps = self._missing_contact_steps + 1 if evidence["finger_contact_side_count"] < 2 else 0
            if self._missing_contact_steps > 100:
                raise HomeSkillError("OBJECT_SLIPPED", "Bilateral grasp lost for 0.2 simulated seconds", evidence)
            if self.grasp_offset is not None:
                drift = float(np.linalg.norm(self.object_in_wrist(self.held_object_id) - self.grasp_offset))
                if drift > .035:
                    raise HomeSkillError("OBJECT_SLIPPED", "Object moved > 35 mm relative to wrist", {"wrist_relative_drift_m": drift})

    def wait(self, seconds: float = .5) -> dict[str, Any]:
        if not math.isfinite(seconds) or not 0 <= seconds <= 30:
            raise HomeSkillError("INVALID_WAIT", "Wait duration must be in [0, 30] seconds")
        self.command_base(0, 0)
        for _ in range(math.ceil(seconds / self.model.opt.timestep)):
            self.step()
        return self.event("wait_completed", seconds=seconds, base_speed=list(self.base_speed()))

    def stop(self, *, emergency: bool = False) -> dict[str, Any]:
        self.stage = "safe_stop" if emergency else "stop"
        self.command_base(0, 0)
        for _ in range(600):
            if emergency:
                # Keep physics and actuator braking active after a latched guard.
                # Recording/viewer callbacks still receive every stopped state.
                try:
                    self.step(check=False)
                except HomeSkillError:
                    pass
            else:
                self.step()
        linear, angular = self.base_speed()
        released_after_slip = None
        if emergency and self.held_object_id:
            name = self.held_object_id
            obj = self.world.objects[name]
            contact = self.object_contact_evidence(name)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                    self.object_body_ids[name], velocity, 0)
            from ..maps.manipulation import lookup_surface
            position = self.data.xpos[self.object_body_ids[name]]
            rotation = self.data.geom_xmat[self.object_geom_ids[name]].reshape(3, 3)
            extents = np.abs(rotation) @ np.asarray(obj.half_size_m)
            stable_supports = []
            for support_id in contact["supports"]:
                support = lookup_surface(self.world, support_id)
                if (abs(position[2] - extents[2] - support.top_z) <= .012
                        and all(abs(position[i] - support.position_m[i]) + extents[i] + .01 <= support.half_size_m[i]
                                for i in (0, 1))):
                    stable_supports.append(support_id)
            # A failed grasp is not a permanent software attachment. Clear it
            # only after braking proves a non-fragile free body has separated
            # from the robot and settled on an independent registered support.
            if ("fragile" not in obj.risk_tags and obj.states.get("damage") != "damaged"
                    and not contact["robot_contact"] and stable_supports
                    and np.linalg.norm(velocity[3:]) <= .025 and np.linalg.norm(velocity[:3]) <= .2):
                released_after_slip = {"object_id": name, "supports": stable_supports,
                    "velocity": velocity.tolist(), "position_m": self.data.xpos[self.object_body_ids[name]].tolist(),
                    "contact_evidence": contact}
                self.held_object_id = None
                self.grasp_offset = self.grasp_start_z = None
                self.target_object_id = self.contact_support_id = None
                self._missing_contact_steps = 0
        evidence = self.event("safe_stop" if emergency else "stopped", linear_speed_m_s=linear,
                              angular_speed_rad_s=angular, arm_controls=self.data.ctrl[2:6].tolist(),
                              held_contact=None if self.held_object_id is None else self.object_contact_evidence(self.held_object_id),
                              released_after_slip=released_after_slip is not None,
                              released_evidence=released_after_slip)
        if linear > .01 or angular > .03:
            raise HomeSkillError("STOP_FAILED", "Base did not settle under wheel braking", evidence)
        if self.held_object_id:
            held = self.object_contact_evidence(self.held_object_id)
            drift = 0.0 if self.grasp_offset is None else float(np.linalg.norm(self.object_in_wrist(self.held_object_id)-self.grasp_offset))
            if held["finger_contact_side_count"] != 2 or held["supports"] or drift > .035:
                raise HomeSkillError("STOP_HOLD_FAILED", "Safe stop did not retain an unsupported bilateral grasp", {"wrist_relative_drift_m": drift, **held})
        return evidence

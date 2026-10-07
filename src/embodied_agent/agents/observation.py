"""Captured world views used by the shared instruction agent, without simulation."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from embodied_agent.models.contracts import PlannerError


@dataclass(frozen=True)
class WorldObservation:
    domain: str
    snapshot: dict
    objects: dict
    supports: dict
    robot: dict
    world_version: Any = None

    @classmethod
    def capture(cls, snapshot: dict, world: Any) -> "WorldObservation":
        if not isinstance(snapshot, dict):
            raise PlannerError("INVALID_OBSERVATION", "Instruction planning requires a captured observation")
        state = copy.deepcopy(snapshot)
        state.pop("untrusted_descriptions", None)
        state.pop("events", None)
        definition = world.to_dict()
        home = "furniture" in definition and "objects" in definition
        if home:
            objects = copy.deepcopy(state.get("objects", {}))
            if set(objects) != set(definition["objects"]):
                raise PlannerError("INVALID_OBSERVATION", "Captured object IDs differ from the registered world")
            supports = copy.deepcopy(definition["furniture"])
            supports["floor"] = {"kind": "floor", "position_m": [0, 0, 0], "top_z_m": 0}
            robot = copy.deepcopy(state.get("robot", {}))
        else:
            objects = copy.deepcopy(state.get("objects", {}))
            # Panda's adapter exposes the simulated cube as one registered entity.
            cube = state.get("cube")
            if not objects and isinstance(cube, dict):
                object_id = cube.get("object_id", "cube")
                objects[object_id] = {**copy.deepcopy(cube),
                    "position_m": cube.get("position_m", cube.get("position_xyz_m")),
                    "operations": ["inspect", "pick", "place"]}
            if not objects:
                position = state.get("cube_xyz_m", state.get("cube_position_m"))
                if position is None:
                    raise PlannerError("INVALID_OBSERVATION", "Captured desktop state lacks registered object positions")
                objects["cube"] = {"position_m": copy.deepcopy(position), "operations": ["inspect", "pick", "place"]}
            supports = copy.deepcopy(state.get("targets", state.get("target_regions", definition.get("targets", {}))))
            if "table" not in supports:
                from embodied_agent.maps.schema import TABLE_CENTER, TABLE_HALF_SIZE, TABLE_TOP_Z
                supports["table"] = {"kind": "table", "position_m": [*TABLE_CENTER, TABLE_TOP_Z],
                                     "half_size_m": [*TABLE_HALF_SIZE, 0.001]}
            robot = copy.deepcopy(state.get("robot", {}))
            if "held_object" not in robot:
                held = state.get("held_estimate", cube.get("held_estimate") if isinstance(cube, dict) else None)
                robot["held_object"] = next(iter(objects)) if held is True else None
        for key, obj in objects.items():
            if not isinstance(obj, dict) or not isinstance(obj.get("position_m"), (list, tuple)):
                raise PlannerError("INVALID_OBSERVATION", f"Captured entity {key!r} lacks a measured position")
        return cls("home" if home else "desktop", state, objects, supports, robot,
                   state.get("world_version", state.get("obs_id")))

    def to_dict(self) -> dict:
        return copy.deepcopy({"domain": self.domain, "objects": self.objects,
            "supports": self.supports, "robot": self.robot, "world_version": self.world_version})

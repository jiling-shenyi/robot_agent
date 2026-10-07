"""Independent, pure Python home map and state-dependent project risk rules.

All distances are metres; geometry uses box half extents. Descriptions are
untrusted text. Only the finite state vocabulary below affects project rules.
Heat, liquids and damage are declared project rules, not material simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from embodied_agent.maps.schema import MAP_ID_PATTERN, WorldError, finite_vector

RULE_VERSION = "home-rules-v1"
DAMAGE_RULE_VERSION = "fragile-project-v1"
DAMAGE_DROP_HEIGHT_M = 0.12
DAMAGE_IMPACT_SPEED_M_S = 0.8
OPERATIONS = frozenset({"inspect", "pick", "carry", "place"})
RISK_TAGS = frozenset({"fragile", "hot", "cleaning", "restricted", "liquid", "electronic"})
STATE_VALUES = {
    "temperature": {"ambient", "hot"}, "powered": {"off", "on"},
    "sealed": {"sealed", "open"}, "leak": {"dry", "leaking"},
    "damage": {"intact", "damaged"},
}
FURNITURE_KINDS = frozenset({"table", "sofa", "cabinet", "storage_box"})
CATEGORIES = frozenset({"box", "remote", "book", "glass", "vase", "kettle", "heater", "cleaning_container", "knife", "medicine", "electronic_device"})


def _keys(value: Any, required: set[str], label: str, optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - (optional or set()):
        raise WorldError("INVALID_HOME_MAP", f"{label} must contain exactly {sorted(required)}")
    return value


def _number(value: Any, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldError("INVALID_HOME_MAP", f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise WorldError("INVALID_HOME_MAP", f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise WorldError("INVALID_HOME_MAP", f"{label} has an invalid value")
    return result


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise WorldError("INVALID_HOME_MAP", f"{label} must be a nonnegative integer")
    return value


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not MAP_ID_PATTERN.fullmatch(value):
        raise WorldError("INVALID_HOME_MAP", f"{label} must be a lowercase identifier")
    return value


def _text(value: Any, label: str, limit: int = 1000) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise WorldError("INVALID_HOME_MAP", f"{label} must be text of at most {limit} characters")
    return value


def _choice(value: Any, allowed: frozenset[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise WorldError("INVALID_HOME_MAP", f"Unsupported {label}")
    return value


def _half(value: Any, label: str) -> tuple[float, float, float]:
    result = finite_vector(value, label)
    if min(result) <= 0:
        raise WorldError("INVALID_HOME_MAP", f"{label} half extents must be positive")
    return result


def _states(value: Any, label: str) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise WorldError("INVALID_HOME_STATE", f"{label} must be a finite state mapping")
    for key, state in value.items():
        if key not in STATE_VALUES or not isinstance(state, str) or state not in STATE_VALUES[key]:
            raise WorldError("INVALID_HOME_STATE", f"Unsupported state {key!r}={state!r}")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class HomeRoom:
    size_m: tuple[float, float, float]
    doorway_position_m: tuple[float, float, float]
    doorway_width_m: float

    @property
    def bounds_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-self.size_m[0] / 2, -self.size_m[1] / 2), (self.size_m[0] / 2, self.size_m[1] / 2))

    @property
    def x_min(self) -> float:
        return -self.size_m[0] / 2

    @property
    def x_max(self) -> float:
        return self.size_m[0] / 2

    @property
    def y_min(self) -> float:
        return -self.size_m[1] / 2

    @property
    def y_max(self) -> float:
        return self.size_m[1] / 2

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HomeRoom":
        _keys(value, {"size_m", "doorway_position_m", "doorway_width_m"}, "room")
        size = _half(value["size_m"], "room.size_m")
        door = finite_vector(value["doorway_position_m"], "room.doorway_position_m")
        width = _number(value["doorway_width_m"], "room.doorway_width_m", 0.4)
        if width >= size[0] or abs(door[0]) + width / 2 > size[0] / 2 or abs(door[1] + size[1] / 2) > 1e-6 or door[2] != 0:
            raise WorldError("INVALID_HOME_MAP", "Doorway must fit within the south wall at floor height")
        return cls(size, door, width)

    def to_dict(self) -> dict[str, Any]:
        return {"size_m": list(self.size_m), "doorway_position_m": list(self.doorway_position_m), "doorway_width_m": self.doorway_width_m}


@dataclass(frozen=True)
class HomeFurniture:
    furniture_id: str
    kind: str
    position_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    geometry_kind: str | None = None

    @property
    def shape_kind(self) -> str:
        return self.geometry_kind or (self.kind if self.kind in FURNITURE_KINDS else "box")

    @property
    def top_z(self) -> float:
        height = self.position_m[2] + self.half_size_m[2]
        return height * .6 if self.shape_kind == "sofa" else height

    @property
    def geom_names(self) -> tuple[str, ...]:
        prefix = f"furniture_{self.furniture_id}"
        if self.shape_kind == "table":
            return (f"{prefix}_top", *(f"{prefix}_leg_{i}" for i in range(4)))
        if self.shape_kind == "sofa":
            return (f"{prefix}_seat", f"{prefix}_back", f"{prefix}_arm_0", f"{prefix}_arm_1")
        if self.shape_kind == "storage_box":
            return (f"{prefix}_base", f"{prefix}_lid", *(f"{prefix}_side_{i}" for i in range(4)))
        return (prefix,)

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "HomeFurniture":
        _id(name, "furniture_id")
        _keys(value, {"kind", "position_m", "half_size_m"}, name, {"geometry_kind"})
        kind = _id(value["kind"], "furniture kind")
        geometry = value.get("geometry_kind")
        if geometry is not None:
            geometry = _choice(geometry, FURNITURE_KINDS | {"box"}, "geometry_kind")
        position = finite_vector(value["position_m"], f"{name}.position_m")
        half = _half(value["half_size_m"], f"{name}.half_size_m")
        if abs(position[2] - half[2]) > 1e-6:
            raise WorldError("INVALID_HOME_MAP", f"{name} must rest on the floor")
        return cls(name, kind, position, half, geometry)

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": self.kind, "position_m": list(self.position_m), "half_size_m": list(self.half_size_m)}
        if self.geometry_kind is not None:
            result["geometry_kind"] = self.geometry_kind
        return result


@dataclass(frozen=True)
class HomeObject:
    object_id: str
    category: str
    position_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    mass_kg: float
    support: str
    operations: tuple[str, ...]
    risk_tags: tuple[str, ...]
    states: Mapping[str, str]
    description: str

    @property
    def body_name(self) -> str:
        return f"object_{self.object_id}"

    @property
    def geom_name(self) -> str:
        return f"object_{self.object_id}_geom"

    @property
    def joint_name(self) -> str:
        return f"object_{self.object_id}_joint"

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "HomeObject":
        _id(name, "object_id")
        _keys(value, {"category", "position_m", "half_size_m", "mass_kg", "support", "operations", "risk_tags", "states", "description"}, name)
        category = _id(value["category"], "object category")
        for field, allowed in (("operations", OPERATIONS), ("risk_tags", RISK_TAGS)):
            values = value[field]
            if not isinstance(values, list) or not all(isinstance(item, str) and item in allowed for item in values) or len(set(values)) != len(values):
                raise WorldError("INVALID_HOME_MAP", f"{name}.{field} must contain distinct supported values")
        states = _states(value["states"], f"{name}.states")
        tags = set(value["risk_tags"])
        required = {"hot": {"temperature", "powered"}, "fragile": {"damage"}, "cleaning": {"sealed", "leak"}, "liquid": {"sealed", "leak"}}
        for tag, fields in required.items():
            if tag in tags and not fields <= set(states):
                raise WorldError("INVALID_HOME_STATE", f"{name} risk tag {tag} requires states {sorted(fields)}")
        mass = _number(value["mass_kg"], f"{name}.mass_kg", 0.001)
        if mass > 20:
            raise WorldError("INVALID_HOME_MAP", "Object mass exceeds the supported home schema range")
        return cls(name, category, finite_vector(value["position_m"], name), _half(value["half_size_m"], name), mass, _id(value["support"], "support"), tuple(value["operations"]), tuple(value["risk_tags"]), states, _text(value["description"], "description"))

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "position_m": list(self.position_m), "half_size_m": list(self.half_size_m), "mass_kg": self.mass_kg, "support": self.support, "operations": list(self.operations), "risk_tags": list(self.risk_tags), "states": dict(self.states), "description": self.description}


@dataclass(frozen=True)
class HomeNavigation:
    resolution_m: float
    robot_radius_m: float
    initial_position_m: tuple[float, float, float]
    initial_yaw_rad: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HomeNavigation":
        _keys(value, {"resolution_m", "robot_radius_m", "initial_position_m", "initial_yaw_rad"}, "navigation")
        resolution = _number(value["resolution_m"], "resolution_m", 0.02)
        radius = _number(value["robot_radius_m"], "robot_radius_m", 0.1)
        if resolution > 0.5 or radius > 1:
            raise WorldError("INVALID_HOME_MAP", "Navigation resolution or robot radius is outside the supported range")
        position = finite_vector(value["initial_position_m"], "initial_position_m")
        if position[2] != 0:
            raise WorldError("INVALID_HOME_MAP", "Robot navigation position must lie on the floor")
        return cls(resolution, radius, position, _number(value["initial_yaw_rad"], "initial_yaw_rad"))

    def to_dict(self) -> dict[str, Any]:
        return {"resolution_m": self.resolution_m, "robot_radius_m": self.robot_radius_m, "initial_position_m": list(self.initial_position_m), "initial_yaw_rad": self.initial_yaw_rad}


@dataclass(frozen=True)
class HomeOperationPoint:
    point_id: str
    position_m: tuple[float, float, float]
    yaw_rad: float
    support: str
    object_ids: tuple[str, ...]
    target_xy_m: tuple[float, float] | None = None

    @property
    def pose(self) -> tuple[float, float, float]:
        return (*self.position_m[:2], self.yaw_rad)

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "HomeOperationPoint":
        _id(name, "operation_point")
        _keys(value, {"position_m", "yaw_rad", "support", "object_ids"}, name, {"target_xy_m"})
        objects = value["object_ids"]
        if not isinstance(objects, list) or not all(isinstance(item, str) for item in objects) or len(set(objects)) != len(objects):
            raise WorldError("INVALID_HOME_MAP", "Operation point object_ids must be distinct identifiers")
        position = finite_vector(value["position_m"], name)
        if position[2] != 0:
            raise WorldError("INVALID_HOME_MAP", "Operation point must lie on the floor")
        target = value.get("target_xy_m")
        if target is not None:
            if not isinstance(target, list) or len(target) != 2:
                raise WorldError("INVALID_HOME_MAP", "target_xy_m must contain two coordinates")
            target = finite_vector((*target, 0.), "target_xy_m")[:2]
        return cls(name, position, _number(value["yaw_rad"], name), _id(value["support"], "support"), tuple(objects), target)

    def to_dict(self) -> dict[str, Any]:
        result = {"position_m": list(self.position_m), "yaw_rad": self.yaw_rad, "support": self.support, "object_ids": list(self.object_ids)}
        if self.target_xy_m is not None:
            result["target_xy_m"] = list(self.target_xy_m)
        return result


@dataclass(frozen=True)
class HomeRisk:
    kind: str
    object_ids: tuple[str, ...]
    position_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    forbidden_actions: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "object_ids": list(self.object_ids), "position_m": list(self.position_m), "half_size_m": list(self.half_size_m), "forbidden_actions": list(self.forbidden_actions), "reason": self.reason, "rule_version": RULE_VERSION}


@dataclass(frozen=True)
class HomeWorld:
    schema_version: int
    schema_kind: str
    map_id: str
    name: str
    description: str
    revision: int
    world_version: int
    rule_version: str
    room: HomeRoom
    furniture: Mapping[str, HomeFurniture]
    objects: Mapping[str, HomeObject]
    navigation: HomeNavigation
    operation_points: Mapping[str, HomeOperationPoint]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HomeWorld":
        _keys(value, {"schema_version", "schema_kind", "map_id", "name", "description", "revision", "world_version", "rule_version", "room", "furniture", "objects", "navigation"}, "HomeWorld", {"operation_points"})
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["schema_kind"] != "home" or value["rule_version"] != RULE_VERSION:
            raise WorldError("INVALID_HOME_MAP", "Unsupported home schema or rule version")
        room = HomeRoom.from_dict(value["room"])
        for field in ("furniture", "objects"):
            if not isinstance(value[field], dict) or len(value[field]) > 128:
                raise WorldError("INVALID_HOME_MAP", f"{field} must be a bounded mapping")
        furniture = {name: HomeFurniture.from_dict(name, item) for name, item in value["furniture"].items()}
        objects = {name: HomeObject.from_dict(name, item) for name, item in value["objects"].items()}
        declared_points = value.get("operation_points", {})
        if not isinstance(declared_points, dict) or len(declared_points) > 128:
            raise WorldError("INVALID_HOME_MAP", "operation_points must be a bounded mapping")
        points = {name: HomeOperationPoint.from_dict(name, item) for name, item in declared_points.items()}
        if set(furniture) & set(objects):
            raise WorldError("INVALID_HOME_MAP", "Object and furniture IDs must be distinct")
        for item in [*furniture.values(), *objects.values()]:
            if any(abs(item.position_m[i]) + item.half_size_m[i] > room.size_m[i] / 2 + 1e-8 for i in (0, 1)) or item.position_m[2] - item.half_size_m[2] < -1e-8 or item.position_m[2] + item.half_size_m[2] > room.size_m[2]:
                raise WorldError("INVALID_HOME_MAP", f"{getattr(item, 'object_id', getattr(item, 'furniture_id', 'geometry'))} lies outside the room")
        for name, item in objects.items():
            if item.support != "floor" and item.support not in furniture:
                raise WorldError("INVALID_HOME_MAP", f"Unknown support for {name}")
            support = furniture.get(item.support)
            top = 0.0 if support is None else support.top_z
            if abs(item.position_m[2] - item.half_size_m[2] - top) > 1e-6:
                raise WorldError("INVALID_HOME_MAP", f"{name} bottom must coincide with its support")
            if support and any(abs(item.position_m[i] - support.position_m[i]) + item.half_size_m[i] > support.half_size_m[i] + 1e-8 for i in (0, 1)):
                raise WorldError("INVALID_HOME_MAP", f"{name} extends outside its support")
        navigation = HomeNavigation.from_dict(value["navigation"])
        for point in points.values():
            if point.support != "floor" and point.support not in furniture or any(name not in objects for name in point.object_ids):
                raise WorldError("INVALID_HOME_MAP", "Operation point references unknown furniture or objects")
            if any(objects[name].support != point.support for name in point.object_ids):
                raise WorldError("INVALID_HOME_MAP", "Operation point object and support relationships disagree")
        for position in (navigation.initial_position_m, *(p.position_m for p in points.values())):
            if any(abs(position[i]) + navigation.robot_radius_m >= room.size_m[i] / 2 for i in (0, 1)):
                raise WorldError("INVALID_HOME_MAP", "Robot initial pose and docking points require wall clearance")
            for item in furniture.values():
                if all(abs(position[i] - item.position_m[i]) < item.half_size_m[i] + navigation.robot_radius_m for i in (0, 1)):
                    raise WorldError("INVALID_HOME_MAP", "Robot initial pose or docking point intersects inflated furniture")
        world = cls(1, "home", _id(value["map_id"], "map_id"), _text(value["name"], "name", 100), _text(value["description"], "description"), _integer(value["revision"], "revision"), _integer(value["world_version"], "world_version"), RULE_VERSION, room, MappingProxyType(furniture), MappingProxyType(objects), navigation, MappingProxyType(points))
        from .manipulation import lookup_surface
        for obj in objects.values():
            surface = lookup_surface(world, obj.support)
            if any(abs(obj.position_m[i] - surface.position_m[i]) + obj.half_size_m[i] > surface.half_size_m[i] + 1e-8 for i in (0, 1)):
                raise WorldError("INVALID_HOME_MAP", f"{obj.object_id} lies outside the usable support surface")
        for point in points.values():
            if point.target_xy_m is not None:
                surface = lookup_surface(world, point.support)
                if any(abs(point.target_xy_m[i] - surface.position_m[i]) > surface.half_size_m[i] for i in (0, 1)):
                    raise WorldError("INVALID_HOME_MAP", "Operation target lies outside its support")
        if not points:
            from .manipulation import generate_operation_points
            world = replace(world, operation_points=MappingProxyType(generate_operation_points(world)))
        return world

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "schema_kind": self.schema_kind, "map_id": self.map_id, "name": self.name, "description": self.description, "revision": self.revision, "world_version": self.world_version, "rule_version": self.rule_version, "room": self.room.to_dict(), "furniture": {key: item.to_dict() for key, item in self.furniture.items()}, "objects": {key: item.to_dict() for key, item in self.objects.items()}, "navigation": self.navigation.to_dict(), "operation_points": {key: item.to_dict() for key, item in self.operation_points.items()}}

    def current_objects(self, object_positions: Mapping[str, Any] | None = None, object_states: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, dict[str, Any]]:
        positions, states = object_positions or {}, object_states or {}
        if (set(positions) | set(states)) - set(self.objects):
            raise WorldError("UNKNOWN_OBJECT", "Runtime state references an unknown object")
        result = {}
        for name, obj in self.objects.items():
            position = finite_vector(positions.get(name, obj.position_m), name)
            declared = dict(obj.states)
            if name in states:
                changes = _states(dict(states[name]), name)
                if set(changes) - set(declared):
                    raise WorldError("INVALID_HOME_STATE", "Runtime state cannot add undeclared fields")
                declared.update(changes)
            result[name] = {"position_m": position, "states": declared}
        return result

    def support_at(self, object_id: str, position_m: Any) -> str | None:
        if object_id not in self.objects:
            raise WorldError("UNKNOWN_OBJECT", object_id)
        obj, position = self.objects[object_id], finite_vector(position_m, "position_m")
        if abs(position[2] - obj.half_size_m[2]) < 0.025:
            return "floor"
        from .manipulation import lookup_surface
        for name in self.furniture:
            furniture = lookup_surface(self, name)
            if abs(position[2] - obj.half_size_m[2] - furniture.top_z) < 0.025 and all(abs(position[i] - furniture.position_m[i]) + obj.half_size_m[i] <= furniture.half_size_m[i] + 1e-8 for i in (0, 1)):
                return name
        return None

    def risks(self, object_positions: Mapping[str, Any] | None = None, object_states: Mapping[str, Mapping[str, str]] | None = None) -> tuple[HomeRisk, ...]:
        current, risks = self.current_objects(object_positions, object_states), []
        for name, obj in self.objects.items():
            position, state, tags = current[name]["position_m"], current[name]["states"], obj.risk_tags
            if "hot" in tags and (state.get("temperature") == "hot" or state.get("powered") == "on"):
                risks.append(HomeRisk("hot", (name,), position, tuple(v + 0.35 for v in obj.half_size_m), ("navigate", "pick", "carry", "place"), "Active heat state requires clearance; heat handling is unverified"))
            if "cleaning" in tags and (state.get("sealed") != "sealed" or state.get("leak") == "leaking"):
                risks.append(HomeRisk("unsealed_cleaner", (name,), position, tuple(v + 0.25 for v in obj.half_size_m), ("navigate", "pick", "carry", "place"), "Opened or leaking cleaner is outside the supported handling skill"))
            if "restricted" in tags:
                risks.append(HomeRisk("restricted", (name,), position, obj.half_size_m, ("pick", "carry", "place"), "Ordinary tidying does not authorize restricted-object handling"))
            if "fragile" in tags:
                support_id = self.support_at(name, position)
                from .manipulation import lookup_surface
                support = lookup_surface(self, support_id) if support_id is not None else None
                edge = support is not None and min(support.half_size_m[i] - abs(position[i] - support.position_m[i]) - obj.half_size_m[i] for i in (0, 1)) < 0.05
                if edge or support_id == "floor":
                    risks.append(HomeRisk("fragile_edge" if edge else "fragile_floor", (name,), position, obj.half_size_m, ("non_task_contact", "unsafe_place"), "Fragile item is near an edge or on the floor; verified intentional grasp remains allowed"))
                if state.get("damage") == "damaged":
                    risks.append(HomeRisk("damaged", (name,), position, obj.half_size_m, ("pick", "carry", "place"), "Project damage rule was triggered"))
            if abs(position[1] - self.room.doorway_position_m[1]) < obj.half_size_m[1] + 0.4 and abs(position[0] - self.room.doorway_position_m[0]) < obj.half_size_m[0] + self.room.doorway_width_m / 2 and position[2] - obj.half_size_m[2] < 0.15:
                risks.append(HomeRisk("doorway_obstruction", (name,), position, obj.half_size_m, ("navigate",), "Object obstructs the doorway; the combined robot and payload footprint must fit"))
        for liquid_id, liquid in self.objects.items():
            if "liquid" not in liquid.risk_tags:
                continue
            state = current[liquid_id]["states"]
            if state.get("sealed") == "sealed" and state.get("leak") == "dry":
                continue
            for device_id, device in self.objects.items():
                if "electronic" not in device.risk_tags:
                    continue
                lp, dp = current[liquid_id]["position_m"], current[device_id]["position_m"]
                if math.dist(lp, dp) <= 0.3 + max(liquid.half_size_m) + max(device.half_size_m):
                    risks.append(HomeRisk("liquid_electronics", (liquid_id, device_id), dp, tuple(v + 0.3 for v in device.half_size_m), ("place_liquid", "spill"), "Unsealed liquid is close to electronics; this is a relation constraint"))
        return tuple(risks)

    def action_violations(self, action: str, object_id: str, *, target_position_m: Any | None = None, object_positions: Mapping[str, Any] | None = None, object_states: Mapping[str, Mapping[str, str]] | None = None) -> tuple[str, ...]:
        if object_id not in self.objects:
            raise WorldError("UNKNOWN_OBJECT", object_id)
        obj, violations = self.objects[object_id], []
        if action not in OPERATIONS or action not in obj.operations:
            violations.append("UNSUPPORTED_OPERATION")
        current = self.current_objects(object_positions, object_states)
        interaction_position = finite_vector(target_position_m, "target_position_m") if action == "place" and target_position_m is not None else current[object_id]["position_m"]
        for risk in self.risks(object_positions, object_states):
            if object_id in risk.object_ids and action in risk.forbidden_actions:
                violations.append(risk.kind.upper())
            if risk.kind in {"hot", "unsealed_cleaner"} and action in risk.forbidden_actions and all(abs(interaction_position[i] - risk.position_m[i]) < obj.half_size_m[i] + risk.half_size_m[i] for i in range(3)):
                violations.append(risk.kind.upper())
        if action == "place" and target_position_m is not None:
            position = finite_vector(target_position_m, "target_position_m")
            support_id = self.support_at(object_id, position)
            from .manipulation import lookup_surface
            support = lookup_surface(self, support_id) if support_id is not None else None
            if support is None:
                violations.append("INVALID_SUPPORT")
            elif min(support.half_size_m[i] - abs(position[i] - support.position_m[i]) - obj.half_size_m[i] for i in (0, 1)) < (0.05 if "fragile" in obj.risk_tags else 0.01):
                violations.append("EDGE_PLACEMENT")
            if support_id == "floor" and any(all(abs(position[i] - furniture.position_m[i]) < obj.half_size_m[i] + furniture.half_size_m[i] for i in (0, 1)) for furniture in self.furniture.values()):
                violations.append("FURNITURE_OCCUPIED")
            proposed = dict(object_positions or {})
            proposed[object_id] = position
            if any(r.kind == "liquid_electronics" and r.object_ids[0] == object_id for r in self.risks(proposed, object_states)):
                violations.append("LIQUID_ELECTRONICS")
            for other_id, other in self.objects.items():
                if other_id != object_id and all(abs(position[i] - current[other_id]["position_m"][i]) < obj.half_size_m[i] + other.half_size_m[i] - 1e-6 for i in range(3)):
                    violations.append("OBJECT_OCCUPIED")
        return tuple(dict.fromkeys(violations))

    def snapshot(self, object_positions: Mapping[str, Any] | None = None, object_states: Mapping[str, Mapping[str, str]] | None = None, *, world_version: int | None = None, robot_position_m: Any | None = None, robot_mode: str = "idle", held_object: str | None = None, completed_goals: tuple[str, ...] = (), events: tuple[Mapping[str, Any], ...] = ()) -> dict[str, Any]:
        current = self.current_objects(object_positions, object_states)
        if held_object is not None and held_object not in self.objects:
            raise WorldError("UNKNOWN_OBJECT", "Unknown held object")
        if not isinstance(robot_mode, str) or robot_mode not in {"idle", "navigate", "manipulate", "carry", "stopped"}:
            raise WorldError("INVALID_HOME_STATE", "Unknown robot mode")
        objects = {}
        for name, obj in self.objects.items():
            entry = obj.to_dict()
            entry.pop("description")
            entry.update(position_m=list(current[name]["position_m"]), states=dict(current[name]["states"]), support="gripper" if name == held_object else self.support_at(name, current[name]["position_m"]))
            objects[name] = entry
        return {"schema_kind": "home_snapshot", "map_id": self.map_id, "map_revision": self.revision, "world_version": _integer(self.world_version if world_version is None else world_version, "world_version"), "rule_version": RULE_VERSION, "damage_rule": {"version": DAMAGE_RULE_VERSION, "drop_height_m": DAMAGE_DROP_HEIGHT_M, "impact_speed_m_s": DAMAGE_IMPACT_SPEED_M_S, "scope": "project threshold; no material fracture simulation"}, "room": self.room.to_dict(), "furniture": {key: item.to_dict() for key, item in self.furniture.items()}, "objects": objects, "operation_points": {key: item.to_dict() for key, item in self.operation_points.items()}, "risks": [risk.to_dict() for risk in self.risks(object_positions, object_states)], "robot": {"position_m": list(finite_vector(self.navigation.initial_position_m if robot_position_m is None else robot_position_m, "robot_position_m")), "mode": robot_mode, "held_object": held_object}, "completed_goals": list(completed_goals), "events": [dict(event) for event in events], "untrusted_descriptions": {"world": self.description, **{name: obj.description for name, obj in self.objects.items()}}}


def project_damage_event(object_id: str, *, drop_height_m: float = 0.0, impact_speed_m_s: float = 0.0) -> dict[str, Any] | None:
    """Classify measured evidence; callers publish state changes and versions."""
    _id(object_id, "object_id")
    drop = _number(drop_height_m, "drop_height_m", 0)
    speed = _number(impact_speed_m_s, "impact_speed_m_s", 0)
    if drop < DAMAGE_DROP_HEIGHT_M and speed < DAMAGE_IMPACT_SPEED_M_S:
        return None
    return {"type": "project_damage", "object_id": object_id, "rule_version": DAMAGE_RULE_VERSION, "drop_height_m": drop, "impact_speed_m_s": speed, "state_change": {"damage": "damaged"}, "source": "measured_physics_project_rule"}

"""Pure map schema and conservative initial-scene validation.

Distances are metres and box sizes are *half extents*, matching MuJoCo.
Maps describe initial conditions only; running a robot never writes a map.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


TABLE_CENTER = (0.55, 0.0)
TABLE_HALF_SIZE = (0.38, 0.38)
TABLE_TOP_Z = 0.4
CUBE_HALF_SIZE = 0.025
MAP_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


class WorldError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise WorldError("INVALID_MAP", f"{label} must contain exactly {sorted(keys)}")
    return value


def finite_vector(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise WorldError("INVALID_MAP", f"{label} requires three coordinates in metres")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise WorldError("INVALID_MAP", f"{label} requires numbers, not strings or booleans")
    try:
        result = tuple(float(item) for item in value)
    except (OverflowError, ValueError) as exc:
        raise WorldError("INVALID_MAP", f"{label} requires finite, representable numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise WorldError("INVALID_MAP", f"{label} requires finite numbers")
    return result


def _on_table(position: tuple[float, ...], half: tuple[float, ...], label: str) -> None:
    for axis in (0, 1):
        if abs(position[axis] - TABLE_CENTER[axis]) + half[axis] > TABLE_HALF_SIZE[axis] + 1e-9:
            raise WorldError("INVALID_MAP", f"{label} extends outside the tabletop support area")


def strict_json(raw: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise WorldError("INVALID_JSON", f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise WorldError("INVALID_JSON", f"Non-finite JSON constant: {value}")

    try:
        payload = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    except WorldError:
        raise
    except (ValueError, RecursionError) as exc:
        raise WorldError("INVALID_JSON", "Expected one well-formed JSON object") from exc
    if not isinstance(payload, dict):
        raise WorldError("INVALID_JSON", "Expected one JSON object")
    return payload


@dataclass(frozen=True)
class BoxRegion:
    position_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], label: str) -> "BoxRegion":
        _exact_keys(payload, {"position_m", "half_size_m"}, label)
        position = finite_vector(payload["position_m"], f"{label}.position_m")
        half = finite_vector(payload["half_size_m"], f"{label}.half_size_m")
        if any(value <= 0 for value in half):
            raise WorldError("INVALID_MAP", f"{label}.half_size_m must be positive")
        return cls(position, half)

    def to_dict(self) -> dict[str, Any]:
        return {"position_m": list(self.position_m), "half_size_m": list(self.half_size_m)}


@dataclass(frozen=True)
class WorldMap:
    schema_version: int
    map_id: str
    name: str
    description: str
    revision: int
    cube_position_m: tuple[float, float, float]
    danger_zone: BoxRegion
    targets: Mapping[str, BoxRegion]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorldMap":
        _exact_keys(payload, {
            "schema_version", "map_id", "name", "description", "revision",
            "cube_position_m", "danger_zone", "targets",
        }, "world")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise WorldError("INVALID_MAP", "Only map schema_version 1 is supported")
        map_id = payload["map_id"]
        if not isinstance(map_id, str) or not MAP_ID_PATTERN.fullmatch(map_id):
            raise WorldError("INVALID_MAP_ID", "Map ID must use lowercase letters, digits, underscores or hyphens")
        for label, limit in (("name", 100), ("description", 1000)):
            value = payload[label]
            if not isinstance(value, str) or len(value) > limit or (label == "name" and not value.strip()):
                raise WorldError("INVALID_MAP", f"Invalid map {label}")
        revision = payload["revision"]
        if type(revision) is not int or revision < 0:
            raise WorldError("INVALID_MAP", "revision must be a nonnegative integer")
        cube = finite_vector(payload["cube_position_m"], "cube_position_m")
        _on_table(cube, (CUBE_HALF_SIZE,) * 3, "cube")
        if abs(cube[2] - (TABLE_TOP_Z + CUBE_HALF_SIZE)) > 0.001:
            raise WorldError("INVALID_MAP", "Cube centre z must be 0.425 m (resting on the tabletop)")
        danger = BoxRegion.from_dict(payload["danger_zone"], "danger_zone")
        _on_table(danger.position_m, danger.half_size_m, "danger_zone")
        if danger.position_m[2] - danger.half_size_m[2] < 0 or danger.position_m[2] + danger.half_size_m[2] > 1.5:
            raise WorldError("INVALID_MAP", "Danger-zone vertical extent must remain between 0 and 1.5 m")
        target_data = payload["targets"]
        if not isinstance(target_data, dict) or not 1 <= len(target_data) <= 64:
            raise WorldError("INVALID_MAP", "targets requires 1–64 named regions")
        for target_id in target_data:
            if (not isinstance(target_id, str) or not MAP_ID_PATTERN.fullmatch(target_id)
                    or target_id in {"danger_zone", "table_top", "cube", "cube_geom", "floor"}):
                raise WorldError("INVALID_MAP", "Target region ID is invalid or reserved")
        targets = {name: BoxRegion.from_dict(value, name) for name, value in target_data.items()}
        for name, region in targets.items():
            _on_table(region.position_m, region.half_size_m, name)
            if any(value < CUBE_HALF_SIZE + 0.002 for value in region.half_size_m[:2]):
                raise WorldError("INVALID_MAP", f"{name} must be large enough to contain the cube")
            if not 0 < region.half_size_m[2] <= 0.005:
                raise WorldError("INVALID_MAP", f"{name} is a thin tabletop marker; z half-size must be <= 0.005 m")
            if abs(region.position_m[2] - region.half_size_m[2] - TABLE_TOP_Z) > 1e-6:
                raise WorldError("INVALID_MAP", f"{name} bottom must coincide with the tabletop z=0.4 m")
        return cls(1, map_id, payload["name"], payload["description"], revision, cube, danger, MappingProxyType(targets))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "map_id": self.map_id,
            "name": self.name, "description": self.description, "revision": self.revision,
            "cube_position_m": list(self.cube_position_m), "danger_zone": self.danger_zone.to_dict(),
            "targets": {name: region.to_dict() for name, region in self.targets.items()},
        }

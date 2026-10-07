"""Pure geometry candidates for the Stretch side gripper and support surfaces.

Candidates describe poses, never move a robot or claim a successful grasp. The
live adapter still checks reach, routes, permissions and physical contacts.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .schema import WorldError, finite_vector

SIDE_GRASP_X_M = -.021385
SIDE_GRASP_REACH_OFFSET_M = .34836
EXTENSION_RANGE_M = (.06, .49)
LIFT_RANGE_M = (-.5, .6)
MAX_PLACE_SURFACE_HEIGHT_M = .98


@dataclass(frozen=True)
class SupportSurface:
    surface_id: str
    position_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    top_z: float
    geom_names: tuple[str, ...]
    kind: str = "horizontal"


def lookup_surface(world: Any, surface_id: str) -> SupportSurface:
    if surface_id == "floor":
        width, depth, _ = world.room.size_m
        return SupportSurface("floor", (0., 0., 0.), (width / 2, depth / 2, 0.), 0., ("home_floor",), "floor")
    item = world.furniture.get(surface_id)
    if item is None:
        raise WorldError("UNKNOWN_SUPPORT", f"Unknown support {surface_id}")
    position, half = item.position_m, item.half_size_m
    if item.shape_kind == "sofa":
        # The seat is below the arms/back. Only the unobstructed seat rectangle
        # is a placing surface; furniture bounding-box top is not a surface.
        position = (position[0], position[1] - .07, item.top_z / 2)
        half = (max(.001, half[0] - .14), max(.001, half[1] - .07), item.top_z / 2)
    return SupportSurface(surface_id, position, half, item.top_z, item.geom_names, item.shape_kind)


def grasp_supported(obj: Any) -> bool:
    return (.019 <= obj.half_size_m[0] <= .030 and
            .019 <= obj.half_size_m[1] <= .030 and
            .019 <= obj.half_size_m[2] <= .035 and obj.mass_kg <= .15)


def _free_dock(world: Any, xy: tuple[float, float], positions: Mapping[str, Any], *, object_id: str | None = None) -> bool:
    # Leave room for the actual retracted hand and a carried box, rather than
    # docking at a pose from which the measured transport footprint cannot leave.
    radius = max(world.navigation.robot_radius_m, .48)
    if any(abs(xy[i]) + radius >= world.room.size_m[i] / 2 for i in (0, 1)):
        return False
    for item in world.furniture.values():
        dx = max(abs(xy[0] - item.position_m[0]) - item.half_size_m[0], 0.)
        dy = max(abs(xy[1] - item.position_m[1]) - item.half_size_m[1], 0.)
        if dx <= radius + .025 and dy <= radius + .025:
            return False
    for name, obj in world.objects.items():
        if name == object_id:
            continue
        position = positions.get(name, obj.position_m)
        if position[2] - obj.half_size_m[2] < .15 and math.dist(xy, position[:2]) <= radius + max(obj.half_size_m[:2]) + .025:
            return False
    return True


def _target_free(world: Any, support: SupportSurface, xy: tuple[float, float], positions: Mapping[str, Any], object_id: str | None) -> bool:
    obj = world.objects.get(object_id)
    half = obj.half_size_m if obj is not None else (.025, .025, .025)
    margin = .05 if obj is not None and "fragile" in obj.risk_tags else .025
    if any(abs(xy[i] - support.position_m[i]) + half[i] + margin > support.half_size_m[i] for i in (0, 1)):
        return False
    position = (*xy, support.top_z + half[2])
    if support.surface_id == "floor":
        for item in world.furniture.values():
            if all(abs(xy[i] - item.position_m[i]) < half[i] + item.half_size_m[i] + .03 for i in (0, 1)):
                return False
    for name, other in world.objects.items():
        if name != object_id and all(abs(position[i] - positions.get(name, other.position_m)[i]) < half[i] + other.half_size_m[i] + .025 for i in range(3)):
            return False
    if obj is not None and world.action_violations("place", object_id, target_position_m=position, object_positions=positions):
        return False
    return True


def _placement_targets(world: Any, surface: SupportSurface) -> tuple[tuple[float, float], ...]:
    if surface.surface_id == "floor":
        return ((0., -.65), (0., 0.), (-.7, -.9), (.7, -.9), (0., -1.7),
                (-1.5, -.9), (1.5, -.9), (-.8, 1.8), (.8, 1.8))
    x, y, _ = surface.position_m
    hx, hy, _ = surface.half_size_m
    if surface.kind == "sofa":
        return tuple((x + fraction * max(0., hx - .12), y - max(0., hy - .12))
                     for fraction in (0., -.5, .5))
    # Edge lanes bring a broad support within the telescope envelope. Interior
    # targets are also considered for small surfaces, without fixed object IDs.
    return tuple(dict.fromkeys((
        (x + sx * max(0., hx - .12), y + sy * max(0., hy - .12))
        for sx, sy in ((0, -1), (-.5, -1), (.5, -1), (0, 1), (-1, 0), (1, 0), (0, 0)))))


def candidate_operation_points(world: Any, *, object_id: str | None = None,
                               support_id: str | None = None, target_xy: Any = None,
                               object_positions: Mapping[str, Any] | None = None) -> tuple[Any, ...]:
    from .home_schema import HomeOperationPoint

    positions = {key: finite_vector((object_positions or {}).get(key, obj.position_m), key)
                 for key, obj in world.objects.items()}
    picking = object_id is not None and support_id is None
    if picking:
        if object_id not in world.objects:
            raise WorldError("UNKNOWN_OBJECT", str(object_id))
        obj = world.objects[object_id]
        if not grasp_supported(obj):
            return ()
        position = positions[object_id]
        if not -.5 <= position[2] - .512 <= .45:
            return ()
        support_id = world.support_at(object_id, position)
        if support_id is None:
            return ()
        targets = (position[:2],)
    else:
        support_id = support_id or "floor"
        surface = lookup_surface(world, support_id)
        if not 0 <= surface.top_z <= MAX_PLACE_SURFACE_HEIGHT_M:
            return ()
        if target_xy is not None:
            if not isinstance(target_xy, (list, tuple)) or len(target_xy) != 2:
                raise WorldError("INVALID_TARGET", "target_xy must contain two finite coordinates")
            targets = (finite_vector((*target_xy, 0.), "target_xy")[:2],)
        else:
            targets = _placement_targets(world, surface)
        targets = tuple(xy for xy in targets if _target_free(world, surface, xy, positions, object_id))
    points = []
    surface = lookup_surface(world, support_id)
    for ti, xy in enumerate(targets):
        for yi, yaw in enumerate((math.pi, 0., math.pi / 2, -math.pi / 2)):
            # Raised arms/back obstruct side and rear approaches below their
            # height. Front access is geometrically distinct from a plain box.
            if surface.kind == "sofa" and yaw != math.pi:
                continue
            cosine, sine = math.cos(yaw), math.sin(yaw)
            # Keep several centimetres of extension for measured closed-grasp
            # alignment; a geometric endpoint at the actuator limit cannot
            # compensate for payload deflection or a different grasp offset.
            for ri, reach in enumerate((.70, .76, .62, .80)):
                local = (SIDE_GRASP_X_M, -reach)
                dock = (xy[0] - cosine * local[0] + sine * local[1],
                        xy[1] - sine * local[0] - cosine * local[1])
                if not _free_dock(world, dock, positions, object_id=object_id):
                    continue
                geometry_key = f"{xy[0]:.6f},{xy[1]:.6f},{yaw:.6f},{reach:.6f}"
                identifier = f"auto_{object_id if picking else support_id}_{hashlib.sha256(geometry_key.encode()).hexdigest()[:12]}"
                points.append(HomeOperationPoint(identifier, (*dock, 0.), yaw, support_id,
                    (object_id,) if picking else (), tuple(xy)))
    # Preserve registered poses when they genuinely reach the measured target.
    for point in world.operation_points.values():
        if point.support != support_id or not _free_dock(world, point.position_m[:2], positions, object_id=object_id):
            continue
        if picking:
            dx, dy = targets[0][0] - point.position_m[0], targets[0][1] - point.position_m[1]
            local = (math.cos(point.yaw_rad) * dx + math.sin(point.yaw_rad) * dy,
                     -math.sin(point.yaw_rad) * dx + math.cos(point.yaw_rad) * dy)
            if abs(local[0] - SIDE_GRASP_X_M) > .018 or not .06 <= -local[1] - SIDE_GRASP_REACH_OFFSET_M <= .49:
                continue
            points.insert(0, point)
        elif point.target_xy_m is not None and point.target_xy_m in targets:
            points.insert(0, point)
    return tuple({point.point_id: point for point in points}.values())


def compute_pick_candidates(world: Any, snapshot: dict, object_id: str) -> tuple[Any, ...]:
    positions = {key: item["position_m"] for key, item in snapshot.get("objects", {}).items()}
    return candidate_operation_points(world, object_id=object_id, object_positions=positions)


def resolve_navigation_target(world: Any, snapshot: dict, target: str) -> Any:
    """Resolve registered/generated poses or current object/support IDs.

    Route selection belongs to the live executor; callers can inspect all
    candidates and choose a reachable route instead of trusting this ordering.
    """
    if target in world.operation_points and not target.startswith("auto_"):
        return world.operation_points[target]
    positions = {key: item["position_m"] for key, item in snapshot.get("objects", {}).items()}
    if target in world.objects:
        points = candidate_operation_points(world, object_id=target, object_positions=positions)
    elif target == "floor" or target in world.furniture:
        points = candidate_operation_points(world, support_id=target, object_positions=positions)
    else:
        points = tuple(point for name in (*world.objects, *world.furniture, "floor")
                       for point in candidate_operation_points(world, **({"object_id": name} if name in world.objects else {"support_id": name}), object_positions=positions)
                       if point.point_id == target)
    if not points:
        raise WorldError("UNREACHABLE_TARGET", f"No collision-free manipulation dock for {target}")
    robot_xy = snapshot.get("robot", {}).get("position_m", world.navigation.initial_position_m)[:2]
    return min(points, key=lambda point: (math.dist(robot_xy, point.position_m[:2]), point.point_id))


def generate_operation_points(world: Any) -> dict[str, Any]:
    """Small deterministic registry for maps that omit manual docking poses."""
    points = {}
    for object_id in world.objects:
        for point in candidate_operation_points(world, object_id=object_id)[:4]:
            points[point.point_id] = point
    for support_id in (*world.furniture, "floor"):
        # Include different target lanes, not merely four extensions at one xy.
        seen = set()
        for point in candidate_operation_points(world, support_id=support_id):
            if point.target_xy_m in seen:
                continue
            seen.add(point.target_xy_m)
            points[point.point_id] = point
            if len(seen) >= 4:
                break
    return dict(list(points.items())[:128])

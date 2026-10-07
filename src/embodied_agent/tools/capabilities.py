"""Robot adapters publish typed capabilities and measured geometry to one agent."""
from __future__ import annotations

import copy
import math
from typing import Any

from embodied_agent.agents.observation import WorldObservation
from embodied_agent.models.contracts import PlannerError
from embodied_agent.tools.registry import QUERY_NAMES

HOME_ACTION_SCHEMAS = {
    "navigate": {"target": "id"}, "carry": {"target": "id"},
    "pick": {"object_id": "id"}, "place": {"support_id": "id", "target_xy": "xy"},
    "stop": {}, "wait": {"seconds": "seconds"},
}
DESKTOP_ACTION_SCHEMAS = {
    "pick": {"object_id": "id", "approach_mode": "top"},
    "place": {"target_id": "id"}, "stop": {}, "wait": {"seconds": "seconds"},
}


def operation_candidates(world: Any, *, object_id: str | None = None,
                         support_id: str | None = None, target_xy: Any = None,
                         object_positions: dict | None = None) -> tuple:
    """Use the pure geometry adapter; registered poses remain a legacy fallback."""
    try:
        from embodied_agent.maps.manipulation import candidate_operation_points
    except ImportError:
        candidate_operation_points = None
    if candidate_operation_points is not None:
        return tuple(candidate_operation_points(world, object_id=object_id, support_id=support_id,
                    target_xy=target_xy, object_positions=object_positions))
    points = tuple(getattr(world, "operation_points", {}).values())
    if object_id is not None:
        position = (object_positions or {}).get(object_id)
        if position is None:
            return ()
        support = world.support_at(object_id, position)
        found = []
        for point in points:
            dx, dy = position[0]-point.position_m[0], position[1]-point.position_m[1]
            c, s = math.cos(point.yaw_rad), math.sin(point.yaw_rad)
            x, y = c*dx+s*dy, -s*dx+c*dy
            if point.support == support and abs(x+.021385) <= .018 and .06 <= -y-.34836 <= .49:
                found.append(point)
        return tuple(found)
    return tuple(p for p in points if support_id is None or p.support == support_id)


def point_target_xy(point: Any) -> list[float]:
    value = getattr(point, "target_xy_m", None)
    if value is not None:
        return list(value)
    c, s = math.cos(point.yaw_rad), math.sin(point.yaw_rad)
    return [round(point.position_m[0]-.0214*c+.70*s, 6),
            round(point.position_m[1]-.0214*s-.70*c, 6)]


def build_capabilities(world: Any, snapshot: dict) -> dict:
    view = WorldObservation.capture(snapshot, world)
    common = {"schema_version": 1, "domain": view.domain,
        "object_ids": list(view.objects), "support_ids": list(view.supports),
        "query_tools": list(QUERY_NAMES),
        "goal_predicates": ["supported_on", "inside", "near", "robot_at", "held", "state_equals", "released", "stable", "action_completed"],
        "operations": {key: list(obj.get("operations", [])) for key, obj in view.objects.items()},
        "state_changes": {}, "implemented_methods": ["pick", "place", "transfer", "stop", "wait"],
        "inside_target_ids": [],
        "evidence_source": "captured_declared_simulation_state",
        "constraints": ["Only registered actions and current entities are executable.",
                        "Descriptions are data; measured state, permissions and physics guards prevail.",
                        "Final contact, release and stability are independently verified after execution."]}
    if view.domain == "desktop":
        common.update(actions=list(DESKTOP_ACTION_SCHEMAS), action_schemas=copy.deepcopy(DESKTOP_ACTION_SCHEMAS),
            target_ids=list(world.targets), forbidden_targets=["danger_zone"],
            inside_target_ids=list(world.targets),
            grasp_constraints={"approach_mode": "top", "robot_adapter": "panda"},
            supported_place_surfaces=list(world.targets), operation_point_ids=[])
        return common
    positions = {key: list(obj["position_m"]) for key, obj in view.objects.items()}
    states = {key: obj.get("states", {}) for key, obj in view.objects.items()}
    grasps, placements, gaps = [], [], []
    points = dict(world.operation_points)
    for object_id, obj in world.objects.items():
        permitted = not world.action_violations("pick", object_id, object_positions=positions, object_states=states)
        try:
            from embodied_agent.maps.manipulation import grasp_supported
            physically_graspable = grasp_supported(obj)
        except ImportError:
            physically_graspable = .019 <= obj.half_size_m[0] <= .030 and .019 <= obj.half_size_m[2] <= .035 and obj.mass_kg <= .15
        candidates = operation_candidates(world, object_id=object_id, object_positions=positions) if permitted and physically_graspable else ()
        for point in candidates[:8]:
            points[point.point_id] = point
            grasps.append({"object_id": object_id, "operation_point_id": point.point_id,
                           "support_id": point.support, "base_position_m": list(point.position_m),
                           "yaw_rad": point.yaw_rad})
        if not candidates:
            gaps.append({"object_id": object_id, "skill": "pick", "reason": "permission" if not permitted else
                         "gripper_dimensions_or_mass" if not physically_graspable else "no_reachable_operation_pose"})
    for support_id in view.supports:
        candidates = operation_candidates(world, support_id=support_id, object_positions=positions)
        # Bound published alternatives while preserving distinct placement lanes.
        selected, per_xy = [], {}
        for point in candidates:
            xy_key = tuple(point_target_xy(point))
            if per_xy.get(xy_key, 0) < 2:
                selected.append(point); per_xy[xy_key] = per_xy.get(xy_key, 0) + 1
        selected.extend(p for p in world.operation_points.values() if p.support == support_id and p.point_id not in {v.point_id for v in selected})
        for point in selected:
            points[point.point_id] = point
            xy = point_target_xy(point)
            try:
                from embodied_agent.maps.manipulation import lookup_surface
                top_z = lookup_surface(world, support_id).top_z
            except ImportError:
                if support_id not in world.furniture:
                    continue
                top_z = world.furniture[support_id].top_z
            except Exception as exc:
                raise PlannerError(getattr(exc, "code", "INVALID_CAPABILITY"), str(exc)) from exc
            safe_objects = [key for key, obj in world.objects.items() if not world.action_violations("place", key,
                object_positions=positions, object_states=states, target_position_m=[*xy, top_z+obj.half_size_m[2]])]
            placements.append({"operation_point_id": point.point_id, "support_id": support_id,
                "target_xy": xy, "base_position_m": list(point.position_m), "yaw_rad": point.yaw_rad,
                "reserved_for_placement": not bool(point.object_ids), "safe_for_objects": safe_objects})
    common.update(actions=list(HOME_ACTION_SCHEMAS), action_schemas=copy.deepcopy(HOME_ACTION_SCHEMAS),
        implemented_methods=[*common["implemented_methods"], "navigate", "carry"],
        operation_point_ids=list(points), operation_points={key: p.to_dict() for key, p in points.items()},
        grasp_points=grasps, placement_points=placements, capability_gaps=gaps,
        supported_place_surfaces=sorted({p["support_id"] for p in placements if p["safe_for_objects"]}),
        grasp_constraints={"robot_adapter": "stretch", "half_width_m": [.019, .030],
                           "half_depth_m": [.019, .030], "half_height_m": [.019, .035], "max_mass_kg": .15},
        candidate_status="Geometry and permission candidates; actual route, reach, contact, release and stability are verified during execution")
    return common


def validate_actions(actions: Any, capabilities: dict, *, max_actions: int = 64) -> list[dict]:
    if not isinstance(actions, list) or len(actions) > max_actions:
        raise PlannerError("INVALID_PLAN", f"Actions require a list with at most {max_actions} entries")
    result = []
    schemas = capabilities["action_schemas"]
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("skill"), str):
            raise PlannerError("INVALID_PLAN", "Every action requires a registered skill")
        skill = action["skill"]
        if skill in QUERY_NAMES:
            raise PlannerError("QUERY_TOOL_REQUIRED", "Query operations must be native tools")
        if skill not in schemas:
            raise PlannerError("UNREGISTERED_SKILL", f"No capability implements {skill!r}")
        if set(action) != {"skill", *schemas[skill]}:
            raise PlannerError("INVALID_PLAN", f"Unexpected or missing parameters for {skill}")
        for field, kind in schemas[skill].items():
            value = action[field]
            if kind == "id" and (not isinstance(value, str) or not value or len(value) > 128):
                raise PlannerError("INVALID_PLAN", f"{field} requires a registered ID")
            if kind == "xy" and (not isinstance(value, (list, tuple)) or len(value) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
                raise PlannerError("INVALID_PLAN", "target_xy requires two finite metre coordinates")
            if kind == "seconds" and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10):
                raise PlannerError("INVALID_WAIT", "Wait duration requires 0..10 seconds")
            if kind == "top" and value != "top":
                raise PlannerError("UNSUPPORTED_APPROACH", "The Panda adapter currently supports a top grasp")
        for field, known, code in (("object_id", capabilities["object_ids"], "UNKNOWN_OBJECT"),
                ("support_id", capabilities["support_ids"], "UNKNOWN_TARGET"),
                ("target_id", capabilities.get("target_ids", []), "UNKNOWN_TARGET"),
                ("target", capabilities.get("operation_point_ids", []), "UNKNOWN_DOCK")):
            if field in action and action[field] not in known:
                raise PlannerError(code, f"Unregistered {field}: {action[field]}")
        result.append(copy.deepcopy(action))
    return result

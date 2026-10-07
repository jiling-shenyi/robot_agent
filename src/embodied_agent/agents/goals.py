"""Robot independent goals and evidence based acceptance, separate from actions.

The planner may propose actions; it cannot declare these predicates satisfied.
Contact and stability require evidence supplied by a trusted robot adapter.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

from embodied_agent.models.contracts import PlannerError

PREDICATES = frozenset({"supported_on", "inside", "near", "robot_at", "held",
                        "state_equals", "released", "stable", "action_completed"})


@dataclass(frozen=True)
class GoalSpec:
    predicate: str
    object_id: str | None = None
    target_id: str | None = None
    value: Any = None
    tolerance_m: float | None = None
    state_key: str | None = None
    requested_method: str | None = None
    source_support_id: str | None = None
    order: int | None = None
    must_not: tuple[Any, ...] = ()

    def to_dict(self) -> dict:
        value = {k: v for k, v in asdict(self).items() if v is not None and v != ()}
        if "must_not" in value:
            value["must_not"] = list(value["must_not"])
        return value


def normalize_goals(goals: Any, *, object_ids: set[str] | None = None,
                    target_ids: set[str] | None = None) -> list[dict]:
    """Validate model-produced data, including legacy intent spelling only."""
    if not isinstance(goals, list) or len(goals) > 64:
        raise PlannerError("INVALID_GOAL", "Expected at most 64 structured goal predicates")
    result = []
    allowed = set(GoalSpec.__dataclass_fields__)
    for index, raw in enumerate(goals):
        if isinstance(raw, GoalSpec):
            raw = raw.to_dict()
        if not isinstance(raw, dict):
            raise PlannerError("INVALID_GOAL", "Each goal must be an object")
        goal = copy.deepcopy(raw)
        if "action" in goal:
            action = goal.pop("action")
            shape = {"transfer": {"object_id", "support_id"}, "place": {"object_id", "support_id"},
                "pick": {"object_id"}, "carry": {"object_id", "support_id"},
                "navigate": {"support_id"}, "stop": set(), "wait": {"seconds"}}
            if not isinstance(action, str) or action not in shape:
                raise PlannerError("UNREGISTERED_GOAL", f"Unregistered structured intent {action!r}")
            extras = set(goal) - {"source_support_id", "requested_method", "order", "must_not"}
            if extras != shape[action]:
                raise PlannerError("INVALID_GOAL", f"Unexpected fields for structured {action} intent")
            goal.setdefault("requested_method", action)
            if action in {"transfer", "place"}:
                goal.update(predicate="supported_on", target_id=goal.pop("support_id"))
            elif action == "pick":
                goal["predicate"] = "held"
            elif action in {"navigate", "carry"}:
                goal.update(predicate="robot_at", target_id=goal.pop("support_id"))
            else:
                goal.update(predicate="action_completed", value=action)
                if action == "wait":
                    goal["value"] = {"skill": "wait", "seconds": goal.pop("seconds")}
            # Separate explicit intents are ordered; a transfer is one predicate.
            goal.setdefault("order", index)
        predicate = goal.get("predicate")
        if not isinstance(predicate, str) or predicate not in PREDICATES or set(goal) - allowed:
            raise PlannerError("INVALID_GOAL", "Goal requires a registered predicate and supported fields")
        needs_object = predicate in {"supported_on", "inside", "near", "held", "state_equals", "released", "stable"}
        needs_target = predicate in {"supported_on", "inside", "near", "robot_at"}
        for field, required, ids in (("object_id", needs_object, object_ids),
                                     ("target_id", needs_target, target_ids)):
            value = goal.get(field)
            if required and value is None:
                raise PlannerError("INVALID_GOAL", f"{predicate} requires {field}")
            if value is not None and (not isinstance(value, str) or not value or len(value) > 128):
                raise PlannerError("INVALID_GOAL", f"{field} requires a bounded entity ID")
            if value is not None and ids is not None and value not in ids:
                raise PlannerError("UNKNOWN_OBJECT" if field == "object_id" else "UNKNOWN_TARGET", f"Unregistered {field}: {value}")
        for field in ("requested_method", "source_support_id", "state_key"):
            if field in goal and (not isinstance(goal[field], str) or not goal[field] or len(goal[field]) > 128):
                raise PlannerError("INVALID_GOAL", f"{field} requires bounded text")
        if "source_support_id" in goal and target_ids is not None and goal["source_support_id"] not in target_ids:
            raise PlannerError("UNKNOWN_TARGET", "Source support is not registered")
        if "order" in goal and (type(goal["order"]) is not int or goal["order"] < 0):
            raise PlannerError("INVALID_GOAL", "Goal order must be a nonnegative integer")
        if "tolerance_m" in goal:
            value = goal["tolerance_m"]
            if type(value) not in (int, float) or not math.isfinite(value) or not value > 0:
                raise PlannerError("INVALID_GOAL", "tolerance_m requires a positive finite value")
        if "must_not" in goal:
            if not isinstance(goal["must_not"], list) or len(goal["must_not"]) > 64:
                raise PlannerError("INVALID_GOAL", "must_not is a bounded list of forbidden actions")
            for constraint in goal["must_not"]:
                if isinstance(constraint, str):
                    if not constraint or len(constraint) > 128:
                        raise PlannerError("INVALID_GOAL", "Forbidden skill names require bounded text")
                elif isinstance(constraint, dict):
                    if not constraint.get("skill") or set(constraint) - {"skill", "object_id", "target_id", "support_id", "target"} or any(
                            not isinstance(v, str) or not v or len(v) > 128 for v in constraint.values()):
                        raise PlannerError("INVALID_GOAL", "Forbidden action requires skill and optional entity IDs")
                    if "object_id" in constraint and object_ids is not None and constraint["object_id"] not in object_ids:
                        raise PlannerError("UNKNOWN_OBJECT", "Forbidden action object must be registered")
                    for field in ("target_id", "support_id", "target"):
                        if field in constraint and target_ids is not None and constraint[field] not in target_ids:
                            raise PlannerError("UNKNOWN_TARGET", "Forbidden action target must be registered")
                else:
                    raise PlannerError("INVALID_GOAL", "Forbidden action requires a skill name or scoped action object")
        if predicate == "state_equals" and (not goal.get("state_key") or "value" not in goal or
                not isinstance(goal["value"], (str, bool, int, float))):
            raise PlannerError("INVALID_GOAL", "state_equals requires state_key and a scalar value")
        if predicate == "state_equals" and type(goal.get("value")) is float and not math.isfinite(goal["value"]):
            raise PlannerError("INVALID_GOAL", "state_equals values must be finite")
        if predicate == "action_completed":
            value = goal.get("value")
            if not isinstance(value, (str, dict)) or not value:
                raise PlannerError("INVALID_GOAL", "action_completed requires a skill or action object")
            if isinstance(value, dict) and (set(value) != {"skill", "seconds"} or value["skill"] != "wait" or
                    type(value["seconds"]) not in (int, float) or not math.isfinite(value["seconds"]) or not 0 <= value["seconds"] <= 10):
                raise PlannerError("INVALID_GOAL", "Wait goals require 0..10 seconds")
        result.append(goal)
    return result


def _position(entity: dict) -> list | tuple | None:
    return entity.get("position_m", entity.get("position_xyz_m", entity.get("center_xyz_m")))


def action_forbidden(action: dict, constraint: str | dict, *, object_id: str | None = None) -> bool:
    if isinstance(constraint, str):
        return action.get("skill") == constraint
    scoped = {**action}
    if object_id is not None:
        scoped.setdefault("object_id", object_id)
    return all(scoped.get(field) == value for field, value in constraint.items())


def _resolved_target(record: dict) -> dict:
    result = record.get("result", record.get("evidence", {}))
    resolved = result.get("resolved_target", {}) if isinstance(result, dict) else {}
    position = resolved.get("position_m") if isinstance(resolved, dict) else None
    if not isinstance(position, (list, tuple)) or len(position) < 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in position):
        return {}
    return resolved


def _target_matches(resolved: dict, target: str) -> bool:
    return target in {resolved.get("point_id"), resolved.get("support_id"), resolved.get("target_id")}


def _superseded(goal: dict, goals: list[dict]) -> bool:
    """A later incompatible user goal turns this goal into a past milestone."""
    if "order" not in goal:
        return False
    later = [g for g in goals if g.get("order", -1) > goal["order"]]
    pred, obj = goal["predicate"], goal.get("object_id")
    for other in later:
        other_pred = other["predicate"]
        if pred == "robot_at" and other_pred == "robot_at" and other.get("target_id") != goal.get("target_id"):
            return True
        if other.get("object_id") != obj or obj is None:
            continue
        if pred == "robot_at" and goal.get("requested_method") == "carry" and other_pred in {"supported_on", "inside", "released"}:
            return True
        if pred in {"supported_on", "inside", "held", "released", "stable"} and other_pred in {"supported_on", "inside", "held", "released"}:
            if pred != other_pred or goal.get("target_id") != other.get("target_id"):
                return True
    return False


def evaluate_goals(goals: list, observation: dict, *, evidence: dict | None = None,
                   world: Any = None) -> dict:
    """Evaluate measured state; missing contact/temporal evidence never passes.

    evidence.placements is a list of independent placement evaluations, each
    containing object_id, support_id/target_id, passed and predicates. evidence
    .action_results contains completed action records {action,result,success}.
    Adapters may also provide evidence.objects[id] with contact/stability facts.
    """
    normalized = normalize_goals(goals)
    evidence = evidence or {}
    objects = observation.get("objects", {})
    if not objects and isinstance(observation.get("cube"), dict):
        cube = observation["cube"]
        objects = {cube.get("object_id", "cube"): cube}
    robot = observation.get("robot", {})
    held = robot.get("held_object")
    if held is None and observation.get("held_estimate") is True and len(objects) == 1:
        held = next(iter(objects))
    supports = observation.get("furniture", observation.get("supports", observation.get("targets", observation.get("target_regions", {}))))
    if world is not None:
        definition = world.to_dict()
        supports = {**definition.get("furniture", definition.get("targets", {})), **supports}
    region_ids = set(observation.get("targets", observation.get("target_regions", {})))
    if world is not None:
        region_ids.update(getattr(world, "targets", {}))
    region_ids.update(key for key, value in supports.items() if value.get("supports_inside") is True)
    records = evidence.get("action_results", [])
    completed, completed_records, actor_ids = [], [], []
    historical_held = evidence.get("initial_observation", evidence.get("initial_snapshot", {})).get("robot", {}).get("held_object")
    for record in records:
        if isinstance(record, dict) and record.get("success", record.get("status", "SUCCESS") == "SUCCESS"):
            action = record.get("action", record)
            if isinstance(action, dict) and "skill" in action:
                completed.append(action)
                completed_records.append(record)
                details = record.get("result", record.get("evidence", {}))
                if not isinstance(details, dict):
                    details = {}
                placement = details.get("independent_placement", details.get("placement", {}))
                actor = action.get("object_id", details.get("object_id", placement.get("object_id", historical_held)))
                actor_ids.append(actor)
                if action["skill"] == "pick":
                    historical_held = actor
                elif action["skill"] == "place":
                    historical_held = None
    predicates, previous_order_index, previous_order = [], -1, None
    unverifiable = False
    for goal in sorted(normalized, key=lambda g: g.get("order", 0)):
        pred, obj_id, target = goal["predicate"], goal.get("object_id"), goal.get("target_id")
        obj = objects.get(obj_id, {})
        facts = evidence.get("objects", {}).get(obj_id, {})
        placements = [p for p in evidence.get("placements", []) if isinstance(p, dict) and
            p.get("object_id") == obj_id and target in {p.get("support_id"), p.get("target_id")}]
        place = placements[-1] if placements else {}
        checks = place.get("predicates", {})
        passed, reason = False, "predicate is not satisfied"
        if pred == "held":
            bilateral = facts.get("finger_contact_side_count", obj.get("finger_contact_side_count", 0)) >= 2
            unsupported = "supports" in facts and not facts["supports"]
            verified = facts.get("held_verified") is True and facts.get("lift_verified") is True
            passed = held == obj_id and (verified or (bilateral and unsupported))
            if held == obj_id and not (verified or bilateral):
                reason = "missing measured bilateral grasp and lift evidence"; unverifiable = True
        elif pred == "released":
            measured = facts.get("released", checks.get("released"))
            passed = measured is True and held != obj_id
            if measured is None:
                reason = "missing measured robot contact/release evidence"; unverifiable = True
        elif pred == "stable":
            measured = facts.get("stable", facts.get("stable_supported"))
            if measured is None and checks:
                measured = checks.get("linear_still") is True and checks.get("angular_still") is True and place.get("stable_window_passed", place.get("passed")) is True
            passed = measured is True
            if measured is None:
                reason = "missing independent stability evidence"; unverifiable = True
        elif pred in {"supported_on", "inside"}:
            if placements:
                passed = place.get("passed") is True
            elif pred == "supported_on" and "supports" in facts:
                passed = target in facts["supports"] and facts.get("released") is True and facts.get("stable") is True
            else:
                reason = "missing independent support/contact/release/stability evidence"; unverifiable = True
            if pred == "inside" and passed:
                if target not in region_ids:
                    passed = False; reason = "support contact does not prove containment inside a container"
                region = supports.get(target, {})
                pos, centre = _position(obj), _position(region)
                half = region.get("half_size_m", region.get("half_size_xyz_m"))
                extents = obj.get("half_size_m", [0, 0, 0])
                passed = bool(passed and pos and centre and half and all(abs(pos[i]-centre[i])+extents[i] <= half[i]+1e-6 for i in (0, 1)))
        elif pred in {"near", "robot_at"}:
            source = robot if pred == "robot_at" else obj
            entity = supports.get(target, objects.get(target, {}))
            pos, centre = _position(source), _position(entity)
            if pred == "robot_at" and world is not None:
                point = getattr(world, "operation_points", {}).get(target)
                if point is not None:
                    centre = point.position_m
                elif hasattr(world, "operation_points"):
                    centres = [p.position_m for p in world.operation_points.values() if p.support == target]
                    try:
                        from embodied_agent.maps.manipulation import candidate_operation_points
                        positions = {key: item["position_m"] for key, item in objects.items()}
                        centres.extend(p.position_m for p in candidate_operation_points(world,
                            support_id=target, object_positions=positions))
                    except (ImportError, ValueError):
                        pass
                    if pos and centres:
                        centre = min(centres, key=lambda p: math.dist(pos[:2], p[:2]))
            if pred == "robot_at":
                for record in reversed(completed_records):
                    resolved = _resolved_target(record)
                    if record.get("action", {}).get("skill") in {"navigate", "carry"} and _target_matches(resolved, target):
                        centre = resolved["position_m"]
                        break
            passed = bool(pos and centre and math.dist(pos[:2], centre[:2]) <= goal.get("tolerance_m", .10))
            if goal.get("requested_method") == "carry":
                verified = facts.get("held_verified") is True and facts.get("lift_verified") is True
                bilateral = facts.get("finger_contact_side_count", 0) >= 2 and "supports" in facts and not facts["supports"]
                passed = passed and held == obj_id and (verified or bilateral)
        elif pred == "state_equals":
            passed = obj.get("states", {}).get(goal["state_key"]) == goal["value"]
        elif pred == "action_completed":
            expected = goal["value"]
            skill = expected if isinstance(expected, str) else expected["skill"]
            matching = [i for i, a in enumerate(completed) if a["skill"] == skill and
                        (not isinstance(expected, dict) or all(a.get(k) == v for k, v in expected.items()))]
            passed = bool(matching)
        if "order" in goal:
            # Ordered predicates are milestones, validated from the actual
            # completed action's contact/settling result, not the final pose.
            historical = []
            for i, record in enumerate(completed_records):
                action = completed[i]
                result = record.get("result", record.get("evidence", {}))
                if not isinstance(result, dict):
                    result = {}
                match = False
                if pred in {"supported_on", "inside", "released", "stable"} and action["skill"] == "place":
                    score = result.get("independent_placement", result.get("placement", result.get("goal_check", {})))
                    score_obj = score.get("object_id", result.get("object_id"))
                    score_target = action.get("support_id", action.get("target_id"))
                    physical = score.get("passed") is True and bool(result.get("stable_steps", score.get("stable_window_passed", result.get("verified"))))
                    match = score_obj == obj_id and (target is None or target == score_target) and physical and (pred != "inside" or target in region_ids)
                elif pred == "held" and action["skill"] == "pick" and action.get("object_id") == obj_id:
                    contact = result.get("grasp", result)
                    match = contact.get("held_verified") is True and contact.get("lift_verified") is True or (
                        contact.get("finger_contact_side_count", 0) >= 2 and not contact.get("supports", []) and
                        contact.get("achieved_lift_m", contact.get("lift_m", 0)) >= .03)
                elif pred == "robot_at" and action["skill"] in {"navigate", "carry"}:
                    action_target = action.get("target")
                    resolved = _resolved_target(record)
                    support = None
                    if world is not None:
                        point = getattr(world, "operation_points", {}).get(action_target)
                        if point is None and hasattr(world, "operation_points"):
                            try:
                                from embodied_agent.maps.manipulation import resolve_navigation_target
                                point = resolve_navigation_target(world, record.get("observation", observation), action_target)
                            except (ImportError, ValueError):
                                point = None
                        support = point.support if point is not None else None
                    match = target in {action_target, support} or _target_matches(resolved, target)
                    measured = record.get("observation", observation)
                    measured_robot = measured.get("robot", {}) if isinstance(measured, dict) else {}
                    arrival_position = _position(measured_robot)
                    dock_position = resolved.get("position_m", point.position_m if world is not None and point is not None else None)
                    match = bool(match and arrival_position and dock_position and math.dist(arrival_position[:2], dock_position[:2]) <= goal.get("tolerance_m", .10))
                    if goal.get("requested_method") == "carry":
                        match = match and action["skill"] == "carry" and result.get("object_id", obj_id) == obj_id and measured_robot.get("held_object") == obj_id
                elif pred == "action_completed":
                    expected = goal["value"]
                    match = action["skill"] == expected if isinstance(expected, str) else all(action.get(k) == v for k, v in expected.items())
                lower = previous_order_index if previous_order == goal["order"] else previous_order_index + 1
                if match and i >= lower:
                    historical.append(i)
            if historical:
                passed = bool(passed or _superseded(goal, normalized))
                previous_order_index = historical[0]; previous_order = goal["order"]
                if not passed:
                    reason = "completed milestone no longer satisfies the measured final state"
            elif previous_order_index >= 0 or not passed:
                passed = False; reason = "ordered milestone lacks actual completed action evidence"
        source = goal.get("source_support_id")
        if source:
            initial = evidence.get("initial_observation", evidence.get("initial_snapshot", {}))
            initial_obj = initial.get("objects", {}).get(obj_id, {})
            initial_support = initial_obj.get("support")
            if initial_support is None:
                passed = False; reason = "missing task-initial source observation"; unverifiable = True
            elif initial_support != source:
                passed = False; reason = "actual initial source differs from the requested source"
        method = goal.get("requested_method")
        if method and method not in {"transfer", "place", "pick", "carry", "navigate", "stop", "wait"}:
            methods = evidence.get("performed_methods", [])
            if method not in methods:
                passed = False; reason = f"requested method {method!r} was not proven"
        forbidden = goal.get("must_not", [])
        if any(action_forbidden(a, constraint, object_id=actor_ids[i]) for i, a in enumerate(completed) for constraint in forbidden):
            passed = False; reason = "a forbidden action was executed"
        predicates.append({**goal, "passed": bool(passed), "reason": "satisfied by measured evidence" if passed else reason})
    passed = all(p["passed"] for p in predicates)
    return {"passed": passed, "predicates": predicates,
            "goal_status": "satisfied" if passed else "unverifiable" if unverifiable else "unsatisfied"}

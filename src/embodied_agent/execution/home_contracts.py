"""Finite home skill contract, independent of simulator and model services."""
from __future__ import annotations

import math
from typing import Any


class HomeExecutionError(ValueError):
    def __init__(self, code: str, message: str, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.evidence = evidence or {}


SKILL_PARAMETERS = {
    "observe": set(), "query_world": {"object_id"}, "inspect": {"object_id"},
    "navigate": {"target"}, "pick": {"object_id"}, "carry": {"target"},
    "place": {"support_id", "target_xy"}, "wait": {"seconds"}, "stop": set(),
}


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def validate_home_plan(payload: Any) -> list[dict[str, Any]]:
    if (not isinstance(payload, dict) or set(payload) != {"schema_version", "actions"}
            or type(payload["schema_version"]) is not int or payload["schema_version"] != 1):
        raise HomeExecutionError("INVALID_PLAN", "Expected home plan schema_version 1 and actions")
    actions = payload["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 32:
        raise HomeExecutionError("INVALID_PLAN", "Plan requires 1..32 actions")
    validated = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("skill"), str):
            raise HomeExecutionError("INVALID_PLAN", "Every action requires a registered skill")
        skill = action["skill"]
        if skill not in SKILL_PARAMETERS:
            raise HomeExecutionError("UNREGISTERED_SKILL", f"Skill {skill!r} is unavailable")
        if set(action) != {"skill", *SKILL_PARAMETERS[skill]}:
            raise HomeExecutionError("INVALID_PLAN", f"Unexpected parameters for {skill}")
        for field in ("object_id", "target", "support_id"):
            if field in action and (not isinstance(action[field], str) or not action[field].strip()):
                raise HomeExecutionError("INVALID_PLAN", f"{field} requires an ID")
        value = dict(action)
        if skill == "place":
            point = action["target_xy"]
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(not _finite(v) for v in point)):
                raise HomeExecutionError("INVALID_PLAN", "target_xy requires two finite metre coordinates")
            value["target_xy"] = list(point)
        if skill == "wait":
            seconds = action["seconds"]
            if not _finite(seconds) or not 0 <= seconds <= 10:
                raise HomeExecutionError("INVALID_PLAN", "Wait requires 0..10 seconds")
        validated.append(value)
    return validated

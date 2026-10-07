"""Pure case catalog validation and independent expected-result scoring."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable
from embodied_agent.paths import PROJECT_ROOT as ROOT
from embodied_agent.execution.home_contracts import HomeExecutionError, validate_home_plan

def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value

def _contains(actual: Any, expected: Any, path: str = "map_snapshot") -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected an object"]
        return [failure for key, value in expected.items()
                for failure in _contains(actual.get(key), value, f"{path}.{key}")]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: expected {expected!r}, got {actual!r}"]
        return [failure for index, value in enumerate(expected)
                for failure in _contains(actual[index], value, f"{path}[{index}]")]
    equal = math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8) if (
        isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    ) else actual == expected
    return [] if equal else [f"{path}: expected {expected!r}, got {actual!r}"]


def check_expected(result: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, value in {"status": "SUCCESS", **expected}.items():
        if key == "map_contains":
            failures.extend(_contains(result.get("map_snapshot"), value))
        elif result.get(key) != value:
            failures.append(f"{key}: expected {value!r}, got {result.get(key)!r}")
    return not failures, failures


def validate_case(case: dict[str, Any], agents: set[str]) -> None:
    if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
        raise ValueError("Each case requires a nonempty case_id")
    if "map_id" in case and (not isinstance(case["map_id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", case["map_id"])):
        raise ValueError(f"Invalid map_id in {case['case_id']}")
    if "initial_overrides" in case:
        overrides = case["initial_overrides"]
        if not isinstance(overrides, dict) or set(overrides) - {"cube_position_m", "danger_zone", "targets"}:
            raise ValueError("initial_overrides only supports cube_position_m, danger_zone and targets")
    if "steps" in case and any(key in case for key in ("instruction", "agent", "expected", "robot_plan", "test_fault_protocol")):
        raise ValueError("A case must use steps or a single instruction, not both")
    steps = case.get("steps", [case])
    if not isinstance(steps, list) or not steps:
        raise ValueError("Case steps must be a nonempty list")
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("instruction"), str) or not step["instruction"].strip():
            raise ValueError("Each step requires a nonempty instruction")
        if step.get("agent", "robot") not in agents:
            raise ValueError(f"Unknown case agent: {step.get('agent')}")
        expected = step.get("expected", {"status": "SUCCESS"})
        if not isinstance(expected, dict) or not expected or set(expected) - {"status", "error_code", "target_id", "map_contains", "transport_success"}:
            raise ValueError("Invalid expected assertion; supported keys: status, error_code, target_id, map_contains, transport_success")
        if expected.get("status", "SUCCESS") not in {"SUCCESS", "FAILED", "ABORTED"}:
            raise ValueError("Invalid expected status")
        for key in ("error_code", "target_id"):
            if key in expected and expected[key] is not None and not isinstance(expected[key], str):
                raise ValueError(f"expected.{key} must be a string or null")
        if "map_contains" in expected and (not isinstance(expected["map_contains"], dict) or not expected["map_contains"]):
            raise ValueError("expected.map_contains must be a nonempty object")
        if "transport_success" in expected and type(expected["transport_success"]) is not bool:
            raise ValueError("expected.transport_success must be a boolean")
        if "robot_plan" in step:
            if step.get("agent", "robot") not in {"robot"}:
                raise ValueError("robot_plan requires the existing robot agent")
            try:
                validate_home_plan(step["robot_plan"])
            except HomeExecutionError as exc:
                raise ValueError(f"Invalid trusted robot_plan: {exc.code}: {exc}") from exc
        if "test_fault_protocol" in step:
            if step["test_fault_protocol"] != "navigation_waypoint_into_tea_table_v1":
                raise ValueError("Unknown trusted test_fault_protocol")
            if "robot_plan" not in step or step["robot_plan"] != {"schema_version": 1, "actions": [{"skill": "navigate", "target": "tea_table"}]}:
                raise ValueError("Navigation fault requires the fixed tea_table navigation fixture")


def load_cases(path: Path | None = None, *, root: Path = ROOT,
               known_agents: Iterable[str] = ("robot", "environment")) -> list[dict[str, Any]]:
    root = Path(root)
    source = Path(path or root / "configs" / "demo_cases.json").resolve()
    payload = _read_json(source)
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("Case file requires schema_version=1 and a cases array")
    if "include_m3_cases" in payload and type(payload["include_m3_cases"]) is not bool:
        raise ValueError("include_m3_cases must be a boolean")
    cases: list[dict[str, Any]] = []
    scenes = {s["scenario_id"]: s for s in _read_json(root / "configs" / "m2_scenarios.json")["scenarios"]}

    def convert_legacy(case: dict[str, Any]) -> dict[str, Any]:
        if "scenario_id" not in case:
            return case
        if case.get("scenario_id") not in scenes:
            raise ValueError(f"Unknown legacy scenario_id: {case.get('scenario_id')}")
        scene = scenes[case["scenario_id"]]
        if case.get("seed") != scene["seed"] or case.get("expected_target_id") != f"target_{scene['target']}":
            raise ValueError(f"Legacy case seed/target mismatch: {case.get('case_id')}")
        return {"case_id": case["case_id"], "name": case["case_id"], "map_id": "classic",
                "agent": "robot", "instruction": case["instruction"],
                "initial_overrides": {"cube_position_m": scene["cube_position_m"]},
                "expected": {"status": "SUCCESS", "target_id": case["expected_target_id"]},
                "legacy_scenario_id": scene["scenario_id"], "legacy_seed": scene["seed"]}

    if payload.get("include_m3_cases", False):
        legacy = _read_json(root / "configs" / "m3_cases.json")
        for case in legacy["cases"]:
            cases.append(convert_legacy(case))
    cases.extend(convert_legacy(case) for case in payload["cases"])
    identifiers: set[str] = set()
    agents = set(known_agents)
    for case in cases:
        validate_case(case, agents)
        if case["case_id"] in identifiers:
            raise ValueError(f"Duplicate case_id: {case['case_id']}")
        identifiers.add(case["case_id"])
    if not cases:
        raise ValueError("Case file contains no cases")
    return cases

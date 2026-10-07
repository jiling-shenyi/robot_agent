"""Natural-language edits compiled into a small, validated map-edit language.

The LLM can propose data only. It cannot choose a file, execute code, change
physics/safety thresholds, or mutate a running MuJoCo simulation.
"""

from __future__ import annotations

import json
import math
import os
import re
from decimal import Decimal
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable

from embodied_agent.models.contracts import PlannerError
from embodied_agent.maps.schema import WorldError, WorldMap, finite_vector, strict_json
from embodied_agent.maps.store import MapStore
from embodied_agent.paths import PROJECT_ROOT
from embodied_agent.models.deepseek import RequestErrors, create_client
from embodied_agent.models.dialogue import run_tool_dialogue
from embodied_agent.prompts import PromptCatalog
from embodied_agent.tools import QueryTools, QueryToolCatalog
from embodied_agent.context import AgentContext
from embodied_agent.models.tracing import record_model_event
from embodied_agent.models.tracing import ModelRecordingError, new_trace_id, trace_scope
from embodied_agent.recording.environment import record_environment_operation, record_rejection, map_diff


class EnvironmentAgentError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EnvironmentEditResult:
    before: WorldMap
    after: WorldMap
    operations: tuple[dict[str, Any], ...]
    planner_kind: str
    persisted: bool
    recording_errors: tuple[dict, ...] = ()
    task_record: dict | None = None
    component_errors: tuple[dict, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": "SUCCESS", "agent": "environment", "planner_kind": self.planner_kind,
            "persisted": self.persisted, "map_id": self.after.map_id,
            "before": self.before.to_dict(), "after": self.after.to_dict(),
            "operations": deepcopy(list(self.operations)),
            "proposal_valid": True, "commit_status": "success" if self.persisted else "not_started",
        }
        if self.recording_errors:
            result["recording_errors"] = deepcopy(list(self.recording_errors))
        if self.component_errors:
            result["component_errors"] = deepcopy(list(self.component_errors))
        if self.task_record:
            result.update(task_record=deepcopy(self.task_record), task_id=self.task_record["task_id"],
                          task_record_path=self.task_record["path"])
        return result


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_VECTOR = re.compile(rf"[\(（\[]?\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[\)）\]]?")
_LABELED_VECTOR = re.compile(rf"x\s*[=:：]\s*({_NUMBER})\s*[,，]?\s*y\s*[=:：]\s*({_NUMBER})\s*[,，]?\s*z\s*[=:：]\s*({_NUMBER})", re.I)
_ENTITIES = (
    ("danger_zone", re.compile(r"危险区(?:域)?|danger[_\s-]*zone", re.I)),
    ("cube", re.compile(r"(?:目标)?方块|cube", re.I)),
    ("target_a", re.compile(r"(?:目标\s*[aA](?:\s*区)?|目标区\s*[aA]|[aA]\s*区|绿色(?:目标)?区|target[_\s-]*a\b)", re.I)),
    ("target_b", re.compile(r"(?:目标\s*[bB](?:\s*区)?|目标区\s*[bB]|[bB]\s*区|蓝色(?:目标)?区|target[_\s-]*b\b)", re.I)),
)
_AXIS = re.compile(r"(?<![A-Za-z0-9_])([xyz])\s*(?:轴|[-\s]*axis\b)", re.I)
_AXIS_INCREASE = re.compile(r"调高|提高|增大|增加|上调|升高|increase|raise|increment", re.I)
_AXIS_DECREASE = re.compile(r"调低|降低|减小|减少|下调|下降|decrease|lower|decrement", re.I)
_AXIS_AMOUNT = re.compile(rf"({_NUMBER})\s*(毫米|厘米|米|mm|cm|m)(?![A-Za-z])", re.I)
_AXIS_SMALL = re.compile(r"一点点|一点|一些|少许|稍微|略微|a\s+little|slightly", re.I)
_SMALL_SHIFT_M = 0.01
_MAX_AXIS_SHIFT_M = 0.5


def _invalid_instruction(message: str) -> EnvironmentAgentError:
    return EnvironmentAgentError("INVALID_ENVIRONMENT_INSTRUCTION", message)


def _entity(clause: str) -> tuple[str, str]:
    found = [(name, match) for name, pattern in _ENTITIES for match in pattern.finditer(clause)]
    if len(found) != 1:
        raise _invalid_instruction("Each edit must name exactly one cube, danger zone, target A or target B")
    name, match = found[0]
    return name, clause[:match.start()] + " OBJECT " + clause[match.end():]


def _rules_axis_shift(template: str, object_id: str) -> dict[str, Any] | None:
    axes = list(_AXIS.finditer(template))
    if not axes:
        return None
    if len(axes) != 1:
        raise _invalid_instruction("A relative edit must name exactly one x, y or z axis")
    directions = [(1, match) for match in _AXIS_INCREASE.finditer(template)]
    directions += [(-1, match) for match in _AXIS_DECREASE.finditer(template)]
    if len(directions) != 1:
        raise _invalid_instruction("A relative axis edit needs one clear increase or decrease direction")
    amounts = list(_AXIS_AMOUNT.finditer(template))
    smalls = list(_AXIS_SMALL.finditer(template))
    if len(amounts) + len(smalls) != 1:
        raise _invalid_instruction("Specify one distance or '一点' (0.01 m) for an axis edit")
    if amounts:
        amount = amounts[0]
        magnitude = float(amount.group(1))
        unit = amount.group(2).lower()
        magnitude *= {"米": 1.0, "m": 1.0, "厘米": 0.01, "cm": 0.01, "毫米": 0.001, "mm": 0.001}[unit]
    else:
        magnitude = _SMALL_SHIFT_M
    if not math.isfinite(magnitude) or not 0 < magnitude <= _MAX_AXIS_SHIFT_M:
        raise _invalid_instruction("Axis shift must be greater than zero and at most 0.5 m")
    matches = [(axes[0], " AXIS "), (directions[0][1], " DIRECTION ")]
    matches.append((amounts[0], " AMOUNT ") if amounts else (smalls[0], " AMOUNT "))
    rest = template
    for match, replacement in sorted(matches, key=lambda item: item[0].start(), reverse=True):
        rest = rest[:match.start()] + replacement + rest[match.end():]
    permitted = re.compile(
        r"(?:OBJECT|AXIS|DIRECTION|AMOUNT|请|将|把|的|初始|中心|位置|坐标|数据|数值|值|往|朝|向|沿|平移|移动|"
        r"[\s:：，,。.!?？]|initial|position|coordinates?|data|value|the|of|by|along|move|shift)",
        re.I,
    )
    if permitted.sub("", rest.strip()):
        raise _invalid_instruction("Rules mode could not understand the complete relative edit")
    return {
        "op": "shift_axis", "object_id": object_id,
        "axis": axes[0].group(1).lower(), "delta_m": directions[0][0] * magnitude,
    }


def _rules_clause(clause: str) -> dict[str, Any]:
    object_id, template = _entity(clause)
    # A complete grammar check after replacing arguments prevents accepting a
    # valid prefix while silently ignoring an unsupported trailing command.
    vector = _LABELED_VECTOR.search(template) or _VECTOR.search(template)
    if vector:
        values = [float(vector.group(index)) for index in (1, 2, 3)]
        rest = template[:vector.start()] + " VECTOR " + template[vector.end():]
        if re.search(r"厘米|\bcm\b|毫米|\bmm\b", rest, re.I):
            raise _invalid_instruction("Rules mode requires metres; convert centimetres/millimetres explicitly")
        half_size = bool(re.search(r"半尺寸|半大小|半边长|半长|half[_\s-]*(?:size|extents?)", rest, re.I))
        size = half_size or bool(re.search(r"尺寸|大小|边长|全长|size|dimensions?", rest, re.I))
        if size and object_id == "cube":
            raise _invalid_instruction("Cube dimensions are fixed by the robot's grasp contract")
        op_words = r"调整到|调整为|设置为|设置到|设为|设到|改为|改到|移到|移动到|移动至|调整|设置|移动|设|set|move|change|update"
        if not re.search(op_words, rest, re.I):
            raise _invalid_instruction("State an explicit set/move operation and a three-dimensional coordinate")
        permitted = re.compile(
            r"(?:OBJECT|VECTOR|请|将|把|的|初始|中心|位置|坐标|新的|新|半尺寸|半大小|半边长|半长|尺寸|大小|边长|全长|"
            + op_words +
            r"|为|到|至|米|[\s:：=。.!]|half[_\s-]*(?:size|extents?)|size|dimensions?|position|coordinates?|initial|centre|center|the|new|of|to|in|m(?:et(?:er|re)s?)?)",
            re.I,
        )
        if permitted.sub("", rest.strip()):
            raise _invalid_instruction("Rules mode could not understand the complete edit; use one explicit edit per clause")
        if size and not half_size:
            values = [value / 2.0 for value in values]
        return {"op": "set_half_size" if size else "set_position", "object_id": object_id, "value": values}

    axis_shift = _rules_axis_shift(template, object_id)
    if axis_shift is not None:
        return axis_shift

    if object_id != "danger_zone":
        raise _invalid_instruction("Provide an explicit xyz position; percentage scaling is supported for danger_zone only")
    # Percentages distinguish 'increase BY 20%' from 'scale TO 120%'.
    percent = re.search(rf"({_NUMBER})\s*[%％]", template)
    multiple = re.search(rf"({_NUMBER})\s*倍", template)
    english_factor = re.search(rf"\bby\s+({_NUMBER})\s*$", template, re.I)
    match = percent or multiple or english_factor
    if not match:
        raise _invalid_instruction("Specify an exact scale, e.g. 将危险区扩大20% or 将危险区扩大到原来的1.2倍")
    amount = float(match.group(1))
    if not math.isfinite(amount) or amount <= 0:
        raise _invalid_instruction("Scaling must use a positive finite amount")
    rest = template[:match.start()] + " AMOUNT " + template[match.end():]
    if percent:
        if re.search(r"(?:扩大|放大|缩小|缩放|调整)(?:到|为)|(?:scale|resize).*?\bto\b", rest, re.I):
            factor = amount / 100.0
        elif re.search(r"缩小|减小|reduce|decrease", rest, re.I):
            factor = 1.0 - amount / 100.0
        elif re.search(r"扩大|放大|增大|increase|enlarge", rest, re.I):
            factor = 1.0 + amount / 100.0
        else:
            raise _invalid_instruction("Specify whether the percentage is an increase or a resulting scale")
    else:
        if multiple and not re.search(r"(?:到|为)(?:原来的)?\s*AMOUNT", rest):
            raise _invalid_instruction("Use 扩大到原来的1.2倍; 扩大1.2倍 is ambiguous")
        if english_factor and not re.search(r"\bscale\b", rest, re.I):
            raise _invalid_instruction("Use scale danger zone by 1.2 for an exact multiplier")
        factor = amount
    allowed = re.compile(r"(?:OBJECT|AMOUNT|请|将|把|的|原来|原|尺寸|大小|统一|整体|扩大|放大|增大|缩小|减小|缩放|调整|到|为|[\s。.!]|scale|resize|increase|enlarge|reduce|decrease|the|by|to|size|of)", re.I)
    if allowed.sub("", rest.strip()):
        raise _invalid_instruction("Rules mode could not understand the complete scaling command")
    return {"op": "scale", "object_id": object_id, "factor": factor}


def rules_plan(instruction: str, world: WorldMap) -> dict[str, Any]:
    clauses = [item.strip() for item in re.split(r"[;；\n]+|然后|并且", instruction) if item.strip()]
    if not 1 <= len(clauses) <= 8:
        raise _invalid_instruction("Use between one and eight explicit edits separated by semicolons")
    return {"schema_version": 1, "base_revision": world.revision, "operations": [_rules_clause(item) for item in clauses]}


_REQUEST_ERRORS = RequestErrors(
    rate_limited="DeepSeek rate limited the environment planning request",
    timeout="Environment planning request timed out",
    connection_failed="Could not connect to the configured DeepSeek endpoint",
    failed_prefix="Environment planning request failed",
    empty_response="DeepSeek returned no completion choices",
)


class EnvironmentAgent:
    """Plan -> strict operation validation -> whole-world validation -> CAS save.

    ``plan`` is deliberately public and replaceable: a GUI may run planning in
    a worker while pumping its event loop. Validation and persistence remain
    downstream of planning. ``rules`` is an explicitly labelled offline mode;
    an LLM error never silently falls back to rules.
    """

    def __init__(
        self, store: MapStore, mode: str = "llm", config: dict[str, Any] | None = None,
        planner: Callable[[str, WorldMap], dict[str, Any] | str] | None = None,
        *, prompt_catalog=None, tool_catalog=None, tool_factory=None, context_provider=None,
        recording_enabled: bool = True, records_dir=None,
    ):
        if mode not in {"llm", "rules"}:
            raise EnvironmentAgentError("INVALID_MODE", "Environment planner mode must be llm or rules")
        self.store = store
        self.mode = mode
        self.planner_kind = mode
        self._planner = planner
        self.config = deepcopy(config)
        self.prompt_catalog = prompt_catalog
        self.tool_catalog = tool_catalog
        self.tool_factory = tool_factory if tool_factory is not None else QueryTools
        self.context_provider = context_provider
        self.recording_enabled, self.records_dir = recording_enabled, records_dir
        self.last_record_ref = self.last_recording_error = None
        if config is not None:
            self._configure_components(self.config)
        self.last_dialogue: dict[str, Any] | None = None

    def _configure_components(self, config: dict) -> None:
        if self.prompt_catalog is None:
            self.prompt_catalog = PromptCatalog(config)
        if self.tool_catalog is None:
            self.tool_catalog = QueryToolCatalog(config)
        if self.context_provider is None:
            self.context_provider = AgentContext(config, role="environment")

    def plan(self, instruction: str, world: WorldMap) -> dict[str, Any] | str:
        self.last_dialogue = None
        if self._planner is not None:
            return self._planner(instruction, world)
        if self.mode == "rules":
            return rules_plan(instruction, world)
        return self._llm_plan(instruction, world)

    def _llm_plan(self, instruction: str, world: WorldMap, *, prompt_id: str = "environment.desktop") -> str:
        self.last_dialogue = {
            "request_count": 0, "llm_request_count": 0, "tool_call_count": 0,
            "messages": [], "tool_results": [],
        }
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise PlannerError("MISSING_CREDENTIAL", "DEEPSEEK_API_KEY is not configured; use --environment-planner rules for explicit offline testing")
        config = self.config
        if config is None:
            config_path = PROJECT_ROOT / "configs" / "agent_runtime.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.config = deepcopy(config)
        self._configure_components(config)
        client = create_client(
            api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout_s=float(config["budgets"]["planner_timeout_s"]),
        )
        try:
            query_tools = self.tool_factory(world, edit_mode=True, catalog=self.tool_catalog)
            prompt = self.prompt_catalog.get(prompt_id)
            payload = {"instruction": instruction, "world": world.to_dict()}
            auxiliary_context = self.context_provider.build(instruction)
            if auxiliary_context:
                payload["auxiliary_context"] = auxiliary_context
            record_model_event("agent_components", {"role": "environment", "stage": prompt_id,
                "prompt": prompt.to_dict(), "tool_catalog": self.tool_catalog.fingerprint,
                "context": self.context_provider.describe()})
            dialogue = run_tool_dialogue(
                client, instruction=instruction,
                stage=prompt_id,
                system_prompt=prompt.text,
                user_payload=payload,
                tools=query_tools.schemas, tool_handler=query_tools.call, config=config,
                errors=_REQUEST_ERRORS,
            )
            self.last_dialogue = {
                "request_count": dialogue.request_count,
                "llm_request_count": dialogue.request_count,
                "tool_call_count": dialogue.tool_call_count,
                "messages": deepcopy(dialogue.messages),
                "tool_results": deepcopy(dialogue.tool_results),
                "response": asdict(dialogue.response),
            }
            return dialogue.response.content
        except PlannerError as exc:
            self.last_dialogue.update(deepcopy(exc.details))
            self.last_dialogue["error_code"] = exc.code
            self.last_dialogue["message"] = str(exc)
            request_count = exc.details.get("request_count", exc.details.get("llm_request_count", 0))
            self.last_dialogue["request_count"] = request_count
            self.last_dialogue["llm_request_count"] = request_count
            raise
        finally:
            client.close()

    @record_environment_operation("preview")
    def preview(self, map_id: str, instruction: str, *, expected_revision: int | None = None) -> EnvironmentEditResult:
        self.last_dialogue = None
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 2000:
            raise _invalid_instruction("Environment instructions must contain 1..2000 characters")
        before = self.store.load(map_id)
        record_model_event("environment_state", {"stored_map_before": before.to_dict(), "operation": "preview"})
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision != before.revision):
            raise EnvironmentAgentError("REVISION_CONFLICT", "Displayed map revision has changed; reload the map before editing")
        # Preserve the original language for model interpretation. Only rules
        # mode performs language parsing; the downstream contract validates data.
        raw = self.plan(instruction, before)
        record_model_event("environment_proposal", {"raw_proposal": raw, "base_revision": before.revision})
        if isinstance(raw, str):
            if len(raw) > 16384:
                raise EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", "Environment plan exceeds 16 KiB")
            try:
                payload = strict_json(raw)
            except WorldError as exc:
                raise EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", str(exc)) from exc
        else:
            payload = deepcopy(raw)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "base_revision", "operations"}:
            raise EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", "Unexpected environment-plan schema")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", "Unsupported environment-plan schema version")
        if type(payload["base_revision"]) is not int or payload["base_revision"] != before.revision:
            raise EnvironmentAgentError("REVISION_CONFLICT", "Environment plan refers to a stale map revision")
        operations = payload["operations"]
        if not isinstance(operations, list) or not 1 <= len(operations) <= 8:
            raise EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", "The instruction needs 1..8 supported edits; specify an amount, xyz coordinates, or an axis and direction for a 0.01 m small shift")
        edited = before.to_dict()
        for operation in operations:
            self._apply_operation(edited, operation)
        after = WorldMap.from_dict(edited)
        record_model_event("environment_validated", {"plan": payload, "proposal_valid": True,
            "stored_map_before": before.to_dict(), "candidate_map_after": after.to_dict(),
            "diff": map_diff(before.to_dict(), after.to_dict())})
        return EnvironmentEditResult(before, after, tuple(deepcopy(operations)), self.planner_kind, False)

    @record_environment_operation("apply")
    def apply(self, map_id: str, instruction: str, *, expected_revision: int | None = None) -> EnvironmentEditResult:
        decision_id = new_trace_id("decision")
        with trace_scope(decision_id=decision_id):
            record_model_event("decision_started", {"operation": "map_edit", "instruction": instruction})
            try:
                preview = self.preview(map_id, instruction, expected_revision=expected_revision)
            except Exception as error:
                record_rejection("environment_rejected", {"error_code": getattr(error, "code", "ENVIRONMENT_ERROR"),
                    "message": str(error), "proposal_valid": False}, error)
                raise
            record_model_event("decision_finished", {"proposal_valid": True, "operations": list(preview.operations)})
        with trace_scope(decision_id=decision_id, action_id=new_trace_id("edit")):
            record_model_event("environment_commit_started", {"base_revision": preview.before.revision,
                "stored_map_before": preview.before.to_dict(), "candidate_map_after": preview.after.to_dict(),
                "operations": list(preview.operations)})
            try:
                after = self.store.save(preview.after, expected_revision=preview.before.revision)
            except Exception as error:
                error.commit_status = "rejected" if getattr(error, "code", None) == "REVISION_CONFLICT" else "unknown"
                observed = None
                try:
                    observed = self.store.load(map_id).to_dict()
                    error.observed_persisted_map = observed
                except Exception:
                    pass
                record_rejection("environment_commit_rejected", {"error_code": getattr(error, "code", "SAVE_FAILED"),
                    "message": str(error), "commit_status": error.commit_status,
                    "stored_map_observed_after_error": observed}, error)
                raise
            try:
                record_model_event("environment_commit_finished", {"saved_revision": after.revision,
                    "stored_map_before": preview.before.to_dict(), "stored_map_after": after.to_dict(),
                    "map_after": after.to_dict(), "operations": list(preview.operations), "commit_status": "success"})
            except ModelRecordingError as error:
                return EnvironmentEditResult(preview.before, after, preview.operations, self.planner_kind, True,
                    ({"event": "commit.finished", "message": str(error), "details": error.details},))
            return EnvironmentEditResult(preview.before, after, preview.operations, self.planner_kind, True)

    @staticmethod
    def _apply_operation(world: dict[str, Any], operation: Any) -> None:
        def invalid(message: str) -> None:
            raise EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", message)

        if not isinstance(operation, dict):
            invalid("Each operation must be an object")
        op = operation.get("op")
        object_id = operation.get("object_id")
        if not isinstance(op, str) or op not in {"set_position", "set_half_size", "scale", "shift_axis"}:
            invalid("Only set_position, set_half_size, scale and shift_axis are supported")
        if not isinstance(object_id, str) or object_id not in {"cube", "danger_zone", "target_a", "target_b"}:
            invalid("Unsupported scene entity")
        required = {"op", "object_id"}
        required |= {"factor"} if op == "scale" else {"axis", "delta_m"} if op == "shift_axis" else {"value"}
        if set(operation) != required:
            invalid("Unexpected operation fields")
        if op not in {"set_position", "shift_axis"} and object_id == "cube":
            invalid("Cube size is not editable")
        region = world["danger_zone"] if object_id == "danger_zone" else world["targets"].get(object_id)
        if op == "scale":
            if object_id != "danger_zone":
                invalid("Only danger_zone supports uniform scaling")
            factor = operation["factor"]
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or not 0 < factor <= 10:
                invalid("Scale factor must be finite and between 0 (exclusive) and 10")
            region["half_size_m"] = [value * factor for value in region["half_size_m"]]
        elif op == "shift_axis":
            axis = operation["axis"]
            delta = operation["delta_m"]
            if not isinstance(axis, str) or axis not in {"x", "y", "z"}:
                invalid("shift_axis.axis must be exactly x, y or z")
            if isinstance(delta, bool) or not isinstance(delta, (int, float)):
                invalid("shift_axis.delta_m must be finite, nonzero and at most 0.5 m in magnitude")
            try:
                delta = float(delta)
            except (OverflowError, ValueError):
                invalid("shift_axis.delta_m must be finite, nonzero and at most 0.5 m in magnitude")
            if not math.isfinite(delta) or not 0 < abs(delta) <= _MAX_AXIS_SHIFT_M:
                invalid("shift_axis.delta_m must be finite, nonzero and at most 0.5 m in magnitude")
            position = world["cube_position_m"] if object_id == "cube" else region["position_m"]
            index = "xyz".index(axis)
            # Add the decimal values before serializing as float, avoiding
            # noise such as 0.41000000000000003 in saved map coordinates.
            position[index] = float(Decimal(str(position[index])) + Decimal(str(delta)))
        else:
            try:
                value = list(finite_vector(operation["value"], "operation.value"))
            except WorldError as exc:
                raise EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", str(exc)) from exc
            if object_id == "cube":
                world["cube_position_m"] = value
            elif op == "set_position":
                region["position_m"] = value
            else:
                region["half_size_m"] = value

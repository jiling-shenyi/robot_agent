"""The existing environment agent's finite edits for both Demo map domains."""
from __future__ import annotations

import re
from copy import deepcopy

from embodied_agent.agents.environment import EnvironmentAgent, EnvironmentAgentError, EnvironmentEditResult, _rules_axis_shift
from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.schema import WorldError, finite_vector, strict_json
from embodied_agent.models.tracing import record_model_event
from embodied_agent.recording.environment import record_environment_operation, map_diff

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_VECTOR = rf"[\(（\[]\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[\)）\]]"
_ALIASES = {"remote": "遥控器", "book": "书本|书", "cup": "玻璃杯|杯子|杯", "kettle": "热水壶|水壶",
            "cleaner": "清洁用品|清洁容器", "medicine": "药品", "knife": "刀具", "device": "电子设备"}


def _invalid(message):
    return EnvironmentAgentError("INVALID_ENVIRONMENT_PLAN", message)


def home_rules_plan(instruction, world):
    operations = []
    working_positions = {name: list(obj.position_m) for name, obj in world.objects.items()}
    for clause in re.split(r"[;；]", instruction):
        clause = clause.strip().rstrip("。")
        if not clause:
            raise _invalid("Each edit clause must be explicit")
        renamed = re.fullmatch(r"(?:请)?(?:将|把)?(?:房间|地图)(?:的)?(?:名称|名字)(?:设为|设置为|改为)\s*(.{1,100})", clause)
        if renamed:
            operations.append({"op": "set_name", "value": renamed.group(1).strip().strip('"“”')})
            continue
        world_description = re.fullmatch(r"(?:请)?(?:将|把)?(?:房间|地图)(?:的)?描述(?:设为|设置为|改为)\s*(.{0,1000})", clause)
        if world_description:
            operations.append({"op": "set_description", "object_id": "world", "value": world_description.group(1)})
            continue
        found = []
        entity_text = re.split(r"描述(?:设为|设置为|改为)", clause, maxsplit=1)[0]
        for name in world.objects:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
            alias = _ALIASES.get(name)
            matches = list(re.finditer(pattern + ("|" + alias if alias else ""), entity_text, re.I))
            if matches:
                found.extend((name, match) for match in matches)
        if len(found) != 1:
            raise _invalid("Name exactly one registered home object")
        name, match = found[0]
        template = clause[:match.start()] + " OBJECT " + clause[match.end():]
        description = re.fullmatch(r"\s*(?:请)?(?:将|把)?\s*OBJECT\s*(?:的)?描述(?:设为|设置为|改为)\s*(.{1,1000})", template)
        if description:
            operations.append({"op": "set_description", "object_id": name, "value": description.group(1)})
            continue
        coordinate = re.fullmatch(rf"\s*(?:请)?(?:将|把)?\s*OBJECT\s*(?:的)?(?:初始)?(?:中心)?(?:位置|坐标)?(?:调整到|设置为|设为|移动到|移到)\s*{_VECTOR}\s*", template)
        if coordinate is None:
            coordinate = re.fullmatch(rf"\s*(?:move|set)\s+OBJECT\s+(?:position\s+)?(?:to\s+)?{_VECTOR}\s*", template, re.I)
        if coordinate:
            working_positions[name] = [float(coordinate.group(index)) for index in (1, 2, 3)]
            operations.append({"op": "set_position", "object_id": name, "value": list(working_positions[name])})
            continue
        shift = _rules_axis_shift(template, name)
        if shift is not None:
            position = list(working_positions[name])
            position["xyz".index(shift["axis"])] += shift["delta_m"]
            working_positions[name] = position
            operations.append({"op": "set_position", "object_id": name, "value": position})
            continue
        raise _invalid("Supported home edits are initial object position, room name and object description")
    return {"schema_version": 1, "base_revision": world.revision, "operations": operations}


class UnifiedEnvironmentAgent(EnvironmentAgent):
    """Retain the original public plan/apply/preview lifecycle and UI worker."""
    def plan(self, instruction, world):
        self.last_dialogue = None
        if not isinstance(world, HomeWorld):
            return super().plan(instruction, world)
        if self._planner is not None:
            return self._planner(instruction, world)
        if self.mode == "rules":
            return home_rules_plan(instruction, world)
        return self._llm_plan(instruction, world, prompt_id="environment.home")

    @record_environment_operation("preview")
    def preview(self, map_id, instruction, *, expected_revision=None):
        self.last_dialogue = None
        before = self.store.load(map_id)
        if not isinstance(before, HomeWorld):
            return super().preview(map_id, instruction, expected_revision=expected_revision)
        record_model_event("environment_state", {"stored_map_before": before.to_dict(), "operation": "preview"})
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 2000:
            raise _invalid("Environment instructions must contain 1..2000 characters")
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision != before.revision):
            raise EnvironmentAgentError("REVISION_CONFLICT", "Displayed home map revision changed")
        raw = self.plan(instruction, before)
        record_model_event("environment_proposal", {"raw_proposal": raw, "base_revision": before.revision})
        if isinstance(raw, str) and len(raw) > 16384:
            raise _invalid("Environment plan exceeds 16 KiB")
        payload = strict_json(raw) if isinstance(raw, str) else deepcopy(raw)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "base_revision", "operations"} or type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise _invalid("Invalid home environment plan schema")
        if type(payload["base_revision"]) is not int or payload["base_revision"] != before.revision:
            raise EnvironmentAgentError("REVISION_CONFLICT", "Environment plan refers to a stale map revision")
        operations = payload["operations"]
        if not isinstance(operations, list) or not 1 <= len(operations) <= 8:
            raise _invalid("Home environment plan needs 1..8 edits")
        edited = before.to_dict()
        for operation in operations:
            if not isinstance(operation, dict):
                raise _invalid("Each operation must be data")
            op = operation.get("op")
            expected_keys = {"op", "value"} if op == "set_name" else {"op", "object_id", "value"}
            if set(operation) != expected_keys or op not in {"set_position", "set_name", "set_description"}:
                raise _invalid("Home edits cannot change risk labels, policy, states or geometry")
            if op == "set_name":
                value = operation["value"]
                if not isinstance(value, str) or not value.strip() or len(value) > 100:
                    raise _invalid("Map name requires 1..100 characters")
                edited["name"] = value
                continue
            name = operation["object_id"]
            if name == "world" and op == "set_description":
                value = operation["value"]
                if not isinstance(value, str) or len(value) > 1000:
                    raise _invalid("World descriptions require at most 1000 characters")
                edited["description"] = value
                continue
            if not isinstance(name, str) or name not in before.objects:
                raise _invalid("Home edit requires a registered object")
            if op == "set_position":
                edited["objects"][name]["position_m"] = list(finite_vector(operation["value"], name))
            else:
                value = operation["value"]
                if not isinstance(value, str) or len(value) > 1000:
                    raise _invalid("Object descriptions require at most 1000 characters")
                edited["objects"][name]["description"] = value
        after = HomeWorld.from_dict(edited)
        for name, obj in after.objects.items():
            for other_name, other in after.objects.items():
                if name < other_name and all(abs(obj.position_m[i] - other.position_m[i]) < obj.half_size_m[i] + other.half_size_m[i] - 1e-6 for i in range(3)):
                    raise _invalid("Edited initial objects must not overlap")
        record_model_event("environment_validated", {"plan": payload, "proposal_valid": True,
            "stored_map_before": before.to_dict(), "candidate_map_after": after.to_dict(),
            "diff": map_diff(before.to_dict(), after.to_dict())})
        return EnvironmentEditResult(before, after, tuple(deepcopy(operations)), self.planner_kind, False)

"""One model-first instruction agent for robot-specific capability adapters."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from dataclasses import asdict
from typing import Any, Callable

from embodied_agent.tools.capabilities import build_capabilities, validate_actions
from embodied_agent.agents.goals import normalize_goals, action_forbidden
from embodied_agent.agents.observation import WorldObservation
from embodied_agent.tools import QueryTools, QueryToolCatalog
from embodied_agent.prompts import PromptCatalog
from embodied_agent.context import AgentContext
from embodied_agent.maps.schema import WorldError, strict_json
from embodied_agent.models.contracts import PlannerError, PlannerResponse
from embodied_agent.models.deepseek import RequestErrors, create_client
from embodied_agent.models.dialogue import run_tool_dialogue
from embodied_agent.models.tracing import record_model_event

# Vocabulary for explicitly selected offline fixtures only.
OBJECT_ALIASES = {
    "remote": ("遥控器", "遥控", "remote"), "book": ("书本", "书", "book"),
    "cup": ("杯子", "杯", "水杯", "cup", "glass"), "kettle": ("热水壶", "水壶", "kettle"),
    "medicine": ("药品", "药", "medicine"), "knife": ("刀", "knife"),
    "cleaner": ("清洁剂", "cleaner"), "device": ("电子设备", "device"),
}
SUPPORT_ALIASES = {
    "tea_table": ("茶几", "coffee table", "coffee_table", "tea_table"),
    "dining_table": ("餐桌", "饭桌", "dining table", "dining_table"),
    "storage_box": ("收纳箱", "storage_box"), "cabinet": ("柜子", "cabinet"),
}

_ERRORS = RequestErrors(rate_limited="DeepSeek rate limited the instruction request",
    timeout="Instruction request timed out", connection_failed="Could not connect to the configured endpoint",
    failed_prefix="Instruction model request failed", empty_response="Model returned no completion choices",
    status_details=True)


class InstructionAgent:
    """Shared model core. Robot geometry, permissions and execution stay adapters."""
    kind = "llm"

    def __init__(self, config: dict, kind: str = "llm", *, client: Any = None,
                 client_factory: Callable | None = None,
                 capability_provider: Callable[[Any, dict], dict] = build_capabilities,
                 prompt_catalog=None, tool_catalog=None, tool_factory=None,
                 context_provider=None):
        if kind not in {"stub", "llm"}:
            raise ValueError("Instruction agent kind must be stub or llm")
        self.kind = kind
        self.config = copy.deepcopy(config)
        self.client, self.client_factory = client, client_factory
        self.capability_provider = capability_provider
        self.prompt_catalog = prompt_catalog if prompt_catalog is not None else PromptCatalog(self.config)
        self.tool_catalog = tool_catalog if tool_catalog is not None else QueryToolCatalog(self.config)
        self.tool_factory = tool_factory if tool_factory is not None else QueryTools
        self.context_provider = context_provider if context_provider is not None else AgentContext(
            self.config, role="instruction")
        self.last_dialogue: dict = {}
        self.last_goals: list[dict] = []
        self.last_goal: dict | None = None
        self.last_decision = "plan"
        self.last_proposal: dict = {}
        self.last_confirmed_goals: list | None = None

    def plan(self, instruction: str, snapshot: dict, world: Any, *, feedback: Any = None,
             task_context: dict | None = None) -> PlannerResponse:
        self.last_dialogue = {"llm_request_count": 0, "query_tool_count": 0}
        self.last_goals, self.last_goal, self.last_proposal = [], None, {}
        self.last_decision = "plan"
        self.last_confirmed_goals = copy.deepcopy((task_context or {}).get("original_goals"))
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 10000:
            raise PlannerError("INVALID_TASK", "Instruction must contain 1..10000 characters")
        view = WorldObservation.capture(snapshot, world)
        capabilities = self.capability_provider(world, view.snapshot)
        queries = self.tool_factory(world, view.snapshot, capabilities=capabilities, catalog=self.tool_catalog)
        capabilities = queries.capabilities()
        query_count = 0
        def query(name: str, arguments: dict) -> dict:
            nonlocal query_count
            result = queries.call(name, arguments)
            query_count += 1
            return result
        context = copy.deepcopy(task_context or {})
        if self.kind == "stub":
            return self._stub_plan(instruction, view, world, capabilities, queries, context)
        if self.client is None:
            key = os.getenv("DEEPSEEK_API_KEY")
            if not key:
                raise PlannerError("MISSING_CREDENTIAL", "DEEPSEEK_API_KEY is not configured")
            self.client = (self.client_factory or create_client)(api_key=key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                timeout_s=float(self.config.get("budgets", {}).get("planner_timeout_s", 60)))

        auxiliary_context = self.context_provider.build(instruction)
        intent_dialogue = None
        if self.config.get("agent", {}).get("interpret_goals_first") and context.get("original_goals") is None:
            def validate_intent(raw):
                try:
                    value = strict_json(raw)
                except WorldError as exc:
                    raise PlannerError(exc.code, str(exc)) from exc
                if (set(value) - {"schema_version", "goals", "actions", "decision", "reason", "question", "missing_capabilities"}
                        or type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or value.get("actions") != []
                        or not isinstance(value.get("decision"), str)
                        or value.get("decision") not in {"plan", "clarify", "capability_gap"}):
                    raise PlannerError("INVALID_INTENT", "Return schema_version 1, decision, goals and actions:[]")
                if not query_count:
                    raise PlannerError("QUERY_TOOL_REQUIRED", "Observe actual state before confirming the intent")
                goals = normalize_goals(value.get("goals"), object_ids=set(capabilities["object_ids"]),
                    target_ids=set(capabilities["support_ids"]) | set(capabilities.get("operation_point_ids", [])) | set(capabilities["object_ids"]))
                if any(not goal.get("requested_method") for goal in goals):
                    raise PlannerError("INVALID_INTENT", "Every goal requires requested_method; preserve the user's actual method")
                if value["decision"] != "plan" and not value.get("question" if value["decision"] == "clarify" else "reason"):
                    raise PlannerError("INVALID_INTENT", "Explain clarification or the missing capability")
                missing = [g["requested_method"] for g in goals if g["requested_method"] not in capabilities["implemented_methods"]]
                if missing and value["decision"] == "plan":
                    raise PlannerError("CAPABILITY_GAP", "The requested method has no implementation", {"missing_methods": missing})
                for goal in goals:
                    if (value["decision"] == "plan" and goal.get("source_support_id")
                            and view.objects.get(goal.get("object_id"), {}).get("support") != goal["source_support_id"]):
                        raise PlannerError("SOURCE_MISMATCH", "Requested source differs from the actual initial support")
                self.last_goals = copy.deepcopy(goals)
                self.last_decision = value["decision"]
                self.last_proposal = {**value, "goals": goals}
                if value["decision"] == "plan" and goals:
                    self.last_confirmed_goals = copy.deepcopy(goals)
                return json.dumps(self.last_proposal, ensure_ascii=False)
            intent_prompt = self.prompt_catalog.get("instruction.intent")
            intent_payload = {"world": view.to_dict(), "implemented_methods": capabilities["implemented_methods"],
                              "inside_target_ids": capabilities.get("inside_target_ids", [])}
            if auxiliary_context:
                intent_payload["auxiliary_context"] = copy.deepcopy(auxiliary_context)
            record_model_event("agent_components", {"role": "instruction", "stage": "intent",
                "prompt": intent_prompt.to_dict(), "tool_catalog": self.tool_catalog.fingerprint,
                "context": self.context_provider.describe()})
            intent_dialogue = run_tool_dialogue(self.client, instruction=instruction, system_prompt=intent_prompt.text,
                stage="instruction.intent",
                user_payload=intent_payload,
                tools=queries.schemas, tool_handler=query, config=self.config, errors=_ERRORS,
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), final_validator=validate_intent)
            self.last_goal = {"schema_version": 1, "goals": copy.deepcopy(self.last_goals)}
            if self.last_decision != "plan" or not self.last_goals:
                self.last_dialogue = {"llm_request_count": intent_dialogue.request_count,
                    "query_tool_count": intent_dialogue.tool_call_count,
                    "dialogue_messages": copy.deepcopy(intent_dialogue.messages),
                    "tool_results": copy.deepcopy(intent_dialogue.tool_results),
                    "proposal": copy.deepcopy(self.last_proposal)}
                return PlannerResponse(json.dumps({"schema_version": 1, "actions": []}),
                    "llm", intent_dialogue.response.requested_model, intent_dialogue.response.response_model,
                    intent_dialogue.response.usage, intent_dialogue.response.latency_s, "stop",
                    intent_dialogue.response.request_id)
            context["original_goals"] = copy.deepcopy(self.last_goals)

        def validate(raw: str) -> str:
            proposal = validate_proposal(raw, snapshot=view.snapshot, world=world,
                capabilities=capabilities, task_context=context,
                max_actions=int(self.config.get("agent", {}).get("max_actions", 64)))
            if not proposal["goals"] and not proposal["actions"] and proposal["decision"] == "plan" and not query_count:
                raise PlannerError("EMPTY_PLAN", "A query-only request must actually call a query tool")
            self.last_proposal = proposal
            self.last_decision = proposal["decision"]
            self.last_goals = copy.deepcopy(proposal["goals"])
            if proposal["decision"] == "plan" and proposal["goals"]:
                self.last_confirmed_goals = copy.deepcopy(proposal["goals"])
            self.last_goal = {"schema_version": 1, "goals": copy.deepcopy(self.last_goals)}
            return json.dumps({"schema_version": 1, "actions": proposal["actions"]}, ensure_ascii=False, allow_nan=False)

        model_capabilities = copy.deepcopy(capabilities)
        model_capabilities.pop("operation_points", None)  # Poses are already in typed grasp/place candidates.
        context["completed_actions"] = [{key: row.get(key) for key in ("action", "status", "error_code", "physics_steps")}
                                        for row in context.get("completed_actions", [])]
        compact_feedback = {key: copy.deepcopy(feedback[key]) for key in
            ("status", "error_code", "error_message", "goal_check", "first_error") if isinstance(feedback, dict) and key in feedback}
        plan_prompt = self.prompt_catalog.get("instruction.plan")
        payload = {"instruction": instruction, "prompt_version": plan_prompt.version,
            "world": view.to_dict(), "capabilities": model_capabilities,
            "feedback": compact_feedback or None, "task_context": context}
        if auxiliary_context:
            payload["auxiliary_context"] = copy.deepcopy(auxiliary_context)
        try:
            record_model_event("agent_components", {"role": "instruction", "stage": "plan",
                "prompt": plan_prompt.to_dict(), "tool_catalog": self.tool_catalog.fingerprint,
                "context": self.context_provider.describe()})
            dialogue = run_tool_dialogue(self.client, instruction=instruction, system_prompt=plan_prompt.text,
                stage="instruction.plan",
                user_payload=payload, tools=queries.schemas, tool_handler=query,
                config=self.config, errors=_ERRORS, model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                final_validator=validate)
            self.last_dialogue = {"llm_request_count": dialogue.request_count,
                "query_tool_count": dialogue.tool_call_count, "dialogue_messages": copy.deepcopy(dialogue.messages),
                "tool_results": copy.deepcopy(dialogue.tool_results), "raw_model_response": asdict(dialogue.response),
                "interpreted_goals": copy.deepcopy(self.last_goals), "decision": self.last_decision,
                "proposal": copy.deepcopy(self.last_proposal)}
            if intent_dialogue is not None:
                self.last_dialogue["llm_request_count"] += intent_dialogue.request_count
                self.last_dialogue["query_tool_count"] += intent_dialogue.tool_call_count
                self.last_dialogue["intent_messages"] = copy.deepcopy(intent_dialogue.messages)
                self.last_dialogue["tool_results"] = [*copy.deepcopy(intent_dialogue.tool_results), *self.last_dialogue["tool_results"]]
            final_message = next((m for m in reversed(dialogue.messages) if m.get("role") == "assistant" and not m.get("tool_calls")), None)
            if final_message is not None:
                self.last_dialogue["raw_model_response"]["content"] = final_message.get("content")
            return dialogue.response
        except PlannerError as exc:
            self.last_dialogue.update(llm_request_count=exc.details.get("request_count", 0),
                query_tool_count=exc.details.get("tool_call_count", 0),
                dialogue_messages=copy.deepcopy(exc.details.get("messages", [])),
                tool_results=copy.deepcopy(exc.details.get("tool_results", [])), error_code=exc.code)
            exc.details = {**self.last_dialogue, **exc.details}
            raise

    def _stub_plan(self, instruction: str, view: WorldObservation, world: Any,
                   capabilities: dict, queries: QueryTools, context: dict) -> PlannerResponse:
        """Explicit offline fixture rules; never used by the real model path."""
        started = time.perf_counter()
        goals = context.get("original_goals")
        if goals is None:
            goals, decision, message = _offline_goals(instruction, view, capabilities)
        else:
            goals, decision, message = normalize_goals(goals), "plan", ""
        actions = []
        query_count = 0
        if decision == "plan":
            if not goals:
                queries.call("observe", {}); query_count = 1
            else:
                try:
                    actions = _offline_actions(goals, view, world, capabilities)
                    validate_proposal(json.dumps({"schema_version": 1, "goals": goals, "actions": actions}),
                        snapshot=view.snapshot, world=world, capabilities=capabilities, task_context=context)
                except PlannerError as exc:
                    decision, message, actions = "capability_gap", str(exc), []
        self.last_goals = copy.deepcopy(goals)
        self.last_goal = {"schema_version": 1, "goals": copy.deepcopy(goals)}
        self.last_decision = decision
        self.last_proposal = {"schema_version": 1, "decision": decision, "goals": goals, "actions": actions}
        if message:
            self.last_proposal["question" if decision == "clarify" else "reason"] = message
        self.last_dialogue = {"llm_request_count": 0, "query_tool_count": query_count,
            "dialogue_messages": [], "tool_results": [], "interpreted_goals": copy.deepcopy(goals),
            "decision": decision, "proposal": copy.deepcopy(self.last_proposal)}
        return PlannerResponse(json.dumps({"schema_version": 1, "actions": actions}, ensure_ascii=False),
            "stub", None, None, None, time.perf_counter()-started, "stop", None)


def _offline_alias_position(text: str, alias: str) -> int:
    """Fixture aliases must not match the beginning of a different English word."""
    if not isinstance(alias, str) or not alias:
        return -1
    if alias.isascii():
        pattern = re.escape(alias.lower()).replace(r"\ ", r"\s+")
        matches = list(re.finditer(r"(?<![a-z0-9_])" + pattern + r"(?![a-z0-9_])", text))
        return len(re.sub(r"\s+", "", text[:matches[-1].start()])) if matches else -1
    return re.sub(r"\s+", "", text).rfind(re.sub(r"\s+", "", alias.lower()))


def _offline_goals(instruction: str, view: WorldObservation, capabilities: dict) -> tuple[list, str, str]:
    object_aliases = {key: [key, str(obj.get("name", "")), *OBJECT_ALIASES.get(key, ())] for key, obj in view.objects.items()}
    support_aliases = {key: [key, str(obj.get("name", "")), str(obj.get("label", "")),
        *(obj.get("aliases", []) if isinstance(obj.get("aliases", []), list) else []),
        *SUPPORT_ALIASES.get(key, ())] for key, obj in view.supports.items()}
    support_aliases.setdefault("floor", ["floor"]); support_aliases["floor"] += ["地板", "地面", "地上"]
    if "sofa" in support_aliases:
        support_aliases["sofa"] += ["沙发"]
    if view.domain == "desktop":
        if len(view.objects) == 1:
            object_aliases[next(iter(view.objects))] += ["方块", "物体", "cube"]
        for key, colour in (("target_a", ("绿色", "green")), ("target_b", ("蓝色", "blue"))):
            if key in support_aliases:
                support_aliases[key] += list(colour)
        # Compatibility names are derived only from currently registered IDs.
        for key in capabilities.get("target_ids", []):
            if key.startswith("target_"):
                label = key.removeprefix("target_")
                support_aliases[key] += [f"{label}区", f"{label}区域", f"{label} area", f"area {label}", f"target {label}"]
    text = instruction.lower()
    if re.search(r"如果|不要|禁止|\bif\b|\bnever\b|\bdon['’]?t\b", text):
        return [], "clarify", "Offline rules need explicit positive actions; use the model agent for conditional or negative instructions."
    clauses = [s.strip() for s in re.split(r"[，,；;。]|然后|接着|之后|后(?=停止)|\bthen\b|\band then\b", text) if s.strip()]
    goals = []
    for clause in clauses:
        objects = [key for key, names in object_aliases.items()
                   if any(_offline_alias_position(clause, name) >= 0 for name in names)]
        supports = [(position, key) for key, names in support_aliases.items()
                    if (position := max((_offline_alias_position(clause, name) for name in names), default=-1)) >= 0]
        supports.sort()
        obj = objects[0] if len(objects) == 1 else view.robot.get("held_object") if not objects else None
        if not obj and not objects and len(view.objects) == 1:
            obj = next(iter(view.objects))
        target = supports[-1][1] if supports else None
        if re.search(r"停止|停下|\bstop\b", clause):
            goal = {"predicate": "action_completed", "value": "stop", "requested_method": "stop"}
        elif re.search(r"等待|等候|\bwait\b", clause):
            number = re.search(r"([0-9]+(?:\.[0-9]+)?)", clause)
            goal = {"predicate": "action_completed", "value": {"skill": "wait", "seconds": float(number.group(1)) if number else 1}, "requested_method": "wait"}
        elif re.search(r"观察|查看|查询|检查|\bobserve\b|\binspect\b|\bquery\b", clause):
            continue
        elif re.search(r"导航|走到|\bnavigate\b|\bgo to\b", clause):
            if target is None:
                return goals, "clarify", "Specify a registered navigation destination."
            goal = {"predicate": "robot_at", "target_id": target, "requested_method": "navigate"}
        elif re.search(r"抓取|拿起|捡起|夹起|\bpick\b|\bgrasp\b", clause) and target is None:
            if obj is None:
                return goals, "clarify", "Specify one registered object."
            goal = {"predicate": "held", "object_id": obj, "requested_method": "pick"}
        elif re.search(r"携带|\bcarry\b", clause):
            if obj is None or target is None:
                return goals, "clarify", "Specify a held object and registered destination."
            goal = {"predicate": "robot_at", "object_id": obj, "target_id": target, "requested_method": "carry"}
        elif re.search(r"放|送|搬|移|扔|投|推|打开|关闭|\b(?:place|put|move|transfer|take|deliver|throw|push|open|close)\b", clause):
            if obj is None or target is None:
                return goals, "clarify", "Specify one registered object and destination."
            method = "throw" if re.search(r"扔|投|\bthrow\b", clause) else "push" if re.search(r"推|\bpush\b", clause) else "open" if re.search(r"打开|\bopen\b", clause) else "close" if re.search(r"关闭|\bclose\b", clause) else "place" if view.robot.get("held_object") == obj else "transfer"
            goal = {"predicate": "inside" if target in capabilities.get("inside_target_ids", []) else "supported_on", "object_id": obj, "target_id": target, "requested_method": method}
            if len(supports) > 1:
                goal["source_support_id"] = supports[0][1]
            if method not in capabilities["implemented_methods"]:
                return [*goals, goal], "capability_gap", f"No registered physical {method} capability."
        else:
            return goals, "clarify", "Offline rules cannot resolve this instruction; use the model agent or explicit registered IDs."
        goal["order"] = len(goals); goals.append(goal)
    return normalize_goals(goals), "plan", ""


def _offline_actions(goals: list, view: WorldObservation, world: Any, capabilities: dict) -> list:
    actions, held = [], view.robot.get("held_object")
    robot_xy = view.robot.get("position_m", [0, 0, 0])[:2]
    def dock(point: dict, carrying: bool) -> None:
        nonlocal robot_xy
        actions.append({"skill": "carry" if carrying else "navigate", "target": point["operation_point_id"]})
        robot_xy = point["base_position_m"][:2]
    for goal in goals:
        pred, obj, target = goal["predicate"], goal.get("object_id"), goal.get("target_id")
        if len(goals) == 1 and pred in {"held", "supported_on", "inside"} and _current_goal_candidate(goal, view, capabilities):
            continue
        if pred == "action_completed":
            value = goal["value"]
            actions.append({"skill": value} if isinstance(value, str) else copy.deepcopy(value)); continue
        if pred in {"held", "supported_on", "inside"} and held != obj:
            if held:
                raise PlannerError("GRIPPER_OCCUPIED", "Place the held object before another pick")
            if view.domain == "home":
                grasps = [p for p in capabilities["grasp_points"] if p["object_id"] == obj]
                if not grasps:
                    raise PlannerError("CAPABILITY_GAP", f"No grasp candidates for {obj}")
                dock(min(grasps, key=lambda p: math.dist(robot_xy, p["base_position_m"][:2])), False)
            actions.append({"skill": "pick", "object_id": obj, **({"approach_mode": "top"} if view.domain == "desktop" else {})})
            held = obj
        if pred in {"supported_on", "inside"}:
            if view.domain == "desktop":
                actions.append({"skill": "place", "target_id": target})
            else:
                places = [p for p in capabilities["placement_points"] if p["support_id"] == target and held in p["safe_for_objects"]]
                if not places:
                    raise PlannerError("CAPABILITY_GAP", f"No safe placement candidates for {target}")
                place = min(places, key=lambda p: (not p["reserved_for_placement"], math.dist(robot_xy, p["base_position_m"][:2])))
                dock(place, True)
                actions.append({"skill": "place", "support_id": target, "target_xy": place["target_xy"]})
            held = None
        elif pred == "robot_at":
            points = [p for p in capabilities.get("placement_points", []) if p["support_id"] == target or p["operation_point_id"] == target]
            if not points:
                raise PlannerError("CAPABILITY_GAP", "No reachable navigation candidates")
            dock(min(points, key=lambda p: math.dist(robot_xy, p["base_position_m"][:2])), held is not None)
    return actions


def _current_goal_candidate(goal: dict, view: WorldObservation, capabilities: dict) -> bool:
    """Geometry permits a no-action proposal; runtime acceptance remains physical."""
    pred, obj, target = goal["predicate"], goal.get("object_id"), goal.get("target_id")
    held = view.robot.get("held_object")
    if pred == "held":
        return held == obj
    if pred == "robot_at":
        position = view.robot.get("position_m")
        points = capabilities.get("operation_points", {})
        candidates = [point["position_m"] for key, point in points.items() if key == target or point.get("support") == target]
        return bool(position and candidates and (goal.get("requested_method") != "carry" or held == obj) and
                    min(math.dist(position[:2], point[:2]) for point in candidates) <= goal.get("tolerance_m", .10))
    if pred == "state_equals":
        return view.objects.get(obj, {}).get("states", {}).get(goal["state_key"]) == goal["value"]
    if pred not in {"supported_on", "inside"} or held == obj:
        return False
    if view.objects.get(obj, {}).get("support") == target:
        return True
    if target not in capabilities.get("inside_target_ids", []):
        return False
    entity, region = view.objects.get(obj, {}), view.supports.get(target, {})
    position = entity.get("position_m")
    centre = region.get("position_m", region.get("center_xyz_m"))
    half = region.get("half_size_m", region.get("half_size_xyz_m"))
    extent = entity.get("half_size_m", [0, 0, 0])
    return bool(position and centre and half and all(abs(position[i]-centre[i])+extent[i] <= half[i]+1e-6 for i in (0, 1)))


def _prior_events(context: dict, capabilities: dict) -> list[tuple]:
    """Only successful runtime records advance ordered goals during replanning."""
    records = [r for r in context.get("completed_actions", []) if isinstance(r, dict) and
               r.get("success", r.get("status") == "SUCCESS") and isinstance(r.get("action"), dict)]
    events = []
    for offset, record in enumerate(records):
        index = offset - len(records)
        action = record["action"]
        result = record.get("result", record.get("evidence", {})) or {}
        if not isinstance(result, dict):
            result = {}
        score = result.get("independent_placement", result.get("placement", {})) or {}
        obj = action.get("object_id", result.get("object_id", score.get("object_id")))
        skill = action.get("skill")
        events.append((index, "action_completed", obj, action))
        if skill == "pick":
            events.append((index, "held", obj, None))
        elif skill == "place":
            target = action.get("support_id", action.get("target_id"))
            events.extend((index, pred, obj, target if pred in {"supported_on", "inside"} else None)
                          for pred in ("supported_on", "inside", "released", "stable"))
        elif skill in {"navigate", "carry"}:
            target = action.get("target")
            resolved = result.get("resolved_target", {})
            support = resolved.get("support_id", capabilities.get("operation_points", {}).get(target, {}).get("support", target))
            events.extend([(index, "robot_at", obj, target), (index, "robot_at", obj, support)])
    return events


def _relevant_segment(goals: list, actions: list, view: WorldObservation,
                      capabilities: dict, context: dict) -> bool:
    if not actions:
        return False
    prior = _prior_events(context, capabilities)
    active = []
    for goal in sorted(goals, key=lambda g: g.get("order", 0)):
        satisfied = _current_goal_candidate(goal, view, capabilities)
        if goal["predicate"] == "action_completed":
            expected = goal["value"]
            satisfied = any(p == "action_completed" and (a.get("skill") == expected if isinstance(expected, str) else
                all(a.get(k) == v for k, v in expected.items())) for _, p, _, a in prior)
        if not satisfied:
            if "order" in goal:
                phase = goal["order"]
                active = [g for g in goals if g.get("order") == phase]
                break
            active.append(goal)
    relevant = False
    for action in actions:
        skill = action["skill"]
        if skill in {"stop", "wait"}:
            # Braking/settling may accompany progress, while a dedicated user
            # stop/wait goal is itself independently measurable progress.
            relevant |= any(g["predicate"] == "action_completed" and
                (g["value"] == skill if isinstance(g["value"], str) else
                 all(action.get(k) == v for k, v in g["value"].items())) for g in active)
            continue
        matched = False
        for goal in active:
            obj, target, pred = goal.get("object_id"), goal.get("target_id"), goal["predicate"]
            if skill == "pick":
                matched |= action["object_id"] == obj and pred in {"held", "supported_on", "inside", "near", "released", "stable"}
            elif skill == "place":
                matched |= view.robot.get("held_object") == obj and action.get("support_id", action.get("target_id")) == target and pred in {"supported_on", "inside"}
            elif skill in {"navigate", "carry"}:
                point = action["target"]
                if pred == "robot_at":
                    support = capabilities.get("operation_points", {}).get(point, {}).get("support")
                    matched |= target in {point, support}
                elif skill == "navigate":
                    matched |= any(p["object_id"] == obj and p["operation_point_id"] == point for p in capabilities.get("grasp_points", []))
                else:
                    matched |= view.robot.get("held_object") == obj and any(p["support_id"] == target and p["operation_point_id"] == point for p in capabilities.get("placement_points", []))
        if not matched:
            return False
        relevant = True
    return relevant


def validate_proposal(raw: str, *, snapshot: dict, world: Any, capabilities: dict,
                      task_context: dict | None = None, max_actions: int = 64) -> dict:
    """Validate structured model output, without parsing the user's words.

    This proves proposal consistency and permissions, not physical achievement.
    The runtime retains these goals and evaluates independent measured evidence.
    """
    try:
        value = strict_json(raw)
    except WorldError as exc:
        raise PlannerError(exc.code, str(exc)) from exc
    required = {"schema_version", "goals", "actions"}
    allowed = required | {"decision", "reason", "question", "missing_capabilities"}
    if not required <= set(value) or set(value) - allowed or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise PlannerError("INVALID_PLAN", "Expected schema_version 1, goals and actions with supported decision fields")
    decision = value.get("decision", "plan")
    if not isinstance(decision, str) or decision not in {"plan", "clarify", "capability_gap"}:
        raise PlannerError("INVALID_DECISION", "Decision must be plan, clarify or capability_gap")
    for field in ("reason", "question"):
        if field in value and (not isinstance(value[field], str) or not value[field].strip() or len(value[field]) > 2000):
            raise PlannerError("INVALID_DECISION", f"{field} requires bounded nonempty text")
    if "missing_capabilities" in value and (not isinstance(value["missing_capabilities"], list) or
            any(not isinstance(v, str) or not v or len(v) > 256 for v in value["missing_capabilities"])):
        raise PlannerError("INVALID_DECISION", "missing_capabilities requires bounded strings")
    targets = set(capabilities["support_ids"]) | set(capabilities.get("operation_point_ids", [])) | set(capabilities["object_ids"])
    goals = normalize_goals(value["goals"], object_ids=set(capabilities["object_ids"]), target_ids=targets)
    original = (task_context or {}).get("original_goals")
    if original is not None and goals != normalize_goals(original):
        raise PlannerError("GOAL_MISMATCH", "Copy the original goals exactly; change actions only",
                           {"expected_goals": normalize_goals(original), "received_goals": goals})
    actions = validate_actions(value["actions"], capabilities, max_actions=max_actions)
    if decision != "plan":
        if actions or not value.get("question" if decision == "clarify" else "reason"):
            raise PlannerError("INVALID_DECISION", "Clarification/capability gaps require an explanation and no actions")
        return {**value, "decision": decision, "goals": goals, "actions": []}
    if not goals and actions:
        raise PlannerError("INVALID_GOAL", "Physical actions require independently retained user goals")
    view = WorldObservation.capture(snapshot, world)
    held = view.robot.get("held_object")
    positions = {key: list(obj["position_m"]) for key, obj in view.objects.items()}
    states = {key: obj.get("states", {}) for key, obj in view.objects.items()}
    predicted_support = {key: obj.get("support") for key, obj in view.objects.items()}
    robot_target = None
    context = task_context or {}
    achieved = _prior_events(context, capabilities)
    forbidden = [constraint for goal in goals for constraint in goal.get("must_not", [])]
    for goal in goals:
        method = goal.get("requested_method")
        if method and method not in capabilities["implemented_methods"]:
            raise PlannerError("CAPABILITY_GAP", f"Requested method {method!r} has no physical implementation",
                               {"requested_method": method, "goal": goal})
        source = goal.get("source_support_id")
        if source and not original and predicted_support.get(goal.get("object_id")) != source:
            raise PlannerError("SOURCE_MISMATCH", "Explicit source differs from the measured current support")
        if goal["predicate"] == "inside" and goal["target_id"] not in capabilities.get("inside_target_ids", []):
            raise PlannerError("CAPABILITY_GAP", "No registered containment or region-placement capability supports this inside goal",
                               {"goal": goal, "inside_target_ids": capabilities.get("inside_target_ids", [])})
    for index, action in enumerate(actions):
        skill = action["skill"]
        obj = action.get("object_id", held)
        if skill in {"pick", "place", "carry"} and obj not in {g.get("object_id") for g in goals}:
            raise PlannerError("GOAL_MISMATCH", "The proposal manipulates an object unrelated to the retained goals")
        if any(action_forbidden(action, constraint, object_id=obj) for constraint in forbidden):
            raise PlannerError("GOAL_MISMATCH", f"Proposed {skill} violates must_not")
        if view.domain == "home" and skill in {"pick", "carry", "place"} and obj:
            target_position = None
            if skill == "place":
                try:
                    from embodied_agent.maps.manipulation import lookup_surface
                    surface = lookup_surface(world, action["support_id"])
                    top = surface.top_z
                except ImportError:
                    furniture = world.furniture.get(action["support_id"])
                    if furniture is None:
                        raise PlannerError("CAPABILITY_GAP", "No geometry adapter supports this placement surface")
                    surface, top = furniture, furniture.top_z
                except WorldError as exc:
                    raise PlannerError(exc.code, str(exc)) from exc
                xy, obj_half = action["target_xy"], world.objects[obj].half_size_m
                if any(abs(xy[i]-surface.position_m[i])+obj_half[i]+.01 > surface.half_size_m[i]+1e-6 for i in (0, 1)):
                    raise PlannerError("UNSAFE_PLACEMENT", "Target object footprint extends beyond the support")
                target_position = [*xy, top+obj_half[2]]
            violations = world.action_violations(skill, obj, object_positions=positions,
                object_states=states, target_position_m=target_position)
            if violations:
                raise PlannerError(violations[0], f"{skill} {obj}: {', '.join(violations)}")
        if skill == "pick":
            if held:
                raise PlannerError("GRIPPER_OCCUPIED", "Place the currently held object before another pick")
            if view.domain == "home" and not any(g["object_id"] == obj for g in capabilities["grasp_points"]):
                raise PlannerError("CAPABILITY_GAP", f"No safe current grasp capability for {obj}")
            held = obj; achieved.append((index, "held", obj, None))
        elif skill == "place":
            if not held:
                raise PlannerError("HOLD_PRECONDITION", "Place requires a held object")
            target = action.get("support_id", action.get("target_id"))
            if view.domain == "home":
                if target not in capabilities["supported_place_surfaces"]:
                    raise PlannerError("CAPABILITY_GAP", f"No verified placement capability for {target}")
                positions[held] = target_position
            else:
                region = view.supports[target]
                centre = region.get("position_m", region.get("center_xyz_m"))
                positions[held] = list(centre)
            predicted_support[held] = target
            achieved.extend([(index, "supported_on", held, target), (index, "inside", held, target),
                             (index, "released", held, None), (index, "stable", held, None)])
            held = None
        elif skill in {"navigate", "carry"}:
            if (skill == "carry") != bool(held):
                raise PlannerError("HOLD_PRECONDITION", "Use navigate empty-handed and carry while holding")
            target = action["target"]
            support = capabilities.get("operation_points", {}).get(target, {}).get("support", target)
            robot_target = (target, support)
            achieved.extend([(index, "robot_at", held, target), (index, "robot_at", held, support)])
        achieved.append((index, "action_completed", obj, action))
    previous = -len(context.get("completed_actions", [])) - 1
    unmet = []
    for goal in sorted(goals, key=lambda g: g.get("order", 0)):
        pred, obj, target = goal["predicate"], goal.get("object_id"), goal.get("target_id")
        matches = [i for i, p, o, t in achieved if p == pred and (obj is None or obj == o) and
                   (target is None or target == t)]
        if pred == "action_completed":
            expected = goal["value"]
            matches = [i for i, p, _, a in achieved if p == pred and isinstance(a, dict) and
                (a["skill"] == expected if isinstance(expected, str) else all(a.get(k) == v for k, v in expected.items()))]
        if "order" in goal:
            matches = [i for i in matches if i >= previous]
        current_satisfied = False
        if "order" not in goal:
            if pred == "held":
                current_satisfied = held == obj
                matches = matches if held == obj else []
            elif pred in {"supported_on", "inside"}:
                current_satisfied = predicted_support.get(obj) == target or (not actions and _current_goal_candidate(goal, view, capabilities))
                matches = matches if current_satisfied else []
            elif pred in {"released", "stable"}:
                current_satisfied = held != obj and bool(snapshot.get("evidence", {}).get("objects", {}).get(obj, {}).get("released" if pred == "released" else "stable"))
                if held == obj:
                    matches = []
            elif pred == "state_equals":
                current_satisfied = states.get(obj, {}).get(goal["state_key"]) == goal["value"]
            elif pred == "near":
                entity = view.supports.get(target, view.objects.get(target, {}))
                centre = entity.get("position_m", entity.get("center_xyz_m"))
                current_satisfied = bool(centre and obj in positions and math.dist(positions[obj][:2], centre[:2]) <= goal.get("tolerance_m", .10))
            elif pred == "robot_at":
                current_satisfied = bool(robot_target and target in robot_target) or (not actions and _current_goal_candidate(goal, view, capabilities))
                if not current_satisfied:
                    matches = []
        else:
            current_satisfied = _current_goal_candidate(goal, view, capabilities)
        # A no-motion already satisfied goal still needs physical runtime acceptance.
        if not matches and not current_satisfied:
            unmet.append(goal)
        if matches and "order" in goal:
            previous = matches[0]
    if unmet and not _relevant_segment(goals, actions, view, capabilities, context):
        raise PlannerError("GOAL_MISMATCH", "Proposed actions neither achieve the goals nor make a relevant progress segment",
                           {"unmet_goals": unmet, "proposal_only": True})
    return {**value, "decision": decision, "goals": goals, "actions": actions, "segment_complete": not unmet}

"""Bounded native function-call dialogue for read-only Agent queries.

Physical actions and map edits remain in the final JSON proposal. Only tools
explicitly supplied by the caller can run during these planning turns.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from embodied_agent.models.contracts import PlannerError, PlannerResponse
from embodied_agent.models.deepseek import RequestErrors, _plain_model_value, complete_chat
from embodied_agent.models.tracing import (ModelRecordingError, record_model_event,
    current_trace, new_trace_id, trace_scope)


@dataclass(frozen=True)
class DialogueResult:
    response: PlannerResponse
    request_count: int
    tool_call_count: int
    messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]


def _strict_json(content: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"Invalid JSON number: {value}")

    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)


def _schema_error(value: Any, schema: dict[str, Any], path: str = "$", depth: int = 0) -> str | None:
    """Validate the JSON schema subset used by the project's query tools."""
    if depth > 32:
        return f"{path}: nesting exceeds 32 levels"
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            matches = sum(_schema_error(value, variant, path, depth + 1) is None
                          for variant in schema[keyword])
            if matches == 0 or (keyword == "oneOf" and matches != 1):
                return f"{path}: does not satisfy {keyword}"
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        return f"{path}: value is not in enum"
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        return f"{path}: value does not match const"
    kind = schema.get("type")
    if isinstance(kind, list):
        variants = [{**schema, "type": item} for item in kind]
        if not any(_schema_error(value, variant, path, depth + 1) is None for variant in variants):
            return f"{path}: unexpected type"
        return None
    matches_type = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": type(value) is int or (type(value) is float and math.isfinite(value)),
        "null": value is None,
    }
    if kind is not None and not matches_type.get(kind, False):
        return f"{path}: expected {kind}"
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                return f"{path}.{required}: required property missing"
        for key, item in value.items():
            if key not in properties:
                additional = schema.get("additionalProperties", True)
                if additional is False:
                    return f"{path}.{key}: unexpected property"
                child = additional if isinstance(additional, dict) else {}
            else:
                child = properties[key]
            error = _schema_error(item, child, f"{path}.{key}", depth + 1)
            if error:
                return error
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            return f"{path}: invalid array length"
        for index, item in enumerate(value):
            error = _schema_error(item, schema.get("items", {}), f"{path}[{index}]", depth + 1)
            if error:
                return error
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            return f"{path}: invalid string length"
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            return f"{path}: string does not match pattern"
    if type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            return f"{path}: non-finite number"
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            return f"{path}: number is outside bounds"
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return f"{path}: number is outside bounds"
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return f"{path}: number is outside bounds"
    return None


def _check_schema(schema: Any, depth: int = 0) -> None:
    """Reject unsupported schema constraints rather than silently ignoring them."""
    if not isinstance(schema, dict) or depth > 32:
        raise PlannerError("INVALID_TOOL_SCHEMA", "Query schema must be a bounded object")
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "const",
        "anyOf", "oneOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "pattern", "minItems", "maxItems", "description", "title",
        "default", "examples", "$schema"}
    if set(schema) - allowed:
        raise PlannerError("INVALID_TOOL_SCHEMA", "Query schema contains unsupported constraints")
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind is not None else []
    if not all(item in {"object", "array", "string", "boolean", "integer", "number", "null"} for item in kinds):
        raise PlannerError("INVALID_TOOL_SCHEMA", "Query schema contains an invalid type")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if (not isinstance(properties, dict) or not isinstance(required, list)
            or any(not isinstance(item, str) or item not in properties for item in required)):
        raise PlannerError("INVALID_TOOL_SCHEMA", "Query schema properties or required list is invalid")
    for child in properties.values():
        _check_schema(child, depth + 1)
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, dict):
        _check_schema(additional, depth + 1)
    elif type(additional) is not bool:
        raise PlannerError("INVALID_TOOL_SCHEMA", "additionalProperties must be boolean or a schema")
    if "items" in schema:
        _check_schema(schema["items"], depth + 1)
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            variants = schema[keyword]
            if not isinstance(variants, list) or not variants:
                raise PlannerError("INVALID_TOOL_SCHEMA", f"{keyword} must contain schema objects")
            for variant in variants:
                _check_schema(variant, depth + 1)
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise PlannerError("INVALID_TOOL_SCHEMA", "enum must be a nonempty list")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise PlannerError("INVALID_TOOL_SCHEMA", "pattern must be a string")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise PlannerError("INVALID_TOOL_SCHEMA", "pattern is not a valid regular expression") from exc
    for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
        if keyword in schema and (type(schema[keyword]) is not int or schema[keyword] < 0):
            raise PlannerError("INVALID_TOOL_SCHEMA", f"{keyword} must be a nonnegative integer")
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        value = schema.get(keyword)
        if keyword in schema and (type(value) not in (int, float) or (type(value) is float and not math.isfinite(value))):
            raise PlannerError("INVALID_TOOL_SCHEMA", f"{keyword} must be a finite number")


def _positive_budget(agent: dict[str, Any], name: str, default: int) -> int:
    value = agent.get(name, default)
    if type(value) is not int or value <= 0:
        raise PlannerError("INVALID_AGENT_CONFIG", f"agent.{name} must be a positive integer")
    return value


def _read(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _record_proposal_rejection(detail: dict[str, Any], original_error: Exception) -> bool:
    """Keep a validation failure primary if its audit record cannot be written."""
    try:
        record_model_event("proposal_rejected", detail)
        return True
    except Exception as recording_error:
        failures = list(getattr(original_error, "recording_errors", []))
        failures.append({"event": "proposal_rejected", "type": type(recording_error).__name__,
                         "message": str(recording_error)})
        original_error.recording_errors = failures
        if isinstance(original_error, PlannerError):
            original_error.details = {**original_error.details, "recording_errors": copy.deepcopy(failures)}
        return False


def _run_tool_dialogue(
    client: Any,
    *,
    instruction: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    tools: list[dict[str, Any]],
    tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]],
    config: dict[str, Any],
    errors: RequestErrors,
    model: str | None = None,
    final_validator: Callable[[str], str] | None = None,
) -> DialogueResult:
    """Query tools over several turns, then return the final JSON response.

    Every received batch is validated in full before any handler runs. Invalid
    protocol, exhausted budgets and recording failures terminate the dialogue;
    domain query errors are returned to the model as tool results.
    """
    started = time.perf_counter()
    request_count, tool_call_count = 0, 0
    messages: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    received: dict[str, Any] | None = None
    validation_phase = None
    requested_model = model if model is not None else os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    try:
        agent = config.get("agent", {})
        if not isinstance(agent, dict):
            raise PlannerError("INVALID_AGENT_CONFIG", "agent must be an object")
        max_rounds = _positive_budget(agent, "max_rounds", 8)
        max_calls = _positive_budget(agent, "max_tool_calls", 24)
        max_tokens = _positive_budget(agent, "max_output_tokens", 4096)
        max_repairs = _positive_budget(agent, "max_proposal_repairs", 2)
        repair_count = 0
        max_result_chars = _positive_budget(agent, "max_tool_result_chars", 65536)
        max_argument_chars = _positive_budget(agent, "max_tool_arguments_chars", 16384)
        if not isinstance(instruction, str) or not isinstance(user_payload, dict):
            raise PlannerError("INVALID_INPUT", "instruction must be a string and user_payload an object")
        if not isinstance(tools, list):
            raise PlannerError("INVALID_TOOL_SCHEMA", "tools must be a list")
        schemas: dict[str, dict[str, Any]] = {}
        for descriptor in tools:
            function = _read(descriptor, "function")
            if not isinstance(descriptor, dict) or descriptor.get("type") != "function" or not isinstance(function, dict):
                raise PlannerError("INVALID_TOOL_SCHEMA", "Only function tool descriptors are supported")
            name = function.get("name")
            schema = function.get("parameters")
            if (not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name)
                    or name in schemas or not isinstance(schema, dict) or schema.get("type") != "object"):
                raise PlannerError("INVALID_TOOL_SCHEMA", "Tool names must be unique and parameters an object schema")
            _check_schema(schema)
            schemas[name] = schema
        payload = copy.deepcopy(user_payload)
        payload["instruction"] = instruction
        messages.extend([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
        ])
        pending_feedback = []
        if payload.get("feedback") is not None:
            pending_feedback.append({"feedback_id": current_trace().get("feedback_id") or new_trace_id("feedback"),
                "message_index": 1, "field_path": ["feedback"], "source": "caller_feedback"})
        used_call_ids: set[str] = set()
        planner = config.get("planner", {})
        for round_number in range(1, max_rounds + 1):
            validation_phase = None
            model_call_id = new_trace_id("model")
            round_identity = {"model_call_id": model_call_id, "round": round_number}
            record_model_event("context_assembled", {**round_identity,
                "messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools),
                "visibility": "prepared_model_input"})
            for included in pending_feedback:
                record_model_event("feedback_included", {**round_identity, **included,
                    "delivery": "included_in_prepared_messages", "remote_read": None})
            pending_feedback = []
            try:
                with trace_scope(**round_identity):
                    response = complete_chat(client, messages=messages, tools=tools, errors=errors,
                        model=requested_model, max_tokens=max_tokens,
                        reasoning_effort=planner.get("reasoning_effort"), thinking=planner.get("thinking"),
                        model_call_id=model_call_id)
            except PlannerError as exc:
                if exc.details.get("request_issued"):
                    request_count += 1
                if exc.details.get("received_response") is not None:
                    received = exc.details["received_response"]
                raise
            request_count += 1
            record_model_event("model_response_delivered", {**round_identity,
                "request_issued": True, "recipient": "dialogue_response_parser",
                "native_request_id": _read(response, "_request_id")})
            choices = _read(response, "choices")
            choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
            received = {"model_call_id": model_call_id,
                "response_text": _read(_read(choice, "message"), "content"),
                "tool_calls": _plain_model_value(_read(_read(choice, "message"), "tool_calls")),
                "finish_reason": _read(choice, "finish_reason"), "response_model": _read(response, "model"),
                "request_id": _read(response, "_request_id")}
            validation_phase = "response_protocol"
            if choice is None:
                raise PlannerError("EMPTY_RESPONSE", errors.empty_response)
            message = _read(choice, "message")
            if message is None:
                raise PlannerError("INVALID_RESPONSE", "Assistant response is missing its message")
            content = _read(message, "content")
            calls = _read(message, "tool_calls")
            finish_reason = _read(choice, "finish_reason")
            if finish_reason == "length":
                raise PlannerError("TRUNCATED_RESPONSE", "Agent response reached the output token limit")
            if content is not None and not isinstance(content, str):
                raise PlannerError("INVALID_RESPONSE", "Assistant content must be a string or null")
            if calls is not None and not isinstance(calls, (list, tuple)):
                raise PlannerError("INVALID_TOOL_CALL", "Assistant tool_calls must be a list")
            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            reasoning = _read(message, "reasoning_content")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise PlannerError("INVALID_RESPONSE", "Assistant reasoning_content must be a string")
                # DeepSeek thinking tool turns require replay of this field.
                assistant["reasoning_content"] = reasoning
            if not calls:
                if finish_reason != "stop":
                    raise PlannerError("INVALID_RESPONSE", "Final assistant response must finish with stop")
                if not content or not content.strip():
                    raise PlannerError("EMPTY_RESPONSE", errors.empty_response)
                validation_phase = None
                messages.append(assistant)
                raw_proposal = content
                record_model_event("proposal_received", {**round_identity, "raw_proposal": raw_proposal,
                    "message_index": len(messages) - 1})
                if final_validator is not None:
                    record_model_event("proposal_validation_started", {**round_identity,
                        "raw_proposal": raw_proposal, "validation_kind": "caller_validator"})
                    try:
                        with trace_scope(**round_identity):
                            content = final_validator(content)
                    except PlannerError as exc:
                        if exc.code in {"RECORDING_FAILED", "TASK_CANCELLED", "VIEWER_CLOSED", "TASK_BUDGET_EXCEEDED"}:
                            raise
                        detail = {"round": round_number, "repair": repair_count + 1,
                                  "error": {"code": exc.code, "message": str(exc), "details": exc.details}}
                        if not _record_proposal_rejection({**round_identity, **detail,
                                "raw_proposal": raw_proposal, "normalized_proposal": None}, exc):
                            raise
                        if repair_count >= max_repairs or round_number == max_rounds:
                            raise
                        repair_count += 1
                        messages.append({"role": "user", "content": json.dumps({
                            "runtime_feedback": "Proposal validation failed before execution. Correct the proposal using the error and available capabilities. No action has executed.",
                            **detail}, ensure_ascii=False, allow_nan=False)})
                        repair_feedback = {"feedback_id": new_trace_id("feedback"),
                            "message_index": len(messages) - 1, "source": "proposal_validation"}
                        record_model_event("feedback_created", {**round_identity, **repair_feedback,
                            "message": copy.deepcopy(messages[-1]), "delivered": False})
                        pending_feedback.append(repair_feedback)
                        continue
                    except ModelRecordingError:
                        raise
                    except Exception as exc:
                        _record_proposal_rejection({**round_identity,
                            "raw_proposal": raw_proposal, "normalized_proposal": None,
                            "error": {"code": type(exc).__name__, "message": str(exc)},
                            "repair_attempted": False}, exc)
                        raise
                record_model_event("proposal_accepted", {**round_identity,
                    "raw_proposal": raw_proposal,
                    "normalized_proposal": content if final_validator is not None else None,
                    "validation_kind": "caller_validator" if final_validator is not None else "protocol_only",
                    "execution_authorized": False})
                final = PlannerResponse(content=content, planner_kind="llm", requested_model=requested_model,
                    response_model=_read(response, "model"), usage=_plain_model_value(_read(response, "usage")),
                    latency_s=time.perf_counter() - started, finish_reason=finish_reason,
                    request_id=_read(response, "_request_id"))
                return DialogueResult(final, request_count, tool_call_count,
                    copy.deepcopy(messages), copy.deepcopy(tool_results))
            if finish_reason != "tool_calls":
                raise PlannerError("INVALID_RESPONSE", "Tool turn must finish with tool_calls")
            if tool_call_count + len(calls) > max_calls:
                raise PlannerError("TOOL_BUDGET_EXCEEDED", "Agent query tool budget exhausted",
                    {"max_tool_calls": max_calls, "requested_tool_calls": len(calls)})
            if round_number == max_rounds:
                raise PlannerError("DIALOGUE_BUDGET_EXCEEDED", "Agent dialogue round budget exhausted",
                    {"max_rounds": max_rounds})
            validation_phase = "tool_call_schema"
            batch: list[dict[str, Any]] = []
            batch_ids: set[str] = set()
            for call in calls:
                call_id, call_type = _read(call, "id"), _read(call, "type")
                function = _read(call, "function")
                name, arguments = _read(function, "name"), _read(function, "arguments")
                if (not isinstance(call_id, str) or not call_id or len(call_id) > 256
                        or call_id in used_call_ids or call_id in batch_ids or call_type != "function"):
                    raise PlannerError("INVALID_TOOL_CALL", "Tool call ID or type is invalid")
                if not isinstance(name, str) or name not in schemas:
                    raise PlannerError("UNKNOWN_TOOL", "Assistant requested an unregistered query tool",
                        {"tool_name": name if isinstance(name, str) else None})
                if not isinstance(arguments, str) or len(arguments) > max_argument_chars:
                    raise PlannerError("INVALID_TOOL_ARGUMENTS", "Tool arguments must be bounded JSON text")
                try:
                    parsed = _strict_json(arguments)
                except (ValueError, TypeError, RecursionError) as exc:
                    raise PlannerError("INVALID_TOOL_ARGUMENTS", "Tool arguments are not strict JSON") from exc
                if not isinstance(parsed, dict):
                    raise PlannerError("INVALID_TOOL_ARGUMENTS", "Tool arguments must be a JSON object")
                error = _schema_error(parsed, schemas[name])
                if error:
                    raise PlannerError("INVALID_TOOL_ARGUMENTS", error, {"tool_name": name})
                batch.append({"tool_call_id": new_trace_id("tool"), "native_tool_call_id": call_id,
                    "name": name, "arguments": parsed,
                    "wire_arguments": arguments})
                batch_ids.add(call_id)
            validation_phase = None
            assistant["tool_calls"] = [{"id": row["native_tool_call_id"], "type": "function",
                "function": {"name": row["name"], "arguments": row["wire_arguments"]}} for row in batch]
            messages.append(assistant)
            used_call_ids.update(batch_ids)
            for row in batch:
                detail = {**round_identity, "tool_call_id": row["tool_call_id"],
                    "native_tool_call_id": row["native_tool_call_id"],
                    "name": row["name"], "arguments": copy.deepcopy(row["arguments"])}
                record_model_event("agent_tool_call", detail)
                tool_call_count += 1
                handler_error = None
                try:
                    with trace_scope(**round_identity, tool_call_id=row["tool_call_id"],
                                     native_tool_call_id=row["native_tool_call_id"]):
                        result = tool_handler(row["name"], copy.deepcopy(row["arguments"]))
                except ModelRecordingError:
                    raise
                except PlannerError as exc:
                    if exc.code == "RECORDING_FAILED":
                        raise
                    result = {"ok": False, "error": {"code": exc.code, "message": str(exc), "details": exc.details}}
                    handler_error = copy.deepcopy(result["error"])
                except Exception as exc:
                    result = {"ok": False, "error": {"code": "QUERY_FAILED", "message": type(exc).__name__}}
                    handler_error = copy.deepcopy(result["error"])
                raw_result = _plain_model_value(result) if handler_error is None else None
                raw_result_type = type(result).__name__ if handler_error is None else None
                raw_availability = "available" if handler_error is None else "handler_failed"
                try:
                    json.dumps(raw_result, ensure_ascii=False, allow_nan=False)
                except (ValueError, TypeError, RecursionError):
                    raw_result, raw_availability = None, "unserializable"
                if result is not None and raw_result is None and handler_error is None:
                    raw_availability = "unsupported_result_type"
                try:
                    if not isinstance(result, dict):
                        raise TypeError("Query result must be an object")
                    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
                except (ValueError, TypeError, RecursionError):
                    result = {"ok": False, "error": {"code": "INVALID_TOOL_RESULT", "message": "Query result is not a JSON object"}}
                    encoded = json.dumps(result, ensure_ascii=False)
                if len(encoded) > max_result_chars:
                    result = {"ok": False, "error": {"code": "TOOL_RESULT_TOO_LARGE",
                        "message": "Query result exceeds the configured character budget"}}
                    encoded = json.dumps(result, ensure_ascii=False)
                    if len(encoded) > max_result_chars:
                        raise PlannerError("TOOL_RESULT_TOO_LARGE", "Query error result exceeds the configured character budget")
                feedback_id = new_trace_id("feedback")
                result_row = {**detail, "result": copy.deepcopy(result),
                    "raw_handler_result": raw_result, "raw_handler_result_availability": raw_availability,
                    "raw_handler_result_type": raw_result_type,
                    "handler_error": handler_error, "presented_result": copy.deepcopy(result),
                    "feedback_id": feedback_id, "message_index": len(messages),
                    "wire_result": encoded}
                tool_results.append(result_row)
                record_model_event("agent_tool_result", result_row)
                messages.append({"role": "tool", "tool_call_id": row["native_tool_call_id"], "content": encoded})
                tool_feedback = {"feedback_id": feedback_id, "tool_call_id": row["tool_call_id"],
                    "native_tool_call_id": row["native_tool_call_id"], "message_index": len(messages) - 1,
                    "source": "query_tool_result"}
                record_model_event("feedback_created", {**round_identity, **tool_feedback,
                    "message": copy.deepcopy(messages[-1]), "delivered": False})
                pending_feedback.append(tool_feedback)
        raise PlannerError("DIALOGUE_BUDGET_EXCEEDED", "Agent dialogue round budget exhausted")
    except ModelRecordingError as exc:
        error = PlannerError(exc.code, str(exc), {**exc.details, "request_issued": request_count > 0})
    except PlannerError as exc:
        error = exc
        if (received is not None and validation_phase in {"response_protocol", "tool_call_schema"}
                and exc.code in {"EMPTY_RESPONSE", "INVALID_RESPONSE", "UNKNOWN_TOOL",
                                 "INVALID_TOOL_CALL", "INVALID_TOOL_ARGUMENTS"}):
            _record_proposal_rejection({**round_identity, "source": "llm",
                "reason": validation_phase, "error_code": exc.code,
                "error": {"code": exc.code, "message": str(exc), "details": copy.deepcopy(exc.details)},
                "native_payload": copy.deepcopy(received), "raw_proposal": received.get("response_text"),
                "normalized_proposal": None, "repair_attempted": False,
                "tool_batch_executed": False, "execution_authorized": False}, exc)
    except Exception as exc:
        error = PlannerError("PLANNER_ERROR", f"{errors.failed_prefix}: {type(exc).__name__}",
            {"recording_errors": copy.deepcopy(exc.recording_errors)} if getattr(exc, "recording_errors", None) else None)
    error.details = {**current_trace(), **error.details,
        "request_count": request_count, "tool_call_count": tool_call_count,
        "messages": copy.deepcopy(messages), "tool_results": copy.deepcopy(tool_results)}
    if received is not None:
        error.details["received_response"] = copy.deepcopy(received)
        error.details["response_text"] = received.get("response_text")
    raise error


def run_tool_dialogue(client: Any, *, instruction: str, system_prompt: str,
                     user_payload: dict[str, Any], tools: list[dict[str, Any]],
                     tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]],
                     config: dict[str, Any], errors: RequestErrors, model: str | None = None,
                     final_validator: Callable[[str], str] | None = None,
                     stage: str | None = None) -> DialogueResult:
    """Run one independent dialogue; stages and IDs survive copied contexts."""
    identity = {"dialogue_id": new_trace_id("dialogue")}
    if stage is not None:
        identity["stage"] = stage
    with trace_scope(**identity):
        return _run_tool_dialogue(client, instruction=instruction, system_prompt=system_prompt,
            user_payload=user_payload, tools=tools, tool_handler=tool_handler, config=config,
            errors=errors, model=model, final_validator=final_validator)

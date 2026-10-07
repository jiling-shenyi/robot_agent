"""Shared native tool transport with lazy SDK imports and bounded requests.

Roles provide their prompts, budgets and existing error wording. The adapter
does not validate plans, choose permissions or execute robot/environment edits.
"""

from __future__ import annotations

import json
import os
import time
import copy
from dataclasses import dataclass
from typing import Any

from embodied_agent.models.contracts import PlannerError, PlannerResponse
from embodied_agent.models.tracing import ModelRecordingError, record_model_event, new_trace_id, trace_scope
from embodied_agent.models.budget import current_budget


@dataclass(frozen=True)
class RequestErrors:
    rate_limited: str
    timeout: str
    connection_failed: str
    failed_prefix: str
    empty_response: str
    status_details: bool = False


def create_client(*, api_key: str, base_url: str, timeout_s: float) -> Any:
    """Create the existing SDK client without introducing automatic retries."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)


def _bounded_timeout(client: Any, task_budget: Any) -> float:
    """Keep the configured request timeout within the remaining task deadline."""
    original = getattr(client, "timeout", None)
    if type(original) not in (int, float):
        original = getattr(original, "read", None)
    remaining = task_budget.remaining_s
    return min(remaining, original) if type(original) in (int, float) and original > 0 else remaining


def completion_choice(response: Any, errors: RequestErrors) -> Any:
    choice = response.choices[0] if response.choices else None
    if choice is None:
        raise PlannerError("EMPTY_RESPONSE", errors.empty_response)
    return choice


def _plain_model_value(value: Any) -> Any:
    """Convert SDK metadata and lightweight fixtures to recordable JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain_model_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_model_value(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain_model_value(dump())
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {key: _plain_model_value(item) for key, item in attributes.items()
                if not key.startswith("_")}
    return None


def _complete_chat(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    errors: RequestErrors,
    model: str | None = None,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    thinking: str | None = None,
    model_call_id: str | None = None,
) -> Any:
    """Issue one native tool-capable turn, without JSON mode or SDK retries.

    The caller owns the conversation and validates tool requests. Evidence is
    durable before each SDK request; recording failures never trigger retries.
    """
    from openai import (
        APIConnectionError, APIStatusError, APITimeoutError,
        AuthenticationError, RateLimitError,
    )

    issued = False
    model_call_id = model_call_id or new_trace_id("model")
    received: dict[str, Any] | None = None

    def failure(code: str, message: str, details: dict | None = None) -> PlannerError:
        known = {**(details or {}), "model_call_id": model_call_id, "request_issued": issued}
        if received is None:
            try:
                record_model_event("model_failed", {**known,
                    "error": {"code": code, "message": message}, "usage": None,
                    "token_ids": None, "logprobs": None,
                    "availability": {"usage": "unknown", "token_ids": "unsupported",
                                     "logprobs": "unsupported_or_not_returned"},
                    "remote_delivery": "unknown", "remote_processing": "unknown"})
            except ModelRecordingError as recording_error:
                return PlannerError(recording_error.code, str(recording_error), {
                    **recording_error.details, **known,
                    "original_request_error": {"code": code, "message": message}})
        else:
            known["received_response"] = received
        return PlannerError(code, message, known)
    try:
        parameters: dict[str, Any] = {
            "model": model if model is not None else os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None:
            parameters["reasoning_effort"] = reasoning_effort
        if thinking is not None:
            parameters["extra_body"] = {"thinking": {"type": thinking}}
        # Reject unserializable caller data before recording or issuing a call.
        json.dumps(parameters, ensure_ascii=False, allow_nan=False)
        task_budget = current_budget()
        if task_budget is not None:
            parameters["max_tokens"] = task_budget.output_limit(parameters["max_tokens"])
            task_budget.check()
            parameters["timeout"] = _bounded_timeout(client, task_budget)
        record_model_event("model_request", {"model_call_id": model_call_id, "request": parameters,
            "token_ids": None, "logprobs": None,
            "availability": {"token_ids": "unsupported", "logprobs": "not_requested",
                "provider_defaults": "unknown"},
            "sampling": {name: {"value": parameters.get(name),
                "availability": "explicit" if name in parameters else "provider_default_unknown"}
                for name in ("temperature", "top_p", "seed", "frequency_penalty", "presence_penalty")}})
        started = time.perf_counter()
        if task_budget is not None:
            task_budget.record_request()
        record_model_event("model_dispatch_started", {"model_call_id": model_call_id,
            "request_issued": False, "boundary": "before_sdk_invocation",
            "remote_delivery": "unknown", "remote_processing": "unknown"})
        issued = True
        response = client.chat.completions.create(**parameters)
        choices = getattr(response, "choices", None)
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        received = {
            "model_call_id": model_call_id,
            "requested_model": parameters["model"],
            "response_model": getattr(response, "model", None),
            "response_text": getattr(message, "content", None),
            "tool_calls": _plain_model_value(getattr(message, "tool_calls", None)),
            "reasoning_content": getattr(message, "reasoning_content", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": _plain_model_value(getattr(response, "usage", None)),
            "request_id": getattr(response, "_request_id", None),
            "latency_s": time.perf_counter() - started,
            "request_issued": True, "remote_delivery": "response_received",
            "token_ids": None, "logprobs": _plain_model_value(getattr(choice, "logprobs", None)),
            "availability": {"token_ids": "unsupported",
                "logprobs": "available" if getattr(choice, "logprobs", None) is not None else "not_returned",
                "usage": "available" if getattr(response, "usage", None) is not None else "unknown"},
        }
        record_model_event("model_response", received)
        if task_budget is not None:
            task_budget.record_usage(getattr(response, "usage", None))
            task_budget.check()
        return response
    except ModelRecordingError as exc:
        details = {**exc.details, "model_call_id": model_call_id, "request_issued": issued}
        if received is not None:
            details["received_response"] = received
        raise PlannerError(exc.code, str(exc), details) from exc
    except PlannerError as exc:
        traced = failure(exc.code, str(exc), exc.details)
        if traced.code != exc.code:
            raise traced from exc
        exc.details = traced.details
        raise
    except AuthenticationError as exc:
        raise failure("AUTHENTICATION_FAILED", "DeepSeek rejected the configured credential") from exc
    except RateLimitError as exc:
        raise failure("RATE_LIMITED", errors.rate_limited) from exc
    except APITimeoutError as exc:
        raise failure("TIMEOUT", errors.timeout) from exc
    except APIConnectionError as exc:
        raise failure("CONNECTION_FAILED", errors.connection_failed) from exc
    except APIStatusError as exc:
        details = {"request_issued": issued}
        if errors.status_details:
            details.update(http_status=int(exc.status_code), request_id=exc.request_id)
        raise failure("API_STATUS_ERROR", f"DeepSeek returned HTTP {exc.status_code}", details) from exc
    except Exception as exc:
        details = {"request_issued": issued}
        if received is not None:
            details["received_response"] = received
        raise failure("PLANNER_ERROR", f"{errors.failed_prefix}: {type(exc).__name__}", details) from exc


def complete_chat(client: Any, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                  errors: RequestErrors, model: str | None = None, max_tokens: int = 4096,
                  reasoning_effort: str | None = None, thinking: str | None = None,
                  model_call_id: str | None = None) -> Any:
    """Issue one traced native turn; preserve identity inside SDK-side callbacks."""
    identity = model_call_id or new_trace_id("model")
    with trace_scope(model_call_id=identity):
        return _complete_chat(client, messages=messages, tools=tools, errors=errors, model=model,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort, thinking=thinking,
            model_call_id=identity)


# Both names describe the same native protocol; callers may choose either.
complete_tools = complete_chat


def model_response(
    response: Any,
    *,
    requested_model: str,
    planner_kind: str,
    started: float,
    errors: RequestErrors,
    truthy_usage: bool = False,
) -> PlannerResponse:
    """Preserve response metadata used by planning and interpretation evidence."""
    choice = completion_choice(response, errors)
    content = choice.message.content if isinstance(choice.message.content, str) else ""
    usage_present = bool(response.usage) if truthy_usage else response.usage is not None
    usage = response.usage.model_dump() if usage_present else None
    return PlannerResponse(
        content=content,
        planner_kind=planner_kind,
        requested_model=requested_model,
        response_model=str(response.model),
        usage=usage,
        latency_s=time.perf_counter() - started,
        finish_reason=choice.finish_reason,
        request_id=getattr(response, "_request_id", None),
    )

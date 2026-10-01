"""Deterministic and DeepSeek planning adapters for the constrained M3 schema."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol


class PlannerError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class PlannerResponse:
    content: str
    planner_kind: str
    requested_model: str | None
    response_model: str | None
    usage: dict[str, Any] | None
    latency_s: float
    finish_reason: str | None
    request_id: str | None


class Planner(Protocol):
    kind: str

    def plan(self, goal: dict[str, Any], observation: dict[str, Any]) -> PlannerResponse: ...


class StubPlanner:
    kind = "stub"

    def plan(self, goal: dict[str, Any], observation: dict[str, Any]) -> PlannerResponse:
        started = time.perf_counter()
        plan = {
            "schema_version": 1,
            "base_obs_id": int(observation["obs_id"]),
            "steps": [
                {"skill": "pick", "object_id": "cube", "approach_mode": "top"},
                {"skill": "place", "target_id": goal["target_id"]},
            ],
        }
        return PlannerResponse(
            content=json.dumps(plan, separators=(",", ":")),
            planner_kind=self.kind,
            requested_model=None,
            response_model=None,
            usage=None,
            latency_s=time.perf_counter() - started,
            finish_reason="stop",
            request_id=None,
        )


class DeepSeekPlanner:
    kind = "llm"

    def __init__(self, config: dict[str, Any]):
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise PlannerError("MISSING_CREDENTIAL", "DEEPSEEK_API_KEY is not configured")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.timeout_s = float(config["budgets"]["planner_timeout_s"])
        self.max_tokens = int(config["budgets"]["max_output_tokens"])
        self.reasoning_effort = str(config["planner"]["reasoning_effort"])
        self.thinking = str(config["planner"]["thinking"])
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            max_retries=0,
        )
        self.system_prompt = str(config["planner"]["system_prompt"])

    def plan(self, goal: dict[str, Any], observation: dict[str, Any]) -> PlannerResponse:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
        )

        request = {
            "task_goal": {
                "object_id": goal["object_id"],
                "target_id": goal["target_id"],
            },
            "instruction": goal["instruction"],
            "observation": observation,
            "skill_catalog": [
                {
                    "skill": "pick",
                    "object_id": "cube",
                    "approach_mode": "top",
                    "description": "Grasp the cube from above and verify a bilateral grasp and lift.",
                },
                {
                    "skill": "place",
                    "target_id": goal["target_id"],
                    "description": "Place and release the held cube in the requested target region.",
                },
            ],
            "required_plan": {
                "schema_version": 1,
                "base_obs_id": int(observation["obs_id"]),
                "steps": [
                    {"skill": "pick", "object_id": "cube", "approach_mode": "top"},
                    {"skill": "place", "target_id": goal["target_id"]},
                ],
            },
        }
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                stream=False,
                reasoning_effort=self.reasoning_effort,
                extra_body={"thinking": {"type": self.thinking}},
            )
        except AuthenticationError as exc:
            raise PlannerError("AUTHENTICATION_FAILED", "DeepSeek rejected the configured credential") from exc
        except RateLimitError as exc:
            raise PlannerError("RATE_LIMITED", "DeepSeek rate limited the planning request") from exc
        except APITimeoutError as exc:
            raise PlannerError("TIMEOUT", "DeepSeek planning request timed out") from exc
        except APIConnectionError as exc:
            raise PlannerError("CONNECTION_FAILED", "Could not connect to the configured DeepSeek endpoint") from exc
        except APIStatusError as exc:
            raise PlannerError(
                "API_STATUS_ERROR",
                f"DeepSeek returned HTTP {exc.status_code}",
                {"http_status": int(exc.status_code), "request_id": exc.request_id},
            ) from exc
        except Exception as exc:
            raise PlannerError("PLANNER_ERROR", f"Planner request failed: {type(exc).__name__}") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise PlannerError("EMPTY_RESPONSE", "DeepSeek returned no completion choices")
        message = choice.message
        content = message.content if isinstance(message.content, str) else ""
        usage = response.usage.model_dump() if response.usage is not None else None
        return PlannerResponse(
            content=content,
            planner_kind=self.kind,
            requested_model=self.model,
            response_model=str(response.model),
            usage=usage,
            latency_s=time.perf_counter() - started,
            finish_reason=choice.finish_reason,
            request_id=getattr(response, "_request_id", None),
        )

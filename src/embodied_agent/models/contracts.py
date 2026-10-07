"""Transport-independent model response and planning error contracts."""

from __future__ import annotations

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

    def plan(self, instruction: str, snapshot: dict[str, Any], world: Any, *,
             feedback: Any = None, task_context: dict[str, Any] | None = None) -> PlannerResponse: ...

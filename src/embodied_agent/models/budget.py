"""One thread-safe budget and cancellation signal for the entire user task."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time
import math
from typing import Any, Callable

from .contracts import PlannerError


class TaskBudget:
    def __init__(self, config: dict[str, Any] | None = None, *, clock: Callable = time.monotonic):
        limits = (config or {}).get("task_budget", config or {})
        self.max_requests = int(limits.get("max_requests", 24))
        self.max_tokens = int(limits.get("max_tokens", 131072))
        self.max_actions = int(limits.get("max_actions", 128))
        self.max_replans = int(limits.get("max_replans", 4))
        self.timeout_s = float(limits.get("timeout_s", 300))
        if min(self.max_requests, self.max_tokens, self.max_actions) < 1 or self.max_replans < 0 or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Task budget limits must be positive; max_replans may be zero")
        self._clock, self._started = clock, clock()
        self._lock = threading.RLock()
        self.cancel_reason: str | None = None
        self.cancel_code = "TASK_CANCELLED"
        self.requests = self.prompt_tokens = self.completion_tokens = 0
        self.actions = self.replans = 0

    def cancel(self, reason: str = "Task cancelled", *, code: str = "TASK_CANCELLED") -> None:
        with self._lock:
            if self.cancel_reason is None:
                self.cancel_reason, self.cancel_code = reason, code

    def check(self) -> None:
        with self._lock:
            if self.cancel_reason:
                raise PlannerError(self.cancel_code, self.cancel_reason, {"budget": self.snapshot()})
            if self.remaining_s <= 0:
                raise PlannerError("TASK_BUDGET_EXCEEDED", "Task wall-clock budget exhausted", {"budget": self.snapshot()})
            if self.prompt_tokens + self.completion_tokens >= self.max_tokens:
                raise PlannerError("TASK_BUDGET_EXCEEDED", "Task token budget exhausted", {"budget": self.snapshot()})

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.timeout_s - (self._clock() - self._started))

    def output_limit(self, requested: int) -> int:
        with self._lock:
            self.check()
            return min(requested, self.max_tokens - self.prompt_tokens - self.completion_tokens)

    def record_request(self) -> None:
        with self._lock:
            self.check()
            if self.requests >= self.max_requests:
                raise PlannerError("TASK_BUDGET_EXCEEDED", "Task model request budget exhausted", {"budget": self.snapshot()})
            self.requests += 1

    def record_usage(self, usage: Any) -> None:
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        if not isinstance(usage, dict):
            return
        with self._lock:
            prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            total = int(usage.get("total_tokens", prompt + completion) or 0)
            self.prompt_tokens += max(prompt, total - completion)
            self.completion_tokens += completion

    def record_action(self) -> None:
        with self._lock:
            self.check()
            if self.actions >= self.max_actions:
                raise PlannerError("TASK_BUDGET_EXCEEDED", "Task action budget exhausted", {"budget": self.snapshot()})
            self.actions += 1

    def record_replan(self) -> None:
        with self._lock:
            self.check()
            if self.replans >= self.max_replans:
                raise PlannerError("TASK_BUDGET_EXCEEDED", "Task recovery budget exhausted", {"budget": self.snapshot()})
            self.replans += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"requests": self.requests, "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                    "actions": self.actions, "replans": self.replans,
                    "elapsed_s": self._clock() - self._started,
                    "remaining_s": self.remaining_s, "cancelled": bool(self.cancel_reason),
                    "limits": {"max_requests": self.max_requests, "max_tokens": self.max_tokens,
                               "max_actions": self.max_actions, "max_replans": self.max_replans,
                               "timeout_s": self.timeout_s}}


_current: ContextVar[TaskBudget | None] = ContextVar("embodied_task_budget", default=None)


def current_budget() -> TaskBudget | None:
    return _current.get()


@contextmanager
def budget_context(budget: TaskBudget):
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)

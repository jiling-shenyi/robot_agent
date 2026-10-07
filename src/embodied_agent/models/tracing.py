"""Task-scoped observations of actual model requests, independent of transport."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator
import uuid

_RECORDER: ContextVar[Callable[[str, dict[str, Any]], None] | None] = ContextVar(
    "model_request_recorder", default=None
)
_IDENTITY: ContextVar[dict[str, Any]] = ContextVar("recording_trace_identity", default={})


def new_trace_id(kind: str) -> str:
    """Allocate a local identity independent of provider and record filenames."""
    return f"{kind}_{uuid.uuid4().hex}"


def current_trace() -> dict[str, Any]:
    """Return detached task/agent/attempt/decision identity for the current work."""
    return copy.deepcopy(_IDENTITY.get())


@contextmanager
def trace_scope(**identity: Any) -> Iterator[dict[str, Any]]:
    """Bind immutable IDs; copy_context carries the scope into language workers.

    Nested scopes override only their own fields and restore the parent's IDs
    on exit. They do not change the recorder or any execution permissions.
    """
    value = {**current_trace(), **copy.deepcopy(identity)}
    token = _IDENTITY.set(value)
    try:
        yield copy.deepcopy(value)
    finally:
        _IDENTITY.reset(token)


class ModelRecordingError(RuntimeError):
    """Persistence failed at a known model boundary; never retry the request."""

    code = "RECORDING_FAILED"

    def __init__(self, event: str, detail: dict[str, Any], cause: Exception):
        super().__init__(f"Cannot record {event}: {cause}")
        self.details = {
            "recording_event": {"event": event, "detail": copy.deepcopy(detail)},
            "request_issued": detail.get("request_issued", event in {
                "model_response", "model_failed", "model_response_delivered"}),
            "original_error": {"type": type(cause).__name__, "message": str(cause)},
        }


@contextmanager
def capture_model_requests(recorder: Callable[[str, dict[str, Any]], None]) -> Iterator[None]:
    token = _RECORDER.set(recorder)
    try:
        yield
    finally:
        _RECORDER.reset(token)


def record_model_event(event: str, detail: dict[str, Any]) -> None:
    recorder = _RECORDER.get()
    if recorder is not None:
        observed = {**current_trace(), **copy.deepcopy(detail)}
        try:
            recorder(event, observed)
        except ModelRecordingError:
            raise
        except Exception as exc:
            raise ModelRecordingError(event, observed, exc) from exc

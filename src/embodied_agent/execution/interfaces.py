"""Public execution interface implemented by live simulator episodes."""

from __future__ import annotations

from typing import Any, Protocol


class LiveRobotEpisode(Protocol):
    """Display and lifecycle interface shared by arm and mobile robot maps.

    Robot-specific controllers remain behind this interface; the Demo owns
    requests, map persistence, case execution and evidence output.
    """

    robot_kind: str
    world: Any
    model: Any
    data: Any
    total_steps: int

    def prepare(self, *, check_safety: bool = True) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def set_viewer_status(self, title: str, detail: str) -> None: ...
    def close(self) -> None: ...

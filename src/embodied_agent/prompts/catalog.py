"""Resolve default or explicitly configured prompt text without backend imports."""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from embodied_agent.paths import PROJECT_ROOT


_DEFAULTS = {
    "instruction.intent": ("instruction-agent-v2", "instruction.intent.txt"),
    "instruction.plan": ("instruction-agent-v2", "instruction.plan.txt"),
    "environment.desktop": ("environment-agent-desktop-v1", "environment.desktop.txt"),
    "environment.home": ("environment-agent-home-v1", "environment.home.txt"),
}


@dataclass(frozen=True)
class PromptSpec:
    """One immutable, versioned prompt with a digest of its exact UTF-8 text."""

    id: str
    version: str
    text: str

    def __post_init__(self) -> None:
        for name in ("id", "version", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Prompt {name} must be a nonempty string")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, str]:
        """Return detached evidence sufficient to identify and inspect the text."""
        return {"id": self.id, "version": self.version,
                "sha256": self.sha256, "text": self.text}


class PromptCatalog:
    """Resolve prompts using a detached snapshot of a full runtime config.

    Overrides live at ``components.prompts.overrides`` and replace the whole
    prompt, including environment query instructions. Each override supplies
    a version and exactly one of text/path. Relative paths use project_root.
    Files are read for each get(), so replacing a file has no stale cache.
    To change an inline override, construct a catalog with the new config.
    """

    def __init__(self, config: Mapping[str, Any] | None = None, *,
                 project_root: str | Path | None = None):
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Prompt configuration must be a runtime mapping")
        self._config = copy.deepcopy(dict(config) if config is not None else {})
        self._project_root = Path(project_root if project_root is not None else PROJECT_ROOT).resolve()
        current = self._config
        for key in ("components", "prompts", "overrides"):
            current = current.get(key, {})
            if not isinstance(current, Mapping):
                raise ValueError(f"Prompt configuration {key} must be a mapping")
        self._overrides = dict(current)
        for prompt_id, override in self._overrides.items():
            if prompt_id not in _DEFAULTS:
                raise ValueError(f"Unknown prompt override: {prompt_id!r}")
            if not isinstance(override, Mapping):
                raise ValueError(f"Prompt override {prompt_id!r} must be a mapping")
            if set(override) not in ({"text", "version"}, {"path", "version"}):
                raise ValueError(f"Prompt override {prompt_id!r} requires version and exactly one of text/path")
            if not isinstance(override["version"], str) or not override["version"].strip():
                raise ValueError(f"Prompt override {prompt_id!r} requires a nonempty version")
            field = "text" if "text" in override else "path"
            if not isinstance(override[field], str) or not override[field].strip():
                raise ValueError(f"Prompt override {prompt_id!r} {field} must be a nonempty string")

    def get(self, prompt_id: str) -> PromptSpec:
        """Read one prompt; unknown IDs fail rather than selecting another role."""
        if prompt_id not in _DEFAULTS:
            raise KeyError(f"Unknown prompt: {prompt_id!r}")
        override = self._overrides.get(prompt_id)
        if override is not None:
            if "text" in override:
                text = override["text"]
            else:
                path = Path(override["path"])
                if not path.is_absolute():
                    path = self._project_root / path
                # Decode bytes to preserve intentional CRLF and terminal newlines.
                text = path.read_bytes().decode("utf-8")
            return PromptSpec(prompt_id, override["version"], text)
        version, filename = _DEFAULTS[prompt_id]
        text = resources.files("embodied_agent.prompts").joinpath("resources", filename).read_bytes().decode("utf-8")
        return PromptSpec(prompt_id, version, text)

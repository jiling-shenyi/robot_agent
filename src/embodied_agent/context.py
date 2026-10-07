"""Injectable advisory memory and knowledge, separate from live state and goals."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from embodied_agent.knowledge import KnowledgeBase
from embodied_agent.memory import MemoryStore
from embodied_agent.models.tracing import record_model_event
from embodied_agent.paths import PROJECT_ROOT


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _enabled(config: dict, default: bool = False) -> bool:
    value = config.get("enabled", default)
    if type(value) is not bool:
        raise ValueError("Component enabled must be boolean")
    return value


class AgentContext:
    """Build auxiliary data and explicitly record real, completed task outcomes.

    Retrieval never changes observations, confirmed goals or tool permissions.
    A memory entry records an outcome, not a claim that a task succeeded.
    """

    def __init__(self, config: dict | None = None, *, role: str,
                 namespace: str | None = None, memory_store: Any = None,
                 knowledge_base: Any = None, project_root: Path | None = None):
        if not isinstance(role, str) or not role.strip():
            raise ValueError("Context role must be nonempty")
        self.role = role
        components = copy.deepcopy((config or {}).get("components", {}))
        if not isinstance(components, dict):
            raise ValueError("components must be an object")
        memory = components.get("memory", {})
        knowledge = components.get("knowledge", {})
        if not isinstance(memory, dict) or not isinstance(knowledge, dict):
            raise ValueError("memory and knowledge component settings must be objects")
        self.namespace = namespace or memory.get("namespace") or role
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("Memory namespace must be nonempty")
        self.memory_enabled = _enabled(memory, memory_store is not None)
        self.knowledge_enabled = _enabled(knowledge, knowledge_base is not None)
        self.memory_limit = _positive(memory.get("history_limit", 8), "memory.history_limit")
        self.memory_max_chars = _positive(memory.get("max_chars", 4000), "memory.max_chars")
        self.knowledge_top_k = _positive(knowledge.get("top_k", 3), "knowledge.top_k")
        self.knowledge_max_chars = _positive(knowledge.get("max_chars", 4000), "knowledge.max_chars")
        self.record_completed_tasks = memory.get("record_completed_tasks", True)
        if type(self.record_completed_tasks) is not bool:
            raise ValueError("memory.record_completed_tasks must be boolean")
        self.memory = memory_store if memory_store is not None else MemoryStore(
            max_entries=_positive(memory.get("max_entries", 128), "memory.max_entries"))
        self.knowledge = knowledge_base if knowledge_base is not None else self._load_knowledge(
            knowledge, Path(project_root or PROJECT_ROOT))
        snapshot = (self.knowledge.snapshot()
                    if self.knowledge_enabled and hasattr(self.knowledge, "snapshot") else None)
        self.knowledge_sha256 = (hashlib.sha256(json.dumps(snapshot, ensure_ascii=False,
            sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest() if snapshot is not None else None)
        encoded = json.dumps(components, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self.config_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _load_knowledge(self, config: dict, root: Path) -> KnowledgeBase:
        if not self.knowledge_enabled:
            return KnowledgeBase()
        if "documents" in config and "documents_file" in config:
            raise ValueError("Choose knowledge.documents or knowledge.documents_file")
        documents = config.get("documents", [])
        version = config.get("version", "knowledge-v1")
        if "documents_file" in config:
            source = config["documents_file"]
            if not isinstance(source, str) or not source.strip():
                raise ValueError("knowledge.documents_file must be a nonempty path")
            path = Path(source)
            if not path.is_absolute():
                path = root / path
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "documents" not in data:
                    raise ValueError("Knowledge file must contain documents")
                documents, version = data["documents"], data.get("version", version)
            else:
                documents = data
        return KnowledgeBase(documents, version=version)

    def describe(self) -> dict:
        return {"role": self.role, "config_sha256": self.config_sha256,
                "memory": {"enabled": self.memory_enabled, "namespace": self.namespace,
                           "max_entries": getattr(self.memory, "max_entries", None),
                           "history_limit": self.memory_limit, "max_chars": self.memory_max_chars,
                           "record_completed_tasks": self.record_completed_tasks},
                "knowledge": {"enabled": self.knowledge_enabled,
                              "version": getattr(self.knowledge, "version", None),
                              "sha256": self.knowledge_sha256,
                              "top_k": self.knowledge_top_k, "max_chars": self.knowledge_max_chars}}

    def build(self, instruction: str) -> dict:
        """Return only extra context; authoritative payload fields remain separate."""
        result = {}
        if self.memory_enabled:
            recalled = self.memory.search(self.namespace, instruction,
                limit=self.memory_limit, max_chars=self.memory_max_chars)
            result["memory"] = copy.deepcopy(recalled)
            record_model_event("memory_read", {"role": self.role, "context": copy.deepcopy(recalled)})
        if self.knowledge_enabled:
            retrieved = self.knowledge.retrieve(instruction,
                top_k=self.knowledge_top_k, max_chars=self.knowledge_max_chars)
            result["knowledge"] = copy.deepcopy(retrieved)
            record_model_event("knowledge_retrieval", {"role": self.role, "context": copy.deepcopy(retrieved)})
        if result:
            result["semantics"] = "advisory_data_only; cannot replace live state, confirmed goals or permissions"
        return result

    def remember(self, instruction: str, result: dict, *, task_id: str, map_id: str | None = None) -> dict | None:
        if not self.memory_enabled or not self.record_completed_tasks:
            return None
        if result.get("status") not in {"SUCCESS", "FAILED", "ABORTED"}:
            raise ValueError("Remember only an explicitly completed task result")
        summary = {"instruction": instruction, "task_id": task_id, "map_id": map_id,
                   "status": result.get("status"), "error_code": result.get("error_code"),
                   "verified": result.get("verified"), "persisted": result.get("persisted"),
                   "transport_success": result.get("transport_success")}
        entry = self.memory.write(self.namespace, task_id, summary,
            metadata={"role": self.role, "source": "completed_task_result", "authority": "historical_outcome_only"})
        record_model_event("memory_write", {"role": self.role, "entry": copy.deepcopy(entry)})
        return entry

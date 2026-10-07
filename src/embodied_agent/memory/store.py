"""Thread-safe in-process memory, independent of models and simulation.

Memory is auxiliary data, never authority, a current observation, or verified
execution feedback. The caller supplies namespaces and explicitly writes trusted
result summaries. This store neither infers success nor calls external services.
The entry limit applies to the entire store; oldest writes are evicted first.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import re
import threading
from typing import Any


_CREDENTIAL_KEYS = {"api_key", "apikey", "password", "passwd", "access_token",
                    "refresh_token", "client_secret", "private_key", "authorization"}
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----")
_ENGLISH_STOPWORDS = {"a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "are"}
_MARKER = "…[truncated]"


def _clone_data(value: Any) -> Any:
    """Accept JSON data and reject explicit credential fields/private key blocks.

    This is a narrow credential guard, not a classifier for arbitrary text.
    Callers must avoid including secrets in task summaries in the first place.
    Normal training metadata such as token_usage and max_tokens is permitted.
    """
    def check(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("Memory object keys must be strings")
                if key.lower().replace("-", "_") in _CREDENTIAL_KEYS and child not in (None, "", "[REDACTED]"):
                    raise ValueError("Credential fields cannot be stored in auxiliary memory")
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, str) and _PRIVATE_KEY.search(item):
            raise ValueError("Private keys cannot be stored in auxiliary memory")

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Memory values must be finite JSON data") from error
    check(value)
    return json.loads(encoded)


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must contain 1..256 characters")
    return value


def _count(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _tokens(text: str, *, query: bool = False) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower())) - _ENGLISH_STOPWORDS
    for segment in re.findall(r"[\u3400-\u9fff]+", text):
        if not query or len(segment) == 1:
            words.update(segment)
        if len(segment) > 1:
            words.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return words


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def _budget_entry(entry: dict, remaining: int) -> dict | None:
    if _size(entry) <= remaining:
        return entry
    # Structured values are indivisible: a prefix must not masquerade as a
    # different complete result. Text excerpts are explicitly marked instead.
    if not isinstance(entry["value"], str):
        return None
    clipped = deepcopy(entry)
    clipped["truncated"] = True
    low, high = 0, len(entry["value"])
    best = None
    while low <= high:
        middle = (low + high) // 2
        clipped["value"] = entry["value"][:middle] + _MARKER
        if _size(clipped) <= remaining:
            best = deepcopy(clipped)
            low = middle + 1
        else:
            high = middle - 1
    return best


class MemoryStore:
    """Explicit namespaces, monotonic namespace revisions and copied JSON data."""

    def __init__(self, max_entries: int = 128):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._revisions: dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def _next_revision(self, namespace: str) -> int:
        revision = self._revisions.get(namespace, 0) + 1
        self._revisions[namespace] = revision
        return revision

    def write(self, namespace: str, key: str, value: Any, *, metadata: dict | None = None) -> dict:
        namespace, key = _identifier(namespace, "namespace"), _identifier(key, "key")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        content, annotations = _clone_data(value), _clone_data(metadata or {})
        with self._lock:
            revision = self._next_revision(namespace)
            entry = {"namespace": namespace, "key": key, "revision": revision,
                     "value": content, "metadata": annotations}
            identity = (namespace, key)
            self._entries.pop(identity, None)
            self._entries[identity] = entry
            while len(self._entries) > self.max_entries:
                (evicted_namespace, _), _ = self._entries.popitem(last=False)
                if evicted_namespace != namespace:
                    self._next_revision(evicted_namespace)
            return deepcopy(entry)

    def read(self, namespace: str, key: str) -> dict | None:
        namespace, key = _identifier(namespace, "namespace"), _identifier(key, "key")
        with self._lock:
            return deepcopy(self._entries.get((namespace, key)))

    def snapshot(self, namespace: str) -> dict:
        namespace = _identifier(namespace, "namespace")
        with self._lock:
            return {"namespace": namespace, "revision": self._revisions.get(namespace, 0),
                    "entries": deepcopy([entry for (scope, _), entry in self._entries.items()
                                         if scope == namespace])}

    def search(self, namespace: str, query: str, *, limit: int = 8, max_chars: int = 4000) -> dict:
        """Rank lexical matches; budget is compact JSON characters of entries.

        Empty/irrelevant queries return no entries. Structured values that do
        not fit are omitted intact; string excerpts include a truncation marker.
        Metadata, keys and the score consume the same visible-content budget.
        """
        limit, max_chars = _count(limit, "limit"), _count(max_chars, "max_chars")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        snapshot = self.snapshot(namespace)
        query_tokens = _tokens(query, query=True)
        ranked = []
        if query_tokens and limit and max_chars:
            for entry in snapshot["entries"]:
                visible = entry["key"] + " " + json.dumps(
                    {"value": entry["value"], "metadata": entry["metadata"]}, ensure_ascii=False)
                matched = query_tokens & _tokens(visible)
                if not matched:
                    continue
                row = {**entry, "score": len(matched) / len(query_tokens), "truncated": False,
                       "original_chars": len(entry["value"]) if isinstance(entry["value"], str)
                       else _size(entry["value"])}
                ranked.append(row)
        ranked.sort(key=lambda row: (-row["score"], row["key"]))
        entries, used, omitted = [], 0, 0
        for row in ranked:
            if len(entries) >= limit:
                break
            visible_row = _budget_entry(row, max_chars - used)
            if visible_row is None:
                omitted += 1
                continue
            entries.append(visible_row)
            used += _size(visible_row)
        return {"namespace": snapshot["namespace"], "revision": snapshot["revision"],
                "query": query, "entries": entries, "max_chars": max_chars,
                "returned_chars": used, "omitted_count": omitted}

    def clear(self, namespace: str) -> dict:
        namespace = _identifier(namespace, "namespace")
        with self._lock:
            identities = [identity for identity in self._entries if identity[0] == namespace]
            for identity in identities:
                del self._entries[identity]
            return {"namespace": namespace, "revision": self._next_revision(namespace),
                    "removed_count": len(identities)}

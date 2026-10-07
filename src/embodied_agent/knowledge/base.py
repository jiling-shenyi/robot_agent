"""Immutable auxiliary knowledge, without model, vector or simulation dependencies.

Retrieved documents are untrusted supporting data. They cannot confer tool
permissions, replace live observations, or assert that execution succeeded.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Iterable


_CREDENTIAL_KEYS = {"api_key", "apikey", "password", "passwd", "access_token",
                    "refresh_token", "client_secret", "private_key", "authorization"}
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----")
_ENGLISH_STOPWORDS = {"a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "are"}
_MARKER = "…[truncated]"


def _clone_data(value: Any) -> Any:
    def check(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("Knowledge object keys must be strings")
                if key.lower().replace("-", "_") in _CREDENTIAL_KEYS and child not in (None, "", "[REDACTED]"):
                    raise ValueError("Credential fields cannot be stored as knowledge")
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, str) and _PRIVATE_KEY.search(item):
            raise ValueError("Private keys cannot be stored as knowledge")

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Knowledge documents must contain finite JSON data") from error
    check(value)
    return json.loads(encoded)


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


def _count(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _budget_document(document: dict, remaining: int) -> dict | None:
    if _size(document) <= remaining:
        return document
    clipped = deepcopy(document)
    clipped["truncated"] = True
    low, high, best = 0, len(document["text"]), None
    while low <= high:
        middle = (low + high) // 2
        clipped["text"] = document["text"][:middle] + _MARKER
        if _size(clipped) <= remaining:
            best = deepcopy(clipped)
            low = middle + 1
        else:
            high = middle - 1
    return best


class KnowledgeBase:
    """Read-only copied documents; stable IDs and version identify evidence."""

    def __init__(self, documents: Iterable[dict] | None = None, *, version: str = "knowledge-v1"):
        if not isinstance(version, str) or not version.strip() or len(version) > 256:
            raise ValueError("version must contain 1..256 characters")
        self._version = version
        if documents is not None and isinstance(documents, (dict, str, bytes)):
            raise ValueError("documents must be an iterable of document objects")
        copied, identities = [], set()
        for document in documents if documents is not None else ():
            if not isinstance(document, dict) or not {"id", "text"} <= set(document) or set(document) - {"id", "text", "metadata"}:
                raise ValueError("A document requires id, text and optional metadata")
            identity, text, metadata = document["id"], document["text"], document.get("metadata", {})
            if not isinstance(identity, str) or not identity.strip() or len(identity) > 256:
                raise ValueError("Document id must contain 1..256 characters")
            if identity in identities:
                raise ValueError("Document ids must be unique")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Document text must be a nonempty string")
            if not isinstance(metadata, dict):
                raise ValueError("Document metadata must be a JSON object")
            copied.append(_clone_data({"id": identity, "text": text, "metadata": metadata}))
            identities.add(identity)
        self._documents = tuple(copied)
        self._index = tuple(_tokens(document["text"]) for document in self._documents)

    @property
    def version(self) -> str:
        return self._version

    @classmethod
    def from_file(cls, path: str | Path, *, version: str | None = None) -> "KnowledgeBase":
        """Load a JSON list, or {version, documents}; no network fetching."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "documents" not in payload or set(payload) - {"version", "documents"}:
                raise ValueError("Knowledge file requires documents and optional version")
            return cls(payload["documents"], version=version if version is not None
                       else payload.get("version", "knowledge-v1"))
        if not isinstance(payload, list):
            raise ValueError("Knowledge file must contain a document list or catalog")
        return cls(payload, version=version if version is not None else "knowledge-v1")

    def snapshot(self) -> dict:
        return {"version": self.version, "documents": deepcopy(list(self._documents))}

    def retrieve(self, query: str, *, top_k: int = 3, max_chars: int = 4000) -> dict:
        """Return real lexical matches with bounded compact-JSON document data.

        English whole words and Chinese adjacent-character pairs are matched.
        No unrelated documents fill an empty result. Truncated text is marked;
        metadata stays intact, and also consumes the visible-content budget.
        """
        top_k, max_chars = _count(top_k, "top_k"), _count(max_chars, "max_chars")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        query_tokens, ranked = _tokens(query, query=True), []
        if query_tokens and top_k and max_chars:
            for document, tokens in zip(self._documents, self._index):
                matched = query_tokens & tokens
                if matched:
                    ranked.append({**deepcopy(document), "score": len(matched) / len(query_tokens),
                                   "truncated": False, "original_chars": len(document["text"])})
        ranked.sort(key=lambda row: (-row["score"], row["id"]))
        documents, used, omitted = [], 0, 0
        for document in ranked:
            if len(documents) >= top_k:
                break
            visible = _budget_document(document, max_chars - used)
            if visible is None:
                omitted += 1
                continue
            documents.append(visible)
            used += _size(visible)
        return {"version": self.version, "query": query, "documents": documents,
                "max_chars": max_chars, "returned_chars": used, "omitted_count": omitted}

"""Strict JSON and durable atomic files for recording facts and projections."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
import tempfile
from typing import Any

SCHEMA_VERSION = 2
SHANGHAI = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")


class TaskRecordError(ValueError):
    """Invalid recording data, lifecycle or source integrity."""


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def timestamp(value: dt.datetime) -> str:
    if not isinstance(value, dt.datetime) or value.tzinfo is None:
        raise TaskRecordError("Recording clock must return a timezone-aware datetime")
    return value.astimezone(SHANGHAI).isoformat(timespec="microseconds")


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TaskRecordError("JSON object keys must be strings")
        return {key: json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return json_value(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TaskRecordError(f"Unsupported JSON value: {type(value).__name__}")


def encode(value: Any, *, pretty: bool = False) -> bytes:
    try:
        options = {"indent": 2} if pretty else {"separators": (",", ":")}
        text = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                          allow_nan=False, **options)
        return (text + ("\n" if pretty else "")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise TaskRecordError(f"Invalid recording JSON: {error}") from error


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TaskRecordError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(value: bytes | str) -> Any:
    def reject(constant):
        raise TaskRecordError(f"Non-finite JSON constant: {constant}")
    def finite_float(text):
        number = float(text)
        if not math.isfinite(number):
            raise TaskRecordError("Non-finite JSON float")
        return number
    try:
        return json.loads(value, object_pairs_hook=_object, parse_constant=reject, parse_float=finite_float)
    except (UnicodeError, ValueError, TypeError) as error:
        raise TaskRecordError(f"Invalid recording JSON: {error}") from error


def clone(value: Any) -> Any:
    return decode(encode(value))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_bytes(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Write and fsync a temporary file, then publish a complete file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # Hard-link publication is atomic and cannot overwrite an existing
            # source artifact/manifest. NTFS and ordinary Linux filesystems support it.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    atomic_bytes(path, encode(value, pretty=True), exclusive=exclusive)

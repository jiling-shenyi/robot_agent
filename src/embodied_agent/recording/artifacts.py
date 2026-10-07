"""Content-addressed immutable artifacts, published before journal references."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from embodied_agent.recording.common import TaskRecordError, atomic_bytes, decode, digest, encode


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def put(self, value: Any, *, media_type: str = "application/json") -> dict:
        if not isinstance(media_type, str) or not media_type.strip():
            raise TaskRecordError("Artifact media_type must be nonempty")
        binary = isinstance(value, (bytes, bytearray, memoryview))
        data = bytes(value) if binary else encode(value)
        if binary and media_type == "application/json":
            media_type = "application/octet-stream"
        checksum = digest(data)
        suffix = ".json" if media_type == "application/json" else ".bin"
        relative = Path("artifacts") / "sha256" / checksum[:2] / f"{checksum}{suffix}"
        path = self.root / relative
        reference = {"sha256": checksum, "size_bytes": len(data), "media_type": media_type,
                     "path": relative.as_posix()}
        if path.exists():
            self.get(reference)
            return reference
        try:
            atomic_bytes(path, data, exclusive=True)
        except FileExistsError:
            self.get(reference)  # A concurrent writer must have published identical bytes.
        return reference

    def get(self, reference: dict) -> Any:
        if not isinstance(reference, dict) or not {"sha256", "size_bytes", "media_type", "path"} <= reference.keys():
            raise TaskRecordError("Malformed artifact reference")
        checksum, size, media_type = reference["sha256"], reference["size_bytes"], reference["media_type"]
        if (not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or type(size) is not int or size < 0 or not isinstance(media_type, str)):
            raise TaskRecordError("Malformed artifact hash/size/type")
        suffix = ".json" if media_type == "application/json" else ".bin"
        expected = (Path("artifacts") / "sha256" / checksum[:2] / f"{checksum}{suffix}").as_posix()
        if reference["path"] != expected:
            raise TaskRecordError("Artifact path does not match its content address")
        path = (self.root / expected).resolve()
        if not path.is_relative_to(self.root):
            raise TaskRecordError("Artifact path escapes the records root")
        try:
            data = path.read_bytes()
        except OSError as error:
            raise TaskRecordError(f"Missing or unreadable artifact {expected}: {error}") from error
        if len(data) != size or digest(data) != checksum:
            raise TaskRecordError(f"Artifact integrity mismatch: {expected}")
        return decode(data) if media_type == "application/json" else data

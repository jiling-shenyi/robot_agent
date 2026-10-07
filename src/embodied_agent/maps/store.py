"""Revision-checked, atomic persistence for validated initial maps."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from embodied_agent.maps.schema import MAP_ID_PATTERN, WorldMap, WorldError, strict_json
from embodied_agent.paths import PROJECT_ROOT

DEFAULT_MAPS_DIR = PROJECT_ROOT / "configs" / "maps"

class MapStore:
    """Only direct, validated <map_id>.json children can be read or written.

    A short-lived exclusive lock serializes writers across threads/processes.
    The expected revision is checked *inside* that lock before atomic replace.
    A busy (or abandoned) lock fails explicitly instead of overwriting a map.
    """

    def __init__(self, root: str | Path = DEFAULT_MAPS_DIR):
        self.root = Path(root).resolve()

    def _path(self, map_id: str) -> Path:
        if not isinstance(map_id, str) or not MAP_ID_PATTERN.fullmatch(map_id):
            raise WorldError("INVALID_MAP_ID", "Invalid map ID")
        path = self.root / f"{map_id}.json"
        if path.resolve().parent != self.root or path.is_symlink():
            raise WorldError("INVALID_MAP_PATH", "Map files must be direct, non-symlink children of the map directory")
        return path

    def list_maps(self) -> list[WorldMap]:
        if not self.root.exists():
            return []
        maps = []
        for path in sorted(self.root.glob("*.json")):
            path = self._path(path.stem)
            if path.stat().st_size > 262144:
                raise WorldError("INVALID_MAP", "Map file exceeds the supported domain size limit")
            # The home domain has its own validator and execution adapter.
            # Keep malformed/unknown desktop maps visible as errors.
            payload = strict_json(path.read_text(encoding="utf-8"))
            if payload.get("schema_kind") == "home":
                continue
            maps.append(self.load(path.stem))
        return maps

    def load(self, map_id: str) -> WorldMap:
        path = self._path(map_id)
        try:
            if path.stat().st_size > 65536:
                raise WorldError("INVALID_MAP", "Map file exceeds 64 KiB")
            payload = strict_json(path.read_text(encoding="utf-8"))
            if payload.get("schema_kind") == "home":
                raise WorldError("INCOMPATIBLE_MAP", "Home maps require HomeMapStore and the home robot entry point")
            world = WorldMap.from_dict(payload)
        except FileNotFoundError as exc:
            raise WorldError("MAP_NOT_FOUND", f"Map {map_id!r} does not exist") from exc
        if world.map_id != map_id:
            raise WorldError("INVALID_MAP", "Map ID must match its filename")
        return world

    def save(self, world: WorldMap, expected_revision: int | None = None) -> WorldMap:
        world = WorldMap.from_dict(world.to_dict())
        expected = world.revision if expected_revision is None else expected_revision
        if type(expected) is not int or expected < 0:
            raise WorldError("INVALID_MAP", "expected_revision must be a nonnegative integer")
        path = self._path(world.map_id)
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f".{world.map_id}.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise WorldError("MAP_BUSY", "Another edit owns this map's lock; retry after it finishes") from exc
        temporary_path: Path | None = None
        try:
            os.close(lock_fd)
            current_revision = self.load(world.map_id).revision if path.exists() else 0
            if current_revision != expected:
                raise WorldError("REVISION_CONFLICT", f"Map changed during editing: expected revision {expected}, found {current_revision}")
            updated = replace(world, revision=current_revision + 1)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=self.root, prefix=f".{world.map_id}.", suffix=".tmp", delete=False) as stream:
                temporary_path = Path(stream.name)
                json.dump(updated.to_dict(), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
            return updated
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)

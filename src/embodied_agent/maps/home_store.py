"""Separate atomic initial-home-map storage; runtime states are never saved."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.schema import MAP_ID_PATTERN, WorldError, strict_json
from embodied_agent.paths import PROJECT_ROOT

DEFAULT_HOME_MAPS_DIR = PROJECT_ROOT / "configs" / "maps"


class HomeMapStore:
    def __init__(self, root: str | Path = DEFAULT_HOME_MAPS_DIR):
        self.root = Path(root).resolve()

    def _path(self, map_id: str) -> Path:
        if not isinstance(map_id, str) or not MAP_ID_PATTERN.fullmatch(map_id):
            raise WorldError("INVALID_MAP_ID", "Invalid home map ID")
        path = self.root / f"{map_id}.json"
        if path.is_symlink() or path.resolve().parent != self.root:
            raise WorldError("INVALID_MAP_PATH", "Home maps must be direct non-symlink children")
        return path

    def load(self, map_id: str = "home_living_room") -> HomeWorld:
        path = self._path(map_id)
        try:
            if path.stat().st_size > 262144:
                raise WorldError("INVALID_HOME_MAP", "Home map exceeds 256 KiB")
            payload = strict_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorldError("MAP_NOT_FOUND", f"Home map {map_id!r} does not exist") from exc
        if payload.get("schema_kind") != "home":
            raise WorldError("INCOMPATIBLE_MAP", "This map is not a home map")
        world = HomeWorld.from_dict(payload)
        if world.map_id != map_id:
            raise WorldError("INVALID_HOME_MAP", "Home map ID must match the filename")
        return world

    def list_maps(self) -> list[HomeWorld]:
        if not self.root.exists():
            return []
        result = []
        for path in sorted(self.root.glob("*.json")):
            path = self._path(path.stem)
            if path.stat().st_size > 262144:
                raise WorldError("INVALID_HOME_MAP", "Map exceeds the supported domain size limit")
            payload = strict_json(path.read_text(encoding="utf-8"))
            if payload.get("schema_kind") == "home":
                result.append(self.load(path.stem))
        return result

    def save(self, world: HomeWorld, expected_revision: int | None = None) -> HomeWorld:
        """Trusted initial-map editor; execution never calls this method."""
        world = HomeWorld.from_dict(world.to_dict())
        expected = world.revision if expected_revision is None else expected_revision
        if type(expected) is not int or expected < 0:
            raise WorldError("INVALID_HOME_MAP", "Expected revision must be a nonnegative integer")
        path = self._path(world.map_id)
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / f".{world.map_id}.home.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise WorldError("MAP_BUSY", "Another editor owns the home map lock") from exc
        temporary: Path | None = None
        try:
            os.close(fd)
            revision = self.load(world.map_id).revision if path.exists() else 0
            if revision != expected:
                raise WorldError("REVISION_CONFLICT", f"Expected revision {expected}, found {revision}")
            updated = replace(world, revision=revision + 1)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=self.root, prefix=f".{world.map_id}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(updated.to_dict(), stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            return updated
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)

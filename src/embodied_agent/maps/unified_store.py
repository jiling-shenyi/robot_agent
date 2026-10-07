"""The existing Demo's map catalog, preserving each domain's own schema."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.home_store import HomeMapStore
from embodied_agent.maps.schema import WorldError, WorldMap, strict_json
from embodied_agent.maps.store import DEFAULT_MAPS_DIR, MapStore

DemoWorld = HomeWorld | WorldMap


def parse_demo_world(payload: dict[str, Any]) -> DemoWorld:
    return HomeWorld.from_dict(payload) if payload.get("schema_kind") == "home" else WorldMap.from_dict(payload)


class UnifiedMapStore:
    """One catalog for map selection; storage retains revision checks by type."""
    def __init__(self, root: str | Path = DEFAULT_MAPS_DIR):
        self.root = Path(root).resolve()
        self.desktop = MapStore(self.root)
        self.home = HomeMapStore(self.root)

    def load(self, map_id: str) -> DemoWorld:
        path = self.desktop._path(map_id)
        try:
            if path.stat().st_size > 262144:
                raise WorldError("INVALID_MAP", "Map exceeds the supported domain size limit")
            payload = strict_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorldError("MAP_NOT_FOUND", f"Map {map_id!r} does not exist") from exc
        if payload.get("schema_kind") != "home" and path.stat().st_size > 65536:
            raise WorldError("INVALID_MAP", "Desktop map exceeds 64 KiB")
        world = parse_demo_world(payload)
        if world.map_id != map_id:
            raise WorldError("INVALID_MAP", "Map ID must match its filename")
        return world

    def list_maps(self) -> list[DemoWorld]:
        return [self.load(path.stem) for path in sorted(self.root.glob("*.json"))] if self.root.exists() else []

    def save(self, world: DemoWorld, expected_revision: int | None = None) -> DemoWorld:
        if isinstance(world, HomeWorld):
            return self.home.save(world, expected_revision)
        if isinstance(world, WorldMap):
            return self.desktop.save(world, expected_revision)
        raise WorldError("INVALID_MAP", "Demo maps must use a registered domain schema")

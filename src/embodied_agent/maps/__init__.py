"""Validated initial maps, atomic storage, and lazy scene integration."""

from embodied_agent.maps.schema import (
    BoxRegion, WorldMap, WorldError, finite_vector, strict_json,
    TABLE_CENTER, TABLE_HALF_SIZE, TABLE_TOP_Z, CUBE_HALF_SIZE,
)
from embodied_agent.maps.store import MapStore, DEFAULT_MAPS_DIR
from embodied_agent.maps.scene import apply_world_to_model
from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.home_store import HomeMapStore

__all__ = [
    "BoxRegion", "WorldMap", "WorldError", "MapStore", "DEFAULT_MAPS_DIR",
    "finite_vector", "strict_json", "apply_world_to_model",
    "TABLE_CENTER", "TABLE_HALF_SIZE", "TABLE_TOP_Z", "CUBE_HALF_SIZE",
    "HomeWorld", "HomeMapStore",
]

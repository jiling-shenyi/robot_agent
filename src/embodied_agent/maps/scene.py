"""Apply initial-map geometry to MuJoCo without importing it until use."""

from __future__ import annotations

import math
from typing import Any

from embodied_agent.maps.schema import WorldMap, WorldError

def apply_world_to_model(model: Any, world: WorldMap) -> None:
    """Apply static geometry before reset/forward and Episode geometry caches.

    The caller supplies world.cube_position_m as the scenario's initial cube
    pose; no active simulation data or robot state is modified here.
    """
    import mujoco

    world = WorldMap.from_dict(world.to_dict())
    for unused in {"target_a", "target_b"} - set(world.targets):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, unused)
        if geom_id >= 0:
            model.geom_rgba[geom_id, 3] = 0
    regions = {"danger_zone": world.danger_zone, **world.targets}
    ids: dict[str, int] = {}
    for name in regions:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX or model.geom_bodyid[geom_id] != 0:
            raise WorldError("INCOMPATIBLE_SCENE", f"Scene requires a world-body box geom named {name}")
        ids[name] = geom_id
    for name, region in regions.items():
        geom_id = ids[name]
        model.geom_pos[geom_id] = region.position_m
        model.geom_size[geom_id] = region.half_size_m
        model.geom_rbound[geom_id] = math.sqrt(sum(value * value for value in region.half_size_m))

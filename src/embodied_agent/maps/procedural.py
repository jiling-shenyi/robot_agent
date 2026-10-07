"""Deterministic same-domain layouts without hand-authored operation points."""
from __future__ import annotations

import random

from .home_schema import HomeWorld, RULE_VERSION


def generate_home_world(seed: int = 0, *, target: str = "floor", source: str = "table") -> HomeWorld:
    if type(seed) is not int or not 0 <= seed <= 999999:
        raise ValueError("seed must be an integer in 0..999999")
    if target not in {"floor", "sofa", "platform", "table"} or source not in {"table", "floor"}:
        raise ValueError("supported targets are floor/sofa/platform/table; sources table/floor")
    rng = random.Random(seed)
    source_x, source_y = -1. + rng.uniform(-.12, .12), .8 + rng.uniform(-.08, .08)
    source_height = rng.choice((.45, .60, .72))
    furniture = {"workbench": {"kind": "workbench", "geometry_kind": "table",
        "position_m": [source_x, source_y, source_height / 2], "half_size_m": [.55, .45, source_height / 2]}}
    if target != "floor":
        height = .80 if target == "sofa" else (.42 if target == "platform" else .65)
        furniture["destination"] = {"kind": "couch" if target == "sofa" else target,
            "geometry_kind": "sofa" if target == "sofa" else ("box" if target == "platform" else "table"),
            "position_m": [1.35, .8, height / 2], "half_size_m": [.5, .4, height / 2]}
    position = [source_x, source_y - .32, source_height + .025] if source == "table" else [0., -.65, .025]
    return HomeWorld.from_dict({"schema_version": 1, "schema_kind": "home", "rule_version": RULE_VERSION,
        "map_id": f"generated_{source}_{target}_{seed}", "name": f"Generated {source} to {target}",
        "description": "Seeded proxy-box manipulation benchmark; geometry and skill constraints still apply.",
        "revision": 0, "world_version": 0,
        "room": {"size_m": [6., 5., 2.4], "doorway_position_m": [0., -2.5, 0.], "doorway_width_m": 1.2},
        "navigation": {"resolution_m": .1, "robot_radius_m": .30,
            "initial_position_m": [0., -1.4, 0.], "initial_yaw_rad": 0.},
        "furniture": furniture,
        "objects": {"parcel": {"category": "personal_item", "position_m": position,
            "half_size_m": [.025, .025, .025], "mass_kg": .08,
            "support": "workbench" if source == "table" else "floor",
            "operations": ["inspect", "pick", "carry", "place"], "risk_tags": [], "states": {},
            "description": "Free-body proxy object; no welded grasp or pose assignment."}}})

"""Frozen and seed-derived episode scenarios and JSON configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import PROJECT_ROOT as ROOT

SCENARIOS_PATH = ROOT / "configs" / "m2_scenarios.json"
THRESHOLDS_PATH = ROOT / "configs" / "m2_thresholds.json"

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_scenario(seed: int, target: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    for item in entries:
        if int(item["seed"]) == seed and item["target"] == target:
            return dict(item)
    rng = np.random.default_rng(seed)
    return {
        "scenario_id": f"adhoc_seed_{seed}_{target}",
        "seed": seed,
        "target": target,
        "cube_position_m": [
            round(float(rng.uniform(0.400, 0.440)), 6),
            round(float(rng.uniform(-0.290, -0.250)), 6),
            0.425,
        ],
    }

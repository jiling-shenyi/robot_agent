"""Conservative deterministic path prechecks for allow-listed Panda skills."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..contracts import ContractError

def _point_to_box_distance(point: np.ndarray, center: np.ndarray, half_size: np.ndarray) -> float:
    outside = np.maximum(np.abs(point - center) - half_size, 0.0)
    return float(np.linalg.norm(outside))


def check_skill_path(episode: Any, skill: str, config: dict[str, Any]) -> dict[str, Any]:
    """Conservatively sample the mocap path against an expanded visible danger box."""
    start = episode.data.mocap_pos[episode.mocap_id].copy()
    cube = episode.data.xpos[episode.cube_body_id].copy()
    center = episode.data.geom_xpos[episode.danger_zone_geom_id].copy()
    half = episode.model.geom_size[episode.danger_zone_geom_id].copy()
    path_config = config["path_precheck"]
    expansion = float(path_config["danger_zone_clearance_m"]) + float(
        path_config["ee_payload_envelope_radius_m"]
    )
    expanded_half = half + expansion
    spacing = float(path_config["sample_spacing_m"])
    if skill == "pick":
        pregrasp = cube.copy()
        pregrasp[2] += float(episode.thresholds["pregrasp_clearance_m"])
        grasp = cube.copy()
        lift = grasp.copy()
        lift[2] += float(episode.thresholds["lift_height_m"])
        waypoints = [start, pregrasp, grasp, lift]
    elif skill == "place":
        transit = start.copy()
        transit[:2] = episode.target_center[:2]
        place = transit.copy()
        place[2] = (
            episode.table_top_z
            + float(episode.cube_half_size[2])
            + float(episode.thresholds["place_clearance_m"])
        )
        retreat = place.copy()
        retreat[2] += float(episode.thresholds["pregrasp_clearance_m"])
        waypoints = [start, transit, place, retreat]
    else:
        raise ContractError("INVALID_PLAN", f"No path generator exists for skill {skill}")

    minimum_distance = math.inf
    checked_samples = 0
    for first, second in zip(waypoints, waypoints[1:]):
        length = float(np.linalg.norm(second - first))
        count = max(1, int(math.ceil(length / spacing)))
        for index in range(count + 1):
            point = first + (second - first) * (index / count)
            distance = _point_to_box_distance(point, center, expanded_half)
            minimum_distance = min(minimum_distance, distance)
            checked_samples += 1
            if distance <= 0.0:
                raise ContractError(
                    "PATH_REJECTED",
                    f"{skill} mocap path enters danger-zone envelope "
                    f"(centerline clearance={distance:.6f} m at {point.tolist()})",
                )
    return {
        "skill": skill,
        "waypoints_m": [point.tolist() for point in waypoints],
        "danger_zone_expansion_m": expansion,
        "minimum_sampled_centerline_clearance_m": minimum_distance,
        "samples_checked": checked_samples,
    }

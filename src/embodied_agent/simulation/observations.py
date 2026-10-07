"""Convert live simulator evidence into the shared pure Observation contract."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ..contracts import Observation

def make_observation(
    episode: Any,
    obs_id: int,
    completed_steps: list[str],
    previous_error: dict[str, str] | None = None,
) -> Observation:
    data, model = episode.data, episode.model
    ee_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(ee_quaternion, data.site_xmat[episode.ee_site_id])
    cube_velocity, cube_angular_velocity = episode.cube_speed()
    cube_position = data.xpos[episode.cube_body_id].copy()
    target_regions: dict[str, dict[str, Any]] = {}
    for name in getattr(getattr(episode, "world", None), "targets", ("target_a", "target_b")):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"Missing target region: {name}")
        target_regions[name] = {
            "center_xyz_m": data.geom_xpos[geom_id].copy().tolist(),
            "half_size_xyz_m": model.geom_size[geom_id].copy().tolist(),
            "aliases": ["green"] if name == "target_a" else ["blue"],
        }
    zone_id = episode.danger_zone_geom_id
    side_count = len(episode.finger_contact_sides())
    lifted = cube_position[2] >= episode.table_top_z + episode.cube_half_size[2] + 0.03
    held: bool | None
    if side_count >= 2 and lifted and episode.grasp_relative_position is not None:
        held = True
    elif not lifted:
        held = False
    else:
        held = None
    return Observation(
        obs_id=obs_id,
        sim_time_s=float(data.time),
        ee_pose_actual={
            "position_xyz_m": data.site_xpos[episode.ee_site_id].copy().tolist(),
            "quaternion_wxyz": ee_quaternion.tolist(),
        },
        gripper_opening_m=data.qpos[episode.finger_qpos_addresses].copy().tolist(),
        cube={
            "object_id": "cube",
            "position_xyz_m": cube_position.tolist(),
            "linear_speed_m_s": cube_velocity,
            "angular_speed_rad_s": cube_angular_velocity,
            "held_estimate": held,
            "finger_contact_side_count": side_count,
        },
        target_regions=target_regions,
        danger_zone={
            "center_xyz_m": data.geom_xpos[zone_id].copy().tolist(),
            "half_size_xyz_m": model.geom_size[zone_id].copy().tolist(),
            "required_clearance_m": episode.danger_zone_clearance_m,
        },
        held_estimate=held,
        completed_steps=list(completed_steps),
        previous_error=previous_error,
    )

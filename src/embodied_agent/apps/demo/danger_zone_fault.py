"""Drive the Panda through M2's real control path and verify danger-zone stopping."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from embodied_agent.simulation.episode import Episode
from embodied_agent.simulation.errors import M2Failure
from embodied_agent.simulation.scenarios import load_json
from embodied_agent.simulation.model import name_id
from embodied_agent.evaluation.evidence import jsonable, sha256


from embodied_agent.paths import PROJECT_ROOT as ROOT
from embodied_agent.evaluation.provenance import source_hashes
SCENE = ROOT / "assets" / "scene" / "panda_task.xml"
THRESHOLDS_PATH = ROOT / "configs" / "m2_thresholds.json"
EXPECTED_CENTER_M = np.array([0.82, 0.28, 0.535], dtype=np.float64)
EXPECTED_HALF_SIZE_M = np.array([0.06, 0.06, 0.135], dtype=np.float64)
EXPECTED_RGBA = np.array([0.95, 0.08, 0.05, 0.34], dtype=np.float64)


def inspect_m1_danger_zone(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    zone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "danger_zone")
    checks: dict[str, bool] = {"named_geom_exists": zone_id >= 0}
    if zone_id < 0:
        return {"checks": checks, "passed": False}

    table_id = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    world_id = name_id(model, mujoco.mjtObj.mjOBJ_BODY, "world")
    center = model.geom_pos[zone_id].copy()
    half_size = model.geom_size[zone_id].copy()
    rgba = model.geom_rgba[zone_id].copy()
    table_center = data.geom_xpos[table_id]
    table_half_size = model.geom_size[table_id]
    table_top_z = float(table_center[2] + table_half_size[2])
    bottom_z = float(center[2] - half_size[2])
    checks.update(
        {
            "fixed_in_world": int(model.geom_bodyid[zone_id]) == world_id,
            "box_matches_visible_volume": int(model.geom_type[zone_id])
            == int(mujoco.mjtGeom.mjGEOM_BOX),
            "expected_center": bool(np.allclose(center, EXPECTED_CENTER_M, atol=1e-9, rtol=0.0)),
            "expected_half_size": bool(
                np.allclose(half_size, EXPECTED_HALF_SIZE_M, atol=1e-9, rtol=0.0)
            ),
            "expected_red_translucent_color": bool(
                np.allclose(rgba, EXPECTED_RGBA, atol=1e-6, rtol=0.0)
            ),
            "visual_only_no_physics_contact": int(model.geom_contype[zone_id]) == 0
            and int(model.geom_conaffinity[zone_id]) == 0,
            "axis_aligned": bool(
                np.allclose(model.geom_quat[zone_id], [1.0, 0.0, 0.0, 0.0], atol=1e-9, rtol=0.0)
            ),
            "bottom_aligned_with_tabletop": abs(bottom_z - table_top_z) <= 1e-9,
            "inside_table_footprint": bool(
                np.all(center[:2] - half_size[:2] >= table_center[:2] - table_half_size[:2])
                and np.all(center[:2] + half_size[:2] <= table_center[:2] + table_half_size[:2])
            ),
            "finite_positive_volume": bool(np.isfinite(half_size).all() and np.all(half_size > 0)),
        }
    )
    return {
        "geom": "danger_zone",
        "world_center_m": center.tolist(),
        "half_size_m": half_size.tolist(),
        "rgba": rgba.tolist(),
        "bottom_z_m": bottom_z,
        "table_top_z_m": table_top_z,
        "collision_masks": {
            "contype": int(model.geom_contype[zone_id]),
            "conaffinity": int(model.geom_conaffinity[zone_id]),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "m2" / "danger_zone_fault_injection",
        help="directory for the expected-failure record (default: results/m2/danger_zone_fault_injection)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace summary.json and episodes.jsonl in the output directory",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a MuJoCo Viewer (the default opens and keeps the Viewer visible)",
    )
    args = parser.parse_args()
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    evidence_paths = [output_dir / "summary.json", output_dir / "episodes.jsonl"]
    if not args.overwrite and any(path.exists() for path in evidence_paths):
        parser.error(f"Evidence already exists in {output_dir}; choose another --output or pass --overwrite")

    os.chdir(ROOT)
    model = mujoco.MjModel.from_xml_path("assets/scene/panda_task.xml")
    data = mujoco.MjData(model)
    key_id = name_id(model, mujoco.mjtObj.mjOBJ_KEY, "home_scene")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    m1_validation = inspect_m1_danger_zone(model, data)
    zone_id = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "danger_zone")
    target_xyz = data.geom_xpos[zone_id].copy()

    report: dict[str, Any] = {
        "source_hashes": source_hashes(ROOT),
        "phase": "M1/M2 danger-zone robot-motion fault injection",
        "scene_xml": "assets/scene/panda_task.xml",
        "scene_xml_sha256": sha256(SCENE),
        "m1_danger_zone_validation": m1_validation,
        "fault_injection": {
            "expected_error_code": "DANGER_ZONE_VIOLATION",
            "observed_error_code": None,
            "control_path": "Episode.prepare -> Episode.move_ee -> Episode.step -> Episode.check_step_safety -> Episode.check_safety",
            "target_xyz_m": target_xyz.tolist(),
            "ee_start_xyz_m": None,
            "ee_at_stop_xyz_m": None,
            "ee_displacement_m": None,
            "step_count": 0,
            "danger_zone_clear": None,
            "violation": None,
            "robot_geom_triggered": False,
        },
        "events": [],
        "passed": False,
    }

    episode: Episode | None = None
    unexpected_error: str | None = None
    if m1_validation["passed"]:
        try:
            thresholds = load_json(THRESHOLDS_PATH)
            scenario = {
                "scenario_id": "danger_zone_robot_motion_fault_injection",
                "seed": -1,
                "target": "a",
                "cube_position_m": [0.42, -0.27, 0.425],
            }
            episode = Episode(scenario, thresholds, output_dir, enable_viewer=not args.headless)
            episode.prepare()
            ee_start = episode.data.site_xpos[episode.ee_site_id].copy()

            if episode.viewer is not None and episode.viewer.is_running():
                with episode.viewer.lock():
                    episode.viewer.cam.lookat[:] = np.array([0.70, 0.16, 0.49])
                    episode.viewer.cam.distance = 1.15
                    episode.viewer.cam.azimuth = 135.0
                    episode.viewer.cam.elevation = -25.0
                episode.set_viewer_status(
                    "PANDA DANGER-ZONE MOTION",
                    "Moving the real arm toward the red volume; M2 checks every physics step",
                )

            observed_code: str | None = None
            observed_message: str | None = None
            try:
                episode.move_ee("danger_zone_fault_injection", target_xyz)
            except M2Failure as error:
                observed_code = error.code
                observed_message = str(error)
                episode.record_failure(error.code, str(error))

            ee_at_stop = episode.data.site_xpos[episode.ee_site_id].copy()
            displacement = float(np.linalg.norm(ee_at_stop - ee_start))
            violation = episode.danger_zone_violation
            robot_geom_triggered = bool(
                violation is not None
                and not str(violation.get("geom", "")).startswith("cube:")
            )
            report["fault_injection"].update(
                {
                    "observed_error_code": observed_code,
                    "observed_error_message": observed_message,
                    "ee_start_xyz_m": ee_start.tolist(),
                    "ee_at_stop_xyz_m": ee_at_stop.tolist(),
                    "ee_displacement_m": displacement,
                    "step_count": episode.total_steps,
                    "sim_time_s": float(episode.data.time),
                    "danger_zone_clear": violation is None,
                    "violation": jsonable(violation),
                    "robot_geom_triggered": robot_geom_triggered,
                }
            )
            report["events"] = jsonable(episode.events)
            report["passed"] = bool(
                observed_code == "DANGER_ZONE_VIOLATION"
                and violation is not None
                and robot_geom_triggered
                and displacement > 0.10
                and not report["fault_injection"]["danger_zone_clear"]
            )
            if report["passed"]:
                geom = violation["geom"]
                distance_mm = float(violation["distance_m"]) * 1000.0
                episode.set_viewer_status(
                    "DANGER_ZONE_VIOLATION TRIGGERED",
                    f"{geom} moved {displacement * 1000:.0f} mm; stopped at {distance_mm:.2f} mm clearance",
                )
            elif episode.viewer is not None:
                episode.set_viewer_status(
                    "DANGER-ZONE DEMO FAILED",
                    observed_message or "The robot did not trigger the expected guarded failure",
                )
        except Exception as error:
            unexpected_error = f"{type(error).__name__}: {error}"
            report["unexpected_error"] = unexpected_error

    output_dir.mkdir(parents=True, exist_ok=True)
    if episode is not None and episode.events:
        payload = {"scenario_id": "danger_zone_robot_motion_fault_injection", **report}
        evidence_paths[1].write_text(
            json.dumps(jsonable(payload), ensure_ascii=False) + "\n", encoding="utf-8"
        )
    else:
        evidence_paths[1].write_text("", encoding="utf-8")
    evidence_paths[0].write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"M1 danger-zone configuration: {'PASS' if m1_validation['passed'] else 'FAIL'}")
    for name, passed in m1_validation["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(
        "M2 robot-motion fault injection: "
        f"expected DANGER_ZONE_VIOLATION, observed "
        f"{report['fault_injection']['observed_error_code'] or unexpected_error or 'no violation'}"
    )
    if report["fault_injection"]["violation"]:
        violation = report["fault_injection"]["violation"]
        print(
            f"  geom={violation['geom']} separation={violation['distance_m'] * 1000:.3f} mm "
            f"required={violation['required_clearance_m'] * 1000:.1f} mm"
        )
        print(f"  actual Panda hand displacement={report['fault_injection']['ee_displacement_m'] * 1000:.1f} mm")
    print(f"Result: {'PASS' if report['passed'] else 'FAIL'}; evidence: {output_dir}")
    if episode is not None:
        episode.wait_for_viewer_close()
    return 0 if report["passed"] else 1

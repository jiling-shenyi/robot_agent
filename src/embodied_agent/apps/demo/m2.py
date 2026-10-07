"""Run deterministic MuJoCo Panda pick-and-place episodes for M2."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from embodied_agent.paths import PROJECT_ROOT as ROOT
from embodied_agent.evaluation.provenance import source_hashes
from embodied_agent.simulation.episode import Episode
from embodied_agent.simulation.errors import M2Failure, ViewerClosed
from embodied_agent.simulation.model import SCENE, SAMPLE_EVERY_STEPS, WARNING_TYPES, load_scene_model, name_id, warning_counts
from embodied_agent.simulation.scenarios import SCENARIOS_PATH, THRESHOLDS_PATH, load_json, make_scenario
from embodied_agent.evaluation.evidence import jsonable, sha256

def run_one(
    scenario: dict[str, Any],
    thresholds: dict[str, Any],
    output_dir: Path,
    enable_viewer: bool = False,
    show_ui: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode = Episode(scenario, thresholds, output_dir, enable_viewer, show_ui)
    result = episode.run()
    episode.wait_for_viewer_close()
    return result, episode.samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--batch", action="store_true", help="run the frozen 10-scene list")
    selection.add_argument("--target", choices=("a", "b"), help="run one episode for target a or b")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without the live viewer (single episodes show it by default; batches are headless by default)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="show the live viewer for each episode in a batch",
    )
    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="show MuJoCo's native side panels with the live viewer",
    )
    parser.add_argument("--seed", type=int, default=0, help="scenario seed for a single episode")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "m2",
        help="output directory (existing evidence files are never overwritten by default)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace episodes.jsonl, trajectory.csv, and summary.json in the output directory",
    )
    args = parser.parse_args()
    if args.headless and args.viewer:
        parser.error("--headless and --viewer cannot be used together")
    if args.viewer and not args.batch:
        parser.error("--viewer is only needed with --batch; single episodes show the viewer by default")

    os.chdir(ROOT)
    scenario_data = load_json(SCENARIOS_PATH)
    thresholds = load_json(THRESHOLDS_PATH)
    if args.batch:
        if scenario_data.get("status") != "frozen":
            parser.error("--batch requires configs/m2_scenarios.json status=frozen")
        if thresholds.get("status") != "frozen":
            parser.error("--batch requires configs/m2_thresholds.json status=frozen")
        scenarios = list(scenario_data["scenarios"])
        if len(scenarios) != 10:
            parser.error(f"M2 acceptance requires exactly 10 scenarios, found {len(scenarios)}")
        if sum(item["target"] == "a" for item in scenarios) < 5 or sum(
            item["target"] == "b" for item in scenarios
        ) < 5:
            parser.error("The frozen scenario list must include at least five episodes per target")
    else:
        scenarios = [make_scenario(args.seed, args.target, scenario_data.get("scenarios", []))]
    enable_viewer = not args.headless and (not args.batch or args.viewer)

    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / "episodes.jsonl", output_dir / "trajectory.csv", output_dir / "summary.json"]
    if not args.overwrite and any(path.exists() for path in output_paths):
        parser.error(
            f"Output evidence already exists in {output_dir}; choose another --output or pass --overwrite"
        )

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    episode_results: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(
            f"[{scenario['scenario_id']}] target={scenario['target']} seed={scenario['seed']} "
            f"cube={scenario['cube_position_m']}"
        )
        result, samples = run_one(
            scenario,
            thresholds,
            output_dir,
            enable_viewer=enable_viewer,
            show_ui=args.show_ui,
        )
        result["run_id"] = run_id
        episode_results.append(result)
        all_samples.extend(
            {"run_id": run_id, "scenario_id": scenario["scenario_id"], **sample}
            for sample in samples
        )
        print(
            f"  {result['status']} {result['error_code'] or ''}; "
            f"steps={result['step_count']} max_ee_error={result['max_ee_position_error_m']:.6f} m"
        )

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for result in episode_results:
            stream.write(json.dumps(jsonable(result), ensure_ascii=False, separators=(",", ":")) + "\n")
    trajectory_fields = [
        "run_id",
        "scenario_id",
        "phase",
        "step",
        "sim_time_s",
        "ee_target_xyz_m",
        "ee_actual_xyz_m",
        "ee_position_error_m",
        "cube_xyz_m",
        "cube_linear_speed_m_s",
        "cube_angular_speed_rad_s",
        "gripper_ctrl",
        "finger_qpos_m",
        "contacts",
        "minimum_contact_distance_so_far_m",
        "minimum_danger_zone_separation_so_far_m",
        "danger_zone_clear",
        "warnings",
        "finite_state",
    ]
    with (output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=trajectory_fields)
        writer.writeheader()
        for sample in all_samples:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in sample.items()
                }
            )

    model = mujoco.MjModel.from_xml_path("assets/scene/panda_task.xml")
    danger_zone_geom_id = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "danger_zone")
    summary = {
        "source_hashes": source_hashes(ROOT),
        "run_id": run_id,
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": "M2",
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "scene_xml": "assets/scene/panda_task.xml",
        "scene_xml_sha256": sha256(SCENE),
        "panda_mocap_xml_sha256": sha256(
            ROOT / "assets" / "third_party" / "franka_emika_panda" / "panda_mocap.xml"
        ),
        "scenario_config": "configs/m2_scenarios.json",
        "scenario_config_sha256": sha256(SCENARIOS_PATH),
        "threshold_config": "configs/m2_thresholds.json",
        "threshold_config_sha256": sha256(THRESHOLDS_PATH),
        "danger_zone": {
            "geom": "danger_zone",
            "center_m": model.geom_pos[danger_zone_geom_id].tolist(),
            "half_size_m": model.geom_size[danger_zone_geom_id].tolist(),
            "clearance_m": float(thresholds["danger_zone_clearance_m"]),
            "validated_geoms": "Panda collision geoms and cube_geom, every physics step",
        },
        "scenario_count": len(episode_results),
        "success_count": sum(result["status"] == "SUCCESS" for result in episode_results),
        "failure_count": sum(result["status"] != "SUCCESS" for result in episode_results),
        "all_pass": all(result["status"] == "SUCCESS" for result in episode_results),
        "episodes": [
            {
                "scenario_id": result["scenario_id"],
                "seed": result["seed"],
                "target": result["target"],
                "status": result["status"],
                "error_code": result["error_code"],
                "step_count": result["step_count"],
                "max_ee_position_error_m": result["max_ee_position_error_m"],
                "minimum_contact_distance_m": result["minimum_contact_distance_m"],
                "penetration_exceptions_seen_m": result["penetration_exceptions_seen_m"],
                "max_unilateral_contact_steps": result["max_unilateral_contact_steps"],
                "minimum_danger_zone_separation_m": result[
                    "minimum_danger_zone_separation_m"
                ],
                "danger_zone_violation": result["danger_zone_violation"],
                "stable_steps_observed": result["stable_steps_observed"],
                "postconditions": result["postconditions"],
            }
            for result in episode_results
        ],
        "model_dimensions": {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nu": int(model.nu),
            "neq": int(model.neq),
            "timestep_s": float(model.opt.timestep),
        },
        "thresholds": thresholds,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"M2 result: {summary['success_count']}/{summary['scenario_count']} passed. "
        f"Evidence: {output_dir}"
    )
    return 0 if summary["all_pass"] else 1

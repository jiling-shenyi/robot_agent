"""Run one bounded natural-language Panda pick/place task or the frozen M3 set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mujoco
from dotenv import load_dotenv

from embodied_agent.contracts import ContractError, resolve_task
from embodied_agent.planner import DeepSeekPlanner, PlannerError, StubPlanner
from embodied_agent.runtime import M3EpisodeRunner, M3RunWriter, sha256
from m2_pick_place import Episode, load_json, make_scenario


def _manifest(run_id: str, planner_kind: str, runtime: dict[str, Any]) -> dict[str, Any]:
    prompt = runtime["planner"]["system_prompt"].encode("utf-8")
    return {
        "phase": "M3",
        "planner_kind": planner_kind,
        "requested_model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        if planner_kind == "llm"
        else None,
        "python_version": sys.version.split()[0],
        "mujoco_version": mujoco.__version__,
        "scene_xml": "assets/scene/panda_task.xml",
        "scene_xml_sha256": sha256(ROOT / "assets" / "scene" / "panda_task.xml"),
        "scenario_config_sha256": sha256(ROOT / "configs" / "m2_scenarios.json"),
        "threshold_config_sha256": sha256(ROOT / "configs" / "m2_thresholds.json"),
        "runtime_config": "configs/m3_runtime.json",
        "runtime_config_sha256": sha256(ROOT / "configs" / "m3_runtime.json"),
        "case_config_sha256": sha256(ROOT / "configs" / "m3_cases.json"),
        "prompt_version": runtime["planner"]["system_prompt_version"],
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "schema_version": runtime["planner"]["schema_version"],
        "budgets": runtime["budgets"],
        "path_precheck": runtime["path_precheck"],
        "credential_configured": bool(os.getenv("DEEPSEEK_API_KEY")) if planner_kind == "llm" else None,
    }


def _make_planner(kind: str, runtime: dict[str, Any]):
    if kind == "stub":
        return StubPlanner()
    try:
        return DeepSeekPlanner(runtime)
    except PlannerError as exc:
        unavailable_error = exc

        class UnavailablePlanner:
            kind = "llm"

            def plan(self, goal: dict[str, Any], observation: dict[str, Any]):
                raise unavailable_error

        return UnavailablePlanner()


def _select_scenario(
    instruction: str,
    seed: int,
    runtime: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any] | None:
    goal = resolve_task(instruction, runtime)
    return make_scenario(seed, goal.target_id.removeprefix("target_"), scenarios)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--instruction", help="one explicit cube-to-target instruction")
    select.add_argument("--batch", action="store_true", help="run the frozen 20-case M3 set")
    parser.add_argument("--planner", choices=("llm", "stub"), default="llm")
    parser.add_argument("--seed", type=int, default=0, help="frozen or generated scenario seed for one task")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable the live viewer (single tasks show it by default; batch runs are headless by default)",
    )
    parser.add_argument("--viewer", action="store_true", help="show the live viewer in batch mode")
    parser.add_argument("--show-ui", action="store_true", help="show MuJoCo's native side panels")
    parser.add_argument(
        "--output",
        type=Path,
        help="new output directory (default: results/m3/<run_id>; existing evidence is never overwritten)",
    )
    args = parser.parse_args()
    if args.headless and args.viewer:
        parser.error("--headless and --viewer cannot be used together")
    if args.viewer and not args.batch:
        parser.error("--viewer is only used for batch mode; single tasks show the viewer by default")

    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    runtime = load_json(ROOT / "configs" / "m3_runtime.json")
    scenario_config = load_json(ROOT / "configs" / "m2_scenarios.json")
    thresholds = load_json(ROOT / "configs" / "m2_thresholds.json")
    if runtime.get("status") != "frozen" or scenario_config.get("status") != "frozen":
        parser.error("M3 requires frozen runtime and M2 scenario configurations")
    scenarios = list(scenario_config["scenarios"])
    by_id = {str(row["scenario_id"]): row for row in scenarios}

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output or ROOT / "results" / "m3" / run_id
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory already contains data: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    planner = _make_planner(args.planner, runtime)
    writer = M3RunWriter(output_dir, run_id)
    manifest = _manifest(run_id, args.planner, runtime)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    runner = M3EpisodeRunner(Episode, runtime, writer, planner, run_id)
    enable_viewer = not args.headless and (not args.batch or args.viewer)
    results: list[dict[str, Any]] = []

    if args.batch:
        cases_data = load_json(ROOT / "configs" / "m3_cases.json")
        if cases_data.get("status") != "frozen" or len(cases_data.get("cases", [])) != 20:
            parser.error("--batch requires the frozen 20-case configs/m3_cases.json")
        cases = list(cases_data["cases"])
    else:
        case = {
            "case_id": f"single_seed_{args.seed}",
            "seed": args.seed,
            "instruction": args.instruction,
            "expected_target_id": None,
        }
        try:
            scenario = _select_scenario(args.instruction, args.seed, runtime, scenarios)
        except ContractError:
            scenario = None
        cases = [case]

    interrupted_at: int | None = None
    for index, case in enumerate(cases):
        case_id = str(case["case_id"])
        if args.batch:
            scenario = by_id.get(str(case["scenario_id"]))
            if scenario is None or int(scenario["seed"]) != int(case["seed"]):
                scenario = None
        episode_id = f"{run_id}-{index + 1:03d}-{case_id}"
        print(f"[{index + 1}/{len(cases)}] {case_id}: {case['instruction']}")
        try:
            result = runner.run_episode(
                instruction=str(case["instruction"]),
                scenario=scenario,
                thresholds=thresholds,
                output_dir=output_dir,
                enable_viewer=enable_viewer,
                show_ui=args.show_ui,
                episode_id=episode_id,
                case_id=case_id,
                expected_target_id=case.get("expected_target_id"),
            )
        except KeyboardInterrupt:
            interrupted_at = index
            result = {
                "run_id": run_id,
                "episode_id": episode_id,
                "case_id": case_id,
                "instruction": case["instruction"],
                "status": "ABORTED",
                "error_code": "INTERRUPTED",
                "error_message": "Run interrupted by the user",
                "step_count": 0,
                "planner_kind": args.planner,
                "llm_request_count": 0,
            }
            writer.append_episode(result)
        results.append(result)
        print(
            f"  {result['status']} {result.get('error_code') or ''}; "
            f"steps={result.get('step_count', 0)} target={result.get('target_id')}"
        )
        if interrupted_at is not None:
            break

    if interrupted_at is not None:
        for index, case in enumerate(cases[interrupted_at + 1 :], start=interrupted_at + 1):
            episode_id = f"{run_id}-{index + 1:03d}-{case['case_id']}"
            result = {
                "run_id": run_id,
                "episode_id": episode_id,
                "case_id": case["case_id"],
                "instruction": case["instruction"],
                "status": "NOT_RUN",
                "error_code": "RUN_INTERRUPTED",
                "error_message": "Not started because an earlier episode was interrupted",
                "step_count": 0,
                "planner_kind": args.planner,
                "llm_request_count": 0,
            }
            writer.append_episode(result)
            results.append(result)

    summary = writer.finish(results, manifest)
    print(
        f"M3 result: {summary['success_count']}/{summary['episode_count']} successful, "
        f"{summary['failure_count']} failed, {summary['aborted_count']} aborted. Evidence: {output_dir}"
    )
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

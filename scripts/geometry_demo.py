"""Generated-layout physical acceptance; one shared live Viewer by default."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import sys
from datetime import datetime
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=("floor", "sofa", "platform", "table"), default="floor")
    parser.add_argument("--source", choices=("table", "floor"), default="table")
    parser.add_argument("--headless", action="store_true", help="Explicit automated physical acceptance mode")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "geometry" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--records-dir", type=Path, default=ROOT / "records", help="Unified execution records root")
    args = parser.parse_args()
    from embodied_agent.maps.geometry_acceptance import run_geometry_case
    from embodied_agent.evaluation.task_writer import TaskRunWriter
    from embodied_agent.models.tracing import trace_scope
    from embodied_agent.simulation.stretch import HomeSkillError
    args.output.mkdir(parents=True, exist_ok=True)
    writer = TaskRunWriter(args.output, "geometry_" + uuid.uuid4().hex, records_dir=args.records_dir,
        versions={"geometry_capabilities": "stretch-side-grasp-v2", "entry": "geometry_demo"})
    viewer = None
    world_view = None
    viewer_status = None
    viewer_closed = False
    live_episode = None
    task_scopes = ExitStack()
    def ready(episode):
        nonlocal viewer, world_view, viewer_status, live_episode
        live_episode = episode
        if not args.headless:
            import tkinter as tk
            from tkinter import ttk
            from embodied_agent.visualization.world_view import WorldView
            viewer = tk.Tk()
            viewer.title(f"MuJoCo Geometry · {episode.world.map_id}")
            viewer.geometry("960x720")
            viewer_status = tk.StringVar(value=f"{args.source} → {args.target} · executing")
            ttk.Label(viewer, textvariable=viewer_status).pack(fill="x", padx=10, pady=8)
            world_view = WorldView(viewer, width=940, height=660)
            world_view.pack(fill="both", expand=True)
            def close_viewer():
                nonlocal viewer_closed
                viewer_closed = True
                world_view.close()
                viewer.destroy()
            viewer.protocol("WM_DELETE_WINDOW", close_viewer)
            viewer.update()
            world_view.render(episode)
        writer.begin_task(instruction=f"将程序化地图中的 parcel 从{args.source}搬到{args.target}",
            map_id=episode.world.map_id, map_definition=episode.world.to_dict(), initial_state=episode.observe(),
            metadata={"planner_source": "registered_geometry_routine", "headless": args.headless,
                "seed": args.seed, "language_model_used": False,
                "viewer_backend": None if args.headless else "demo_world_view"})
        task_scopes.enter_context(trace_scope(**writer.trace_identity()))
    def frame(episode):
        if viewer is not None:
            if viewer_closed:
                raise HomeSkillError("VIEWER_CLOSED", "Viewer was closed during physical acceptance")
            if episode.total_steps % 80 == 0:
                viewer_status.set(f"{args.source} → {args.target} · {episode.active_skill}")
                world_view.render(episode)
                viewer.update()
    try:
        result = run_geometry_case(args.seed, target=args.target, source=args.source, on_ready=ready, on_frame=frame,
            record_event=writer.record_event, trace_identity=writer.trace_identity)
        result["headless"] = args.headless
        result["viewer_backend"] = None if args.headless else "demo_world_view"
        writer.finish_task(result, final_state=result["final_state"], actual_execution={
            "episode_events": result["episode_events"], "physics_events": result["physics_events"],
            "trajectory": result["trajectory"], "range": {"start_step": 0, "end_step": result["physics_steps"]}})
        report = writer.report_result(result)
        report["world_artifact_ref"] = writer.store.put_artifact(result["world"])
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (args.output / "map.json").write_text(json.dumps({"artifact_ref": report["world_artifact_ref"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result.get(key) for key in ("status", "map_id", "transport_success", "physics_steps", "error_code", "error_message")}, ensure_ascii=False))
        print(f"Evidence: {args.output.resolve()}")
        if viewer is not None and not viewer_closed:
            viewer_status.set(f"{args.source} → {args.target} · {result['status']} · close window to exit")
            world_view.render(live_episode)
            # Export the same rendered live model/data shown in the original
            # Demo viewport; this does not create a second simulation episode.
            if world_view._photo is not None:
                world_view._photo.write(str(args.output / "viewer.png"), format="png")
            viewer.update()
            viewer.mainloop()
        return 0 if result["status"] == "SUCCESS" else 1
    finally:
        original_error = sys.exception()
        cleanup_errors = []
        cleanup = [writer.close]
        if live_episode is not None:
            cleanup.append(live_episode.close)
        if world_view is not None:
            cleanup.append(world_view.close)
        if viewer is not None and not viewer_closed:
            cleanup.append(viewer.destroy)
        cleanup.append(task_scopes.close)
        for close in cleanup:
            try:
                close()
            except Exception as error:
                cleanup_errors.append(error)
                if original_error is not None:
                    original_error.add_note(f"Geometry cleanup also failed: {type(error).__name__}: {error}")
        if original_error is None and cleanup_errors:
            raise cleanup_errors[0]


if __name__ == "__main__":
    raise SystemExit(main())

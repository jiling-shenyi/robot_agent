"""Physical acceptance of generated layouts through the normal live adapter."""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any

from embodied_agent.models.tracing import new_trace_id, trace_scope

from .manipulation import candidate_operation_points
from .procedural import generate_home_world


class AcceptanceFailure(RuntimeError):
    def __init__(self, code: str, message: str, status: str = "FAILED"):
        super().__init__(message)
        self.code, self.status = code, status


def run_geometry_case(seed: int = 0, *, target: str = "floor", source: str = "table",
                      on_ready=None, on_frame=None, record_event=None, trace_identity=None) -> dict[str, Any]:
    from embodied_agent.evaluation.home import placement_evidence
    from embodied_agent.simulation.robot_episode import StretchDemoEpisode

    world = generate_home_world(seed, target=target, source=source)
    episode = StretchDemoEpisode(world, on_frame=on_frame, step_budget=180000)
    try:
        if on_ready:
            on_ready(episode)
        initial = episode.observe()
        identity = trace_identity() if callable(trace_identity) else trace_identity or {}
    except BaseException as error:
        try:
            episode.close()
        except Exception as cleanup_error:
            error.add_note(f"Closing geometry episode also failed: {cleanup_error}")
        raise
    results = []
    recording_errors = []

    def record(name, detail, observation=None):
        if record_event is None:
            return None
        try:
            return record_event(name, {**detail, "source": "registered_geometry_routine",
                "observation_only": True, "state_restorable": False}, observation)
        except Exception as error:
            recording_errors.append({"event": name, "type": type(error).__name__, "message": str(error)})
            raise AcceptanceFailure("RECORDING_FAILED", f"Cannot record {name}: {error}") from error

    def execute(action):
        action_id = new_trace_id("action")
        with trace_scope(**identity, action_id=action_id):
            started = record("action_dispatch_started", {"action": action, "action_id": action_id,
                "dispatch_issued": False, "boundary": "before_episode_run_action"}, episode.observe())
            with trace_scope(parent_event_id=started):
                try:
                    result = episode.run_action(action, record_event=record if record_event else None)
                except Exception as error:
                    try:
                        record("action_dispatch_finished", {"action": action, "action_id": action_id,
                            "dispatch_issued": True, "dispatch_returned": False, "executed": None,
                            "error_code": getattr(error, "code", type(error).__name__), "error_message": str(error)})
                    except Exception as recording_error:
                        error.add_note(f"Recording dispatch failure also failed: {recording_error}")
                    raise
                result = asdict(result) if is_dataclass(result) else result
                result = {**result, "action_id": action_id}
                results.append(result)
                try:
                    record("action_dispatch_finished", {"action": action, "action_id": action_id,
                        "dispatch_issued": True, "dispatch_returned": True, "executed": result.get("executed"),
                        "status": result["status"], "error_code": result.get("error_code"),
                        "physics_steps": result.get("physics_steps")}, result.get("observation"))
                except AcceptanceFailure:
                    if result["status"] == "SUCCESS":
                        raise
                if result["status"] != "SUCCESS":
                    raise AcceptanceFailure(result["error_code"], result["error_message"], result["status"])
    try:
        episode.sim.wait(.5)

        def select(points):
            nav = episode._navigator(episode.positions())
            choices = []
            for point in points:
                try:
                    path = nav.plan(episode.sim.base_pose()[:2], point.position_m[:2])
                except Exception as exc:
                    if getattr(exc, "code", None) == "PATH_BLOCKED":
                        continue
                    raise
                cost = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                choices.append((cost, point.point_id, point))
            if not choices:
                raise RuntimeError("No generated dock has a current collision-free route")
            return min(choices, key=lambda item: item[:2])[2]

        dock = select(candidate_operation_points(world, object_id="parcel", object_positions=episode.positions()))
        execute({"skill": "navigate", "target": dock.point_id})
        execute({"skill": "pick", "object_id": "parcel"})
        support_id = "floor" if target == "floor" else "destination"
        destination = select(candidate_operation_points(world, object_id="parcel", support_id=support_id,
            object_positions=episode.positions()))
        execute({"skill": "carry", "target": destination.point_id})
        execute({"skill": "place", "support_id": support_id, "target_xy": list(destination.target_xy_m)})
        score = placement_evidence(episode.sim, world, "parcel", support_id, destination.target_xy_m)
        return {"status": "SUCCESS" if score["passed"] else "FAILED", "map_id": world.map_id,
            "transport_success": bool(score["passed"]), "postconditions": score,
            "world": world.to_dict(), "initial_state": initial, "final_state": episode.observe(),
            "actions": results, "physics_steps": episode.total_steps,
            "physics_events": episode.sim.events, "episode_events": episode.events, "trajectory": episode.samples,
            "llm_request_count": 0, "source": source, "target": target, "seed": seed,
            "recording_errors": recording_errors}
    except Exception as exc:
        cleanup_errors = []
        if not results or results[-1]["status"] == "SUCCESS":
            try:
                episode.safe_stop()
            except Exception as stop_error:
                cleanup_errors.append({"phase": "stop", "message": str(stop_error)})
        return {"status": getattr(exc, "status", "FAILED"), "error_code": getattr(exc, "code", type(exc).__name__),
            "error_message": str(exc), "map_id": world.map_id, "transport_success": False,
            "world": world.to_dict(), "initial_state": initial, "final_state": episode.observe(),
            "actions": results, "physics_steps": episode.total_steps,
            "physics_events": episode.sim.events, "episode_events": episode.events, "trajectory": episode.samples,
            "llm_request_count": 0, "source": source, "target": target, "seed": seed,
            "cleanup_errors": cleanup_errors, "recording_errors": recording_errors}

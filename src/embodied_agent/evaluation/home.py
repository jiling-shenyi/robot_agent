"""Independent home placement predicates over actual MuJoCo state/contact."""
from __future__ import annotations

import math
from typing import Any


def summarize_home_results(results: list[dict], planned_cases: list[str], *, headless: bool) -> dict:
    completed = {r["case_id"] for r in results}
    complete = completed == set(planned_cases) and len(results) == len(planned_cases)
    return {"case_count": len(results), "planned_case_count": len(planned_cases),
        "complete": complete, "pending_cases": [name for name in planned_cases if name not in completed],
        "passed_count": sum(bool(r.get("expected_pass")) for r in results),
        "success_count": sum(r["status"] == "SUCCESS" for r in results),
        "transport_success_count": sum(bool(r["transport_success"]) for r in results),
        "expected_rejection_count": sum(bool(r.get("expected_pass")) and r["status"] == "FAILED"
                                         and not r.get("test_fault_protocol") for r in results),
        "expected_fault_count": sum(bool(r.get("expected_pass")) and r["status"] == "FAILED"
                                     and bool(r.get("test_fault_protocol")) for r in results),
        "aborted_count": sum(r["status"] == "ABORTED" for r in results),
        "all_pass": complete and all(r.get("expected_pass", False) for r in results),
        "headless": headless, "results": results}


def placement_evidence(sim: Any, world: Any, object_id: str, support_id: str,
                       target_xy: list[float] | tuple[float, float]) -> dict[str, Any]:
    import mujoco
    import numpy as np
    from embodied_agent.maps.manipulation import lookup_surface
    obj, support = world.objects[object_id], lookup_surface(world, support_id)
    body = sim.object_body_ids[object_id]
    position = sim.data.xpos[body].copy()
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    geom = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, obj.geom_name)
    # Use actual oriented-box extents; centre alone cannot prove safe support.
    rotation = sim.data.geom_xmat[geom].reshape(3, 3)
    extents = np.abs(rotation) @ np.asarray(obj.half_size_m)
    support_geoms = {mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                     for name in support.geom_names}
    pairs = []
    for index, contact in enumerate(sim.data.contact[:sim.data.ncon]):
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, index, force)
        pairs.append(((int(contact.geom1), int(contact.geom2)), float(force[0])))
    touching_support = any(geom in pair and force > 0.05 and any(g in support_geoms for g in pair if g != geom) for pair, force in pairs)
    touching_robot = any(geom in pair and any(g in sim.robot_geom_ids for g in pair if g != geom) for pair, _ in pairs)
    margin = min(support.half_size_m[i] - abs(position[i] - support.position_m[i]) - extents[i]
                 for i in (0, 1))
    predicates = {
        "target_xy": math.dist(position[:2], target_xy) <= 0.045,
        "support_height": abs(position[2] - extents[2] - support.top_z) <= 0.012,
        "edge_margin": margin >= (0.05 if "fragile" in obj.risk_tags else 0.01),
        "physical_support_contact": touching_support,
        "released": not touching_robot and getattr(sim, "held_object_id", None) != object_id,
        "linear_still": float(np.linalg.norm(velocity[3:])) <= 0.025,
        "angular_still": float(np.linalg.norm(velocity[:3])) <= 0.2,
    }
    predicates = {name: bool(value) for name, value in predicates.items()}
    return {"passed": all(predicates.values()), "predicates": predicates,
            "position_m": position.tolist(), "target_xy": list(target_xy),
            "edge_margin_m": float(margin), "velocity": velocity.tolist(),
            "support_id": support_id, "object_id": object_id,
            "evaluator_version": "home-placement-v1"}

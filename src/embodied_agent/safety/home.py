"""State rules and path clearance enforced by the home skill dispatcher.

The simulator also checks actual registered geometry at every physics step.
This is the bounded A2/A3 execution guard, not the later full A4 Hook/permit
or A5 rehearsal pipeline.
"""
from __future__ import annotations

import math
from typing import Any

from embodied_agent.execution.home_contracts import HomeExecutionError
from embodied_agent.skills.navigation import Navigator, Obstacle


class HomeSafety:
    def __init__(self, world: Any):
        self.world = world

    def object_action(self, skill: str, object_id: str, positions: dict,
                      states: dict, target_position=None) -> None:
        violations = self.world.action_violations(skill, object_id,
            object_positions=positions, object_states=states, target_position_m=target_position)
        if violations:
            raise HomeExecutionError(violations[0], f"{skill} {object_id}: {', '.join(violations)}",
                {"rule_version": self.world.rule_version, "object_id": object_id,
                 "position_m": positions.get(object_id), "target_position_m": target_position,
                 "violations": list(violations)})

    def navigator(self, positions: dict, states: dict, *, carrying: bool = False,
                  carried_half_size=None, footprint_radius: float | None = None,
                  object_velocities: dict | None = None, prediction_horizon_s: float = 0.5,
                  held_object_id: str | None = None) -> Navigator:
        obstacles = []
        for name, item in self.world.furniture.items():
            x, y = item.position_m[:2]
            sx, sy = item.half_size_m[:2]
            obstacles.append(Obstacle(x - sx, y - sy, x + sx, y + sy, f"furniture:{name}"))
        for risk in self.world.risks(positions, states):
            if "navigate" not in risk.forbidden_actions:
                continue
            x, y = risk.position_m[:2]
            sx, sy = risk.half_size_m[:2]
            obstacles.append(Obstacle(x - sx, y - sy, x + sx, y + sy, f"risk:{risk.kind}"))
        for name, position in positions.items():
            if name == held_object_id:
                continue
            obj = self.world.objects[name]
            # Floor objects are physical obstacles even without a risk tag.
            if position[2] - obj.half_size_m[2] < 0.15:
                x, y = position[:2]
                sx, sy = obj.half_size_m[:2]
                velocity = (object_velocities or {}).get(name, (0.0, 0.0))
                if (len(velocity) < 2 or any(not math.isfinite(float(v)) for v in velocity[:2])
                        or not math.isfinite(prediction_horizon_s) or prediction_horizon_s < 0):
                    raise HomeExecutionError("INVALID_OBSERVATION", "Obstacle motion needs finite xy velocity and horizon")
                end_x, end_y = x + velocity[0] * prediction_horizon_s, y + velocity[1] * prediction_horizon_s
                # Swept AABB covers motion during the local reaction horizon.
                obstacles.append(Obstacle(min(x, end_x) - sx, min(y, end_y) - sy,
                    max(x, end_x) + sx, max(y, end_y) + sy, f"object:{name}"))
        width, depth = self.world.room.size_m[:2]
        radius = self.world.navigation.robot_radius_m
        if carrying:
            # Verified transport is retracted; reserve clearance for payload
            # half-diagonal and tracking error beyond the registered base disk.
            radius += math.hypot(*(carried_half_size or (0.025, 0.025))[:2]) + 0.02
        if footprint_radius is not None:
            radius = max(radius, footprint_radius)
        return Navigator((-width / 2, -depth / 2, width / 2, depth / 2),
                         obstacles, radius, self.world.navigation.resolution_m)

    def during_motion(self, sim: Any, positions: dict, states: dict,
                      footprint_radius: float | None = None) -> None:
        x, y, _ = sim.base_pose()
        radius = self.world.navigation.robot_radius_m
        radius = max(radius, sim.transport_footprint_radius() if footprint_radius is None else footprint_radius)
        for risk in self.world.risks(positions, states):
            if "navigate" not in risk.forbidden_actions:
                continue
            px, py = risk.position_m[:2]
            sx, sy = risk.half_size_m[:2]
            distance = math.hypot(max(abs(x - px) - sx, 0), max(abs(y - py) - sy, 0))
            if distance <= radius:
                raise HomeExecutionError("RISK_CLEARANCE", f"Robot footprint entered {risk.kind} constraint for {risk.object_ids}",
                    {"risk": risk.to_dict(), "base_pose": list(sim.base_pose()),
                     "required_distance_m": radius, "actual_distance_m": distance})

    def during_geometry(self, sim: Any, positions: dict, states: dict) -> None:
        """Check current collision geometry against active 3D state constraints."""
        import numpy as np
        risks = [r for r in self.world.risks(positions, states)
                 if r.kind in {"hot", "unsealed_cleaner"}]
        if not risks:
            return
        geoms = set(sim.robot_geom_ids)
        held = getattr(sim, "held_object_id", None)
        if held:
            geoms.add(sim.object_geom_ids[held])
        for geom in geoms:
            if not (sim.model.geom_contype[geom] or sim.model.geom_conaffinity[geom]):
                continue
            rotation = sim.data.geom_xmat[geom].reshape(3, 3)
            aabb = sim.model.geom_aabb[geom]
            center = sim.data.geom_xpos[geom] + rotation @ aabb[:3]
            extents = np.abs(rotation) @ aabb[3:]
            for risk in risks:
                if np.all(np.abs(center - np.asarray(risk.position_m)) <= extents + np.asarray(risk.half_size_m)):
                    raise HomeExecutionError("RISK_CLEARANCE",
                        f"Geometry {sim.model.geom(geom).name} entered {risk.kind} region of {risk.object_ids}",
                        {"geom": sim.model.geom(geom).name, "risk": risk.to_dict(),
                         "geometry_center_m": center.tolist(), "geometry_half_size_m": extents.tolist(),
                         "criterion": "current collision-geometry AABB intersects the active 3D risk region"})

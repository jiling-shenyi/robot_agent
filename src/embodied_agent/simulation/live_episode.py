"""Live map episode with display callbacks and continuous instruction state."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
from embodied_agent.paths import PROJECT_ROOT as ROOT
from embodied_agent.maps import WorldMap, apply_world_to_model
from embodied_agent.simulation.episode import Episode, load_scene_model, SAMPLE_EVERY_STEPS
from embodied_agent.execution.interfaces import LiveRobotEpisode

FrameCallback = Callable[[Any], None]
StatusCallback = Callable[[str, str], None]

class DemoEpisode(Episode):
    """The existing safety-monitored M2 skills with an externally owned display."""

    robot_kind = "panda"

    def __init__(self, world: WorldMap, thresholds: dict[str, Any], output_dir: Path,
                 *, root: Path = ROOT, on_frame: FrameCallback | None = None,
                 on_status: StatusCallback | None = None):
        model = load_scene_model(root, regions=world.targets)
        apply_world_to_model(model, world)
        self.world = world
        self.on_frame = on_frame
        self.on_status = on_status
        scenario = {"scenario_id": f"map_{world.map_id}", "seed": 0, "target": "a",
                    "cube_position_m": list(world.cube_position_m)}
        super().__init__(scenario, thresholds, output_dir, model=model)
        if "target_a" not in world.targets:
            target_id = next(iter(world.targets))
            self.target_geom_id = model.geom(target_id).id
            self.target_center = self.data.geom_xpos[self.target_geom_id].copy()

    def prepare(self, *, check_safety: bool = True) -> dict[str, Any]:
        if not self.prepared:
            return super().prepare(check_safety=check_safety)
        if check_safety:
            self.check_safety()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Measured telemetry, not a complete, restorable physics checkpoint."""
        return super().snapshot()

    def _open_viewer(self) -> None:
        # Renderer/view controls belong to the caller and use this model/data.
        pass

    def _sync_viewer(self) -> None:
        if self.on_frame is not None and self.total_steps % SAMPLE_EVERY_STEPS == 0:
            self.on_frame(self)

    def set_viewer_status(self, title: str, detail: str) -> None:
        if self.on_status is not None:
            self.on_status(title, detail)

    def wait_for_viewer_close(self) -> None:
        # The outer demo retains the final world and its window after a task.
        pass


def create_demo_episode(world: Any, thresholds: dict[str, Any], output_dir: Path,
                        *, root: Path = ROOT, on_frame: FrameCallback | None = None,
                        on_status: StatusCallback | None = None) -> LiveRobotEpisode:
    """Choose robot mechanics while retaining the original Demo lifecycle."""
    from embodied_agent.maps.home_schema import HomeWorld

    if isinstance(world, HomeWorld):
        from embodied_agent.simulation.robot_episode import StretchDemoEpisode

        episode_type = StretchDemoEpisode
    elif isinstance(world, WorldMap):
        episode_type = DemoEpisode
    else:
        raise TypeError("Unsupported Demo map schema")
    return episode_type(world, thresholds, output_dir, root=root,
                        on_frame=on_frame, on_status=on_status)

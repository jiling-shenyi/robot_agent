"""MuJoCo model loading, identifiers and simulator telemetry helpers."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import mujoco

from ..paths import PROJECT_ROOT as ROOT

SCENE = ROOT / "assets" / "scene" / "panda_task.xml"
SAMPLE_EVERY_STEPS = 25
WARNING_TYPES = {
    int(getattr(mujoco.mjtWarning, name)): name
    for name in dir(mujoco.mjtWarning)
    if name.startswith("mjWARN_")
}
_SCENE_LOAD_LOCK = threading.RLock()

def load_scene_model(root: Path = ROOT, *, regions=None) -> mujoco.MjModel:
    """Load relative asset paths, including on Windows workspaces with Unicode names."""
    with _SCENE_LOAD_LOCK:
        previous_directory = Path.cwd()
        try:
            os.chdir(root)
            extra = set(regions or {}) - {"target_a", "target_b"}
            if extra:
                spec = mujoco.MjSpec.from_file("assets/scene/panda_task.xml")
                for name in sorted(extra):
                    region = regions[name]
                    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                        pos=region.position_m, size=region.half_size_m,
                        contype=0, conaffinity=0, rgba=[.25, .75, .75, .45])
                return spec.compile()
            return mujoco.MjModel.from_xml_path("assets/scene/panda_task.xml")
        finally:
            os.chdir(previous_directory)


def name_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise RuntimeError(f"Required MuJoCo name not found: {name}")
    return int(value)


def warning_counts(data: mujoco.MjData) -> dict[str, int]:
    return {
        WARNING_TYPES[index]: int(count)
        for index, count in enumerate(data.warning.number)
        if count and index in WARNING_TYPES
    }

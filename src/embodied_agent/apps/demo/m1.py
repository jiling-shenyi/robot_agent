"""Run the M1 Panda mocap-following, gripper, and scene smoke checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from embodied_agent.paths import PROJECT_ROOT as ROOT
from embodied_agent.simulation.tracking import (
    ViewerClosed,
    _name_id,
    _sha256,
    _site_quat,
    _warning_counts,
    _write_png,
    _contact_snapshot,
    SCENE,
    MODEL_SOURCE,
    MODEL_DERIVATIVE,
    ERROR_LIMIT_M,
    PENETRATION_LIMIT_M,
    SAMPLE_EVERY_STEPS,
    DWELL_STEPS,
    MOTION_STEPS,
    DANGER_ZONE_CENTER_M,
    DANGER_ZONE_HALF_SIZE_M,
    DANGER_ZONE_RGBA,
    run_tracking,
)

__all__ = ['main', 'ViewerClosed', '_name_id', '_sha256', '_site_quat', '_warning_counts', '_write_png', '_contact_snapshot', 'SCENE', 'MODEL_SOURCE', 'MODEL_DERIVATIVE', 'ERROR_LIMIT_M', 'PENETRATION_LIMIT_M', 'SAMPLE_EVERY_STEPS', 'DWELL_STEPS', 'MOTION_STEPS', 'DANGER_ZONE_CENTER_M', 'DANGER_ZONE_HALF_SIZE_M', 'DANGER_ZONE_RGBA', 'ROOT', 'run_tracking']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "m1",
        help="directory for CSV, JSON summary, and screenshot (default: results/m1)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run as fast as possible without opening the live MuJoCo viewer",
    )
    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="show native debug panels (hidden by default; affected AMD Windows drivers corrupt these panels)",
    )
    args = parser.parse_args()
    return run_tracking(args)

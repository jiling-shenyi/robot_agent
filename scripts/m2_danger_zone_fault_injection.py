"""Compatibility entry for embodied_agent.apps.demo.danger_zone_fault."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.apps.demo.danger_zone_fault import *

if __name__ == "__main__":
    raise SystemExit(main())

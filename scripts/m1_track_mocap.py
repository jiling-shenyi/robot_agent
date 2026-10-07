"""Compatibility entry for embodied_agent.apps.demo.m1."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_agent.apps.demo.m1 import *

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ViewerClosed:
        print("Live viewer closed before the episode completed; no final evidence was written.")
        raise SystemExit(130)

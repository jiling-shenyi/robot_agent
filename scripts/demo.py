"""Compatibility launcher for the shared demo application."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_agent.apps.demo.cli import build_parser, main

if __name__ == "__main__":
    raise SystemExit(main())

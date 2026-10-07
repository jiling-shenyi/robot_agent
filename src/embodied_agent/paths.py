"""Project paths shared by library modules and application launchers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
ASSETS_DIR = PROJECT_ROOT / "assets"
RESULTS_DIR = PROJECT_ROOT / "results"

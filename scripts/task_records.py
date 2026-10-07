"""Query versioned task records from the project package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_agent.evaluation.task_query import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Content fingerprints for run reproduction, excluding secrets and outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_hashes(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    paths = set((root / "src").rglob("*.py")) | set((root / "scripts").glob("*.py"))
    paths.update((root / "configs").rglob("*.json"))
    paths.update((root / "src" / "embodied_agent" / "prompts").rglob("*.txt"))
    paths.update(root / name for name in ("requirements.txt", "requirements-lock.txt"))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}

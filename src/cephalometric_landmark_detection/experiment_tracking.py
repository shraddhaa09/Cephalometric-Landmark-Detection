from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .config import ROOT_DIR


class ExperimentTracker:
    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir or ROOT_DIR / "runs")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.run_dir / "metadata.json"

    def log(self, payload: Dict[str, Any]) -> None:
        with open(self.metadata_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

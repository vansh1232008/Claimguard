"""Put the backend package on sys.path so ml/ scripts can reuse app code.

The training scripts import ``app.services.features`` on purpose: training and
serving must build features with the same function, not two copies of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"

for path in (str(BACKEND), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

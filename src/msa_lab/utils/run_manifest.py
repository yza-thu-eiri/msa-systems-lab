from __future__ import annotations

from pathlib import Path
import json
import platform
import subprocess
import sys
from typing import Any

import torch


def _git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_run_manifest(path: str | Path, *, command: list[str], cwd: Path) -> None:
    payload: dict[str, Any] = {
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(cwd),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

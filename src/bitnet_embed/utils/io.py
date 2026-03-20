from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(payload)!r}")
    return payload


def dump_json(path: Path | str, payload: Any) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def get_git_revision(cwd: Path | str | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None

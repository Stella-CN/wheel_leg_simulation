"""Load validated JSON configurations stored under ``configs/``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import SIMULATION_CONFIG_DIR, project_path


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = project_path(path)
    with resolved.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a JSON object: {resolved}")
    return data


def load_simulation_config(path: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    """Return a simulation config and its resolved path.

    ``None`` intentionally means no override. This lets command-line arguments
    retain the same defaults when no dedicated scenario file is selected.
    """
    if path is None:
        return {}, None
    resolved = project_path(path)
    data = load_json(resolved)
    data.pop("name", None)
    return data, resolved


def default_simulation_config(name: str) -> Path:
    return SIMULATION_CONFIG_DIR / f"{name}.json"

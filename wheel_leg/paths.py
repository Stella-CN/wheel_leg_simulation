"""Canonical project paths used by builders, simulations and reports."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_CONFIG_DIR = CONFIG_DIR / "model"
SIMULATION_CONFIG_DIR = CONFIG_DIR / "simulations"
MODEL_DIR = PROJECT_ROOT / "models"
GENERATED_MODEL_DIR = MODEL_DIR / "generated"
REPORT_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
CURRENT_RESULTS_DIR = RESULTS_DIR / "current"
ARCHIVE_RESULTS_DIR = RESULTS_DIR / "archive"
PHYSICAL_PARAMETERS_PATH = MODEL_CONFIG_DIR / "physical_parameters.json"


def project_path(path: str | Path) -> Path:
    """Resolve a user-supplied path relative to the project root."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


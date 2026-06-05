"""Project-level defaults for the industrial anomaly detection coursework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

MAIN_CATEGORIES = ("bottle", "hazelnut", "metal_nut")


@dataclass(frozen=True)
class EpochPreset:
    name: str
    max_epochs: int
    description: str


EPOCH_PRESETS = {
    "env": EpochPreset("env", 1, "Environment check"),
    "smoke": EpochPreset("smoke", 5, "Smoke test"),
    "initial20": EpochPreset("initial20", 20, "Initial result"),
    "initial50": EpochPreset("initial50", 50, "Initial stronger result"),
    "final100": EpochPreset("final100", 100, "Final experiment"),
    "final200": EpochPreset("final200", 200, "Final long experiment"),
}

DEFAULT_IMAGE_SIZE = 256
DEFAULT_FIXED_THRESHOLD = 0.5
DEFAULT_PERCENTILE = 99.0
DEFAULT_SMOOTH_SIGMA = 0.0


def categories_from_arg(category: str) -> list[str]:
    if category == "all":
        return list(MAIN_CATEGORIES)
    if category not in MAIN_CATEGORIES:
        allowed = ", ".join((*MAIN_CATEGORIES, "all"))
        raise ValueError(f"Unknown category '{category}'. Allowed: {allowed}")
    return [category]

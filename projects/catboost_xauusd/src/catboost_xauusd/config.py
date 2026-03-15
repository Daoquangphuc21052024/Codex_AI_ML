from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MT5Config:
    source: str
    symbol: str
    timeframe: str
    bars: int
    timezone: str
    login: int | None
    password: str | None
    server: str | None
    csv_path: str | None = None


@dataclass(frozen=True)
class LabelingConfig:
    horizon_bars: int
    tp_points: list[float]
    sl_points: list[float]
    tie_breaker: str


@dataclass(frozen=True)
class FeatureConfig:
    max_features: int
    windows: list[int]
    corr_threshold: float


@dataclass(frozen=True)
class TrainConfig:
    n_splits: int
    min_train_size: int
    val_size: int
    test_size: int
    random_state: int
    tuning_trials: int
    iterations: int
    threshold_no_trade: float


@dataclass(frozen=True)
class PathsConfig:
    raw_data: str
    reports_dir: str
    artifacts_dir: str
    logs_dir: str


@dataclass(frozen=True)
class AppConfig:
    mt5: MT5Config
    labeling: LabelingConfig
    features: FeatureConfig
    train: TrainConfig
    paths: PathsConfig


def resolve_config_path(config_path: str | Path) -> Path:
    input_path = Path(config_path)
    if input_path.exists():
        return input_path

    cwd = Path.cwd()
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        cwd / input_path,
        cwd / "configs" / "config.yaml",
        cwd / "config.yaml",
        project_root / "configs" / "config.yaml",
        project_root / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {path}" for path in [input_path, *candidates])
    raise FileNotFoundError(
        "Config file not found.\n"
        f"Current working directory: {cwd}\n"
        "Tried the following locations:\n"
        f"{tried}\n"
        "Hint: run with '--config configs/config.yaml' from project root."
    )


def load_config(config_path: str | Path) -> AppConfig:
    resolved = resolve_config_path(config_path)
    with resolved.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    return AppConfig(
        mt5=MT5Config(**raw["mt5"]),
        labeling=LabelingConfig(**raw["labeling"]),
        features=FeatureConfig(**raw["features"]),
        train=TrainConfig(**raw["train"]),
        paths=PathsConfig(**raw["paths"]),
    )

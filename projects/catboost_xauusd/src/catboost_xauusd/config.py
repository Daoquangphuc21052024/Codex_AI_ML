from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MT5Config:
    symbol: str
    timeframe: str
    bars: int
    timezone: str
    login: int | None
    password: str | None
    server: str | None


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


TIMEFRAME_MAP: dict[str, int] = {
    "H1": 16385,
}


def load_config(config_path: str | Path) -> AppConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    return AppConfig(
        mt5=MT5Config(**raw["mt5"]),
        labeling=LabelingConfig(**raw["labeling"]),
        features=FeatureConfig(**raw["features"]),
        train=TrainConfig(**raw["train"]),
        paths=PathsConfig(**raw["paths"]),
    )

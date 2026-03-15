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
    timezone: str
    start_utc: str | None
    end_utc: str | None
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
    min_train_days: int
    val_days: int
    test_days: int
    step_days: int
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




def _reject_legacy_keys(raw: dict[str, Any]) -> None:
    mt5_legacy = [k for k in ["bars"] if k in raw.get("mt5", {})]
    train_legacy = [k for k in ["min_train_size", "val_size", "test_size"] if k in raw.get("train", {})]
    if mt5_legacy or train_legacy:
        raise ValueError(
            "Detected legacy config keys from v0.1.x that are no longer supported. "
            f"mt5 legacy keys: {mt5_legacy or 'none'}, train legacy keys: {train_legacy or 'none'}. "
            "Please migrate to v0.5.0 config schema using start_utc/end_utc and day-based walk-forward windows."
        )

def load_config(config_path: str | Path) -> AppConfig:
    resolved = resolve_config_path(config_path)
    with resolved.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    _reject_legacy_keys(raw)

    return AppConfig(
        mt5=MT5Config(**raw["mt5"]),
        labeling=LabelingConfig(**raw["labeling"]),
        features=FeatureConfig(**raw["features"]),
        train=TrainConfig(**raw["train"]),
        paths=PathsConfig(**raw["paths"]),
    )

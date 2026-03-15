from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TrainConfig


@dataclass
class FoldIndices:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_walk_forward_folds(df: pd.DataFrame, cfg: TrainConfig) -> list[FoldIndices]:
    if "time" not in df.columns:
        raise ValueError("Dataframe must include 'time' column for time-series folds")

    ordered = df.sort_values("time").reset_index(drop=True)
    time_series = pd.to_datetime(ordered["time"], utc=True)
    start_time = time_series.min()
    last_time = time_series.max()

    folds: list[FoldIndices] = []
    train_delta = pd.Timedelta(days=cfg.min_train_days)
    val_delta = pd.Timedelta(days=cfg.val_days)
    test_delta = pd.Timedelta(days=cfg.test_days)
    step_delta = pd.Timedelta(days=cfg.step_days)

    fold = 0
    train_end_time = start_time + train_delta
    while fold < cfg.n_splits:
        val_end_time = train_end_time + val_delta
        test_end_time = val_end_time + test_delta
        if test_end_time > last_time:
            break

        train_idx = ordered.index[time_series < train_end_time].to_numpy()
        val_idx = ordered.index[(time_series >= train_end_time) & (time_series < val_end_time)].to_numpy()
        test_idx = ordered.index[(time_series >= val_end_time) & (time_series < test_end_time)].to_numpy()

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            train_end_time += step_delta
            continue

        folds.append(FoldIndices(fold=fold, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
        fold += 1
        train_end_time += step_delta

    if not folds:
        raise ValueError("Not enough time-range data for configured walk-forward windows")
    return folds

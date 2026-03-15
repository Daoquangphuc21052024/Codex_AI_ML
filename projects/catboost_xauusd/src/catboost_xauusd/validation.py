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
    n = len(df)
    folds: list[FoldIndices] = []
    start = cfg.min_train_size

    for fold in range(cfg.n_splits):
        train_end = start + fold * cfg.test_size
        val_end = train_end + cfg.val_size
        test_end = val_end + cfg.test_size
        if test_end > n:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, test_end)
        folds.append(FoldIndices(fold=fold, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))

    if not folds:
        raise ValueError("Not enough data for configured walk-forward splits")
    return folds

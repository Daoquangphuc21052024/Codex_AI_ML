from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


def _train_quantile(series: pd.Series, quantiles: list[float], train_ratio: float = 0.6) -> list[float]:
    split = int(len(series) * train_ratio)
    train = series.iloc[:split] if split > 10 else series
    return train.quantile(quantiles).to_list()


def create_labels(
    close: pd.Series,
    rolling_periods: list[int],
    markup: float,
    threshold: float = 0.5,
    vol_window: int = 30,
    min_l: int = 1,
    max_l: int = 15,
    seed: int = SEED,
) -> pd.DataFrame:
    close = close.astype(float)
    n = len(close)
    rng = np.random.default_rng(seed)

    buy_votes = np.zeros(n, dtype=np.int16)
    sell_votes = np.zeros(n, dtype=np.int16)

    ret_std = close.pct_change().rolling(vol_window, min_periods=vol_window).std()
    for span in rolling_periods:
        ema = close.ewm(span=span, adjust=False).mean()
        trend = ema.diff()
        norm = trend / (ret_std + 1e-12)
        buy_votes += (norm > threshold).astype(np.int16).to_numpy()
        sell_votes += (norm < -threshold).astype(np.int16).to_numpy()

    majority = len(rolling_periods) // 2 + 1
    candidate = np.full(n, 2, dtype=np.int16)
    candidate[(buy_votes >= majority) & (sell_votes == 0)] = 0
    candidate[(sell_votes >= majority) & (buy_votes == 0)] = 1

    labels = np.full(n, 2, dtype=np.int16)
    steps = rng.integers(min_l, max_l + 1, size=n)
    cvals = close.to_numpy()

    for i in range(n):
        if candidate[i] == 2:
            continue
        j = i + int(steps[i])
        if j >= n:
            continue
        if candidate[i] == 0 and cvals[j] >= cvals[i] + markup:
            labels[i] = 0
        elif candidate[i] == 1 and cvals[j] <= cvals[i] - markup:
            labels[i] = 1

    # meta target: volatility regime from train-only quantiles
    abs_ret = close.pct_change().abs().fillna(0.0)
    q20, q80 = _train_quantile(abs_ret, [0.2, 0.8])
    meta_target = ((abs_ret >= q20) & (abs_ret <= q80)).astype(np.int16)

    return pd.DataFrame({"labels": labels, "meta_target": meta_target}, index=close.index)

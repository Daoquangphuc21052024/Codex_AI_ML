from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    return _true_range(df).rolling(window=window, min_periods=window).mean()


def engineer_features(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["log_ret_1"] = np.log(out["close"] / out["close"].shift(1))
    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["body_size"] = (out["close"] - out["open"]).abs() / out["close"].replace(0, np.nan)
    out["body_range_ratio"] = (out["close"] - out["open"]).abs() / (out["high"] - out["low"]).replace(0, np.nan)

    out["true_range"] = _true_range(out)
    base_window = cfg.windows[0]
    out["atr"] = _atr(out, base_window)
    out["rolling_std_ret"] = out["ret_1"].rolling(base_window, min_periods=base_window).std()

    out["movement"] = out["close"].diff()
    roll_mean = out["movement"].rolling(base_window, min_periods=base_window).mean()
    roll_std = out["movement"].rolling(base_window, min_periods=base_window).std()
    out["zscore_move"] = (out["movement"] - roll_mean) / roll_std.replace(0, np.nan)

    for w in cfg.windows:
        out[f"volatility_{w}"] = out["log_ret_1"].rolling(w, min_periods=w).std()
        out[f"velocity_{w}"] = out["close"].pct_change(w)
        out[f"intensity_{w}"] = out["body_size"].rolling(w, min_periods=w).mean()
        out[f"tr_norm_{w}"] = out["true_range"].rolling(w, min_periods=w).mean() / out["close"].replace(0, np.nan)

    out["volatility_regime"] = np.where(
        out[f"volatility_{cfg.windows[-1]}"] > out[f"volatility_{cfg.windows[-1]}"].rolling(cfg.windows[-1], min_periods=cfg.windows[-1]).median(),
        1.0,
        0.0,
    )

    candidate_features = [
        "atr",
        "true_range",
        "rolling_std_ret",
        "ret_1",
        "log_ret_1",
        "hl_range",
        "body_size",
        "body_range_ratio",
        "zscore_move",
        "volatility_regime",
    ] + [
        f"volatility_{w}" for w in cfg.windows
    ] + [
        f"velocity_{w}" for w in cfg.windows
    ] + [
        f"intensity_{w}" for w in cfg.windows
    ] + [
        f"tr_norm_{w}" for w in cfg.windows
    ]

    out = out.dropna(subset=candidate_features).reset_index(drop=True)
    return out, candidate_features

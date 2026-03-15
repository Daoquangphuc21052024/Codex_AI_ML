from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureSet:
    data: pd.DataFrame
    main_features: list[str]
    meta_features: list[str]


def build_features(df: pd.DataFrame, periods: list[int], periods_meta: list[int], atr_window: int = 14) -> FeatureSet:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    main_features: list[str] = []
    meta_features: list[str] = []

    for n in periods:
        ma = close.rolling(n).mean()
        rolling_std = close.rolling(n).std()
        long_std = close.rolling(n * 3).std()
        roll_low = close.rolling(n).min()
        roll_high = close.rolling(n).max()

        feats = {
            f"ma_dev_{n}": (close / ma) - 1,
            f"mom_{n}": (close / close.shift(n)) - 1,
            f"volratio_{n}": rolling_std / (long_std + 1e-12),
            f"pos_{n}": (close - roll_low) / ((roll_high - roll_low) + 1e-12),
            f"rsi_like_{n}": (close.diff() > 0).rolling(n).mean(),
            f"vol_chg_{n}": (volume / (volume.rolling(n).mean() + 1e-12)) - 1,
        }
        for k, v in feats.items():
            out[k] = v
            main_features.append(k)

    atr = (high - low).rolling(atr_window).mean() / close
    for n in periods_meta:
        s = close.rolling(n)
        out[f"skew_meta_{n}"] = s.skew()
        out[f"kurt_meta_{n}"] = s.kurt()
        out[f"trend_strength_meta_{n}"] = close.ewm(span=n, adjust=False).mean().diff().abs()
        meta_features.extend([f"skew_meta_{n}", f"kurt_meta_{n}", f"trend_strength_meta_{n}"])

    out["atr_meta"] = atr
    meta_features.append("atr_meta")

    out = out.dropna().copy()
    return FeatureSet(data=out, main_features=main_features, meta_features=meta_features)

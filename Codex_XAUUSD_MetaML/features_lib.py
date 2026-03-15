from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


@dataclass
class FeatureSet:
    data: pd.DataFrame
    main_features: list[str]
    meta_features: list[str]


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    removed_zero_importance: list[str]
    removed_high_correlation: list[str]
    importance_table: pd.DataFrame


def build_features(df: pd.DataFrame, periods: list[int], periods_meta: list[int], atr_window: int = 14) -> FeatureSet:
    """Build causal features without fragmented DataFrame writes."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    main_features: list[str] = []
    meta_features: list[str] = []
    feature_series: dict[str, pd.Series] = {}

    returns = close.pct_change()
    prev_close = close.shift(1)
    candle_body_raw = (close - out["open"]) / (prev_close.abs() + 1e-12)
    intrabar_range_raw = (high - low) / (prev_close.abs() + 1e-12)

    for n in periods:
        ma = close.rolling(n).mean()
        rolling_std = close.rolling(n).std()
        long_std = close.rolling(n * 3).std()
        rolling_ret_std = returns.rolling(n).std()
        roll_low = close.rolling(n).min()
        roll_high = close.rolling(n).max()
        ema = close.ewm(span=n, adjust=False).mean()
        breakout_up = (close - roll_high.shift(1)) / (prev_close.abs() + 1e-12)
        breakout_down = (roll_low.shift(1) - close) / (prev_close.abs() + 1e-12)
        downside_vol = returns.clip(upper=0).rolling(n).std()
        upside_vol = returns.clip(lower=0).rolling(n).std()

        feats = {
            f"ma_dev_{n}": (close / ma) - 1,
            f"mom_{n}": (close / close.shift(n)) - 1,
            f"volratio_{n}": rolling_std / (long_std + 1e-12),
            f"pos_{n}": (close - roll_low) / ((roll_high - roll_low) + 1e-12),
            f"rsi_like_{n}": (close.diff() > 0).rolling(n).mean(),
            f"vol_chg_{n}": (volume / (volume.rolling(n).mean() + 1e-12)) - 1,
            f"ema_dev_{n}": (close / (ema + 1e-12)) - 1,
            f"ret_zscore_{n}": returns / (rolling_ret_std + 1e-12),
            f"candle_body_{n}": candle_body_raw.rolling(n).mean(),
            f"range_norm_{n}": intrabar_range_raw.rolling(n).mean(),
            f"breakout_up_{n}": breakout_up.rolling(n).mean(),
            f"breakout_down_{n}": breakout_down.rolling(n).mean(),
            f"vol_skew_{n}": (upside_vol - downside_vol) / (rolling_ret_std + 1e-12),
        }

        feature_series.update(feats)
        main_features.extend(list(feats.keys()))

    atr = (high - low).rolling(atr_window).mean() / close
    for n in periods_meta:
        s = close.rolling(n)
        meta_dict = {
            f"skew_meta_{n}": s.skew(),
            f"kurt_meta_{n}": s.kurt(),
            f"trend_strength_meta_{n}": close.ewm(span=n, adjust=False).mean().diff().abs(),
        }
        feature_series.update(meta_dict)
        meta_features.extend(list(meta_dict.keys()))

    feature_series["atr_meta"] = atr
    meta_features.append("atr_meta")

    feature_frame = pd.concat(feature_series, axis=1)
    out = pd.concat([out, feature_frame], axis=1).dropna().copy()
    return FeatureSet(data=out, main_features=main_features, meta_features=meta_features)


def select_main_features_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    corr_threshold: float = 0.96,
    min_features: int = 12,
    random_seed: int = 42,
) -> FeatureSelectionResult:
    """Train-only feature filtering: remove zero-importance then high-correlation duplicates."""
    X = X_train[feature_names].copy()
    non_constant = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[non_constant]

    selector_model = CatBoostClassifier(
        iterations=180,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=random_seed,
        verbose=False,
    )
    selector_model.fit(X, y_train)
    importances = selector_model.get_feature_importance()
    imp_tbl = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)

    keep = imp_tbl[imp_tbl["importance"] > 0]["feature"].tolist()
    if len(keep) < min_features:
        keep = imp_tbl.head(min_features)["feature"].tolist()
    removed_zero = [f for f in X.columns if f not in keep]

    corr = X[keep].corr().abs().fillna(0.0)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    rank = {row.feature: float(row.importance) for row in imp_tbl.itertuples(index=False)}

    dropped_corr: set[str] = set()
    for col in upper.columns:
        high_corr_cols = [idx for idx, val in upper[col].items() if val > corr_threshold]
        if not high_corr_cols:
            continue
        for peer in high_corr_cols:
            if peer in dropped_corr or col in dropped_corr:
                continue
            if rank.get(peer, 0.0) >= rank.get(col, 0.0):
                dropped_corr.add(col)
            else:
                dropped_corr.add(peer)

    selected = [f for f in keep if f not in dropped_corr]
    if len(selected) < min_features:
        for f in imp_tbl["feature"].tolist():
            if f not in selected:
                selected.append(f)
            if len(selected) >= min_features:
                break

    return FeatureSelectionResult(
        selected_features=selected,
        removed_zero_importance=removed_zero,
        removed_high_correlation=sorted(dropped_corr),
        importance_table=imp_tbl,
    )

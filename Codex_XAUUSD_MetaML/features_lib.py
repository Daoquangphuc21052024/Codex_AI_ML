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
    feature_groups: dict[str, str]


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    removed_zero_importance: list[str]
    removed_high_correlation: list[str]
    importance_table: pd.DataFrame


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(n).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(n).mean() / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.rolling(n).mean()
    return adx, plus_di, minus_di


def build_features(
    df: pd.DataFrame,
    periods: list[int],
    periods_meta: list[int],
    atr_window: int = 14,
) -> FeatureSet:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    tickvol = out.get("volume", pd.Series(0.0, index=out.index)).astype(float)
    spread = out.get("spread", pd.Series(0.0, index=out.index)).astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    ret1 = close.pct_change()

    feat: dict[str, pd.Series] = {}
    groups: dict[str, str] = {}

    def add(name: str, series: pd.Series, group: str):
        feat[name] = series
        groups[name] = group

    # Trend
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    add("ema_20", ema20, "trend")
    add("ema_50", ema50, "trend")
    add("ema_200", ema200, "trend")
    add("ema_20_slope_3", ema20.diff(3), "trend")
    add("ema_50_slope_5", ema50.diff(5), "trend")
    add("ema_200_slope_10", ema200.diff(10), "trend")
    add("close_to_ema20", (close / (ema20 + 1e-12)) - 1, "trend")
    add("close_to_ema50", (close / (ema50 + 1e-12)) - 1, "trend")
    add("close_to_ema200", (close / (ema200 + 1e-12)) - 1, "trend")
    ema_stack_bull = ((ema20 > ema50) & (ema50 > ema200)).astype(float)
    ema_stack_bear = ((ema20 < ema50) & (ema50 < ema200)).astype(float)
    add("ema_stack_bull", ema_stack_bull, "trend")
    add("ema_stack_bear", ema_stack_bear, "trend")
    add("trend_persistence_12", (np.sign(ema20.diff()) > 0).rolling(12).mean(), "trend")
    add("trend_persistence_24", (np.sign(ema50.diff()) > 0).rolling(24).mean(), "trend")

    # Momentum
    for p in [3, 6, 12, 24]:
        add(f"roc_{p}", close.pct_change(p), "momentum")
    rsi14 = _rsi(close, 14)
    add("rsi_14", rsi14, "momentum")
    add("rsi_change_3", rsi14.diff(3), "momentum")
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_hist = (macd_fast - macd_slow) - (macd_fast - macd_slow).ewm(span=9, adjust=False).mean()
    add("macd_hist", macd_hist, "momentum")
    add("macd_hist_slope", macd_hist.diff(3), "momentum")
    adx14, plus14, minus14 = _adx(high, low, close, 14)
    add("adx_14", adx14, "momentum")
    add("plus_di_14", plus14, "momentum")
    add("minus_di_14", minus14, "momentum")
    add("di_spread", plus14 - minus14, "momentum")

    # Volatility / expansion
    range_3 = (high.rolling(3).max() - low.rolling(3).min()) / (close + 1e-12)
    range_6 = (high.rolling(6).max() - low.rolling(6).min()) / (close + 1e-12)
    range_12 = (high.rolling(12).max() - low.rolling(12).min()) / (close + 1e-12)
    range_24 = (high.rolling(24).max() - low.rolling(24).min()) / (close + 1e-12)
    add("atr_14", atr_14, "volatility")
    add("atr_rank_100", atr_14.rolling(100).rank(pct=True), "volatility")
    add("atr_rank_250", atr_14.rolling(250).rank(pct=True), "volatility")
    add("true_range_z_50", (tr - tr.rolling(50).mean()) / (tr.rolling(50).std() + 1e-12), "volatility")
    add("range_3", range_3, "volatility")
    add("range_6", range_6, "volatility")
    add("range_12", range_12, "volatility")
    add("range_24", range_24, "volatility")
    add("range_expansion_6_24", range_6 / (range_24 + 1e-12), "volatility")
    add("range_expansion_3_12", range_3 / (range_12 + 1e-12), "volatility")
    add("compression_score_12", tr.rolling(12).mean() / (tr.rolling(48).mean() + 1e-12), "volatility")

    # Structure / breakout
    for p in [6, 12, 24]:
        add(f"break_high_{p}", (close > high.rolling(p).max().shift(1)).astype(float), "structure")
        add(f"break_low_{p}", (close < low.rolling(p).min().shift(1)).astype(float), "structure")
    add("dist_to_high_12", (high.rolling(12).max() - close) / (close + 1e-12), "structure")
    add("dist_to_high_24", (high.rolling(24).max() - close) / (close + 1e-12), "structure")
    add("dist_to_low_12", (close - low.rolling(12).min()) / (close + 1e-12), "structure")
    add("dist_to_low_24", (close - low.rolling(24).min()) / (close + 1e-12), "structure")
    add("hh_count_12", (high > high.shift(1)).rolling(12).sum(), "structure")
    add("ll_count_12", (low < low.shift(1)).rolling(12).sum(), "structure")
    inside = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(float)
    outside = ((high >= high.shift(1)) & (low <= low.shift(1))).astype(float)
    add("inside_bar_flag", inside, "structure")
    add("outside_bar_flag", outside, "structure")

    # Candle microstructure
    candle_range = (high - low).replace(0, np.nan)
    body = close - open_
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    add("body_ratio", body.abs() / (candle_range + 1e-12), "microstructure")
    add("upper_wick_ratio", upper_wick / (candle_range + 1e-12), "microstructure")
    add("lower_wick_ratio", lower_wick / (candle_range + 1e-12), "microstructure")
    add("wick_imbalance", (upper_wick - lower_wick) / (candle_range + 1e-12), "microstructure")
    add("close_near_high", (high - close) / (candle_range + 1e-12), "microstructure")
    add("close_near_low", (close - low) / (candle_range + 1e-12), "microstructure")
    add("bull_body_flag", (close > open_).astype(float), "microstructure")
    add("bear_body_flag", (close < open_).astype(float), "microstructure")

    # Mean reversion
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    std20 = close.rolling(20).std()
    std50 = close.rolling(50).std()
    add("zscore_close_20", (close - ma20) / (std20 + 1e-12), "mean_reversion")
    add("zscore_close_50", (close - ma50) / (std50 + 1e-12), "mean_reversion")
    add("price_vs_mean_20", (close / (ma20 + 1e-12)) - 1, "mean_reversion")
    add("price_vs_mean_50", (close / (ma50 + 1e-12)) - 1, "mean_reversion")
    add("distance_from_ema200", (close / (ema200 + 1e-12)) - 1, "mean_reversion")
    add("reversion_pressure_12", ((close - ema20) / (atr_14 + 1e-12)).rolling(12).mean(), "mean_reversion")

    # Tick volume / effort-result
    tickvol_ma20 = tickvol.rolling(20).mean()
    add("tickvol", tickvol, "volume")
    add("tickvol_ma_20", tickvol_ma20, "volume")
    add("tickvol_z_50", (tickvol - tickvol.rolling(50).mean()) / (tickvol.rolling(50).std() + 1e-12), "volume")
    add("tickvol_rank_100", tickvol.rolling(100).rank(pct=True), "volume")
    add("tickvol_ratio_20", tickvol / (tickvol_ma20 + 1e-12), "volume")
    add("volume_expansion", tickvol.rolling(5).mean() / (tickvol.rolling(30).mean() + 1e-12), "volume")
    add("effort_result_1", ret1.abs() / (tickvol + 1e-12), "volume")
    add("effort_result_3", close.pct_change(3).abs() / (tickvol.rolling(3).sum() + 1e-12), "volume")

    # Regime composite (H1-only)
    bull_score = (
        0.35 * ema_stack_bull
        + 0.25 * (ema20.diff(3) > 0).astype(float)
        + 0.20 * (ema50.diff(5) > 0).astype(float)
        + 0.20 * (adx14 > 20).astype(float)
    )
    bear_score = (
        0.35 * ema_stack_bear
        + 0.25 * (ema20.diff(3) < 0).astype(float)
        + 0.20 * (ema50.diff(5) < 0).astype(float)
        + 0.20 * (adx14 > 20).astype(float)
    )
    trend_strength = (ema20.diff(3).abs() + ema50.diff(5).abs()) / (atr_14 + 1e-12) + (adx14 / 100.0)
    volatility_regime = 0.6 * atr_14.rolling(100).rank(pct=True) + 0.4 * ((tr - tr.rolling(50).mean()) / (tr.rolling(50).std() + 1e-12))
    add("bull_regime_score", bull_score.clip(0, 1), "regime")
    add("bear_regime_score", bear_score.clip(0, 1), "regime")
    add("trend_strength_score", trend_strength, "regime")
    add("volatility_regime_score", volatility_regime, "regime")

    # Compatibility legacy params used elsewhere for export metadata
    meta_feat: dict[str, pd.Series] = {}
    for n in periods_meta:
        s = close.rolling(n)
        meta_feat[f"skew_meta_{n}"] = s.skew()
        meta_feat[f"kurt_meta_{n}"] = s.kurt()
        meta_feat[f"trend_strength_meta_{n}"] = close.ewm(span=n, adjust=False).mean().diff().abs()
    meta_feat["atr_meta"] = atr_14 / (close + 1e-12)

    feature_frame = pd.concat(feat, axis=1)
    meta_frame = pd.concat(meta_feat, axis=1)
    merged = pd.concat([out, feature_frame, meta_frame], axis=1).dropna().copy()

    return FeatureSet(
        data=merged,
        main_features=list(feature_frame.columns),
        meta_features=list(meta_frame.columns),
        feature_groups=groups,
    )


def select_main_features_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    corr_threshold: float = 0.96,
    min_features: int = 20,
    random_seed: int = 42,
) -> FeatureSelectionResult:
    X = X_train[feature_names].copy()
    X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]

    selector_model = CatBoostClassifier(
        iterations=220,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=random_seed,
        verbose=False,
    )
    selector_model.fit(X, y_train)
    imp = selector_model.get_feature_importance()
    imp_tbl = pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values("importance", ascending=False)

    keep = imp_tbl[imp_tbl["importance"] > 0]["feature"].tolist()
    if len(keep) < min_features:
        keep = imp_tbl.head(min_features)["feature"].tolist()
    removed_zero = [f for f in X.columns if f not in keep]

    corr = X[keep].corr().abs().fillna(0.0)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    rank = {row.feature: float(row.importance) for row in imp_tbl.itertuples(index=False)}
    dropped_corr: set[str] = set()
    for col in upper.columns:
        peers = [idx for idx, val in upper[col].items() if val > corr_threshold]
        for peer in peers:
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

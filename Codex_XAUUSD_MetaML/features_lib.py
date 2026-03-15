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
    mtf_context: dict[str, pd.DataFrame] | None = None,
) -> FeatureSet:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]
    tickvol = out.get("volume", pd.Series(0.0, index=out.index)).astype(float)
    spread = out.get("spread", pd.Series(0.0, index=out.index)).astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    feat: dict[str, pd.Series] = {}
    groups: dict[str, str] = {}

    def add(name: str, series: pd.Series, grp: str):
        feat[name] = series
        groups[name] = grp

    # A. Trend
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    add("ema_20", ema20, "trend")
    add("ema_50", ema50, "trend")
    add("ema_100", ema100, "trend")
    add("ema_200", ema200, "trend")
    add("ema_20_slope_3", ema20.diff(3), "trend")
    add("ema_50_slope_5", ema50.diff(5), "trend")
    add("ema_100_slope_8", ema100.diff(8), "trend")
    add("ema_200_slope_13", ema200.diff(13), "trend")
    add("close_to_ema20", (close / (ema20 + 1e-12)) - 1, "trend")
    add("close_to_ema50", (close / (ema50 + 1e-12)) - 1, "trend")
    add("close_to_ema200", (close / (ema200 + 1e-12)) - 1, "trend")
    add("ema_stack_bull", ((ema20 > ema50) & (ema50 > ema100) & (ema100 > ema200)).astype(float), "trend")
    add("ema_stack_bear", ((ema20 < ema50) & (ema50 < ema100) & (ema100 < ema200)).astype(float), "trend")

    # B. Momentum
    for p in [3, 6, 12, 24]:
        add(f"roc_{p}", close.pct_change(p), "momentum")
    add("rsi_7", _rsi(close, 7), "momentum")
    add("rsi_14", _rsi(close, 14), "momentum")
    add("rsi_21", _rsi(close, 21), "momentum")
    adx14, plus14, minus14 = _adx(high, low, close, 14)
    add("adx_14", adx14, "momentum")
    add("plus_di_14", plus14, "momentum")
    add("minus_di_14", minus14, "momentum")
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd = macd_fast - macd_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    add("macd_hist", macd_hist, "momentum")
    add("macd_hist_slope", macd_hist.diff(3), "momentum")

    # C. Volatility / range
    add("atr_14", atr_14, "volatility")
    add("atr_14_pct_close", atr_14 / (close + 1e-12), "volatility")
    add("atr_rank_100", atr_14.rolling(100).rank(pct=True), "volatility")
    add("atr_rank_250", atr_14.rolling(250).rank(pct=True), "volatility")
    for p in [1, 3, 6, 12]:
        hi = high.rolling(p).max()
        lo = low.rolling(p).min()
        add(f"range_{p}", (hi - lo) / (close + 1e-12), "volatility")
    add("true_range_z_50", (tr - tr.rolling(50).mean()) / (tr.rolling(50).std() + 1e-12), "volatility")
    candle_range = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    add("body_to_range", body / (candle_range + 1e-12), "volatility")
    add("upper_wick_to_range", (high - np.maximum(open_, close)) / (candle_range + 1e-12), "volatility")
    add("lower_wick_to_range", (np.minimum(open_, close) - low) / (candle_range + 1e-12), "volatility")

    # D. Breakout / structure
    for p in [6, 12, 24, 48]:
        add(f"break_high_{p}", (close > high.rolling(p).max().shift(1)).astype(float), "structure")
        add(f"break_low_{p}", (close < low.rolling(p).min().shift(1)).astype(float), "structure")
    add("dist_to_high_24", (high.rolling(24).max() - close) / (close + 1e-12), "structure")
    add("dist_to_low_24", (close - low.rolling(24).min()) / (close + 1e-12), "structure")
    add("dist_to_high_72", (high.rolling(72).max() - close) / (close + 1e-12), "structure")
    add("dist_to_low_72", (close - low.rolling(72).min()) / (close + 1e-12), "structure")
    add("hh_count_12", (high > high.shift(1)).rolling(12).sum(), "structure")
    add("ll_count_12", (low < low.shift(1)).rolling(12).sum(), "structure")
    inside = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(float)
    add("inside_bar_count_6", inside.rolling(6).sum(), "structure")
    add("compression_score_12", (tr.rolling(12).mean() / (tr.rolling(48).mean() + 1e-12)), "structure")

    # E. Mean reversion context
    for p in [20, 50, 100]:
        add(f"zscore_close_{p}", (close - close.rolling(p).mean()) / (close.rolling(p).std() + 1e-12), "mean_reversion")
    add("distance_from_ema200", (close - ema200) / (close + 1e-12), "mean_reversion")
    add("reversion_pressure_12", ((close - ema20) / (atr_14 + 1e-12)).rolling(12).mean(), "mean_reversion")

    # F. Candle microstructure
    add("body_pct", (close - open_) / (open_.abs() + 1e-12), "microstructure")
    add("bull_body", (close > open_).astype(float), "microstructure")
    add("bear_body", (close < open_).astype(float), "microstructure")
    add("upper_wick_pct", (high - np.maximum(open_, close)) / (open_.abs() + 1e-12), "microstructure")
    add("lower_wick_pct", (np.minimum(open_, close) - low) / (open_.abs() + 1e-12), "microstructure")
    add("close_near_high", (high - close) / (candle_range + 1e-12), "microstructure")
    add("close_near_low", (close - low) / (candle_range + 1e-12), "microstructure")

    # G. Tick volume
    add("tickvol", tickvol, "volume")
    tv_ma20 = tickvol.rolling(20).mean()
    tv_ma50 = tickvol.rolling(50).mean()
    add("tickvol_ma_20", tv_ma20, "volume")
    add("tickvol_ratio_20", tickvol / (tv_ma20 + 1e-12), "volume")
    add("tickvol_ratio_50", tickvol / (tv_ma50 + 1e-12), "volume")
    add("tickvol_z_50", (tickvol - tickvol.rolling(50).mean()) / (tickvol.rolling(50).std() + 1e-12), "volume")
    add("effort_result_1", (close.diff(1).abs()) / (tickvol + 1e-12), "volume")
    add("effort_result_3", (close.diff(3).abs()) / (tickvol.rolling(3).sum() + 1e-12), "volume")
    add("spread_efficiency", tr / (spread + 1e-12), "volume")

    # H. Session / time context
    if isinstance(out.index, pd.DatetimeIndex):
        hour = pd.Series(out.index.hour, index=out.index)
        dow = pd.Series(out.index.dayofweek, index=out.index)
    else:
        dt = pd.to_datetime(out.index)
        hour = pd.Series(dt.hour, index=out.index)
        dow = pd.Series(dt.dayofweek, index=out.index)
    add("hour_of_day", hour.astype(float), "session")
    add("day_of_week", dow.astype(float), "session")
    is_asia = ((hour >= 0) & (hour < 8)).astype(float)
    is_london = ((hour >= 8) & (hour < 16)).astype(float)
    is_ny = ((hour >= 13) & (hour < 22)).astype(float)
    add("is_asia", is_asia, "session")
    add("is_london", is_london, "session")
    add("is_ny", is_ny, "session")
    add("is_overlap_london_ny", ((hour >= 13) & (hour < 16)).astype(float), "session")

    session_key = pd.Series(np.where(hour < 8, "asia", np.where(hour < 16, "london", "ny")), index=out.index)
    day_key = pd.Series(pd.to_datetime(out.index).date, index=out.index)
    grp = pd.MultiIndex.from_arrays([day_key, session_key])
    session_hi = high.groupby(grp).cummax()
    session_lo = low.groupby(grp).cummin()
    add("session_range_so_far", (session_hi - session_lo) / (close + 1e-12), "session")
    add("session_breakout_up", (close > session_hi.shift(1)).astype(float), "session")
    add("session_breakout_down", (close < session_lo.shift(1)).astype(float), "session")

    # I. MTF hooks (optional)
    if mtf_context:
        h4 = mtf_context.get("H4")
        d1 = mtf_context.get("D1")
        if h4 is not None and "close" in h4:
            h4_close = h4["close"].reindex(out.index, method="ffill")
            h4_ema50 = h4_close.ewm(span=50, adjust=False).mean()
            h4_ema200 = h4_close.ewm(span=200, adjust=False).mean()
            add("H4_ema50_slope", h4_ema50.diff(3), "mtf")
            add("H4_close_to_ema200", (h4_close / (h4_ema200 + 1e-12)) - 1, "mtf")
            add("H4_rsi14", _rsi(h4_close, 14), "mtf")
        if d1 is not None and "close" in d1:
            d1_close = d1["close"].reindex(out.index, method="ffill")
            add("D1_roc_3", d1_close.pct_change(3), "mtf")
            add("D1_trend_flag", (d1_close > d1_close.ewm(span=50, adjust=False).mean()).astype(float), "mtf")

    # legacy features preserved for compatibility
    for n in periods:
        ma = close.rolling(n).mean()
        rolling_std = close.rolling(n).std()
        long_std = close.rolling(n * 3).std()
        roll_low = close.rolling(n).min()
        roll_high = close.rolling(n).max()
        add(f"ma_dev_{n}", (close / (ma + 1e-12)) - 1, "legacy")
        add(f"mom_{n}", (close / (close.shift(n) + 1e-12)) - 1, "legacy")
        add(f"volratio_{n}", rolling_std / (long_std + 1e-12), "legacy")
        add(f"pos_{n}", (close - roll_low) / ((roll_high - roll_low) + 1e-12), "legacy")
        add(f"rsi_like_{n}", (close.diff() > 0).rolling(n).mean(), "legacy")
        add(f"vol_chg_{n}", (tickvol / (tickvol.rolling(n).mean() + 1e-12)) - 1, "legacy")

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
    main_features = list(feature_frame.columns)
    meta_features = list(meta_frame.columns)
    return FeatureSet(data=merged, main_features=main_features, meta_features=meta_features, feature_groups=groups)


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

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


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def _fit(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    return series.rolling(window, min_periods=window).apply(_fit, raw=True)


def engineer_features(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    out = df.copy()
    eps = 1e-9

    out["ret_1"] = out["close"].pct_change(1)
    out["log_ret_1"] = np.log(out["close"] / out["close"].shift(1))

    out["true_range"] = _true_range(out)
    for w in [6, 12, 24]:
        out[f"atr_{w}"] = _atr(out, w)

    # 1) PRICE RETURN FEATURES
    for w in [3, 6, 12, 24]:
        out[f"ret_{w}"] = out["close"].pct_change(w)
        out[f"cum_ret_{w}"] = out["ret_1"].rolling(w, min_periods=w).sum()
    out["ret_accel_1"] = out["ret_1"] - out["ret_1"].shift(1)
    out["dir_strength_6"] = np.sign(out["ret_1"]).rolling(6, min_periods=6).mean()
    out["upside_ret_12"] = out["ret_1"].clip(lower=0).rolling(12, min_periods=12).sum()
    out["downside_ret_12"] = out["ret_1"].clip(upper=0).rolling(12, min_periods=12).sum()

    # 2) RANGE / VOLATILITY FEATURES
    out["hl_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    for w in [6, 12, 24]:
        out[f"natr_{w}"] = out[f"atr_{w}"] / out["close"].replace(0, np.nan)
        out[f"std_ret_{w}"] = out["ret_1"].rolling(w, min_periods=w).std()
        out[f"realized_vol_{w}"] = np.sqrt((out["log_ret_1"] ** 2).rolling(w, min_periods=w).sum())
        out[f"range_mean_{w}"] = out["hl_range"].rolling(w, min_periods=w).mean()
    out["range_expansion_1"] = out["true_range"] / out["atr_12"].replace(0, np.nan)
    out["range_compress_24"] = out["hl_range"].rolling(6, min_periods=6).mean() / out["hl_range"].rolling(24, min_periods=24).mean().replace(0, np.nan)
    out["vol_rank_24_120"] = out["std_ret_24"].rolling(120, min_periods=120).rank(pct=True)

    # 3) BODY / CANDLE STRUCTURE FEATURES
    out["body_size"] = (out["close"] - out["open"]).abs() / out["close"].replace(0, np.nan)
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["close"].replace(0, np.nan)
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["close"].replace(0, np.nan)
    out["body_range_ratio"] = (out["close"] - out["open"]).abs() / (out["high"] - out["low"]).replace(0, np.nan)
    out["wick_asym"] = (out["upper_wick"] - out["lower_wick"]) / (out["hl_range"].replace(0, np.nan))
    out["close_loc_range"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    out["is_expansion_bar"] = (out["true_range"] > out["true_range"].rolling(24, min_periods=24).quantile(0.8)).astype(float)
    out["is_contraction_bar"] = (out["true_range"] < out["true_range"].rolling(24, min_periods=24).quantile(0.2)).astype(float)
    out["bull_streak_5"] = (out["close"] > out["open"]).astype(int).rolling(5, min_periods=5).sum()
    out["bear_streak_5"] = (out["close"] < out["open"]).astype(int).rolling(5, min_periods=5).sum()

    # 4) MOMENTUM / VELOCITY FEATURES
    for w in [3, 6, 12, 24]:
        out[f"velocity_{w}"] = out["close"].pct_change(w)
    out["zscore_move_12"] = (out["close"].diff() - out["close"].diff().rolling(12, min_periods=12).mean()) / out["close"].diff().rolling(12, min_periods=12).std().replace(0, np.nan)
    out["momentum_impulse_12"] = (out["close"] - out["close"].shift(12)) / out["atr_12"].replace(0, np.nan)
    out["momentum_change_6"] = out["velocity_6"] - out["velocity_6"].shift(1)
    out["move_eff_12"] = (out["close"] - out["close"].shift(12)).abs() / out["close"].diff().abs().rolling(12, min_periods=12).sum().replace(0, np.nan)
    out["mom_persist_12"] = np.sign(out["ret_1"]).rolling(12, min_periods=12).mean()

    # 5) TREND / REGIME FEATURES
    for w in [12, 24, 48]:
        out[f"sma_{w}"] = out["close"].rolling(w, min_periods=w).mean()
        out[f"dist_sma_{w}"] = (out["close"] - out[f"sma_{w}"]) / out["atr_12"].replace(0, np.nan)
    out["sma_spread_12_24"] = (out["sma_12"] - out["sma_24"]) / out["atr_12"].replace(0, np.nan)
    out["sma_spread_24_48"] = (out["sma_24"] - out["sma_48"]) / out["atr_24"].replace(0, np.nan)
    out["slope_sma_24"] = _rolling_slope(out["sma_24"], 12)
    out["slope_sma_48"] = _rolling_slope(out["sma_48"], 12)
    out["trend_persist_24"] = np.sign(out["close"] - out["sma_24"]).rolling(24, min_periods=24).mean()
    out["trend_strength_24"] = (out["close"] - out["close"].shift(24)).abs() / out["atr_24"].replace(0, np.nan)
    out["volatility_regime"] = (out["std_ret_24"] > out["std_ret_24"].rolling(120, min_periods=120).median()).astype(float)
    out["directional_regime"] = (out["sma_spread_12_24"] > 0).astype(float)

    # 6) MICROSTRUCTURE-LITE
    out["tick_vol_chg_1"] = out["tick_volume"].pct_change(1)
    out["tick_vol_chg_6"] = out["tick_volume"].pct_change(6)
    out["tick_vol_z_24"] = (out["tick_volume"] - out["tick_volume"].rolling(24, min_periods=24).mean()) / out["tick_volume"].rolling(24, min_periods=24).std().replace(0, np.nan)
    out["pv_div_12"] = out["ret_12"] - out["tick_vol_chg_6"]
    out["range_vol_interact"] = out["hl_range"] * out["tick_vol_z_24"]
    out["body_vol_interact"] = out["body_size"] * out["tick_vol_z_24"]
    out["effort_result_12"] = out["tick_vol_z_24"] / (out["ret_1"].abs() + eps)

    # 7) SESSION / TIME FEATURES
    ts = pd.to_datetime(out["time"], utc=True)
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["session_london"] = ((hour >= 7) & (hour <= 16)).astype(float)
    out["session_ny"] = ((hour >= 12) & (hour <= 21)).astype(float)
    out["session_overlap"] = ((hour >= 12) & (hour <= 16)).astype(float)

    # 8) NORMALIZED FEATURES
    out["atr_norm_ret_1"] = out["ret_1"] / (out["natr_12"] + eps)
    out["atr_norm_body"] = out["body_size"] / (out["natr_12"] + eps)
    out["atr_norm_range"] = out["hl_range"] / (out["natr_12"] + eps)
    out["atr_norm_true_range"] = out["true_range"] / out["atr_24"].replace(0, np.nan)
    out["z_ret_24"] = (out["ret_1"] - out["ret_1"].rolling(24, min_periods=24).mean()) / out["ret_1"].rolling(24, min_periods=24).std().replace(0, np.nan)

    # 9) CONTEXT FEATURES
    out["roll_rank_close_48"] = out["close"].rolling(48, min_periods=48).rank(pct=True)
    out["breakout_prox_24"] = (out["close"] - out["high"].rolling(24, min_periods=24).max()) / out["atr_12"].replace(0, np.nan)
    out["meanrev_dist_24"] = (out["close"] - out["close"].rolling(24, min_periods=24).mean()) / out["atr_12"].replace(0, np.nan)
    out["compression_pre_expand"] = out["std_ret_6"] / out["std_ret_24"].replace(0, np.nan)
    out["swing_pressure_12"] = (np.sign(out["close"] - out["open"]) * out["hl_range"]).rolling(12, min_periods=12).sum()

    # 10) LABEL-ALIGNED FEATURES
    out["tp_potential_buy"] = (out["high"].rolling(12, min_periods=12).max() - out["close"]) / out["atr_12"].replace(0, np.nan)
    out["tp_potential_sell"] = (out["close"] - out["low"].rolling(12, min_periods=12).min()) / out["atr_12"].replace(0, np.nan)
    out["adverse_proxy_buy"] = (out["close"] - out["low"].rolling(12, min_periods=12).min()) / out["atr_12"].replace(0, np.nan)
    out["adverse_proxy_sell"] = (out["high"].rolling(12, min_periods=12).max() - out["close"]) / out["atr_12"].replace(0, np.nan)
    out["rr_context_buy"] = out["tp_potential_buy"] / (out["adverse_proxy_buy"].replace(0, np.nan))
    out["rr_context_sell"] = out["tp_potential_sell"] / (out["adverse_proxy_sell"].replace(0, np.nan))
    out["setup_quality"] = out[["move_eff_12", "trend_strength_24", "body_range_ratio"]].mean(axis=1)

    feature_families: dict[str, str] = {}
    family_map = {
        "price_return": [
            "ret_1", "log_ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "cum_ret_3", "cum_ret_6", "cum_ret_12", "cum_ret_24", "ret_accel_1", "dir_strength_6", "upside_ret_12", "downside_ret_12",
        ],
        "range_volatility": [
            "true_range", "atr_6", "atr_12", "atr_24", "natr_6", "natr_12", "natr_24", "std_ret_6", "std_ret_12", "std_ret_24", "realized_vol_6", "realized_vol_12", "realized_vol_24", "hl_range", "range_mean_6", "range_mean_12", "range_mean_24", "range_expansion_1", "range_compress_24", "vol_rank_24_120",
        ],
        "candle_structure": [
            "body_size", "upper_wick", "lower_wick", "body_range_ratio", "wick_asym", "close_loc_range", "is_expansion_bar", "is_contraction_bar", "bull_streak_5", "bear_streak_5",
        ],
        "momentum_velocity": [
            "velocity_3", "velocity_6", "velocity_12", "velocity_24", "zscore_move_12", "momentum_impulse_12", "momentum_change_6", "move_eff_12", "mom_persist_12",
        ],
        "trend_regime": [
            "sma_12", "sma_24", "sma_48", "dist_sma_12", "dist_sma_24", "dist_sma_48", "sma_spread_12_24", "sma_spread_24_48", "slope_sma_24", "slope_sma_48", "trend_persist_24", "trend_strength_24", "volatility_regime", "directional_regime",
        ],
        "microstructure_lite": [
            "tick_vol_chg_1", "tick_vol_chg_6", "tick_vol_z_24", "pv_div_12", "range_vol_interact", "body_vol_interact", "effort_result_12",
        ],
        "session_time": [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_london", "session_ny", "session_overlap",
        ],
        "normalized": [
            "atr_norm_ret_1", "atr_norm_body", "atr_norm_range", "atr_norm_true_range", "z_ret_24",
        ],
        "context": [
            "roll_rank_close_48", "breakout_prox_24", "meanrev_dist_24", "compression_pre_expand", "swing_pressure_12",
        ],
        "label_aligned": [
            "tp_potential_buy", "tp_potential_sell", "adverse_proxy_buy", "adverse_proxy_sell", "rr_context_buy", "rr_context_sell", "setup_quality",
        ],
    }

    for fam, feats in family_map.items():
        for f in feats:
            feature_families[f] = fam

    candidate_features = [f for f in feature_families.keys() if f in out.columns]
    candidate_features = list(dict.fromkeys(candidate_features))

    out = out.dropna(subset=candidate_features).reset_index(drop=True)
    if len(candidate_features) > 120:
        raise ValueError("Unexpectedly large candidate feature count; please review feature definitions")
    if cfg.max_features > 60:
        raise ValueError("features.max_features must be <= 60 by design")

    return out, candidate_features, feature_families

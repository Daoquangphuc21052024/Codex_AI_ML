from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelingConfig
from .labeling import simulate_signal_outcome


def _select_trade_params(cfg: LabelingConfig) -> tuple[float, float]:
    tp = float(np.median(np.array(cfg.tp_points, dtype=float)))
    sl = float(np.median(np.array(cfg.sl_points, dtype=float)))
    return tp, sl


def run_backtest(pred_df: pd.DataFrame, cfg: LabelingConfig) -> pd.DataFrame:
    out = pred_df.sort_values("time").reset_index(drop=True).copy()
    if "signal" not in out.columns:
        out["signal"] = out["pred"].astype(int)

    tp, sl = _select_trade_params(cfg)
    horizon = cfg.horizon_bars

    realized_r = np.zeros(len(out), dtype=float)
    for i in range(0, len(out) - horizon):
        entry = float(out.iloc[i]["close"])
        future = out.iloc[i + 1 : i + 1 + horizon]
        future_high = future["high"].to_numpy(dtype=float)
        future_low = future["low"].to_numpy(dtype=float)
        signal = int(out.iloc[i]["signal"])
        realized_r[i] = simulate_signal_outcome(
            future_high=future_high,
            future_low=future_low,
            entry=entry,
            signal=signal,
            tp=tp,
            sl=sl,
            tie_breaker=cfg.tie_breaker,
        )

    out["pnl_r"] = realized_r
    out["equity"] = (1.0 + 0.01 * out["pnl_r"]).cumprod()
    out["rolling_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["rolling_max"] - 1.0
    out["is_win"] = out["pnl_r"] > 0
    return out

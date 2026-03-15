from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(pred_df: pd.DataFrame, threshold_no_trade: float) -> pd.DataFrame:
    out = pred_df.copy()

    if "signal" in out.columns:
        signal = out["signal"].astype(int).to_numpy()
    else:
        signal = np.where(
            out["prob_0"] >= threshold_no_trade,
            0,
            np.where(out["prob_1"] >= out["prob_2"], 1, 2),
        )
    out["signal"] = signal

    ret = out["close"].pct_change().fillna(0.0)
    pnl = np.where(out["signal"] == 1, ret, np.where(out["signal"] == 2, -ret, 0.0))
    out["pnl"] = pnl
    out["equity"] = (1.0 + out["pnl"]).cumprod()
    out["rolling_max"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["rolling_max"] - 1.0

    out["is_win"] = out["pnl"] > 0
    return out

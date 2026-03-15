from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig, LabelingConfig
from .labeling import simulate_signal_outcome


@dataclass
class TradeRecord:
    time: str
    fold: int
    source_index: int
    predicted_class: int
    actual_label: int
    entry_price: float
    exit_price: float
    exit_reason: str
    holding_bars: int
    tp_points: float
    sl_points: float
    spread_points: float
    slippage_points: float
    commission_points: float
    confidence: float
    pnl_points: float
    pnl_r: float


def _select_trade_params(cfg: LabelingConfig) -> tuple[float, float]:
    return float(np.median(np.array(cfg.tp_points, dtype=float))), float(np.median(np.array(cfg.sl_points, dtype=float)))


def run_backtest(
    pred_df: pd.DataFrame,
    market_df: pd.DataFrame,
    label_cfg: LabelingConfig,
    bt_cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred = pred_df.sort_values("time").reset_index(drop=True).copy()
    market = market_df.sort_values("time").reset_index(drop=True).copy()

    tp_points, sl_points = _select_trade_params(label_cfg)
    trade_rows: list[TradeRecord] = []

    for row in pred.itertuples(index=False):
        signal = int(row.signal)
        confidence = float(max(row.prob_1, row.prob_2))
        if signal == 0 or confidence < bt_cfg.min_confidence:
            continue

        src_idx = int(row.source_index)
        entry_idx = src_idx if label_cfg.entry_mode == "signal_close" else src_idx + 1
        if entry_idx + label_cfg.horizon_bars >= len(market):
            continue

        entry_base = float(market.iloc[entry_idx]["open"] if label_cfg.entry_mode == "next_open" else market.iloc[src_idx]["close"])
        direction = 1 if signal == 1 else -1
        execution_cost = bt_cfg.spread_points + bt_cfg.slippage_points
        entry_price = entry_base + direction * execution_cost

        future = market.iloc[entry_idx : entry_idx + label_cfg.horizon_bars]
        future_high = future["high"].to_numpy(dtype=float)
        future_low = future["low"].to_numpy(dtype=float)

        outcome_r, exit_reason, holding_bars = simulate_signal_outcome(
            future_high=future_high,
            future_low=future_low,
            entry=entry_price,
            signal=signal,
            tp=tp_points,
            sl=sl_points,
            tie_breaker=label_cfg.tie_breaker,
        )

        if signal == 1:
            if exit_reason == "tp":
                exit_price = entry_price + tp_points
            elif exit_reason == "sl":
                exit_price = entry_price - sl_points
            else:
                exit_price = float(future.iloc[-1]["close"])
            pnl_points = exit_price - entry_price
        else:
            if exit_reason == "tp":
                exit_price = entry_price - tp_points
            elif exit_reason == "sl":
                exit_price = entry_price + sl_points
            else:
                exit_price = float(future.iloc[-1]["close"])
            pnl_points = entry_price - exit_price

        pnl_points -= bt_cfg.commission_points
        pnl_r = pnl_points / max(sl_points, 1e-9)

        trade_rows.append(
            TradeRecord(
                time=str(row.time),
                fold=int(row.fold),
                source_index=src_idx,
                predicted_class=signal,
                actual_label=int(row.label),
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                holding_bars=int(holding_bars),
                tp_points=tp_points,
                sl_points=sl_points,
                spread_points=bt_cfg.spread_points,
                slippage_points=bt_cfg.slippage_points,
                commission_points=bt_cfg.commission_points,
                confidence=confidence,
                pnl_points=float(pnl_points),
                pnl_r=float(pnl_r),
            )
        )

    trade_log = pd.DataFrame([asdict(t) for t in trade_rows])
    if trade_log.empty:
        pred["pnl_r"] = 0.0
        pred["equity"] = 1.0
        pred["rolling_max"] = 1.0
        pred["drawdown"] = 0.0
        pred["is_win"] = False
        return pred, trade_log, {"total_trades": 0}

    curve = pred[["time", "fold", "source_index"]].copy()
    curve["pnl_r"] = 0.0
    trade_pnl = trade_log.groupby("source_index")["pnl_r"].sum()
    curve.loc[curve["source_index"].isin(trade_pnl.index), "pnl_r"] = curve.loc[
        curve["source_index"].isin(trade_pnl.index), "source_index"
    ].map(trade_pnl)

    curve["equity"] = (1.0 + bt_cfg.risk_per_trade * curve["pnl_r"]).cumprod()
    curve["rolling_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["rolling_max"] - 1.0
    curve["is_win"] = curve["pnl_r"] > 0

    wins = trade_log.loc[trade_log["pnl_r"] > 0, "pnl_r"]
    losses = trade_log.loc[trade_log["pnl_r"] < 0, "pnl_r"]
    summary = {
        "total_trades": int(len(trade_log)),
        "winrate": float((trade_log["pnl_r"] > 0).mean()),
        "average_win": float(wins.mean() if not wins.empty else 0.0),
        "average_loss": float(losses.mean() if not losses.empty else 0.0),
        "payoff_ratio": float((wins.mean() / abs(losses.mean())) if (not wins.empty and not losses.empty and losses.mean() != 0) else 0.0),
        "profit_factor": float((wins.sum() / abs(losses.sum())) if (not wins.empty and not losses.empty and losses.sum() != 0) else 0.0),
        "expectancy": float(trade_log["pnl_r"].mean()),
        "max_drawdown": float(curve["drawdown"].min()),
        "final_equity": float(curve["equity"].iloc[-1]),
    }
    return curve, trade_log, summary

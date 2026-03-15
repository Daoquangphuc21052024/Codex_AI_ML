from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity - peak


def _profit_factor(pnl: pd.Series) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def backtest_signals(
    dataset: pd.DataFrame,
    stop: float,
    take: float,
    markup: float,
    max_hold: int = 120,
    signal_shift: int = 1,
) -> tuple[pd.DataFrame, dict]:
    df = dataset.copy()
    required = {"close", "labels", "meta_labels"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    signal = df["labels"].shift(signal_shift)
    confirm = df["meta_labels"].shift(signal_shift)

    trades = []
    close_vals = df["close"].to_numpy()
    idx = df.index

    for i in range(signal_shift, len(df) - 2):
        if confirm.iloc[i] != 1.0:
            continue
        side = int(signal.iloc[i]) if pd.notna(signal.iloc[i]) else None
        if side not in (0, 1):
            continue

        entry = close_vals[i]
        exit_idx = None
        pnl = 0.0
        exit_reason = "timeout"

        for j in range(i + 1, min(i + 1 + max_hold, len(df))):
            price = close_vals[j]
            if side == 0:
                if price - entry - markup >= take:
                    pnl = take - markup
                    exit_idx = j
                    exit_reason = "tp"
                    break
                if entry - price + markup >= stop:
                    pnl = -(stop + markup)
                    exit_idx = j
                    exit_reason = "sl"
                    break
            else:
                if entry - price - markup >= take:
                    pnl = take - markup
                    exit_idx = j
                    exit_reason = "tp"
                    break
                if price - entry + markup >= stop:
                    pnl = -(stop + markup)
                    exit_idx = j
                    exit_reason = "sl"
                    break

        if exit_idx is None:
            exit_idx = min(i + max_hold, len(df) - 1)
            pnl = (close_vals[exit_idx] - entry) - markup if side == 0 else (entry - close_vals[exit_idx]) - markup

        trades.append(
            {
                "entry_time": idx[i],
                "exit_time": idx[exit_idx],
                "side": "buy" if side == 0 else "sell",
                "entry": float(entry),
                "exit": float(close_vals[exit_idx]),
                "bars_held": int(exit_idx - i),
                "pnl": float(pnl),
                "exit_reason": exit_reason,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "sortino_like": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
        return trades_df, metrics

    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    dd = _drawdown(trades_df["cum_pnl"])
    r = trades_df["pnl"]
    downside = r[r < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    total_std = float(r.std(ddof=0)) if len(r) else 0.0

    metrics = {
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["pnl"] > 0).mean()),
        "avg_trade": float(trades_df["pnl"].mean()),
        "profit_factor": _profit_factor(trades_df["pnl"]),
        "expectancy": float(trades_df["pnl"].mean()),
        "pnl": float(trades_df["pnl"].sum()),
        "max_drawdown": float(dd.min()),
        "sharpe_like": float(r.mean() / (total_std + 1e-12)),
        "sortino_like": float(r.mean() / downside_std) if downside_std > 1e-12 else 0.0,
        "long_trades": int((trades_df["side"] == "buy").sum()),
        "short_trades": int((trades_df["side"] == "sell").sum()),
    }
    return trades_df, metrics


def save_backtest_reports(trades_df: pd.DataFrame, symbol: str, out_dir: str = "reports") -> dict[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if trades_df.empty:
        return {}

    paths: dict[str, str] = {}
    eq = trades_df["cum_pnl"].to_numpy()
    dd = _drawdown(trades_df["cum_pnl"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trades_df["exit_time"], eq, color="#1f77b4")
    ax.set_title(f"{symbol} Equity Curve")
    p = f"{out_dir}/{symbol}_equity_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["equity_png"] = p

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(trades_df["exit_time"], dd.to_numpy(), 0, color="#d62728", alpha=0.4)
    ax.set_title(f"{symbol} Drawdown")
    p = f"{out_dir}/{symbol}_drawdown_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["drawdown_png"] = p

    rolling_win = (trades_df["pnl"] > 0).rolling(30, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(trades_df["exit_time"], rolling_win, color="#2ca02c")
    ax.set_ylim(0, 1)
    ax.set_title(f"{symbol} Rolling Win Rate (30)")
    p = f"{out_dir}/{symbol}_rolling_winrate.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["rolling_winrate_png"] = p

    rolling_pnl = trades_df["pnl"].rolling(30, min_periods=5).sum()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(trades_df["exit_time"], rolling_pnl, color="#9467bd")
    ax.set_title(f"{symbol} Rolling PnL (30)")
    p = f"{out_dir}/{symbol}_rolling_pnl.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["rolling_pnl_png"] = p

    trades_df.to_csv(f"{out_dir}/{symbol}_trade_log.csv", index=False)

    # pandas mới đã deprecate/loại bỏ alias "M", dùng "ME" (month-end)
    monthly_base = trades_df.copy()
    monthly_base["exit_time"] = pd.to_datetime(monthly_base["exit_time"], errors="coerce")
    monthly_base = monthly_base.dropna(subset=["exit_time"])
    monthly = monthly_base.set_index("exit_time")["pnl"].resample("ME").agg(["sum", "count", "mean"])
    monthly.to_csv(f"{out_dir}/{symbol}_monthly_summary.csv")
    paths["trade_log_csv"] = f"{out_dir}/{symbol}_trade_log.csv"
    paths["monthly_summary_csv"] = f"{out_dir}/{symbol}_monthly_summary.csv"
    return paths

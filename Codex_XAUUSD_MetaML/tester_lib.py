from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity - equity.cummax()


def _profit_factor(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)


def _mean_or_zero(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else 0.0


def resolve_actions(
    prob_buy: np.ndarray,
    prob_sell: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
    edge_margin: float,
    conflict_mode: str = "no_trade",
) -> np.ndarray:
    """Map probabilities to actions.

    Actions: BUY=0, SELL=1, NO_TRADE=-1.
    """
    assert prob_buy.shape == prob_sell.shape
    assert 0 <= buy_threshold <= 1 and 0 <= sell_threshold <= 1

    buy_signal = (prob_buy >= buy_threshold) & (prob_buy > prob_sell + edge_margin)
    sell_signal = (prob_sell >= sell_threshold) & (prob_sell > prob_buy + edge_margin)

    action = np.full(prob_buy.shape[0], -1, dtype=np.int8)
    action[buy_signal] = 0
    action[sell_signal] = 1

    conflict = (prob_buy >= buy_threshold) & (prob_sell >= sell_threshold) & (action == -1)
    if conflict.any() and conflict_mode == "stronger_edge":
        action[conflict & (prob_buy > prob_sell)] = 0
        action[conflict & (prob_sell > prob_buy)] = 1

    # assert mutually exclusive
    assert not np.any((action == 0) & (action == 1))
    return action


def _side_metrics(trades_df: pd.DataFrame, side: str) -> dict:
    side_df = trades_df[trades_df["side"] == side]
    wins = side_df[side_df["pnl"] > 0]["pnl"]
    losses = side_df[side_df["pnl"] < 0]["pnl"]
    return {
        f"{side}_trades": int(len(side_df)),
        f"{side}_win_rate": float((side_df["pnl"] > 0).mean()) if len(side_df) else 0.0,
        f"{side}_pnl": float(side_df["pnl"].sum()) if len(side_df) else 0.0,
        f"avg_win_{side}": _mean_or_zero(wins),
        f"avg_loss_{side}": _mean_or_zero(losses),
        f"payoff_ratio_{side}": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0,
        f"profit_factor_{side}": _profit_factor(side_df["pnl"]) if len(side_df) else 0.0,
    }


def backtest_probabilities(
    dataset: pd.DataFrame,
    stop: float,
    take: float,
    markup: float,
    buy_threshold: float,
    sell_threshold: float,
    edge_margin: float,
    max_hold: int = 120,
    signal_shift: int = 1,
    conflict_mode: str = "no_trade",
) -> tuple[pd.DataFrame, dict]:
    required = {"close", "prob_buy", "prob_sell"}
    if not required.issubset(dataset.columns):
        raise ValueError(f"Missing required columns: {required - set(dataset.columns)}")

    df = dataset.copy()
    actions = resolve_actions(
        df["prob_buy"].to_numpy(),
        df["prob_sell"].to_numpy(),
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        edge_margin=edge_margin,
        conflict_mode=conflict_mode,
    )
    signal = pd.Series(actions, index=df.index).shift(signal_shift)

    close_vals = df["close"].to_numpy()
    idx = df.index
    trades = []

    for i in range(signal_shift, len(df) - 2):
        side = int(signal.iloc[i]) if pd.notna(signal.iloc[i]) else -1
        if side not in (0, 1):
            continue
        entry = close_vals[i]
        exit_idx = None
        pnl = 0.0
        reason = "timeout"

        for j in range(i + 1, min(i + 1 + max_hold, len(df))):
            price = close_vals[j]
            if side == 0:
                if price - entry - markup >= take:
                    pnl = take - markup
                    exit_idx = j
                    reason = "tp"
                    break
                if entry - price + markup >= stop:
                    pnl = -(stop + markup)
                    exit_idx = j
                    reason = "sl"
                    break
            else:
                if entry - price - markup >= take:
                    pnl = take - markup
                    exit_idx = j
                    reason = "tp"
                    break
                if price - entry + markup >= stop:
                    pnl = -(stop + markup)
                    exit_idx = j
                    reason = "sl"
                    break

        if exit_idx is None:
            exit_idx = min(i + max_hold, len(df) - 1)
            pnl = (close_vals[exit_idx] - entry - markup) if side == 0 else (entry - close_vals[exit_idx] - markup)

        trades.append(
            {
                "entry_time": idx[i],
                "exit_time": idx[exit_idx],
                "side": "buy" if side == 0 else "sell",
                "entry": float(entry),
                "exit": float(close_vals[exit_idx]),
                "bars_held": int(exit_idx - i),
                "pnl": float(pnl),
                "exit_reason": reason,
            }
        )

    trades_df = pd.DataFrame(trades)
    no_trade_bars = int(np.sum(actions == -1))
    buy_actions = int(np.sum(actions == 0))
    sell_actions = int(np.sum(actions == 1))

    if trades_df.empty:
        return trades_df, {
            "trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "no_trade_bars": no_trade_bars,
            "no_trade_ratio": float(no_trade_bars / max(1, len(actions))),
            "win_rate": 0.0,
            "buy_win_rate": 0.0,
            "sell_win_rate": 0.0,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_win_buy": 0.0,
            "avg_loss_buy": 0.0,
            "avg_win_sell": 0.0,
            "avg_loss_sell": 0.0,
            "payoff_ratio": 0.0,
            "payoff_ratio_buy": 0.0,
            "payoff_ratio_sell": 0.0,
            "pnl": 0.0,
            "buy_pnl": 0.0,
            "sell_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
            "sortino_like": 0.0,
            "profit_factor": 0.0,
            "profit_factor_buy": 0.0,
            "profit_factor_sell": 0.0,
            "side_dominance_ratio": 0.0,
            "average_holding_bars": 0.0,
            "buy_actions": buy_actions,
            "sell_actions": sell_actions,
        }

    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    dd = _drawdown(trades_df["cum_pnl"])
    pnl = trades_df["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    downside_std = float(losses.std(ddof=0)) if len(losses) else 0.0
    total_std = float(pnl.std(ddof=0)) if len(pnl) else 0.0

    side_buy = _side_metrics(trades_df, "buy")
    side_sell = _side_metrics(trades_df, "sell")
    side_total = max(1, side_buy["buy_trades"] + side_sell["sell_trades"])

    metrics = {
        "trades": int(len(trades_df)),
        "buy_trades": side_buy["buy_trades"],
        "sell_trades": side_sell["sell_trades"],
        "no_trade_bars": no_trade_bars,
        "no_trade_ratio": float(no_trade_bars / max(1, len(actions))),
        "win_rate": float((pnl > 0).mean()),
        "buy_win_rate": side_buy["buy_win_rate"],
        "sell_win_rate": side_sell["sell_win_rate"],
        "avg_trade": _mean_or_zero(pnl),
        "avg_win": _mean_or_zero(wins),
        "avg_loss": _mean_or_zero(losses),
        "avg_win_buy": side_buy["avg_win_buy"],
        "avg_loss_buy": side_buy["avg_loss_buy"],
        "avg_win_sell": side_sell["avg_win_sell"],
        "avg_loss_sell": side_sell["avg_loss_sell"],
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0,
        "payoff_ratio_buy": side_buy["payoff_ratio_buy"],
        "payoff_ratio_sell": side_sell["payoff_ratio_sell"],
        "pnl": float(pnl.sum()),
        "buy_pnl": side_buy["buy_pnl"],
        "sell_pnl": side_sell["sell_pnl"],
        "max_drawdown": float(dd.min()),
        "sharpe_like": float(pnl.mean() / (total_std + 1e-12)),
        "sortino_like": float(pnl.mean() / downside_std) if downside_std > 1e-12 else 0.0,
        "profit_factor": _profit_factor(pnl),
        "profit_factor_buy": side_buy["profit_factor_buy"],
        "profit_factor_sell": side_sell["profit_factor_sell"],
        "side_dominance_ratio": float(max(side_buy["buy_trades"], side_sell["sell_trades"]) / side_total),
        "average_holding_bars": float(trades_df["bars_held"].mean()),
        "buy_actions": buy_actions,
        "sell_actions": sell_actions,
    }
    return trades_df, metrics


def save_backtest_reports(trades_df: pd.DataFrame, symbol: str, out_dir: str = "reports") -> dict[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if trades_df.empty:
        return {}

    paths: dict[str, str] = {}
    eq = trades_df["cum_pnl"]
    dd = _drawdown(eq)

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
    monthly_base = trades_df.copy()
    monthly_base["exit_time"] = pd.to_datetime(monthly_base["exit_time"], errors="coerce")
    monthly = monthly_base.dropna(subset=["exit_time"]).set_index("exit_time")["pnl"].resample("ME").agg(["sum", "count", "mean"])
    monthly.to_csv(f"{out_dir}/{symbol}_monthly_summary.csv")

    by_side = trades_df.groupby("side")["pnl"].sum().rename("pnl")
    by_side.to_csv(f"{out_dir}/{symbol}_pnl_by_side.csv")

    paths["trade_log_csv"] = f"{out_dir}/{symbol}_trade_log.csv"
    paths["monthly_summary_csv"] = f"{out_dir}/{symbol}_monthly_summary.csv"
    paths["pnl_by_side_csv"] = f"{out_dir}/{symbol}_pnl_by_side.csv"
    return paths

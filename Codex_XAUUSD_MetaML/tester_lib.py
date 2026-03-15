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


def resolve_actions(
    prob_buy: np.ndarray,
    prob_sell: np.ndarray,
    buy_threshold: float | np.ndarray,
    sell_threshold: float | np.ndarray,
    edge_margin: float,
    conflict_mode: str = "no_trade",
) -> np.ndarray:
    """Map probabilities to actions.

    Actions: BUY=0, SELL=1, NO_TRADE=-1.
    """
    if np.isscalar(buy_threshold):
        buy_thr = np.full(prob_buy.shape[0], float(buy_threshold), dtype=float)
    else:
        buy_thr = np.asarray(buy_threshold, dtype=float)
    if np.isscalar(sell_threshold):
        sell_thr = np.full(prob_sell.shape[0], float(sell_threshold), dtype=float)
    else:
        sell_thr = np.asarray(sell_threshold, dtype=float)

    buy_signal = (prob_buy >= buy_thr) & (prob_buy > prob_sell + edge_margin)
    sell_signal = (prob_sell >= sell_thr) & (prob_sell > prob_buy + edge_margin)

    action = np.full(prob_buy.shape[0], -1, dtype=np.int8)
    action[buy_signal] = 0
    action[sell_signal] = 1

    conflict = (prob_buy >= buy_thr) & (prob_sell >= sell_thr) & (action == -1)
    if conflict.any() and conflict_mode == "stronger_edge":
        action[conflict & (prob_buy > prob_sell)] = 0
        action[conflict & (prob_sell > prob_buy)] = 1

    return action


def _calc_trade_cost(
    spread_points: float,
    commission: float,
    slippage_points: float,
    spread_col_val: float | None = None,
) -> float:
    spread_component = float(spread_col_val) if spread_col_val is not None else float(spread_points)
    return spread_component + float(commission) + float(slippage_points)


def backtest_probabilities(
    dataset: pd.DataFrame,
    stop: float,
    take: float,
    markup: float,
    buy_threshold: float | np.ndarray,
    sell_threshold: float | np.ndarray,
    edge_margin: float,
    max_hold: int = 120,
    signal_shift: int = 1,
    conflict_mode: str = "no_trade",
    allow_overlap: bool = False,
    entry_mode: str = "next_open",
    spread_points: float = 0.0,
    commission: float = 0.0,
    slippage_points: float = 0.0,
    use_spread_column: bool = False,
    same_bar_conflict: str = "sl_first",
) -> tuple[pd.DataFrame, dict]:
    required = {"open", "high", "low", "close", "prob_buy", "prob_sell"}
    if not required.issubset(dataset.columns):
        raise ValueError(f"Missing required columns: {required - set(dataset.columns)}")
    if entry_mode not in {"close", "next_open"}:
        raise ValueError("entry_mode must be close or next_open")
    if same_bar_conflict not in {"sl_first", "tp_first"}:
        raise ValueError("same_bar_conflict must be sl_first or tp_first")

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

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    spr = df["spread"].to_numpy(dtype=float) if "spread" in df.columns else np.zeros(len(df), dtype=float)

    trades: list[dict] = []
    overlap_count = 0
    exposure_bars = np.zeros(len(df), dtype=np.int8)
    active_until = -1

    for i in range(signal_shift, len(df) - 1):
        side = int(signal.iloc[i]) if pd.notna(signal.iloc[i]) else -1
        if side not in (0, 1):
            continue

        if not allow_overlap and i <= active_until:
            continue
        if allow_overlap and i <= active_until:
            overlap_count += 1

        entry_idx = i + 1 if entry_mode == "next_open" and i + 1 < len(df) else i
        entry = o[entry_idx] if entry_mode == "next_open" and entry_idx < len(df) else c[i]
        cost = _calc_trade_cost(
            spread_points=spread_points + markup,
            commission=commission,
            slippage_points=slippage_points,
            spread_col_val=spr[entry_idx] if use_spread_column else None,
        )

        tp_price = entry + take if side == 0 else entry - take
        sl_price = entry - stop if side == 0 else entry + stop

        exit_idx = None
        exit_reason = "timeout"
        raw_move = 0.0

        for j in range(entry_idx + 1, min(entry_idx + 1 + max_hold, len(df))):
            if side == 0:
                hit_tp = h[j] >= tp_price
                hit_sl = l[j] <= sl_price
            else:
                hit_tp = l[j] <= tp_price
                hit_sl = h[j] >= sl_price

            if hit_tp and hit_sl:
                if same_bar_conflict == "sl_first":
                    exit_idx = j
                    exit_reason = "sl_conflict"
                    raw_move = -stop
                else:
                    exit_idx = j
                    exit_reason = "tp_conflict"
                    raw_move = take
                break
            if hit_tp:
                exit_idx = j
                exit_reason = "tp"
                raw_move = take
                break
            if hit_sl:
                exit_idx = j
                exit_reason = "sl"
                raw_move = -stop
                break

        if exit_idx is None:
            exit_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_reason = "timeout"
            raw_move = (c[exit_idx] - entry) if side == 0 else (entry - c[exit_idx])

        pnl = raw_move - cost
        active_until = max(active_until, exit_idx)
        exposure_bars[entry_idx : exit_idx + 1] = 1

        trades.append(
            {
                "entry_time": df.index[entry_idx],
                "exit_time": df.index[exit_idx],
                "side": "buy" if side == 0 else "sell",
                "entry": float(entry),
                "exit": float(c[exit_idx]),
                "bars_held": int(exit_idx - entry_idx),
                "cost": float(cost),
                "raw_move": float(raw_move),
                "pnl": float(pnl),
                "exit_reason": exit_reason,
            }
        )

    trades_df = pd.DataFrame(trades)
    no_trade_bars = int(np.sum(actions == -1))
    buy_actions = int(np.sum(actions == 0))
    sell_actions = int(np.sum(actions == 1))

    base_metrics = {
        "trades": int(len(trades_df)),
        "buy_trades": int((trades_df.get("side") == "buy").sum()) if not trades_df.empty else 0,
        "sell_trades": int((trades_df.get("side") == "sell").sum()) if not trades_df.empty else 0,
        "no_trade_bars": no_trade_bars,
        "no_trade_ratio": float(no_trade_bars / max(1, len(actions))),
        "buy_actions": buy_actions,
        "sell_actions": sell_actions,
        "overlap_count": int(overlap_count),
        "allow_overlap": bool(allow_overlap),
        "cost_model": {
            "spread_points": float(spread_points + markup),
            "commission": float(commission),
            "slippage_points": float(slippage_points),
            "use_spread_column": bool(use_spread_column),
        },
        "entry_mode": entry_mode,
        "exposure_ratio": float(exposure_bars.mean()),
    }

    if trades_df.empty:
        zero = {
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
            "median_holding_bars": 0.0,
            "exit_reason_counts": {},
            "exit_reason_counts_buy": {},
            "exit_reason_counts_sell": {},
            "tp_count": 0,
            "sl_count": 0,
            "timeout_count": 0,
        }
        return trades_df, {**base_metrics, **zero}

    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    pnl = trades_df["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    dd = _drawdown(trades_df["cum_pnl"])

    buy_df = trades_df[trades_df["side"] == "buy"]
    sell_df = trades_df[trades_df["side"] == "sell"]

    def side_stats(side_df: pd.DataFrame, name: str) -> dict:
        p = side_df["pnl"]
        w = p[p > 0]
        ls = p[p < 0]
        return {
            f"{name}_win_rate": float((p > 0).mean()) if len(p) else 0.0,
            f"{name}_pnl": float(p.sum()) if len(p) else 0.0,
            f"avg_win_{name}": float(w.mean()) if len(w) else 0.0,
            f"avg_loss_{name}": float(ls.mean()) if len(ls) else 0.0,
            f"payoff_ratio_{name}": float(w.mean() / abs(ls.mean())) if len(w) and len(ls) else 0.0,
            f"profit_factor_{name}": _profit_factor(p) if len(p) else 0.0,
            f"exit_reason_counts_{name}": side_df["exit_reason"].value_counts().to_dict() if len(side_df) else {},
        }

    s_buy = side_stats(buy_df, "buy")
    s_sell = side_stats(sell_df, "sell")

    downside = pnl[pnl < 0]
    metrics = {
        **base_metrics,
        "win_rate": float((pnl > 0).mean()),
        "avg_trade": float(pnl.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0,
        "pnl": float(pnl.sum()),
        "max_drawdown": float(dd.min()),
        "sharpe_like": float(pnl.mean() / (pnl.std(ddof=0) + 1e-12)),
        "sortino_like": float(pnl.mean() / (downside.std(ddof=0) + 1e-12)) if len(downside) else 0.0,
        "profit_factor": _profit_factor(pnl),
        "side_dominance_ratio": float(max(len(buy_df), len(sell_df)) / max(1, len(trades_df))),
        "average_holding_bars": float(trades_df["bars_held"].mean()),
        "median_holding_bars": float(trades_df["bars_held"].median()),
        "exit_reason_counts": trades_df["exit_reason"].value_counts().to_dict(),
        "tp_count": int(trades_df["exit_reason"].isin(["tp", "tp_conflict"]).sum()),
        "sl_count": int(trades_df["exit_reason"].isin(["sl", "sl_conflict"]).sum()),
        "timeout_count": int((trades_df["exit_reason"] == "timeout").sum()),
        **s_buy,
        **s_sell,
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

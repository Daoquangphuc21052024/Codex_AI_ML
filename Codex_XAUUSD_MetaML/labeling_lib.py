from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def create_dual_edge_labels(
    open_: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_window: int = 14,
    tp_atr_buy: float = 1.2,
    sl_atr_buy: float = 1.0,
    tp_atr_sell: float = 1.2,
    sl_atr_sell: float = 1.0,
    max_holding_bars: int = 12,
    entry_mode: str = "next_open",
    same_bar_conflict: str = "sl_first",
) -> pd.DataFrame:
    """Create H1 dual-edge labels using OHLC first-touch logic.

    y_buy=1 when BUY TP is touched before BUY SL in lookahead window, else 0.
    y_sell=1 when SELL TP is touched before SELL SL in lookahead window, else 0.

    If TP and SL are both touched in the same bar, conservative rule applies:
    that side is treated as failure (SL first).

    entry_mode:
      - "close": entry at close[i]
      - "next_open": entry at open[i+1] (when available)
    """
    if entry_mode not in {"close", "next_open"}:
        raise ValueError("entry_mode must be 'close' or 'next_open'")
    if same_bar_conflict not in {"sl_first", "tp_first"}:
        raise ValueError("same_bar_conflict must be 'sl_first' or 'tp_first'")

    atr = _atr(high, low, close, atr_window)
    n = len(close)
    y_buy = np.zeros(n, dtype=np.int16)
    y_sell = np.zeros(n, dtype=np.int16)

    o = open_.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float)

    buy_tp_count = 0
    sell_tp_count = 0
    buy_fail_count = 0
    sell_fail_count = 0

    for i in range(n - 1):
        if entry_mode == "next_open":
            if i + 1 >= n:
                continue
            entry = o[i + 1]
            start_j = i + 1
        else:
            entry = c[i]
            start_j = i + 1

        if not np.isfinite(entry) or not np.isfinite(a[i]) or a[i] <= 0:
            continue

        buy_tp = entry + tp_atr_buy * a[i]
        buy_sl = entry - sl_atr_buy * a[i]
        sell_tp = entry - tp_atr_sell * a[i]
        sell_sl = entry + sl_atr_sell * a[i]

        buy_done = False
        sell_done = False

        for j in range(start_j, min(start_j + max_holding_bars, n)):
            if not buy_done:
                hit_tp = h[j] >= buy_tp
                hit_sl = l[j] <= buy_sl
                if hit_tp and hit_sl:
                    if same_bar_conflict == "tp_first":
                        y_buy[i] = 1
                        buy_tp_count += 1
                    else:
                        buy_fail_count += 1
                    buy_done = True
                elif hit_tp:
                    y_buy[i] = 1
                    buy_tp_count += 1
                    buy_done = True
                elif hit_sl:
                    buy_fail_count += 1
                    buy_done = True

            if not sell_done:
                hit_tp = l[j] <= sell_tp
                hit_sl = h[j] >= sell_sl
                if hit_tp and hit_sl:
                    if same_bar_conflict == "tp_first":
                        y_sell[i] = 1
                        sell_tp_count += 1
                    else:
                        sell_fail_count += 1
                    sell_done = True
                elif hit_tp:
                    y_sell[i] = 1
                    sell_tp_count += 1
                    sell_done = True
                elif hit_sl:
                    sell_fail_count += 1
                    sell_done = True

            if buy_done and sell_done:
                break

    out = pd.DataFrame({"y_buy": y_buy, "y_sell": y_sell}, index=close.index)
    out["atr"] = atr
    out["direction_label"] = np.select(
        [
            (out["y_buy"] == 1) & (out["y_sell"] == 0),
            (out["y_buy"] == 0) & (out["y_sell"] == 1),
        ],
        [0, 1],
        default=2,
    ).astype(np.int16)
    out.attrs["label_counters"] = {
        "buy_tp_positive_count": buy_tp_count,
        "sell_tp_positive_count": sell_tp_count,
        "buy_failure_count": buy_fail_count,
        "sell_failure_count": sell_fail_count,
        "entry_mode": entry_mode,
        "same_bar_conflict": same_bar_conflict,
        "barrier_type": "atr",
        "max_holding_bars": max_holding_bars,
    }
    return out


def create_labels(close: pd.Series, high: pd.Series, low: pd.Series, train_ratio: float = 0.6) -> pd.DataFrame:
    open_ = close.shift(1).fillna(close)
    return create_dual_edge_labels(open_=open_, close=close, high=high, low=low)


def split_label_diagnostics(y_buy: pd.Series, y_sell: pd.Series, train_ratio: float = 0.6, val_ratio: float = 0.2) -> dict:
    n = len(y_buy)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    def stat(sb: pd.Series, ss: pd.Series) -> dict:
        both_pos = ((sb == 1) & (ss == 1)).sum()
        both_zero = ((sb == 0) & (ss == 0)).sum()
        return {
            "n": int(len(sb)),
            "buy_positive": int(sb.sum()),
            "sell_positive": int(ss.sum()),
            "buy_positive_rate": float(sb.mean()) if len(sb) else 0.0,
            "sell_positive_rate": float(ss.mean()) if len(ss) else 0.0,
            "both_positive_count": int(both_pos),
            "both_positive_rate": float(both_pos / max(1, len(sb))),
            "both_zero_count": int(both_zero),
            "both_zero_rate": float(both_zero / max(1, len(sb))),
        }

    return {
        "semantic": {"action_buy": 0, "action_sell": 1, "action_no_trade": -1},
        "train": stat(y_buy.iloc[:i_val], y_sell.iloc[:i_val]),
        "val": stat(y_buy.iloc[i_val:i_test], y_sell.iloc[i_val:i_test]),
        "test": stat(y_buy.iloc[i_test:], y_sell.iloc[i_test:]),
    }


def evaluate_label_quality(df: pd.DataFrame, out_dir: str = "reports") -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    y_buy = df["y_buy"].astype(int)
    y_sell = df["y_sell"].astype(int)
    both_pos = ((y_buy == 1) & (y_sell == 1)).sum()
    both_zero = ((y_buy == 0) & (y_sell == 0)).sum()

    summary = {
        "buy_positive_rate": float(y_buy.mean()),
        "sell_positive_rate": float(y_sell.mean()),
        "buy_positive_count": int(y_buy.sum()),
        "sell_positive_count": int(y_sell.sum()),
        "buy_failure_count": int((y_buy == 0).sum()),
        "sell_failure_count": int((y_sell == 0).sum()),
        "both_positive_count": int(both_pos),
        "both_positive_rate": float(both_pos / max(1, len(df))),
        "both_zero_count": int(both_zero),
        "both_zero_rate": float(both_zero / max(1, len(df))),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y_buy.value_counts().sort_index().plot(kind="bar", ax=axes[0], color=["#c7e9c0", "#31a354"])
    axes[0].set_title("BUY edge labels (0/1)")
    y_sell.value_counts().sort_index().plot(kind="bar", ax=axes[1], color=["#fcbba1", "#de2d26"])
    axes[1].set_title("SELL edge labels (0/1)")
    png = f"{out_dir}/label_distribution_dual.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame([summary]).to_csv(f"{out_dir}/label_quality_dual.csv", index=False)
    pd.Series(summary).to_json(f"{out_dir}/label_quality_dual.json", indent=2)
    return {"summary": summary, "label_plot": png}

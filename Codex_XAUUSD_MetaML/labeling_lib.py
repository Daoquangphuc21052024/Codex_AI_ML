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
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_window: int = 14,
    tp_atr_buy: float = 1.4,
    sl_atr_buy: float = 1.2,
    tp_atr_sell: float = 1.4,
    sl_atr_sell: float = 1.2,
    max_holding_bars: int = 12,
) -> pd.DataFrame:
    """Create dual-edge labels.

    Semantics:
    - y_buy=1 if BUY TP is touched before BUY SL inside horizon, else 0.
    - y_sell=1 if SELL TP is touched before SELL SL inside horizon, else 0.

    A bar can be (0,0), (1,0), (0,1), or (1,1) depending on first-touch paths
    per side. If TP and SL touch in the same candle, tie-break is conservative:
    SL wins for that side.
    """
    atr = _atr(high, low, close, atr_window)
    n = len(close)
    y_buy = np.zeros(n, dtype=np.int16)
    y_sell = np.zeros(n, dtype=np.int16)

    c = close.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    a = atr.to_numpy()

    for i in range(n - max_holding_bars):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue

        entry = c[i]
        buy_tp = entry + tp_atr_buy * a[i]
        buy_sl = entry - sl_atr_buy * a[i]
        sell_tp = entry - tp_atr_sell * a[i]
        sell_sl = entry + sl_atr_sell * a[i]

        buy_done = False
        sell_done = False

        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            if not buy_done:
                buy_hit_tp = h[j] >= buy_tp
                buy_hit_sl = l[j] <= buy_sl
                if buy_hit_tp and not buy_hit_sl:
                    y_buy[i] = 1
                    buy_done = True
                elif buy_hit_sl:
                    # sl-only OR both touched -> conservative fail
                    y_buy[i] = 0
                    buy_done = True

            if not sell_done:
                sell_hit_tp = l[j] <= sell_tp
                sell_hit_sl = h[j] >= sell_sl
                if sell_hit_tp and not sell_hit_sl:
                    y_sell[i] = 1
                    sell_done = True
                elif sell_hit_sl:
                    y_sell[i] = 0
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
    return out


def create_labels(close: pd.Series, high: pd.Series, low: pd.Series, train_ratio: float = 0.6) -> pd.DataFrame:
    """Compatibility wrapper for previous pipeline entrypoints."""
    return create_dual_edge_labels(close, high, low)


def split_label_diagnostics(y_buy: pd.Series, y_sell: pd.Series, train_ratio: float = 0.6, val_ratio: float = 0.2) -> dict:
    n = len(y_buy)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    def stat(sb: pd.Series, ss: pd.Series) -> dict:
        return {
            "n": int(len(sb)),
            "buy_positive": int(sb.sum()),
            "sell_positive": int(ss.sum()),
            "buy_positive_rate": float(sb.mean()) if len(sb) else 0.0,
            "sell_positive_rate": float(ss.mean()) if len(ss) else 0.0,
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

    summary = {
        "buy_positive_rate": float(y_buy.mean()),
        "sell_positive_rate": float(y_sell.mean()),
        "buy_positive_count": int(y_buy.sum()),
        "sell_positive_count": int(y_sell.sum()),
        "both_positive_count": int(((y_buy == 1) & (y_sell == 1)).sum()),
        "both_zero_count": int(((y_buy == 0) & (y_sell == 0)).sum()),
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

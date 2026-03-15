from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _calc_r2(y: list[float]) -> float:
    if len(y) < 3:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    y_arr = np.asarray(y, dtype=float)
    coef = np.polyfit(x, y_arr, 1)
    pred = np.poly1d(coef)(x)
    ss_res = np.sum((y_arr - pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def _save_report_png(equity_curve, symbol, trades, r2, backward, forward, tag=""):
    os.makedirs("reports", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    eq = np.asarray(equity_curve)
    ax1.plot(eq, color="#1f77b4", lw=1.5, label="Equity")
    ideal = np.poly1d(np.polyfit(range(len(eq)), eq, 1))(range(len(eq)))
    ax1.plot(ideal, "--", color="#ff7f0e", lw=1.5, label="Ideal line")
    ax1.fill_between(range(len(eq)), eq, 0, where=eq >= 0, alpha=0.15, color="green")
    ax1.fill_between(range(len(eq)), eq, 0, where=eq < 0, alpha=0.15, color="red")
    ax1.set_title(f"Equity Curve | Trades: {trades} | R²: {r2:.4f}")
    ax1.legend()

    diffs = np.diff(eq)
    diffs = diffs[diffs != 0]
    pos, neg = diffs[diffs > 0], diffs[diffs < 0]
    if len(pos) > 0:
        ax2.hist(pos, bins=max(1, min(30, len(np.unique(pos)))), color="green", alpha=0.6, label=f"Win ({len(pos)})")
    if len(neg) > 0:
        ax2.hist(neg, bins=max(1, min(30, len(np.unique(neg)))), color="red", alpha=0.6, label=f"Loss ({len(neg)})")
    win_rate = len(pos) / len(diffs) * 100 if len(diffs) > 0 else 0
    ax2.set_title(f"Trade Distribution | Win rate: {win_rate:.1f}%")
    ax2.legend(loc="best")

    plt.suptitle(f"{symbol} | {backward.date()} → {forward.date()}")
    fname = f"reports/{symbol}_{tag}_{datetime.now():%Y%m%d_%H%M}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fname}")
    return fname


def tester(dataset: pd.DataFrame, stop: float, take: float, forward, backward, markup: float, plt_show: bool, symbol: str, tag: str = "") -> float:
    df = dataset[(dataset.index >= backward) & (dataset.index <= forward)].copy()
    if df.empty:
        return float("nan")

    equity = 0.0
    equity_curve = [0.0]
    trades = 0

    for i in range(len(df) - 1):
        if df["meta_labels"].iloc[i] != 1.0:
            equity_curve.append(equity)
            continue

        signal = df["labels"].iloc[i]
        entry = df["close"].iloc[i]
        traded = False

        for j in range(i + 1, min(i + 200, len(df))):
            price = df["close"].iloc[j]
            if signal == 0.0:
                if price - entry - markup >= take:
                    equity += take - markup
                    traded = True
                    break
                if entry - price + markup >= stop:
                    equity -= stop + markup
                    traded = True
                    break
            else:
                if entry - price - markup >= take:
                    equity += take - markup
                    traded = True
                    break
                if price - entry + markup >= stop:
                    equity -= stop + markup
                    traded = True
                    break

        if traded:
            trades += 1
        equity_curve.append(equity)

    r2 = _calc_r2(equity_curve)
    if plt_show:
        _save_report_png(equity_curve, symbol, trades, r2, backward, forward, tag)
    return r2

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42


def _train_quantile(series: pd.Series, quantiles: list[float], train_ratio: float = 0.6) -> list[float]:
    split = int(len(series) * train_ratio)
    train = series.iloc[:split] if split > 10 else series
    return train.quantile(quantiles).to_list()


def create_triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int = 12,
    vol_window: int = 48,
    up_mult: float = 1.5,
    dn_mult: float = 1.5,
) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    vol = ret.rolling(vol_window, min_periods=vol_window).std().fillna(ret.std())
    labels = np.full(len(close), 2, dtype=np.int16)

    c = close.to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    v = vol.to_numpy()

    for i in range(len(close) - horizon):
        entry = c[i]
        up_barrier = entry * (1 + up_mult * v[i])
        dn_barrier = entry * (1 - dn_mult * v[i])

        lbl = 2
        for j in range(i + 1, i + horizon + 1):
            if h[j] >= up_barrier:
                lbl = 0
                break
            if l[j] <= dn_barrier:
                lbl = 1
                break
        labels[i] = lbl

    return pd.Series(labels, index=close.index, name="labels")


def create_meta_target(close: pd.Series, train_ratio: float = 0.6) -> pd.Series:
    abs_ret = close.pct_change().abs().fillna(0.0)
    q20, q80 = _train_quantile(abs_ret, [0.2, 0.8], train_ratio=train_ratio)
    return ((abs_ret >= q20) & (abs_ret <= q80)).astype(np.int16).rename("meta_target")


def create_labels(close: pd.Series, high: pd.Series, low: pd.Series, train_ratio: float = 0.6) -> pd.DataFrame:
    labels = create_triple_barrier_labels(close, high, low)
    meta_target = create_meta_target(close, train_ratio=train_ratio)
    return pd.DataFrame({"labels": labels, "meta_target": meta_target}, index=close.index)


def evaluate_label_quality(df: pd.DataFrame, out_dir: str = "reports") -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    labels = df["labels"].dropna().astype(int)
    binary = labels[labels.isin([0, 1])]

    class_counts = binary.value_counts().to_dict()
    total = int(binary.shape[0]) if len(binary) else 1
    buy_ratio = class_counts.get(0, 0) / total
    sell_ratio = class_counts.get(1, 0) / total
    imbalance = abs(buy_ratio - sell_ratio)

    fwd = df["close"].pct_change(12).shift(-12)
    hit_buy = ((labels == 0) & (fwd > 0)).sum()
    hit_sell = ((labels == 1) & (fwd < 0)).sum()
    denom = ((labels == 0) | (labels == 1)).sum()
    hit_rate = float((hit_buy + hit_sell) / denom) if denom else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    binary.value_counts().sort_index().plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title("Label Distribution (0=Buy, 1=Sell)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    label_png = f"{out_dir}/label_distribution.png"
    fig.savefig(label_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "class_counts": class_counts,
        "buy_ratio": float(buy_ratio),
        "sell_ratio": float(sell_ratio),
        "imbalance_abs": float(imbalance),
        "hit_rate_h12": hit_rate,
        "label_horizon": 12,
        "noise_ratio": float((labels == 2).mean()),
    }

    pd.DataFrame([summary]).to_csv(f"{out_dir}/label_quality.csv", index=False)
    pd.Series(summary).to_json(f"{out_dir}/label_quality.json", indent=2)
    return {"summary": summary, "label_plot": label_png}

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DataLoadConfig:
    filepath: str
    timezone: str = "UTC"


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=mapping)

    aliases = {
        "date": "time",
        "datetime": "time",
        "timestamp": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "volume",
        "real_volume": "volume",
        "tickvol": "volume",
        "vol": "volume",
        "volume": "volume",
    }

    out = {}
    for c in df.columns:
        key = aliases.get(c, c)
        out.setdefault(key, c)

    required = ["time", "open", "high", "low", "close"]
    for c in required:
        if c not in out:
            raise ValueError(f"Missing required column: {c}")

    result = pd.DataFrame({k: df[v] for k, v in out.items() if k in {"time", "open", "high", "low", "close", "volume"}})
    if "volume" not in result.columns:
        result["volume"] = 0.0
    return result


def read_price_csv(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(filepath)

    header = path.open("r", encoding="utf-8").readline().strip()
    if "<DATE>" in header or "<CLOSE>" in header:
        raw = pd.read_csv(path, sep=r"\s+")
        df = pd.DataFrame(
            {
                "time": raw["<DATE>"].astype(str) + " " + raw["<TIME>"].astype(str),
                "open": raw["<OPEN>"],
                "high": raw["<HIGH>"],
                "low": raw["<LOW>"],
                "close": raw["<CLOSE>"],
                "volume": raw.get("<TICKVOL>", 0),
            }
        )
    else:
        raw = pd.read_csv(path)
        df = _standardize_columns(raw)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    df = df.sort_values("time")
    df = df[~df["time"].duplicated(keep="last")]
    df = df.set_index("time")
    return df


def build_split_visualization(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> dict[str, str]:
    import matplotlib.pyplot as plt

    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["close"], color="#1f77b4", linewidth=1)
    ax.axvspan(df.index.min(), df.index[i_train - 1], alpha=0.15, color="green", label="train")
    ax.axvspan(df.index[i_train], df.index[i_val - 1], alpha=0.15, color="orange", label="val")
    ax.axvspan(df.index[i_val], df.index.max(), alpha=0.15, color="red", label="test")
    ax.legend(loc="best")
    ax.set_title("Train/Val/Test split over time")

    out = "reports/split_visualization.png"
    Path("reports").mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"split_plot": out}


def generate_synthetic_ohlc(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="H")
    drift = 0.00005
    vol = 0.003
    returns = drift + rng.normal(0, vol, size=n)
    close = 1200 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0015, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0015, size=n))
    volume = rng.integers(100, 2000, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

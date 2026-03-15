from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out = out[REQUIRED_COLUMNS].drop_duplicates(subset=["time"]).sort_values("time")
    out = out.dropna(subset=REQUIRED_COLUMNS)

    out["time"] = pd.to_datetime(out["time"], utc=True)
    for col in ["open", "high", "low", "close", "tick_volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna().reset_index(drop=True)
    return out

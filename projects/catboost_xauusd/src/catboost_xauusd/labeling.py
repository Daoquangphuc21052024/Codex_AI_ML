from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelingConfig


@dataclass
class LabelDiagnostics:
    label_counts: dict[int, int]
    combos_tested: int


def _first_hit_direction(
    future_high: np.ndarray,
    future_low: np.ndarray,
    entry: float,
    tp: float,
    sl: float,
    tie_breaker: str,
) -> int:
    buy_tp = entry + tp
    buy_sl = entry - sl
    sell_tp = entry - tp
    sell_sl = entry + sl

    for h, l in zip(future_high, future_low):
        buy_tp_hit = h >= buy_tp
        buy_sl_hit = l <= buy_sl
        sell_tp_hit = l <= sell_tp
        sell_sl_hit = h >= sell_sl

        buy_event = None
        if buy_tp_hit and buy_sl_hit:
            buy_event = "tp" if tie_breaker == "tp_priority" else "sl"
        elif buy_tp_hit:
            buy_event = "tp"
        elif buy_sl_hit:
            buy_event = "sl"

        sell_event = None
        if sell_tp_hit and sell_sl_hit:
            sell_event = "tp" if tie_breaker == "tp_priority" else "sl"
        elif sell_tp_hit:
            sell_event = "tp"
        elif sell_sl_hit:
            sell_event = "sl"

        if buy_event == "tp" and sell_event != "tp":
            return 1
        if sell_event == "tp" and buy_event != "tp":
            return 2
        if buy_event == "tp" and sell_event == "tp":
            return 0
        if buy_event == "sl" and sell_event == "sl":
            return 0

    return 0


def create_labels(df: pd.DataFrame, cfg: LabelingConfig) -> tuple[pd.DataFrame, LabelDiagnostics]:
    out = df.copy()
    n = len(out)
    labels = np.zeros(n, dtype=np.int32)

    combos = [(tp, sl) for tp in cfg.tp_points for sl in cfg.sl_points]

    for i in range(0, n - cfg.horizon_bars):
        entry = float(out.iloc[i]["close"])
        future = out.iloc[i + 1 : i + 1 + cfg.horizon_bars]
        future_high = future["high"].to_numpy()
        future_low = future["low"].to_numpy()

        decisions = [
            _first_hit_direction(future_high, future_low, entry, tp, sl, cfg.tie_breaker)
            for tp, sl in combos
        ]
        if 1 in decisions and 2 not in decisions:
            labels[i] = 1
        elif 2 in decisions and 1 not in decisions:
            labels[i] = 2
        else:
            labels[i] = 0

    out["label"] = labels
    out = out.iloc[: n - cfg.horizon_bars].copy()
    out = out.reset_index(drop=True)

    counts = out["label"].value_counts().sort_index().to_dict()
    diagnostics = LabelDiagnostics(label_counts={int(k): int(v) for k, v in counts.items()}, combos_tested=len(combos))
    return out, diagnostics

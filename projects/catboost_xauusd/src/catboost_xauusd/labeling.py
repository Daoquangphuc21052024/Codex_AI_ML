from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelingConfig


@dataclass
class LabelDiagnostics:
    label_counts: dict[int, int]
    combos_tested: int


def _resolve_event(tp_hit: bool, sl_hit: bool, tie_breaker: str) -> str | None:
    if tp_hit and sl_hit:
        return "tp" if tie_breaker == "tp_priority" else "sl"
    if tp_hit:
        return "tp"
    if sl_hit:
        return "sl"
    return None


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
        buy_event = _resolve_event(h >= buy_tp, l <= buy_sl, tie_breaker)
        sell_event = _resolve_event(l <= sell_tp, h >= sell_sl, tie_breaker)

        if buy_event == "tp" and sell_event != "tp":
            return 1
        if sell_event == "tp" and buy_event != "tp":
            return 2
        if buy_event == "tp" and sell_event == "tp":
            return 0
        if buy_event == "sl" and sell_event == "sl":
            return 0

    return 0


def simulate_signal_outcome(
    future_high: np.ndarray,
    future_low: np.ndarray,
    entry: float,
    signal: int,
    tp: float,
    sl: float,
    tie_breaker: str,
) -> float:
    if signal == 0:
        return 0.0

    if signal == 1:
        tp_level = entry + tp
        sl_level = entry - sl
        for h, l in zip(future_high, future_low):
            event = _resolve_event(h >= tp_level, l <= sl_level, tie_breaker)
            if event == "tp":
                return 1.0
            if event == "sl":
                return -1.0
        return 0.0

    tp_level = entry - tp
    sl_level = entry + sl
    for h, l in zip(future_high, future_low):
        event = _resolve_event(l <= tp_level, h >= sl_level, tie_breaker)
        if event == "tp":
            return 1.0
        if event == "sl":
            return -1.0
    return 0.0


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
    out = out.iloc[: n - cfg.horizon_bars].copy().reset_index(drop=True)

    counts = out["label"].value_counts().sort_index().to_dict()
    diagnostics = LabelDiagnostics(label_counts={int(k): int(v) for k, v in counts.items()}, combos_tested=len(combos))
    return out, diagnostics

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import LabelingConfig


@dataclass
class LabelDiagnostics:
    label_counts: dict[int, int]
    combos_tested: int
    ambiguous_count: int
    low_move_rejected: int
    conflict_count: int
    usable_samples: int
    usable_ratio: float
    per_fold_label_distribution: dict[str, dict[str, int]]


@dataclass
class LabelMeta:
    entry_index: int
    entry_price: float
    buy_score: float
    sell_score: float
    ambiguity_score: float
    low_move_reject: bool
    conflict: bool


def _resolve_event(tp_hit: bool, sl_hit: bool, tie_breaker: str) -> str | None:
    if tp_hit and sl_hit:
        return "tp" if tie_breaker == "tp_priority" else "sl"
    if tp_hit:
        return "tp"
    if sl_hit:
        return "sl"
    return None


def _simulate_direction(
    future_high: np.ndarray,
    future_low: np.ndarray,
    entry: float,
    signal: int,
    tp: float,
    sl: float,
    tie_breaker: str,
) -> float:
    if signal == 1:
        tp_level, sl_level = entry + tp, entry - sl
        for h, l in zip(future_high, future_low):
            event = _resolve_event(h >= tp_level, l <= sl_level, tie_breaker)
            if event == "tp":
                return 1.0
            if event == "sl":
                return -1.0
        return 0.0

    tp_level, sl_level = entry - tp, entry + sl
    for h, l in zip(future_high, future_low):
        event = _resolve_event(l <= tp_level, h >= sl_level, tie_breaker)
        if event == "tp":
            return 1.0
        if event == "sl":
            return -1.0
    return 0.0


def _entry_index(i: int, cfg: LabelingConfig) -> int:
    if cfg.entry_mode == "signal_close":
        return i
    if cfg.entry_mode == "next_open":
        return i + 1
    raise ValueError(f"Unsupported labeling.entry_mode={cfg.entry_mode}")


def _label_one(
    df: pd.DataFrame,
    i: int,
    cfg: LabelingConfig,
    combos: list[tuple[float, float]],
) -> tuple[int, LabelMeta]:
    e_idx = _entry_index(i, cfg)
    entry = float(df.iloc[e_idx]["open"] if cfg.entry_mode == "next_open" else df.iloc[i]["close"])

    horizon_start = e_idx
    horizon_end = e_idx + cfg.horizon_bars
    future = df.iloc[horizon_start:horizon_end]
    future_high = future["high"].to_numpy(dtype=float)
    future_low = future["low"].to_numpy(dtype=float)

    outcomes_buy = []
    outcomes_sell = []
    for tp, sl in combos:
        outcomes_buy.append(_simulate_direction(future_high, future_low, entry, 1, tp, sl, cfg.tie_breaker))
        outcomes_sell.append(_simulate_direction(future_high, future_low, entry, 2, tp, sl, cfg.tie_breaker))

    buy_score = float(np.mean(outcomes_buy))
    sell_score = float(np.mean(outcomes_sell))

    max_excursion = float(np.max(np.maximum(np.abs(future_high - entry), np.abs(future_low - entry))))
    atr = float(df.iloc[i]["atr"]) if "atr" in df.columns else np.nan
    low_move_reject = bool(np.isfinite(atr) and max_excursion < cfg.min_move_atr * atr)

    ambiguity = 1.0 - abs(buy_score - sell_score)
    conflict = (buy_score > 0 and sell_score > 0) or (buy_score < 0 and sell_score < 0)

    if low_move_reject:
        label = 0
    elif conflict:
        label = 0
    elif buy_score >= cfg.dominance_threshold and sell_score <= 0:
        label = 1
    elif sell_score >= cfg.dominance_threshold and buy_score <= 0:
        label = 2
    else:
        label = 0

    meta = LabelMeta(
        entry_index=e_idx,
        entry_price=entry,
        buy_score=buy_score,
        sell_score=sell_score,
        ambiguity_score=ambiguity,
        low_move_reject=low_move_reject,
        conflict=conflict,
    )
    return label, meta


def create_labels(df: pd.DataFrame, cfg: LabelingConfig) -> tuple[pd.DataFrame, LabelDiagnostics]:
    out = df.copy().reset_index(drop=True)
    n = len(out)
    combos = [(tp, sl) for tp in cfg.tp_points for sl in cfg.sl_points]

    max_i = n - cfg.horizon_bars - (1 if cfg.entry_mode == "next_open" else 0)
    labels = np.zeros(max_i, dtype=np.int32)
    meta_rows: list[dict] = []

    for i in range(max_i):
        label, meta = _label_one(out, i, cfg, combos)
        labels[i] = label
        meta_rows.append(asdict(meta))

    labeled = out.iloc[:max_i].copy().reset_index(drop=True)
    labeled["label"] = labels
    meta_df = pd.DataFrame(meta_rows)
    labeled = pd.concat([labeled, meta_df], axis=1)

    counts = labeled["label"].value_counts().sort_index().to_dict()
    ambiguous_count = int((labeled["ambiguity_score"] > 0.75).sum())
    low_move_rejected = int(labeled["low_move_reject"].sum())
    conflict_count = int(labeled["conflict"].sum())
    usable = int((labeled["label"] != 0).sum())

    diagnostics = LabelDiagnostics(
        label_counts={int(k): int(v) for k, v in counts.items()},
        combos_tested=len(combos),
        ambiguous_count=ambiguous_count,
        low_move_rejected=low_move_rejected,
        conflict_count=conflict_count,
        usable_samples=usable,
        usable_ratio=float(usable / max(len(labeled), 1)),
        per_fold_label_distribution={},
    )
    return labeled, diagnostics


def simulate_signal_outcome(
    future_high: np.ndarray,
    future_low: np.ndarray,
    entry: float,
    signal: int,
    tp: float,
    sl: float,
    tie_breaker: str,
) -> tuple[float, str, int]:
    if signal == 0:
        return 0.0, "no_trade", 0

    if signal == 1:
        tp_level, sl_level = entry + tp, entry - sl
        for i, (h, l) in enumerate(zip(future_high, future_low), start=1):
            event = _resolve_event(h >= tp_level, l <= sl_level, tie_breaker)
            if event == "tp":
                return 1.0, "tp", i
            if event == "sl":
                return -1.0, "sl", i
        return 0.0, "horizon", len(future_high)

    tp_level, sl_level = entry - tp, entry + sl
    for i, (h, l) in enumerate(zip(future_high, future_low), start=1):
        event = _resolve_event(l <= tp_level, h >= sl_level, tie_breaker)
        if event == "tp":
            return 1.0, "tp", i
        if event == "sl":
            return -1.0, "sl", i
    return 0.0, "horizon", len(future_high)

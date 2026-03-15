from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import LabelingConfig


@dataclass(frozen=True)
class TradeProfile:
    tp_points: float
    sl_points: float


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
    tp_points: float
    sl_points: float
    tie_count: int


@dataclass
class LabelMeta:
    entry_index: int
    entry_price: float
    buy_score: float
    sell_score: float
    ambiguity_score: float
    low_move_reject: bool
    conflict: bool


def get_trade_profile(cfg: LabelingConfig) -> TradeProfile:
    """Single source of truth for TP/SL semantics used by both labels and backtest.

    Contract:
    - Training labels and execution backtest both use one deterministic TP/SL profile.
    - We pick the median of configured arrays to keep stable production behavior while
      still allowing users to specify arrays in config for experimentation.
    """
    tp = float(np.median(np.array(cfg.tp_points, dtype=float)))
    sl = float(np.median(np.array(cfg.sl_points, dtype=float)))
    return TradeProfile(tp_points=tp, sl_points=sl)


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


def _label_one(df: pd.DataFrame, i: int, cfg: LabelingConfig, profile: TradeProfile) -> tuple[int, LabelMeta, bool]:
    e_idx = _entry_index(i, cfg)
    entry = float(df.iloc[e_idx]["open"] if cfg.entry_mode == "next_open" else df.iloc[i]["close"])

    horizon_start = e_idx
    horizon_end = e_idx + cfg.horizon_bars
    future = df.iloc[horizon_start:horizon_end]
    future_high = future["high"].to_numpy(dtype=float)
    future_low = future["low"].to_numpy(dtype=float)

    buy_score = _simulate_direction(
        future_high,
        future_low,
        entry,
        1,
        profile.tp_points,
        profile.sl_points,
        cfg.tie_breaker,
    )
    sell_score = _simulate_direction(
        future_high,
        future_low,
        entry,
        2,
        profile.tp_points,
        profile.sl_points,
        cfg.tie_breaker,
    )

    max_excursion = float(np.max(np.maximum(np.abs(future_high - entry), np.abs(future_low - entry))))
    atr = float(df.iloc[i]["atr_12"]) if "atr_12" in df.columns else np.nan
    low_move_reject = bool(np.isfinite(atr) and max_excursion < cfg.min_move_atr * atr)

    ambiguity = 1.0 - abs(buy_score - sell_score)
    conflict = (buy_score > 0 and sell_score > 0) or (buy_score < 0 and sell_score < 0)
    tie_case = bool(np.isclose(buy_score, sell_score, atol=1e-12))

    if low_move_reject or conflict:
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
        buy_score=float(buy_score),
        sell_score=float(sell_score),
        ambiguity_score=float(ambiguity),
        low_move_reject=low_move_reject,
        conflict=conflict,
    )
    return label, meta, tie_case


def create_labels(df: pd.DataFrame, cfg: LabelingConfig) -> tuple[pd.DataFrame, LabelDiagnostics]:
    out = df.copy().reset_index(drop=True)
    n = len(out)
    profile = get_trade_profile(cfg)

    max_i = n - cfg.horizon_bars - (1 if cfg.entry_mode == "next_open" else 0)
    labels = np.zeros(max_i, dtype=np.int32)
    meta_rows: list[dict] = []
    tie_count = 0

    for i in range(max_i):
        label, meta, tie_case = _label_one(out, i, cfg, profile)
        labels[i] = label
        meta_rows.append(asdict(meta))
        tie_count += int(tie_case)

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
        combos_tested=1,
        ambiguous_count=ambiguous_count,
        low_move_rejected=low_move_rejected,
        conflict_count=conflict_count,
        usable_samples=usable,
        usable_ratio=float(usable / max(len(labeled), 1)),
        per_fold_label_distribution={},
        tp_points=profile.tp_points,
        sl_points=profile.sl_points,
        tie_count=tie_count,
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

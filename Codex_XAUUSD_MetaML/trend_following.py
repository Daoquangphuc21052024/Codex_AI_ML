from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler

from data_lib import build_split_visualization, generate_synthetic_ohlc, read_price_csv
from evaluation_lib import action_semantic_diagnostics, evaluate_dual_classification, save_dual_classification_reports, save_feature_importance
from export_lib import export_artifacts
from features_lib import build_features, select_main_features_train_only
from labeling_lib import SEED, create_dual_edge_labels, evaluate_label_quality, split_label_diagnostics
from search_lib import run_param_search
from tester_lib import backtest_probabilities, resolve_actions, save_backtest_reports

np.random.seed(SEED)
random.seed(SEED)


@dataclass
class HyperParams:
    symbol: str = "XAUUSD_H1"
    export_path: str = os.path.join(os.getcwd(), "exports")
    model_number: int = 0

    # Execution / risk-reward defaults (balanced, not self-defeating)
    stop_loss: float = 6.0
    take_profit: float = 6.0
    spread_points: float = 0.35
    commission: float = 0.0
    slippage_points: float = 0.05

    periods: tuple[int, ...] = (5, 35, 65, 95, 125, 155, 185, 215, 245, 275)
    periods_meta: tuple[int, ...] = (50, 100, 200)
    atr_window: int = 14

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    main_iterations: int = 350

    # Label alignment
    label_entry_mode: str = "next_open"
    label_max_hold: int = 12
    label_tp_buy_atr: float = 1.2
    label_sl_buy_atr: float = 1.0
    label_tp_sell_atr: float = 1.2
    label_sl_sell_atr: float = 1.0
    label_same_bar_conflict: str = "sl_first"

    # Threshold constraints
    min_total_trades: int = 300
    min_buy_trades: int = 80
    min_sell_trades: int = 80
    max_no_trade_ratio: float = 0.90
    max_side_dominance: float = 0.80
    min_buy_profit_factor: float = 0.85
    min_sell_profit_factor: float = 0.85

    # Regime-aware thresholding
    use_regime_adjustment: bool = True
    regime_delta: float = 0.04


HP = HyperParams(export_path=os.environ.get("MT5_INCLUDE_PATH", os.path.join(os.getcwd(), "exports")))


def _time_split_3way(X, y, train_ratio=0.6, val_ratio=0.2):
    n = len(X)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))
    return X.iloc[:i_val], y.iloc[:i_val], X.iloc[i_val:i_test], y.iloc[i_val:i_test], X.iloc[i_test:], y.iloc[i_test:]


def _normalize(X_train, X_val, X_test):
    scaler = RobustScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler


def _build_dataset(use_synthetic_if_missing: bool = False):
    csv_path = f"files/{HP.symbol}.csv"
    if Path(csv_path).exists():
        prices = read_price_csv(csv_path)
    elif use_synthetic_if_missing:
        prices = generate_synthetic_ohlc(n=4000, seed=SEED)
    else:
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    fs = build_features(prices, periods=list(HP.periods), periods_meta=list(HP.periods_meta), atr_window=HP.atr_window)
    labels = create_dual_edge_labels(
        open_=fs.data["open"],
        close=fs.data["close"],
        high=fs.data["high"],
        low=fs.data["low"],
        atr_window=HP.atr_window,
        tp_atr_buy=HP.label_tp_buy_atr,
        sl_atr_buy=HP.label_sl_buy_atr,
        tp_atr_sell=HP.label_tp_sell_atr,
        sl_atr_sell=HP.label_sl_sell_atr,
        max_holding_bars=HP.label_max_hold,
        entry_mode=HP.label_entry_mode,
        same_bar_conflict=HP.label_same_bar_conflict,
    )
    data = fs.data.join(labels[["y_buy", "y_sell", "direction_label"]], how="inner").dropna().copy()
    return data, fs.main_features, fs.feature_groups


def _regime_thresholds(base_buy: float, base_sell: float, regime_bull: pd.Series, regime_bear: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if not HP.use_regime_adjustment:
        n = len(regime_bull)
        return np.full(n, base_buy, dtype=float), np.full(n, base_sell, dtype=float)

    bull = regime_bull.clip(0.0, 1.0).to_numpy(dtype=float)
    bear = regime_bear.clip(0.0, 1.0).to_numpy(dtype=float)

    # Strong bull => easier buy, harder sell; strong bear => opposite
    buy_thr = base_buy + HP.regime_delta * (bear - bull)
    sell_thr = base_sell + HP.regime_delta * (bull - bear)
    return np.clip(buy_thr, 0.35, 0.90), np.clip(sell_thr, 0.35, 0.90)


def _score_threshold_row(row: dict) -> float:
    # explicit emphasis on side quality, especially preventing catastrophic one-side behavior.
    pf_total = min(2.5, row["profit_factor"]) / 2.5
    pf_buy = min(2.5, row["profit_factor_buy"]) / 2.5
    pf_sell = min(2.5, row["profit_factor_sell"]) / 2.5
    expectancy = np.tanh(row["avg_trade"] / 4.0)
    sharpe_like = np.tanh(row["sharpe_like"])
    side_balance = 1.0 - abs(row["buy_trades"] - row["sell_trades"]) / max(1, row["buy_trades"] + row["sell_trades"])
    trade_count = min(1.0, row["trades"] / 600.0)

    return (
        0.20 * pf_total
        + 0.22 * pf_buy
        + 0.28 * pf_sell
        + 0.10 * expectancy
        + 0.08 * sharpe_like
        + 0.07 * side_balance
        + 0.05 * trade_count
    )


def _threshold_search(
    val_df: pd.DataFrame,
    prob_buy: np.ndarray,
    prob_sell: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    buy_grid = [round(x, 3) for x in np.arange(0.48, 0.69, 0.02)]
    sell_grid = [round(x, 3) for x in np.arange(0.48, 0.69, 0.02)]
    margin_grid = [round(x, 3) for x in np.arange(0.00, 0.11, 0.02)]

    for buy_t in buy_grid:
        for sell_t in sell_grid:
            for edge_margin in margin_grid:
                dyn_buy_t, dyn_sell_t = _regime_thresholds(
                    base_buy=buy_t,
                    base_sell=sell_t,
                    regime_bull=val_df.get("bull_regime_score", pd.Series(0.5, index=val_df.index)),
                    regime_bear=val_df.get("bear_regime_score", pd.Series(0.5, index=val_df.index)),
                )

                ds = pd.DataFrame(
                    {
                        "open": val_df["open"],
                        "high": val_df["high"],
                        "low": val_df["low"],
                        "close": val_df["close"],
                        "spread": val_df.get("spread", 0.0),
                        "prob_buy": prob_buy,
                        "prob_sell": prob_sell,
                    },
                    index=val_df.index,
                )
                _, metrics = backtest_probabilities(
                    ds,
                    stop=HP.stop_loss,
                    take=HP.take_profit,
                    markup=0.0,
                    buy_threshold=dyn_buy_t,
                    sell_threshold=dyn_sell_t,
                    edge_margin=edge_margin,
                    max_hold=HP.label_max_hold,
                    signal_shift=0,
                    conflict_mode="no_trade",
                    allow_overlap=False,
                    entry_mode=HP.label_entry_mode,
                    spread_points=HP.spread_points,
                    commission=HP.commission,
                    slippage_points=HP.slippage_points,
                    use_spread_column=True,
                    same_bar_conflict=HP.label_same_bar_conflict,
                    barrier_type="atr",
                    atr_window=HP.atr_window,
                    tp_atr_buy=HP.label_tp_buy_atr,
                    sl_atr_buy=HP.label_sl_buy_atr,
                    tp_atr_sell=HP.label_tp_sell_atr,
                    sl_atr_sell=HP.label_sl_sell_atr,
                )

                hard_fail = (
                    metrics["trades"] < HP.min_total_trades
                    or metrics["buy_trades"] < HP.min_buy_trades
                    or metrics["sell_trades"] < HP.min_sell_trades
                    or metrics["no_trade_ratio"] > HP.max_no_trade_ratio
                    or metrics["side_dominance_ratio"] > HP.max_side_dominance
                    or metrics["profit_factor_buy"] < HP.min_buy_profit_factor
                    or metrics["profit_factor_sell"] < HP.min_sell_profit_factor
                )

                score = _score_threshold_row(metrics)
                if hard_fail:
                    score -= 2.0

                rows.append(
                    {
                        "buy_threshold": buy_t,
                        "sell_threshold": sell_t,
                        "buy_threshold_effective_mean": float(np.mean(dyn_buy_t)),
                        "sell_threshold_effective_mean": float(np.mean(dyn_sell_t)),
                        "edge_margin": edge_margin,
                        "score": score,
                        "hard_fail": int(hard_fail),
                        **metrics,
                    }
                )

    table = pd.DataFrame(rows).sort_values(["hard_fail", "score", "profit_factor", "pnl"], ascending=[True, False, False, False])
    valid = table[table["hard_fail"] == 0]
    if valid.empty:
        raise RuntimeError("No threshold candidates satisfy hard constraints.")
    return valid.iloc[0].to_dict(), table


def train_pipeline(use_synthetic_if_missing: bool = False, run_search: bool = False, disable_meta: bool = True):
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path(HP.export_path).mkdir(parents=True, exist_ok=True)

    data, main_features, feature_groups = _build_dataset(use_synthetic_if_missing=use_synthetic_if_missing)
    evaluate_label_quality(data, out_dir="reports")
    build_split_visualization(data, HP.train_ratio, HP.val_ratio)

    X = data[main_features]
    y_buy = data["y_buy"].astype(int)
    y_sell = data["y_sell"].astype(int)
    assert X.index.equals(y_buy.index) and X.index.equals(y_sell.index), "feature-label index mismatch"

    label_diag = split_label_diagnostics(y_buy, y_sell, HP.train_ratio, HP.val_ratio)
    with open("reports/label_split_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(label_diag, f, indent=2)
    print("Label positive rates by split:")
    print(json.dumps(label_diag, indent=2))

    trX, tr_buy, vaX, va_buy, teX, te_buy = _time_split_3way(X, y_buy, HP.train_ratio, HP.val_ratio)
    _, tr_sell, _, va_sell, _, te_sell = _time_split_3way(X, y_sell, HP.train_ratio, HP.val_ratio)

    fs_buy = select_main_features_train_only(trX, tr_buy, main_features, random_seed=SEED)
    fs_sell = select_main_features_train_only(trX, tr_sell, main_features, random_seed=SEED)
    selected = sorted(set(fs_buy.selected_features).union(fs_sell.selected_features))
    pd.DataFrame({"selected_features": selected}).to_csv("reports/feature_selection_dual.csv", index=False)

    X = X[selected]
    trX, tr_buy, vaX, va_buy, teX, te_buy = _time_split_3way(X, y_buy, HP.train_ratio, HP.val_ratio)
    _, tr_sell, _, va_sell, _, te_sell = _time_split_3way(X, y_sell, HP.train_ratio, HP.val_ratio)

    if run_search:
        search_space = {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.08],
            "l2_leaf_reg": [2.0, 5.0],
            "iterations": [200, 300, 420],
        }
        best_buy_params, _ = run_param_search(X, y_buy, HP.train_ratio, HP.val_ratio, search_space=search_space, out_dir="reports", max_trials=12)
        best_sell_params, _ = run_param_search(X, y_sell, HP.train_ratio, HP.val_ratio, search_space=search_space, out_dir="reports", max_trials=12)
    else:
        best_buy_params = {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "iterations": HP.main_iterations}
        best_sell_params = {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "iterations": HP.main_iterations}

    trX_s, vaX_s, teX_s, scaler = _normalize(trX, vaX, teX)

    model_buy = CatBoostClassifier(
        depth=int(best_buy_params["depth"]),
        learning_rate=float(best_buy_params["learning_rate"]),
        l2_leaf_reg=float(best_buy_params["l2_leaf_reg"]),
        iterations=int(best_buy_params["iterations"]),
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
        use_best_model=True,
        early_stopping_rounds=30,
    )
    model_sell = CatBoostClassifier(
        depth=int(best_sell_params["depth"]),
        learning_rate=float(best_sell_params["learning_rate"]),
        l2_leaf_reg=float(best_sell_params["l2_leaf_reg"]),
        iterations=int(best_sell_params["iterations"]),
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
        use_best_model=True,
        early_stopping_rounds=30,
    )
    model_buy.fit(trX_s, tr_buy, eval_set=(vaX_s, va_buy))
    model_sell.fit(trX_s, tr_sell, eval_set=(vaX_s, va_sell))

    val_prob_buy = model_buy.predict_proba(vaX_s)[:, 1]
    val_prob_sell = model_sell.predict_proba(vaX_s)[:, 1]

    val_df = data.loc[vaX.index, ["open", "high", "low", "close"]].copy()
    if "spread" in data.columns:
        val_df["spread"] = data.loc[vaX.index, "spread"]
    if "bull_regime_score" in data.columns:
        val_df["bull_regime_score"] = data.loc[vaX.index, "bull_regime_score"]
    if "bear_regime_score" in data.columns:
        val_df["bear_regime_score"] = data.loc[vaX.index, "bear_regime_score"]

    best_thr, thr_table = _threshold_search(val_df, val_prob_buy, val_prob_sell)
    thr_table.to_csv("reports/threshold_search_dual.csv", index=False)
    thr_table.head(60).to_csv("reports/threshold_candidates_side_by_side.csv", index=False)

    buy_t = float(best_thr["buy_threshold"])
    sell_t = float(best_thr["sell_threshold"])
    edge_margin = float(best_thr["edge_margin"])

    test_prob_buy = model_buy.predict_proba(teX_s)[:, 1]
    test_prob_sell = model_sell.predict_proba(teX_s)[:, 1]

    dual_cls = evaluate_dual_classification(te_buy, te_sell, test_prob_buy, test_prob_sell, buy_t, sell_t)
    print(
        f"Per-side AUC/PR-AUC | buy: {dual_cls['auc_buy']:.4f}/{dual_cls['pr_auc_buy']:.4f} | "
        f"sell: {dual_cls['auc_sell']:.4f}/{dual_cls['pr_auc_sell']:.4f}"
    )

    test_buy_thr, test_sell_thr = _regime_thresholds(
        buy_t,
        sell_t,
        regime_bull=data.loc[teX.index, "bull_regime_score"] if "bull_regime_score" in data.columns else pd.Series(0.5, index=teX.index),
        regime_bear=data.loc[teX.index, "bear_regime_score"] if "bear_regime_score" in data.columns else pd.Series(0.5, index=teX.index),
    )

    actions = resolve_actions(test_prob_buy, test_prob_sell, test_buy_thr, test_sell_thr, edge_margin)
    act_diag = action_semantic_diagnostics(actions, te_buy, te_sell)

    def _run_backtest_for_split(split_name: str, idx: pd.Index, prob_buy: np.ndarray, prob_sell: np.ndarray, buy_thr, sell_thr):
        ds = data.loc[idx, ["open", "high", "low", "close"]].copy()
        ds["spread"] = data.loc[idx, "spread"] if "spread" in data.columns else 0.0
        ds["prob_buy"] = prob_buy
        ds["prob_sell"] = prob_sell
        split_trades, split_metrics = backtest_probabilities(
            ds,
            stop=HP.stop_loss,
            take=HP.take_profit,
            markup=0.0,
            buy_threshold=buy_thr,
            sell_threshold=sell_thr,
            edge_margin=edge_margin,
            max_hold=HP.label_max_hold,
            signal_shift=0,
            conflict_mode="no_trade",
            allow_overlap=False,
            entry_mode=HP.label_entry_mode,
            spread_points=HP.spread_points,
            commission=HP.commission,
            slippage_points=HP.slippage_points,
            use_spread_column=True,
            same_bar_conflict=HP.label_same_bar_conflict,
            barrier_type="atr",
            atr_window=HP.atr_window,
            tp_atr_buy=HP.label_tp_buy_atr,
            sl_atr_buy=HP.label_sl_buy_atr,
            tp_atr_sell=HP.label_tp_sell_atr,
            sl_atr_sell=HP.label_sl_sell_atr,
        )
        return split_trades, split_metrics

    train_prob_buy = model_buy.predict_proba(trX_s)[:, 1]
    train_prob_sell = model_sell.predict_proba(trX_s)[:, 1]
    val_prob_buy_bt = model_buy.predict_proba(vaX_s)[:, 1]
    val_prob_sell_bt = model_sell.predict_proba(vaX_s)[:, 1]

    train_buy_thr, train_sell_thr = _regime_thresholds(
        buy_t,
        sell_t,
        regime_bull=data.loc[trX.index, "bull_regime_score"] if "bull_regime_score" in data.columns else pd.Series(0.5, index=trX.index),
        regime_bear=data.loc[trX.index, "bear_regime_score"] if "bear_regime_score" in data.columns else pd.Series(0.5, index=trX.index),
    )
    val_buy_thr, val_sell_thr = _regime_thresholds(
        buy_t,
        sell_t,
        regime_bull=data.loc[vaX.index, "bull_regime_score"] if "bull_regime_score" in data.columns else pd.Series(0.5, index=vaX.index),
        regime_bear=data.loc[vaX.index, "bear_regime_score"] if "bear_regime_score" in data.columns else pd.Series(0.5, index=vaX.index),
    )

    train_trades_df, train_trading_metrics = _run_backtest_for_split("train", trX.index, train_prob_buy, train_prob_sell, train_buy_thr, train_sell_thr)
    val_trades_df, val_trading_metrics = _run_backtest_for_split("val", vaX.index, val_prob_buy_bt, val_prob_sell_bt, val_buy_thr, val_sell_thr)
    trades_df, trading_metrics = _run_backtest_for_split("test", teX.index, test_prob_buy, test_prob_sell, test_buy_thr, test_sell_thr)

    split_markers = {
        "train_start": trX.index.min(),
        "train_end": trX.index.max(),
        "val_start": vaX.index.min(),
        "val_end": vaX.index.max(),
        "test_start": teX.index.min(),
        "test_end": teX.index.max(),
    }
    save_backtest_reports(
        train_trades_df,
        HP.symbol,
        out_dir="reports",
        full_close=data["close"],
        split_markers=split_markers,
        tag="_train",
    )
    save_backtest_reports(
        val_trades_df,
        HP.symbol,
        out_dir="reports",
        full_close=data["close"],
        split_markers=split_markers,
        tag="_val",
    )
    save_backtest_reports(
        trades_df,
        HP.symbol,
        out_dir="reports",
        full_close=data["close"],
        split_markers=split_markers,
        tag="_test",
    )
    save_dual_classification_reports(te_buy, te_sell, test_prob_buy, test_prob_sell, HP.symbol, out_dir="reports")
    save_feature_importance(model_buy, selected, HP.symbol, out_dir="reports", top_n=20, tag="_buy", feature_groups=feature_groups)
    save_feature_importance(model_sell, selected, HP.symbol, out_dir="reports", top_n=20, tag="_sell", feature_groups=feature_groups)

    rr_alignment = {
        "label_rr_buy": HP.label_tp_buy_atr / max(1e-12, HP.label_sl_buy_atr),
        "label_rr_sell": HP.label_tp_sell_atr / max(1e-12, HP.label_sl_sell_atr),
        "tester_rr": HP.take_profit / max(1e-12, HP.stop_loss),
        "entry_mode": HP.label_entry_mode,
        "horizon": HP.label_max_hold,
    }
    alignment_report = {
        "entry_alignment": {
            "label_entry_mode": HP.label_entry_mode,
            "tester_entry_mode": HP.label_entry_mode,
            "signal_shift": 0,
            "effective_entry_delay_bars": 1 if HP.label_entry_mode == "next_open" else 0,
            "entry_alignment_ok": True,
        },
        "barrier_alignment": {
            "label_barrier_type": "atr",
            "tester_barrier_type": "atr",
            "label_tp": {"buy": HP.label_tp_buy_atr, "sell": HP.label_tp_sell_atr},
            "label_sl": {"buy": HP.label_sl_buy_atr, "sell": HP.label_sl_sell_atr},
            "tester_tp": {"buy": HP.label_tp_buy_atr, "sell": HP.label_tp_sell_atr},
            "tester_sl": {"buy": HP.label_sl_buy_atr, "sell": HP.label_sl_sell_atr},
            "barrier_alignment_ok": True,
        },
        "horizon_alignment": {
            "label_max_hold": HP.label_max_hold,
            "tester_max_hold": HP.label_max_hold,
            "are_horizons_aligned": True,
        },
        "same_bar_conflict_alignment": {
            "label_same_bar_conflict": HP.label_same_bar_conflict,
            "tester_same_bar_conflict": HP.label_same_bar_conflict,
            "same_bar_alignment_ok": True,
        },
        "threshold_search_execution": {
            "threshold_search_entry_mode": HP.label_entry_mode,
            "threshold_search_signal_shift": 0,
            "threshold_search_barrier_type": "atr",
            "threshold_search_same_bar_conflict": HP.label_same_bar_conflict,
            "threshold_search_allow_overlap": False,
        },
        "final_test_execution": {
            "final_entry_mode": HP.label_entry_mode,
            "final_signal_shift": 0,
            "final_barrier_type": "atr",
            "final_same_bar_conflict": HP.label_same_bar_conflict,
            "final_allow_overlap": False,
        },
    }

    summary = {
        "model_type": "dual_edge",
        "n_features_main": len(selected),
        "n_features_meta": len(selected),
        "n_features_buy": len(selected),
        "n_features_sell": len(selected),
        "buy_feature_names": selected,
        "sell_feature_names": selected,
        "configuration": {
            "symbol": HP.symbol,
            "train_ratio": HP.train_ratio,
            "val_ratio": HP.val_ratio,
            "test_ratio": 1 - HP.train_ratio - HP.val_ratio,
            "action_semantics": {"BUY": 0, "SELL": 1, "NO_TRADE": -1},
            "use_regime_adjustment": HP.use_regime_adjustment,
            "regime_delta": HP.regime_delta,
            "allow_overlap": False,
            "cost_model": {
                "spread_points": HP.spread_points,
                "commission": HP.commission,
                "slippage_points": HP.slippage_points,
                "use_spread_column": True,
            },
            "label_entry_mode": HP.label_entry_mode,
            "label_max_hold": HP.label_max_hold,
        },
        "split_dates": {
            "train": {"start": str(trX.index.min()), "end": str(trX.index.max())},
            "val": {"start": str(vaX.index.min()), "end": str(vaX.index.max())},
            "test": {"start": str(teX.index.min()), "end": str(teX.index.max())},
        },
        "split_markers": {k: str(v) for k, v in split_markers.items()},
        "feature_counts": {"original": len(main_features), "selected": len(selected)},
        "feature_names_used": selected,
        "label_diagnostics_by_split": label_diag,
        "dual_classification_metrics": dual_cls,
        "action_diagnostics": act_diag,
        "trading_metrics": {
            "train": train_trading_metrics,
            "val": val_trading_metrics,
            "test": trading_metrics,
        },
        "threshold_optimization": {
            "best_candidate": best_thr,
            "top_candidates": thr_table.head(20).to_dict(orient="records"),
        },
        "label_tester_alignment": rr_alignment,
        "alignment_report": alignment_report,
        "model_best_iterations": {
            "buy": int(model_buy.get_best_iteration()),
            "sell": int(model_sell.get_best_iteration()),
        },
    }

    with open("reports/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open("reports/trading_metrics_train.json", "w", encoding="utf-8") as f:
        json.dump(train_trading_metrics, f, indent=2)
    with open("reports/trading_metrics_val.json", "w", encoding="utf-8") as f:
        json.dump(val_trading_metrics, f, indent=2)
    with open("reports/trading_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(trading_metrics, f, indent=2)
    with open("reports/classification_metrics_dual_test.json", "w", encoding="utf-8") as f:
        json.dump({"dual": dual_cls, "actions": act_diag}, f, indent=2)

    exported = export_artifacts(
        symbol=HP.symbol,
        model_number=HP.model_number,
        export_path=HP.export_path,
        model=model_buy,
        meta_model=model_sell,
        scaler_main=scaler,
        scaler_meta=scaler,
        report=summary,
        periods=list(HP.periods),
        periods_meta=list(HP.periods_meta),
        feature_names=selected,
        feature_names_meta=selected,
        decision_threshold=sell_t,
        sample_main=teX_s,
        sample_meta=teX_s,
    )

    print("Training complete")
    print(json.dumps({"classification_dual": dual_cls, "trading": trading_metrics, "exported": exported}, indent=2))


def _year_bounds(year_start: int, n_years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year_start, month=1, day=1)
    end = pd.Timestamp(year=year_start + n_years, month=1, day=1) - pd.Timedelta(seconds=1)
    return start, end


def generate_walkforward_windows(
    index: pd.DatetimeIndex,
    train_years: int = 8,
    val_years: int = 2,
    test_years: int = 2,
    step_years: int = 1,
    min_start_year: int | None = None,
    max_end_year: int | None = None,
) -> list[dict]:
    if len(index) == 0:
        return []
    idx_min_year = int(index.min().year)
    idx_max_year = int(index.max().year)
    start_year = max(min_start_year or idx_min_year, idx_min_year)
    end_year_cap = min(max_end_year or idx_max_year, idx_max_year)

    windows: list[dict] = []
    fold = 1
    while True:
        val_start_year = start_year + train_years + (fold - 1) * step_years
        val_end_year = val_start_year + val_years - 1
        test_start_year = val_end_year + 1
        test_end_year = test_start_year + test_years - 1
        train_start_year = start_year
        train_end_year = val_start_year - 1

        if test_end_year > end_year_cap:
            break

        train_start, train_end = _year_bounds(train_start_year, train_end_year - train_start_year + 1)
        val_start, val_end = _year_bounds(val_start_year, val_years)
        test_start, test_end = _year_bounds(test_start_year, test_years)

        windows.append(
            {
                "fold_id": fold,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        fold += 1
    return windows


def _slice_window(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.Index:
    mask = (index >= start) & (index <= end)
    return index[mask]


def _run_backtest_split(ds: pd.DataFrame, buy_threshold, sell_threshold, edge_margin: float):
    return backtest_probabilities(
        ds,
        stop=HP.stop_loss,
        take=HP.take_profit,
        markup=0.0,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        edge_margin=edge_margin,
        max_hold=HP.label_max_hold,
        signal_shift=0,
        conflict_mode="no_trade",
        allow_overlap=False,
        entry_mode=HP.label_entry_mode,
        spread_points=HP.spread_points,
        commission=HP.commission,
        slippage_points=HP.slippage_points,
        use_spread_column=True,
        same_bar_conflict=HP.label_same_bar_conflict,
        barrier_type="atr",
        atr_window=HP.atr_window,
        tp_atr_buy=HP.label_tp_buy_atr,
        sl_atr_buy=HP.label_sl_buy_atr,
        tp_atr_sell=HP.label_tp_sell_atr,
        sl_atr_sell=HP.label_sl_sell_atr,
    )


def run_walkforward(
    use_synthetic_if_missing: bool = False,
    wf_train_years: int = 8,
    wf_val_years: int = 2,
    wf_test_years: int = 2,
    wf_step_years: int = 1,
    wf_min_start_year: int | None = None,
    wf_max_end_year: int | None = None,
):
    wf_root = Path("reports/walkforward")
    wf_root.mkdir(parents=True, exist_ok=True)

    data, main_features, feature_groups = _build_dataset(use_synthetic_if_missing=use_synthetic_if_missing)
    X_all = data[main_features]
    y_buy_all = data["y_buy"].astype(int)
    y_sell_all = data["y_sell"].astype(int)

    windows = generate_walkforward_windows(
        data.index,
        train_years=wf_train_years,
        val_years=wf_val_years,
        test_years=wf_test_years,
        step_years=wf_step_years,
        min_start_year=wf_min_start_year,
        max_end_year=wf_max_end_year,
    )
    if not windows:
        raise RuntimeError("No walk-forward windows generated for given parameters.")

    fold_records: list[dict] = []
    failures: list[dict] = []

    for w in windows:
        fold_id = int(w["fold_id"])
        fold_dir = wf_root / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        try:
            tr_idx = _slice_window(data.index, w["train_start"], w["train_end"])
            va_idx = _slice_window(data.index, w["val_start"], w["val_end"])
            te_idx = _slice_window(data.index, w["test_start"], w["test_end"])

            if min(len(tr_idx), len(va_idx), len(te_idx)) < 200:
                raise RuntimeError("too_few_rows")

            trX, vaX, teX = X_all.loc[tr_idx], X_all.loc[va_idx], X_all.loc[te_idx]
            tr_buy, va_buy, te_buy = y_buy_all.loc[tr_idx], y_buy_all.loc[va_idx], y_buy_all.loc[te_idx]
            tr_sell, va_sell, te_sell = y_sell_all.loc[tr_idx], y_sell_all.loc[va_idx], y_sell_all.loc[te_idx]

            if tr_buy.nunique() < 2 or tr_sell.nunique() < 2:
                raise RuntimeError("degenerate_train_labels")

            fs_buy = select_main_features_train_only(trX, tr_buy, main_features, random_seed=SEED)
            fs_sell = select_main_features_train_only(trX, tr_sell, main_features, random_seed=SEED)
            selected = sorted(set(fs_buy.selected_features).union(fs_sell.selected_features))

            trX, vaX, teX = trX[selected], vaX[selected], teX[selected]
            trX_s, vaX_s, teX_s, scaler = _normalize(trX, vaX, teX)

            model_buy = CatBoostClassifier(
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                iterations=HP.main_iterations,
                eval_metric="F1",
                auto_class_weights="Balanced",
                random_seed=SEED,
                verbose=False,
                use_best_model=True,
                early_stopping_rounds=30,
            )
            model_sell = CatBoostClassifier(
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                iterations=HP.main_iterations,
                eval_metric="F1",
                auto_class_weights="Balanced",
                random_seed=SEED,
                verbose=False,
                use_best_model=True,
                early_stopping_rounds=30,
            )
            model_buy.fit(trX_s, tr_buy, eval_set=(vaX_s, va_buy))
            model_sell.fit(trX_s, tr_sell, eval_set=(vaX_s, va_sell))

            val_prob_buy = model_buy.predict_proba(vaX_s)[:, 1]
            val_prob_sell = model_sell.predict_proba(vaX_s)[:, 1]
            val_df = data.loc[vaX.index, ["open", "high", "low", "close"]].copy()
            val_df["spread"] = data.loc[vaX.index, "spread"] if "spread" in data.columns else 0.0
            if "bull_regime_score" in data.columns:
                val_df["bull_regime_score"] = data.loc[vaX.index, "bull_regime_score"]
            if "bear_regime_score" in data.columns:
                val_df["bear_regime_score"] = data.loc[vaX.index, "bear_regime_score"]

            best_thr, thr_table = _threshold_search(val_df, val_prob_buy, val_prob_sell)
            thr_table.to_csv(fold_dir / "threshold_candidates.csv", index=False)

            buy_t = float(best_thr["buy_threshold"])
            sell_t = float(best_thr["sell_threshold"])
            edge_margin = float(best_thr["edge_margin"])

            test_prob_buy = model_buy.predict_proba(teX_s)[:, 1]
            test_prob_sell = model_sell.predict_proba(teX_s)[:, 1]
            dual_cls = evaluate_dual_classification(te_buy, te_sell, test_prob_buy, test_prob_sell, buy_t, sell_t)

            dyn_buy_t, dyn_sell_t = _regime_thresholds(
                buy_t,
                sell_t,
                regime_bull=data.loc[teX.index, "bull_regime_score"] if "bull_regime_score" in data.columns else pd.Series(0.5, index=teX.index),
                regime_bear=data.loc[teX.index, "bear_regime_score"] if "bear_regime_score" in data.columns else pd.Series(0.5, index=teX.index),
            )
            test_ds = data.loc[teX.index, ["open", "high", "low", "close"]].copy()
            test_ds["spread"] = data.loc[teX.index, "spread"] if "spread" in data.columns else 0.0
            test_ds["prob_buy"] = test_prob_buy
            test_ds["prob_sell"] = test_prob_sell
            trades_df, trading_metrics = _run_backtest_split(test_ds, dyn_buy_t, dyn_sell_t, edge_margin)

            split_markers = {
                "train_start": trX.index.min(),
                "train_end": trX.index.max(),
                "val_start": vaX.index.min(),
                "val_end": vaX.index.max(),
                "test_start": teX.index.min(),
                "test_end": teX.index.max(),
            }
            save_backtest_reports(
                trades_df,
                HP.symbol,
                out_dir=str(fold_dir),
                full_close=data["close"],
                split_markers=split_markers,
                tag="",
            )

            record = {
                "fold_id": fold_id,
                "train_start": str(w["train_start"]),
                "train_end": str(w["train_end"]),
                "val_start": str(w["val_start"]),
                "val_end": str(w["val_end"]),
                "test_start": str(w["test_start"]),
                "test_end": str(w["test_end"]),
                "n_train": int(len(trX)),
                "n_val": int(len(vaX)),
                "n_test": int(len(teX)),
                "classification": dual_cls,
                "threshold": {
                    "buy_threshold": buy_t,
                    "sell_threshold": sell_t,
                    "edge_margin": edge_margin,
                    "threshold_objective_score": float(best_thr.get("score", np.nan)),
                    "top_candidates": thr_table.head(10).to_dict(orient="records"),
                },
                "trading": trading_metrics,
                "alignment": {
                    "entry_mode": HP.label_entry_mode,
                    "signal_shift": 0,
                    "barrier_type": "atr",
                    "same_bar_conflict": HP.label_same_bar_conflict,
                    "allow_overlap": False,
                },
                "model_best_iterations": {
                    "buy": int(model_buy.get_best_iteration()),
                    "sell": int(model_sell.get_best_iteration()),
                },
            }
            with open(fold_dir / "report.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            fold_records.append(record)
        except Exception as exc:  # noqa: BLE001
            fail = {
                "fold_id": fold_id,
                "reason": str(exc),
                "train_start": str(w["train_start"]),
                "test_end": str(w["test_end"]),
            }
            failures.append(fail)
            with open(fold_dir / "report.json", "w", encoding="utf-8") as f:
                json.dump({"status": "failed", **fail}, f, indent=2)

    if not fold_records:
        raise RuntimeError(f"All walk-forward folds failed. Failures: {failures}")

    fold_table = []
    for r in fold_records:
        t = r["trading"]
        c = r["classification"]
        fold_table.append(
            {
                "fold_id": r["fold_id"],
                "auc_buy": c.get("auc_buy", np.nan),
                "auc_sell": c.get("auc_sell", np.nan),
                "profit_factor": t.get("profit_factor", np.nan),
                "profit_factor_buy": t.get("profit_factor_buy", np.nan),
                "profit_factor_sell": t.get("profit_factor_sell", np.nan),
                "pnl": t.get("pnl", np.nan),
                "buy_pnl": t.get("buy_pnl", np.nan),
                "sell_pnl": t.get("sell_pnl", np.nan),
                "max_drawdown": t.get("max_drawdown", np.nan),
                "win_rate": t.get("win_rate", np.nan),
            }
        )
    fold_df = pd.DataFrame(fold_table).sort_values("fold_id")
    fold_df.to_csv(wf_root / "aggregate_fold_table.csv", index=False)

    def _stats(col: str) -> dict:
        s = fold_df[col].dropna()
        return {
            "mean": float(s.mean()) if len(s) else np.nan,
            "median": float(s.median()) if len(s) else np.nan,
            "std": float(s.std(ddof=0)) if len(s) else np.nan,
        }

    agg = {
        "n_folds": int(len(fold_df)),
        "failed_folds": failures,
        "per_metric": {
            "auc_buy": _stats("auc_buy"),
            "auc_sell": _stats("auc_sell"),
            "profit_factor": _stats("profit_factor"),
            "profit_factor_buy": _stats("profit_factor_buy"),
            "profit_factor_sell": _stats("profit_factor_sell"),
            "pnl": _stats("pnl"),
            "buy_pnl": _stats("buy_pnl"),
            "sell_pnl": _stats("sell_pnl"),
            "max_drawdown": _stats("max_drawdown"),
            "win_rate": _stats("win_rate"),
        },
        "robustness": {
            "fraction_pf_gt_1": float((fold_df["profit_factor"] > 1.0).mean()),
            "fraction_buy_pf_gt_1": float((fold_df["profit_factor_buy"] > 1.0).mean()),
            "fraction_sell_pf_gt_1": float((fold_df["profit_factor_sell"] > 1.0).mean()),
            "fraction_positive_pnl": float((fold_df["pnl"] > 0.0).mean()),
            "fraction_both_side_positive_pnl": float(((fold_df["buy_pnl"] > 0.0) & (fold_df["sell_pnl"] > 0.0)).mean()),
        },
        "stability": {
            "worst_fold_pf": float(fold_df["profit_factor"].min()),
            "best_fold_pf": float(fold_df["profit_factor"].max()),
            "worst_fold_pnl": float(fold_df["pnl"].min()),
            "best_fold_pnl": float(fold_df["pnl"].max()),
        },
    }
    agg["conclusion_flags"] = {
        "walkforward_pass_total_pf": bool(agg["robustness"]["fraction_pf_gt_1"] >= 0.6),
        "walkforward_pass_buy_pf": bool(agg["robustness"]["fraction_buy_pf_gt_1"] >= 0.6),
        "walkforward_pass_sell_pf": bool(agg["robustness"]["fraction_sell_pf_gt_1"] >= 0.6),
        "walkforward_pass_positive_pnl_rate": bool(agg["robustness"]["fraction_positive_pnl"] >= 0.6),
    }

    with open(wf_root / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump({**agg, "folds": fold_records}, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fold_df["fold_id"], fold_df["profit_factor"], color="#1f77b4")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Walk-forward Profit Factor by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Profit Factor")
    fig.savefig(wf_root / "per_fold_profit_factor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fold_df["fold_id"], fold_df["pnl"], color="#ff7f0e")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Walk-forward PnL by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("PnL")
    fig.savefig(wf_root / "per_fold_pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fold_df["fold_id"], fold_df["buy_pnl"], label="buy_pnl", color="#2ca02c")
    ax.bar(fold_df["fold_id"], fold_df["sell_pnl"], bottom=fold_df["buy_pnl"], label="sell_pnl", color="#d62728")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Walk-forward Buy/Sell PnL by Fold")
    ax.set_xlabel("Fold")
    ax.legend(loc="best")
    fig.savefig(wf_root / "per_fold_buy_sell_pnl_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fold_df["fold_id"], fold_df["pnl"].cumsum(), marker="o", color="#9467bd")
    ax.set_title("Cumulative Fold PnL")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Cumulative PnL")
    fig.savefig(wf_root / "cumulative_fold_pnl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].boxplot(fold_df["profit_factor"].dropna())
    axes[0].set_title("PF")
    axes[1].boxplot(fold_df["pnl"].dropna())
    axes[1].set_title("PnL")
    axes[2].boxplot(fold_df["max_drawdown"].dropna())
    axes[2].set_title("Max DD")
    fig.suptitle("Walk-forward Metric Boxplots")
    fig.savefig(wf_root / "aggregate_metric_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, data["close"], color="#1f77b4", linewidth=1.0)
    for w in windows:
        ax.axvspan(w["train_start"], w["train_end"], color="#2ca02c", alpha=0.05)
        ax.axvspan(w["val_start"], w["val_end"], color="#ff7f0e", alpha=0.05)
        ax.axvspan(w["test_start"], w["test_end"], color="#d62728", alpha=0.05)
    ax.set_title("Walk-forward Fold Timeline")
    fig.savefig(wf_root / "fold_timeline_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Walk-forward complete")
    print(json.dumps({"n_folds": len(fold_df), "failures": len(failures), "aggregate": agg}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "search", "walkforward"], default="train")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data when files/XAUUSD_H1.csv is missing")
    parser.add_argument("--disable-meta", action="store_true", help="compatibility flag; unused in dual-edge mode")
    parser.add_argument("--wf-train-years", type=int, default=8)
    parser.add_argument("--wf-val-years", type=int, default=2)
    parser.add_argument("--wf-test-years", type=int, default=2)
    parser.add_argument("--wf-step-years", type=int, default=1)
    parser.add_argument("--wf-min-start-year", type=int, default=None)
    parser.add_argument("--wf-max-end-year", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "walkforward":
        run_walkforward(
            use_synthetic_if_missing=args.synthetic,
            wf_train_years=args.wf_train_years,
            wf_val_years=args.wf_val_years,
            wf_test_years=args.wf_test_years,
            wf_step_years=args.wf_step_years,
            wf_min_start_year=args.wf_min_start_year,
            wf_max_end_year=args.wf_max_end_year,
        )
    else:
        train_pipeline(use_synthetic_if_missing=args.synthetic, run_search=args.mode == "search", disable_meta=args.disable_meta)


if __name__ == "__main__":
    main()

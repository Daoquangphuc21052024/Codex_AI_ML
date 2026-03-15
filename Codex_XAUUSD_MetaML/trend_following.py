from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler

from data_lib import build_split_visualization, generate_synthetic_ohlc, read_price_csv
from evaluation_lib import (
    action_semantic_diagnostics,
    evaluate_dual_classification,
    save_dual_classification_reports,
    save_feature_importance,
)
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
    markup: float = 0.35
    stop_loss: float = 8.0
    take_profit: float = 4.0
    periods: tuple[int, ...] = (5, 35, 65, 95, 125, 155, 185, 215, 245, 275)
    periods_meta: tuple[int, ...] = (50, 100, 200)
    atr_window: int = 14
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    main_iterations: int = 300
    min_total_trades: int = 120
    min_buy_trades: int = 20
    min_sell_trades: int = 20
    max_side_dominance: float = 0.88
    use_regime_adjustment: bool = False
    regime_delta: float = 0.02


HP = HyperParams(export_path=os.environ.get("MT5_INCLUDE_PATH", os.path.join(os.getcwd(), "exports")))


def _time_split_3way(X, y, train_ratio=0.6, val_ratio=0.2):
    n = len(X)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))
    return X.iloc[:i_val], y.iloc[:i_val], X.iloc[i_val:i_test], y.iloc[i_val:i_test], X.iloc[i_test:], y.iloc[i_test:]


def _normalize(X_train, X_val, X_test):
    s = RobustScaler()
    return s.fit_transform(X_train), s.transform(X_val), s.transform(X_test), s


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
        close=fs.data["close"],
        high=fs.data["high"],
        low=fs.data["low"],
        atr_window=HP.atr_window,
        tp_atr_buy=1.4,
        sl_atr_buy=1.2,
        tp_atr_sell=1.4,
        sl_atr_sell=1.2,
        max_holding_bars=12,
    )
    data = fs.data.join(labels[["y_buy", "y_sell", "direction_label"]], how="inner").dropna().copy()
    return data, fs.main_features, fs.feature_groups


def _score_threshold_row(row: dict) -> float:
    pf_norm = min(2.0, row["profit_factor"]) / 2.0
    exp_norm = np.tanh(row["avg_trade"] / 4.0)
    pnl_norm = np.tanh(row["pnl"] / 800.0)
    side_balance = 1.0 - abs(row["buy_trades"] - row["sell_trades"]) / max(1, row["buy_trades"] + row["sell_trades"])
    trade_count = min(1.0, row["trades"] / 300.0)
    return 0.50 * pf_norm + 0.20 * exp_norm + 0.15 * pnl_norm + 0.10 * side_balance + 0.05 * trade_count


def _threshold_search(
    val_close: pd.Series,
    prob_buy: np.ndarray,
    prob_sell: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    buy_grid = [round(x, 3) for x in np.arange(0.50, 0.71, 0.02)]
    sell_grid = [round(x, 3) for x in np.arange(0.50, 0.71, 0.02)]
    margin_grid = [round(x, 3) for x in np.arange(0.00, 0.11, 0.02)]

    ds = pd.DataFrame({"close": val_close, "prob_buy": prob_buy, "prob_sell": prob_sell}, index=val_close.index)

    for bt in buy_grid:
        for st in sell_grid:
            for em in margin_grid:
                _, metrics = backtest_probabilities(
                    ds,
                    stop=HP.stop_loss,
                    take=HP.take_profit,
                    markup=HP.markup,
                    buy_threshold=bt,
                    sell_threshold=st,
                    edge_margin=em,
                    max_hold=120,
                    signal_shift=1,
                    conflict_mode="no_trade",
                )
                hard_fail = (
                    metrics["buy_trades"] < HP.min_buy_trades
                    or metrics["sell_trades"] < HP.min_sell_trades
                    or metrics["trades"] < HP.min_total_trades
                    or metrics["side_dominance_ratio"] > HP.max_side_dominance
                )
                score = _score_threshold_row(metrics)
                if hard_fail:
                    score -= 1.0
                rows.append({"buy_threshold": bt, "sell_threshold": st, "edge_margin": em, "score": score, "hard_fail": int(hard_fail), **metrics})

    res = pd.DataFrame(rows).sort_values(["hard_fail", "score", "pnl"], ascending=[True, False, False])
    best = res.iloc[0].to_dict()
    return best, res


def train_pipeline(use_synthetic_if_missing: bool = False, run_search: bool = False, disable_meta: bool = True):
    Path("reports").mkdir(exist_ok=True)
    Path(HP.export_path).mkdir(parents=True, exist_ok=True)

    data, main_features, feature_groups = _build_dataset(use_synthetic_if_missing=use_synthetic_if_missing)
    evaluate_label_quality(data, out_dir="reports")
    build_split_visualization(data, HP.train_ratio, HP.val_ratio)

    X = data[main_features]
    y_buy = data["y_buy"].astype(int)
    y_sell = data["y_sell"].astype(int)

    assert X.index.equals(y_buy.index) and X.index.equals(y_sell.index), "feature/label index misalignment"

    label_diag = split_label_diagnostics(y_buy, y_sell, train_ratio=HP.train_ratio, val_ratio=HP.val_ratio)
    with open("reports/label_split_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(label_diag, f, indent=2)

    trX, tr_buy, vaX, va_buy, teX, te_buy = _time_split_3way(X, y_buy, HP.train_ratio, HP.val_ratio)
    _, tr_sell, _, va_sell, _, te_sell = _time_split_3way(X, y_sell, HP.train_ratio, HP.val_ratio)

    # Feature selection against both sides -> union
    fs_buy = select_main_features_train_only(trX, tr_buy, main_features, random_seed=SEED)
    fs_sell = select_main_features_train_only(trX, tr_sell, main_features, random_seed=SEED)
    selected = sorted(set(fs_buy.selected_features).union(fs_sell.selected_features))

    pd.DataFrame(
        {
            "selected_features": selected,
        }
    ).to_csv("reports/feature_selection_dual.csv", index=False)

    X = X[selected]
    trX, tr_buy, vaX, va_buy, teX, te_buy = _time_split_3way(X, y_buy, HP.train_ratio, HP.val_ratio)
    _, tr_sell, _, va_sell, _, te_sell = _time_split_3way(X, y_sell, HP.train_ratio, HP.val_ratio)

    if run_search:
        search_space = {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.08],
            "l2_leaf_reg": [2.0, 5.0],
            "iterations": [180, 260, 340],
        }
        best_buy_params, _ = run_param_search(X, y_buy, HP.train_ratio, HP.val_ratio, search_space, out_dir="reports", max_trials=12)
        best_sell_params, _ = run_param_search(X, y_sell, HP.train_ratio, HP.val_ratio, search_space, out_dir="reports", max_trials=12)
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
    assert np.all((val_prob_buy >= 0) & (val_prob_buy <= 1))
    assert np.all((val_prob_sell >= 0) & (val_prob_sell <= 1))

    best_thr, thr_table = _threshold_search(data.loc[vaX.index, "close"], val_prob_buy, val_prob_sell)
    thr_table.to_csv("reports/threshold_search_dual.csv", index=False)
    thr_table.head(25).to_csv("reports/threshold_search_dual_top25.csv", index=False)

    buy_t = float(best_thr["buy_threshold"])
    sell_t = float(best_thr["sell_threshold"])
    edge_margin = float(best_thr["edge_margin"])

    test_prob_buy = model_buy.predict_proba(teX_s)[:, 1]
    test_prob_sell = model_sell.predict_proba(teX_s)[:, 1]

    dual_cls = evaluate_dual_classification(te_buy, te_sell, test_prob_buy, test_prob_sell, buy_t, sell_t)
    actions = resolve_actions(test_prob_buy, test_prob_sell, buy_t, sell_t, edge_margin)
    act_diag = action_semantic_diagnostics(actions, te_buy, te_sell)

    test_ds = pd.DataFrame(
        {
            "close": data.loc[teX.index, "close"],
            "prob_buy": test_prob_buy,
            "prob_sell": test_prob_sell,
        },
        index=teX.index,
    )
    trades_df, trading_metrics = backtest_probabilities(
        test_ds,
        stop=HP.stop_loss,
        take=HP.take_profit,
        markup=HP.markup,
        buy_threshold=buy_t,
        sell_threshold=sell_t,
        edge_margin=edge_margin,
        max_hold=120,
        signal_shift=1,
        conflict_mode="no_trade",
    )
    save_backtest_reports(trades_df, HP.symbol, out_dir="reports")

    save_dual_classification_reports(te_buy, te_sell, test_prob_buy, test_prob_sell, HP.symbol, out_dir="reports")
    save_feature_importance(model_buy, selected, HP.symbol, out_dir="reports", top_n=20, tag="_buy", feature_groups=feature_groups)
    save_feature_importance(model_sell, selected, HP.symbol, out_dir="reports", top_n=20, tag="_sell", feature_groups=feature_groups)

    summary = {
        "configuration": {
            "symbol": HP.symbol,
            "train_ratio": HP.train_ratio,
            "val_ratio": HP.val_ratio,
            "test_ratio": 1 - HP.train_ratio - HP.val_ratio,
            "action_semantics": {"BUY": 0, "SELL": 1, "NO_TRADE": -1},
            "meta_gating_used": False,
            "meta_warning": "meta model disabled by default in dual-edge framework",
            "regime_aware_thresholds_used": HP.use_regime_adjustment,
        },
        "split_dates": {
            "train": {"start": str(trX.index.min()), "end": str(trX.index.max())},
            "val": {"start": str(vaX.index.min()), "end": str(vaX.index.max())},
            "test": {"start": str(teX.index.min()), "end": str(teX.index.max())},
        },
        "feature_counts": {"original": len(main_features), "selected": len(selected)},
        "per_side_label_stats": label_diag,
        "model_best_iterations": {
            "buy": int(model_buy.get_best_iteration()),
            "sell": int(model_sell.get_best_iteration()),
        },
        "per_side_classification_metrics": dual_cls,
        "action_diagnostics": act_diag,
        "final_trading_metrics": trading_metrics,
        "threshold_optimization": {
            "buy_threshold": buy_t,
            "sell_threshold": sell_t,
            "edge_margin": edge_margin,
            "score": float(best_thr["score"]),
            "hard_fail": int(best_thr["hard_fail"]),
            "selection_rule": "validation trading objective with side-balance constraints",
        },
    }

    with open("reports/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open("reports/trading_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(trading_metrics, f, indent=2)
    with open("reports/classification_metrics_dual_test.json", "w", encoding="utf-8") as f:
        json.dump({"dual": dual_cls, "actions": act_diag}, f, indent=2)

    # keep export compatibility: use buy model as main and sell model as meta slot
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "search"], default="train")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data when files/XAUUSD_H1.csv is missing")
    parser.add_argument("--disable-meta", action="store_true", help="Kept for backward CLI compatibility; ignored in dual-edge mode")
    args = parser.parse_args()

    train_pipeline(use_synthetic_if_missing=args.synthetic, run_search=args.mode == "search", disable_meta=True)


if __name__ == "__main__":
    main()

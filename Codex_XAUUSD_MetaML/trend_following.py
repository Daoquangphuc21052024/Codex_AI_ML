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
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from data_lib import build_split_visualization, generate_synthetic_ohlc, read_price_csv
from evaluation_lib import classification_metrics, optimize_threshold, save_classification_reports, save_feature_importance
from export_lib import export_artifacts
from features_lib import build_features, select_main_features_train_only
from labeling_lib import SEED, create_labels, evaluate_label_quality
from search_lib import run_param_search
from tester_lib import backtest_signals, save_backtest_reports

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
    n_clusters: int = 5
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    wf_splits: int = 2
    min_f1: float = 0.45
    main_iterations: int = 300
    meta_iterations: int = 250


HP = HyperParams(export_path=os.environ.get("MT5_INCLUDE_PATH", os.path.join(os.getcwd(), "exports")))


def _time_split_3way(X, y, train_ratio=0.6, val_ratio=0.2):
    n = len(X)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))
    return X.iloc[:i_val], y.iloc[:i_val], X.iloc[i_val:i_test], y.iloc[i_val:i_test], X.iloc[i_test:], y.iloc[i_test:]


def _normalize_features(X_train, X_val, X_test, X_meta_train, X_meta_val, X_meta_test):
    scaler_main = RobustScaler()
    scaler_meta = RobustScaler()

    X_train_s = scaler_main.fit_transform(X_train)
    X_val_s = scaler_main.transform(X_val)
    X_test_s = scaler_main.transform(X_test)

    Xm_train_s = scaler_meta.fit_transform(X_meta_train)
    Xm_val_s = scaler_meta.transform(X_meta_val)
    Xm_test_s = scaler_meta.transform(X_meta_test)
    return X_train_s, X_val_s, X_test_s, Xm_train_s, Xm_val_s, Xm_test_s, scaler_main, scaler_meta


def _fit_cluster_train_only(data: pd.DataFrame, meta_features: list[str], n_clusters: int, train_ratio: float):
    meta_X = data[meta_features].dropna()
    split_idx = int(len(meta_X) * train_ratio)
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km.fit(meta_X.iloc[:split_idx])
    labels = km.predict(meta_X)
    return pd.Series(labels, index=meta_X.index, name="clusters")


def _build_dataset(use_synthetic_if_missing: bool = False):
    csv_path = f"files/{HP.symbol}.csv"
    if Path(csv_path).exists():
        prices = read_price_csv(csv_path)
    elif use_synthetic_if_missing:
        prices = generate_synthetic_ohlc(n=4000, seed=SEED)
    else:
        raise FileNotFoundError(f"Missing input file: {csv_path}")

    fs = build_features(prices, periods=list(HP.periods), periods_meta=list(HP.periods_meta), atr_window=HP.atr_window)
    labels = create_labels(fs.data["close"], fs.data["high"], fs.data["low"], train_ratio=HP.train_ratio)
    data = fs.data.join(labels, how="inner")
    data = data[data["labels"].isin([0, 1])].copy()

    clusters = _fit_cluster_train_only(data, fs.meta_features, HP.n_clusters, HP.train_ratio)
    data = data.join(clusters, how="inner")
    data = data.dropna().copy()
    return data, fs.main_features, fs.meta_features


def _signal_from_probability(prob_sell: np.ndarray, sell_threshold: float) -> np.ndarray:
    buy_threshold = 1.0 - sell_threshold
    signal = np.full(len(prob_sell), np.nan, dtype=float)
    signal[prob_sell >= sell_threshold] = 1.0
    signal[prob_sell <= buy_threshold] = 0.0
    return signal


def _select_threshold_by_validation_trading(
    val_index: pd.Index,
    val_close: pd.Series,
    val_prob_main: np.ndarray,
    val_prob_meta: np.ndarray,
    threshold_candidates: list[float],
) -> tuple[float, pd.DataFrame]:
    rows = []
    for t in threshold_candidates:
        pred = pd.DataFrame(index=val_index)
        pred["close"] = val_close.loc[val_index]
        pred["labels"] = _signal_from_probability(val_prob_main, t)
        pred["meta_labels"] = (val_prob_meta >= 0.5).astype(float)

        trades_df, m = backtest_signals(
            pred,
            stop=HP.stop_loss,
            take=HP.take_profit,
            markup=HP.markup,
            max_hold=120,
            signal_shift=1,
        )

        score = m["pnl"]
        if m["trades"] < 50:
            score -= 1000.0
        if m["profit_factor"] < 1.0:
            score -= 500.0

        rows.append({"threshold": t, "score": score, **m})

    res = pd.DataFrame(rows).sort_values(["score", "profit_factor", "pnl"], ascending=False)
    best_t = float(res.iloc[0]["threshold"])
    return best_t, res


def train_pipeline(use_synthetic_if_missing: bool = False, run_search: bool = False):
    Path("reports").mkdir(exist_ok=True)
    Path(HP.export_path).mkdir(parents=True, exist_ok=True)

    data, main_features, meta_features = _build_dataset(use_synthetic_if_missing=use_synthetic_if_missing)
    evaluate_label_quality(data, out_dir="reports")
    build_split_visualization(data, HP.train_ratio, HP.val_ratio)

    X_main = data[main_features]
    X_meta = data[meta_features]
    y_main = data["labels"].astype(int)
    y_meta = data["meta_target"].astype(int)

    train_X_raw, train_y_raw, val_X_raw, val_y, test_X_raw, test_y = _time_split_3way(X_main, y_main, HP.train_ratio, HP.val_ratio)
    train_Xm_raw, train_ym, val_Xm_raw, val_ym, test_Xm_raw, test_ym = _time_split_3way(X_meta, y_meta, HP.train_ratio, HP.val_ratio)

    fs_result = select_main_features_train_only(
        X_train=train_X_raw,
        y_train=train_y_raw,
        feature_names=main_features,
        corr_threshold=0.96,
        min_features=16,
        random_seed=SEED,
    )
    main_features = fs_result.selected_features
    fs_result.importance_table.to_csv("reports/feature_selection_importance.csv", index=False)
    with open("reports/feature_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_original": len(X_main.columns),
                "n_selected": len(main_features),
                "removed_zero_importance": fs_result.removed_zero_importance,
                "removed_high_correlation": fs_result.removed_high_correlation,
                "selected_features": main_features,
            },
            f,
            indent=2,
        )

    X_main = X_main[main_features]
    train_X, train_y, val_X, val_y, test_X, test_y = _time_split_3way(X_main, y_main, HP.train_ratio, HP.val_ratio)
    train_Xm, train_ym, val_Xm, val_ym, test_Xm, test_ym = _time_split_3way(X_meta, y_meta, HP.train_ratio, HP.val_ratio)

    if run_search:
        search_space = {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.08],
            "l2_leaf_reg": [2.0, 5.0, 8.0],
            "iterations": [150, 250, 350],
        }
        best_params, _ = run_param_search(
            X_main,
            y_main,
            train_ratio=HP.train_ratio,
            val_ratio=HP.val_ratio,
            search_space=search_space,
            out_dir="reports",
            max_trials=20,
            seed=SEED,
        )
    else:
        best_params = {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "iterations": HP.main_iterations}

    train_X_s, val_X_s, test_X_s, train_Xm_s, val_Xm_s, test_Xm_s, scaler_main, scaler_meta = _normalize_features(
        train_X, val_X, test_X, train_Xm, val_Xm, test_Xm
    )

    model = CatBoostClassifier(
        depth=int(best_params["depth"]),
        learning_rate=float(best_params["learning_rate"]),
        l2_leaf_reg=float(best_params["l2_leaf_reg"]),
        iterations=int(best_params["iterations"]),
        eval_metric="F1",
        random_seed=SEED,
        verbose=False,
        use_best_model=True,
        early_stopping_rounds=30,
        auto_class_weights="Balanced",
    )
    model.fit(train_X_s, train_y, eval_set=(val_X_s, val_y))

    meta_model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        iterations=HP.meta_iterations,
        eval_metric="F1",
        random_seed=SEED,
        verbose=False,
        use_best_model=True,
        early_stopping_rounds=30,
        auto_class_weights="Balanced",
    )
    meta_model.fit(train_Xm_s, train_ym, eval_set=(val_Xm_s, val_ym))

    val_prob = model.predict_proba(val_X_s)[:, 1]
    meta_val_prob = meta_model.predict_proba(val_Xm_s)[:, 1]

    # (1) ML-based threshold (macro-f1 + anti-collapse)
    best_t_ml, threshold_pack = optimize_threshold(val_y, val_prob)
    pd.DataFrame(threshold_pack["all"]).to_csv("reports/threshold_search_ml.csv", index=False)

    # (2) Trading-based threshold on validation only (no test leakage)
    threshold_candidates = [round(x, 3) for x in np.arange(0.50, 0.70, 0.01)]
    best_t_trade, threshold_trade_df = _select_threshold_by_validation_trading(
        val_index=val_X.index,
        val_close=data["close"],
        val_prob_main=val_prob,
        val_prob_meta=meta_val_prob,
        threshold_candidates=threshold_candidates,
    )
    threshold_trade_df.to_csv("reports/threshold_search_trading.csv", index=False)

    # Pick threshold by validation trading score; fallback to ML threshold if degenerate
    best_t = best_t_trade if np.isfinite(best_t_trade) else best_t_ml

    test_prob = model.predict_proba(test_X_s)[:, 1]
    metrics_cls, _, _ = classification_metrics(test_y, test_prob, threshold=best_t)
    with open("reports/classification_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics_cls, f, indent=2)
    save_classification_reports(test_y, test_prob, best_t, HP.symbol, out_dir="reports")

    meta_prob = meta_model.predict_proba(test_Xm_s)[:, 1]
    pred_df = pd.DataFrame(index=test_X.index)
    pred_df["close"] = data.loc[test_X.index, "close"]
    pred_df["labels"] = _signal_from_probability(test_prob, best_t)
    pred_df["meta_labels"] = (meta_prob >= 0.5).astype(float)

    trades_df, trading_metrics = backtest_signals(
        pred_df,
        stop=HP.stop_loss,
        take=HP.take_profit,
        markup=HP.markup,
        max_hold=120,
        signal_shift=1,
    )
    save_backtest_reports(trades_df, HP.symbol, out_dir="reports")
    with open("reports/trading_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(trading_metrics, f, indent=2)

    save_feature_importance(model, main_features, HP.symbol, out_dir="reports", top_n=20)

    report = {
        "timeframe": "H1",
        "train_ratio": HP.train_ratio,
        "val_ratio": HP.val_ratio,
        "test_ratio": 1 - HP.train_ratio - HP.val_ratio,
        "n_features_main": len(main_features),
        "n_features_meta": len(meta_features),
        "best_threshold_sell": best_t,
        "best_threshold_buy": 1.0 - best_t,
        "threshold_source": "validation_trading",
        "classification_metrics_test": metrics_cls,
        "trading_metrics_test": trading_metrics,
        "best_main_params": {
            "depth": int(best_params["depth"]),
            "learning_rate": float(best_params["learning_rate"]),
            "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
            "iterations": int(best_params["iterations"]),
        },
        "main_model": {"best_iteration": int(model.get_best_iteration()), "best_score": model.get_best_score()},
        "meta_model": {"best_iteration": int(meta_model.get_best_iteration()), "best_score": meta_model.get_best_score()},
        "train_period": {"start": str(train_X.index.min()), "end": str(train_X.index.max())},
        "val_period": {"start": str(val_X.index.min()), "end": str(val_X.index.max())},
        "test_period": {"start": str(test_X.index.min()), "end": str(test_X.index.max())},
        "seed": SEED,
        "feature_selection": {
            "removed_zero_importance": fs_result.removed_zero_importance,
            "removed_high_correlation": fs_result.removed_high_correlation,
            "selected_features": main_features,
        },
    }

    exported = export_artifacts(
        symbol=HP.symbol,
        model_number=HP.model_number,
        export_path=HP.export_path,
        model=model,
        meta_model=meta_model,
        scaler_main=scaler_main,
        scaler_meta=scaler_meta,
        report=report,
        periods=list(HP.periods),
        periods_meta=list(HP.periods_meta),
        feature_names=main_features,
        feature_names_meta=meta_features,
        decision_threshold=best_t,
        sample_main=test_X_s,
        sample_meta=test_Xm_s,
    )

    with open("reports/run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"exported": exported, "report": report}, f, indent=2)

    print("Training complete")
    print(json.dumps({"classification": metrics_cls, "trading": trading_metrics}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "search"], default="train")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data when files/XAUUSD_H1.csv is missing")
    args = parser.parse_args()

    train_pipeline(use_synthetic_if_missing=args.synthetic, run_search=args.mode == "search")


if __name__ == "__main__":
    main()

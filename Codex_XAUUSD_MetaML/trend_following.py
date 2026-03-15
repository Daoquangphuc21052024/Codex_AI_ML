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
from sklearn.metrics import confusion_matrix
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

LABEL_MEANING = {0: "BUY", 1: "SELL"}


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
    min_total_trades: int = 80
    min_side_trades: int = 10
    max_side_dominance: float = 0.92


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


def _assert_label_mapping(y: pd.Series) -> None:
    uniq = set(y.unique().tolist())
    assert uniq.issubset({0, 1}), f"labels must be binary 0/1, got {uniq}"


def _split_label_diagnostics(y: pd.Series, train_ratio: float, val_ratio: float) -> dict:
    _, y_tr, _, y_val, _, y_te = _time_split_3way(y, y, train_ratio, val_ratio)

    def stats(s: pd.Series) -> dict:
        counts = s.value_counts().to_dict()
        total = max(1, len(s))
        return {
            "n": int(len(s)),
            "buy(0)": int(counts.get(0, 0)),
            "sell(1)": int(counts.get(1, 0)),
            "buy_ratio": float(counts.get(0, 0) / total),
            "sell_ratio": float(counts.get(1, 0) / total),
        }

    return {
        "semantic_mapping": {"0": "BUY", "1": "SELL"},
        "train": stats(y_tr),
        "val": stats(y_val),
        "test": stats(y_te),
    }


def _signal_from_probability(prob_sell: np.ndarray, buy_threshold: float, sell_threshold: float) -> np.ndarray:
    assert 0.0 <= buy_threshold < sell_threshold <= 1.0, "require buy_threshold < sell_threshold"
    is_short_signal = prob_sell >= sell_threshold
    is_long_signal = prob_sell <= buy_threshold
    assert not np.any(is_short_signal & is_long_signal), "a bar cannot be long and short simultaneously"

    signal = np.full(len(prob_sell), np.nan, dtype=float)
    signal[is_long_signal] = 0.0
    signal[is_short_signal] = 1.0

    no_trade = ~(is_short_signal | is_long_signal)
    assert np.all(np.isnan(signal[no_trade])), "bars between thresholds must be no-trade"
    return signal


def _binary_signal(prob_sell: np.ndarray) -> np.ndarray:
    return (prob_sell >= 0.5).astype(float)


def _probability_diagnostics(y_true: pd.Series, prob_sell: np.ndarray, buy_t: float, sell_t: float) -> dict:
    y_arr = y_true.to_numpy()
    prob_buy = 1.0 - prob_sell
    no_trade_mask = (prob_sell > buy_t) & (prob_sell < sell_t)
    return {
        "prob_sell_mean": float(prob_sell.mean()),
        "prob_sell_std": float(prob_sell.std()),
        "prob_buy_mean": float(prob_buy.mean()),
        "prob_buy_std": float(prob_buy.std()),
        "count_prob_ge_sell_threshold": int((prob_sell >= sell_t).sum()),
        "count_prob_le_buy_threshold": int((prob_sell <= buy_t).sum()),
        "count_prob_between_thresholds": int(no_trade_mask.sum()),
        "ambiguity_rate_between_thresholds": float(no_trade_mask.mean()),
        "predicted_positive_rate_at_05": float((prob_sell >= 0.5).mean()),
        "true_buy_prob_sell_mean": float(prob_sell[y_arr == 0].mean()) if np.any(y_arr == 0) else 0.0,
        "true_sell_prob_sell_mean": float(prob_sell[y_arr == 1].mean()) if np.any(y_arr == 1) else 0.0,
    }


def _direction_diagnostics(name: str, signals: np.ndarray) -> dict:
    long_n = int(np.sum(signals == 0.0))
    short_n = int(np.sum(signals == 1.0))
    no_trade_n = int(np.sum(np.isnan(signals)))
    total = max(1, len(signals))
    dominance = max(long_n, short_n) / max(1, long_n + short_n)
    warn = ""
    if long_n + short_n > 0 and dominance > HP.max_side_dominance:
        warn = f"directional dominance warning: {dominance:.3f}"
    return {
        "name": name,
        "long_signals": long_n,
        "short_signals": short_n,
        "no_trade_signals": no_trade_n,
        "long_ratio": float(long_n / total),
        "short_ratio": float(short_n / total),
        "no_trade_ratio": float(no_trade_n / total),
        "dominance": float(dominance),
        "warning": warn,
    }


def _semantic_confusion(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "labels": ["BUY(0)", "SELL(1)"],
        "matrix": cm.tolist(),
        "BUY_as_BUY": int(cm[0, 0]),
        "BUY_as_SELL": int(cm[0, 1]),
        "SELL_as_BUY": int(cm[1, 0]),
        "SELL_as_SELL": int(cm[1, 1]),
    }


def _select_thresholds_by_validation_trading(
    val_index: pd.Index,
    val_close: pd.Series,
    val_prob_main: np.ndarray,
    val_prob_meta: np.ndarray,
    use_meta: bool,
) -> tuple[tuple[float, float], pd.DataFrame]:
    rows = []
    buy_candidates = [round(x, 3) for x in np.arange(0.34, 0.50, 0.01)]
    sell_candidates = [round(x, 3) for x in np.arange(0.51, 0.68, 0.01)]

    for buy_t in buy_candidates:
        for sell_t in sell_candidates:
            if buy_t >= sell_t:
                continue

            pred = pd.DataFrame(index=val_index)
            pred["close"] = val_close.loc[val_index]
            pred["labels"] = _signal_from_probability(val_prob_main, buy_t, sell_t)
            pred["meta_labels"] = (val_prob_meta >= 0.5).astype(float) if use_meta else 1.0

            _, m = backtest_signals(pred, stop=HP.stop_loss, take=HP.take_profit, markup=HP.markup, max_hold=120, signal_shift=1)
            side_total = max(1, m["long_trades"] + m["short_trades"])
            side_dominance = max(m["long_trades"], m["short_trades"]) / side_total

            hard_fail = (
                m["trades"] < HP.min_total_trades
                or m["long_trades"] < HP.min_side_trades
                or m["short_trades"] < HP.min_side_trades
                or side_dominance > HP.max_side_dominance
            )

            score = m["pnl"]
            if m["profit_factor"] < 1.0:
                score -= 300.0
            if hard_fail:
                score -= 1500.0

            rows.append(
                {
                    "buy_threshold": buy_t,
                    "sell_threshold": sell_t,
                    "score": score,
                    "hard_fail": int(hard_fail),
                    "side_dominance": side_dominance,
                    **m,
                }
            )

    res = pd.DataFrame(rows).sort_values(["hard_fail", "score", "profit_factor", "pnl"], ascending=[True, False, False, False])
    best = res.iloc[0]
    return (float(best["buy_threshold"]), float(best["sell_threshold"])), res


def _run_meta_diagnostics(
    val_index: pd.Index,
    val_close: pd.Series,
    val_prob_main: np.ndarray,
    val_prob_meta: np.ndarray,
    buy_t: float,
    sell_t: float,
) -> tuple[str, pd.DataFrame]:
    variants = []

    pred_main_only = pd.DataFrame(index=val_index)
    pred_main_only["close"] = val_close.loc[val_index]
    pred_main_only["labels"] = _signal_from_probability(val_prob_main, buy_t, sell_t)
    pred_main_only["meta_labels"] = 1.0
    _, m1 = backtest_signals(pred_main_only, stop=HP.stop_loss, take=HP.take_profit, markup=HP.markup, max_hold=120, signal_shift=1)
    variants.append({"variant": "main_only", **m1})

    pred_main_meta = pred_main_only.copy()
    pred_main_meta["meta_labels"] = (val_prob_meta >= 0.5).astype(float)
    _, m2 = backtest_signals(pred_main_meta, stop=HP.stop_loss, take=HP.take_profit, markup=HP.markup, max_hold=120, signal_shift=1)
    variants.append({"variant": "main_plus_meta", **m2})

    pred_simple = pd.DataFrame(index=val_index)
    pred_simple["close"] = val_close.loc[val_index]
    pred_simple["labels"] = _binary_signal(val_prob_main)
    pred_simple["meta_labels"] = 1.0
    _, m3 = backtest_signals(pred_simple, stop=HP.stop_loss, take=HP.take_profit, markup=HP.markup, max_hold=120, signal_shift=1)
    variants.append({"variant": "main_binary_t05", **m3})

    cmp = pd.DataFrame(variants).sort_values(["pnl", "profit_factor"], ascending=False)
    best_variant = str(cmp.iloc[0]["variant"])
    return best_variant, cmp


def train_pipeline(use_synthetic_if_missing: bool = False, run_search: bool = False, disable_meta: bool = False):
    Path("reports").mkdir(exist_ok=True)
    Path(HP.export_path).mkdir(parents=True, exist_ok=True)

    data, main_features, meta_features = _build_dataset(use_synthetic_if_missing=use_synthetic_if_missing)
    evaluate_label_quality(data, out_dir="reports")
    build_split_visualization(data, HP.train_ratio, HP.val_ratio)

    X_main = data[main_features]
    X_meta = data[meta_features]
    y_main = data["labels"].astype(int)
    y_meta = data["meta_target"].astype(int)

    _assert_label_mapping(y_main)
    label_diag = _split_label_diagnostics(y_main, HP.train_ratio, HP.val_ratio)
    with open("reports/label_split_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(label_diag, f, indent=2)

    train_X_raw, train_y_raw, _, _, _, _ = _time_split_3way(X_main, y_main, HP.train_ratio, HP.val_ratio)
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

    val_prob_sell = model.predict_proba(val_X_s)[:, 1]
    meta_val_prob = meta_model.predict_proba(val_Xm_s)[:, 1]

    best_t_ml, threshold_pack = optimize_threshold(val_y, val_prob_sell)
    pd.DataFrame(threshold_pack["all"]).to_csv("reports/threshold_search_ml.csv", index=False)

    use_meta_for_threshold = not disable_meta
    (buy_t, sell_t), threshold_trade_df = _select_thresholds_by_validation_trading(
        val_index=val_X.index,
        val_close=data["close"],
        val_prob_main=val_prob_sell,
        val_prob_meta=meta_val_prob,
        use_meta=use_meta_for_threshold,
    )
    threshold_trade_df.to_csv("reports/threshold_search_trading.csv", index=False)
    threshold_trade_df.head(20).to_csv("reports/threshold_search_trading_top20.csv", index=False)

    best_variant, meta_cmp = _run_meta_diagnostics(
        val_index=val_X.index,
        val_close=data["close"],
        val_prob_main=val_prob_sell,
        val_prob_meta=meta_val_prob,
        buy_t=buy_t,
        sell_t=sell_t,
    )
    meta_cmp.to_csv("reports/meta_variant_comparison_val.csv", index=False)

    meta_best_iter = int(meta_model.get_best_iteration())
    meta_warning = ""
    use_meta_gate = not disable_meta and best_variant == "main_plus_meta"
    if meta_best_iter <= 1:
        meta_warning = "meta_model best_iteration <= 1: meta gate disabled"
        use_meta_gate = False

    val_prob_diag = _probability_diagnostics(val_y, val_prob_sell, buy_t, sell_t)
    val_signals = _signal_from_probability(val_prob_sell, buy_t, sell_t)
    val_dir_diag = _direction_diagnostics("val", val_signals)

    test_prob_sell = model.predict_proba(test_X_s)[:, 1]
    test_prob_diag = _probability_diagnostics(test_y, test_prob_sell, buy_t, sell_t)
    test_signals = _signal_from_probability(test_prob_sell, buy_t, sell_t)
    test_dir_diag = _direction_diagnostics("test", test_signals)

    with open("reports/probability_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_prob_diag, "test": test_prob_diag}, f, indent=2)
    with open("reports/direction_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_dir_diag, "test": test_dir_diag}, f, indent=2)

    metrics_cls, y_pred_cls, _ = classification_metrics(test_y, test_prob_sell, threshold=sell_t)
    metrics_cls["buy_threshold"] = buy_t
    metrics_cls["sell_threshold"] = sell_t
    metrics_cls["probability_semantic"] = "prob_sell = P(label=SELL=1)"
    metrics_cls["semantic_confusion"] = _semantic_confusion(test_y, y_pred_cls)

    with open("reports/classification_metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics_cls, f, indent=2)
    save_classification_reports(test_y, test_prob_sell, sell_t, HP.symbol, out_dir="reports")

    meta_test = (meta_model.predict_proba(test_Xm_s)[:, 1] >= 0.5).astype(float) if use_meta_gate else np.ones(len(test_X), dtype=float)
    pred_df = pd.DataFrame(index=test_X.index)
    pred_df["close"] = data.loc[test_X.index, "close"]
    pred_df["labels"] = test_signals
    pred_df["meta_labels"] = meta_test

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

    diagnostics = {
        "label_split": label_diag,
        "probability": {"val": val_prob_diag, "test": test_prob_diag},
        "direction": {"val": val_dir_diag, "test": test_dir_diag},
        "threshold_choice": {
            "buy_threshold": buy_t,
            "sell_threshold": sell_t,
            "selection_rule": "validation trading with side-balance constraints",
            "top_candidates": threshold_trade_df.head(10).to_dict(orient="records"),
        },
        "meta": {
            "disable_meta_flag": disable_meta,
            "meta_best_iteration": meta_best_iter,
            "meta_variant_best": best_variant,
            "meta_gate_used": use_meta_gate,
            "warning": meta_warning,
        },
    }
    with open("reports/pipeline_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    report = {
        "timeframe": "H1",
        "train_ratio": HP.train_ratio,
        "val_ratio": HP.val_ratio,
        "test_ratio": 1 - HP.train_ratio - HP.val_ratio,
        "class_semantic": {"0": "BUY", "1": "SELL"},
        "n_features_main": len(main_features),
        "n_features_meta": len(meta_features),
        "best_threshold_sell": sell_t,
        "best_threshold_buy": buy_t,
        "threshold_source": "validation_trading_constrained",
        "classification_metrics_test": metrics_cls,
        "trading_metrics_test": trading_metrics,
        "best_main_params": {
            "depth": int(best_params["depth"]),
            "learning_rate": float(best_params["learning_rate"]),
            "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
            "iterations": int(best_params["iterations"]),
        },
        "main_model": {"best_iteration": int(model.get_best_iteration()), "best_score": model.get_best_score()},
        "meta_model": {
            "best_iteration": meta_best_iter,
            "best_score": meta_model.get_best_score(),
            "meta_gate_used": use_meta_gate,
            "warning": meta_warning,
        },
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
        decision_threshold=sell_t,
        sample_main=test_X_s,
        sample_meta=test_Xm_s,
    )

    with open("reports/run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"exported": exported, "report": report, "diagnostics": diagnostics}, f, indent=2)

    print("Training complete")
    print(json.dumps({"classification": metrics_cls, "trading": trading_metrics, "diagnostics": diagnostics["meta"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "search"], default="train")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data when files/XAUUSD_H1.csv is missing")
    parser.add_argument("--disable-meta", action="store_true", help="Disable meta-model confirmation gate")
    args = parser.parse_args()

    train_pipeline(
        use_synthetic_if_missing=args.synthetic,
        run_search=args.mode == "search",
        disable_meta=args.disable_meta,
    )


if __name__ == "__main__":
    main()

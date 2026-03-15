from __future__ import annotations

import math
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from export_lib import export_artifacts
from labeling_lib import SEED, create_labels
from tester_lib import tester

np.random.seed(SEED)
random.seed(SEED)

hyper_params = {
    "symbol": "XAUUSD_H1",
    "export_path": os.environ.get("MT5_INCLUDE_PATH", os.path.join(os.getcwd(), "exports")),
    "model_number": 0,
    "markup": 0.35,
    "stop_loss": 8.0,
    "take_profit": 4.0,
    "periods": [5, 35, 65, 95, 125, 155, 185, 215, 245, 275],
    "periods_meta": [50, 100, 200],
    "atr_window": 14,
    "backward": datetime(2000, 1, 1),
    "forward": datetime(2024, 1, 1),
    "full_forward": datetime(2026, 1, 1),
    "n_clusters": 5,
    "rolling": [10, 30, 60],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "wf_splits": 2,
    "min_r2": 0.0,
}


def get_prices() -> pd.DataFrame:
    filepath = f"files/{hyper_params['symbol']}.csv"
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    if "<DATE>" in header or "<CLOSE>" in header:
        p = pd.read_csv(filepath, sep=r"\s+")
        df = pd.DataFrame()
        df["time"] = p["<DATE>"].astype(str) + " " + p["<TIME>"].astype(str)
        df["close"] = p["<CLOSE>"]
        df["high"] = p["<HIGH>"]
        df["low"] = p["<LOW>"]
    else:
        p = pd.read_csv(filepath)
        p.columns = [c.strip().lower() for c in p.columns]
        time_col = next(c for c in p.columns if c in {"time", "date", "datetime", "timestamp"})
        close_col = next(c for c in p.columns if c == "close")
        high_col = next(c for c in p.columns if c == "high")
        low_col = next(c for c in p.columns if c == "low")
        df = pd.DataFrame({
            "time": p[time_col],
            "close": p[close_col],
            "high": p[high_col],
            "low": p[low_col],
        })

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df = df.sort_index()
    return df.dropna()


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    for n in hyper_params["periods"]:
        ma = close.rolling(n).mean()
        out[f"ma_dev_{n}"] = (close / ma) - 1
        out[f"mom_{n}"] = (close / close.shift(n)) - 1
        out[f"volratio_{n}"] = close.rolling(n).std() / (close.rolling(n * 3).std() + 1e-12)
        rolling_low = close.rolling(n).min()
        rolling_high = close.rolling(n).max()
        out[f"pos_{n}"] = (close - rolling_low) / ((rolling_high - rolling_low) + 1e-12)
        out[f"rsi_like_{n}"] = (close.diff() > 0).rolling(n).mean()

    atr = (high - low).rolling(hyper_params["atr_window"]).mean() / close
    for n in hyper_params["periods_meta"]:
        out[f"skew_meta_{n}"] = close.rolling(n).skew()
    out["atr_meta"] = atr

    return out.dropna()


def _time_split_3way(X, y, train_ratio=0.6, val_ratio=0.2):
    n = len(X)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))
    return (
        X.iloc[:i_val], y.iloc[:i_val],
        X.iloc[i_val:i_test], y.iloc[i_val:i_test],
        X.iloc[i_test:], y.iloc[i_test:],
    )


def normalize_features(X_train, X_val, X_test, X_meta_train, X_meta_val, X_meta_test):
    scaler_main = RobustScaler()
    X_train_s = scaler_main.fit_transform(X_train)
    X_val_s = scaler_main.transform(X_val)
    X_test_s = scaler_main.transform(X_test)

    scaler_meta = RobustScaler()
    Xm_train_s = scaler_meta.fit_transform(X_meta_train)
    Xm_val_s = scaler_meta.transform(X_meta_val)
    Xm_test_s = scaler_meta.transform(X_meta_test)

    return X_train_s, X_val_s, X_test_s, Xm_train_s, Xm_val_s, Xm_test_s, scaler_main, scaler_meta


def get_volatility_regime(data: pd.DataFrame, n_clusters: int, train_ratio: float = 0.6):
    meta_cols = [c for c in data.columns if c.endswith("_meta")]
    meta_X = data[meta_cols].dropna()
    split_idx = int(len(meta_X) * train_ratio)
    meta_train = meta_X.iloc[:split_idx]

    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km.fit(meta_train)
    labels = km.predict(meta_X)
    return pd.Series(labels, index=meta_X.index, name="clusters")


def walk_forward_score(model, meta_model, X_test_s, X_meta_test_s, test_index, close_series):
    tscv = TimeSeriesSplit(n_splits=hyper_params["wf_splits"])
    scores = []

    X_test_df = pd.DataFrame(X_test_s, index=test_index)
    Xm_test_df = pd.DataFrame(X_meta_test_s, index=test_index)

    for _, test_idx in tscv.split(X_test_df):
        idx = X_test_df.index[test_idx]
        fold_X = X_test_df.iloc[test_idx].to_numpy()
        fold_Xm = Xm_test_df.iloc[test_idx].to_numpy()

        pr_fold = pd.DataFrame(index=idx)
        pr_fold["close"] = close_series.loc[idx]
        pr_fold["labels"] = (model.predict_proba(fold_X)[:, 1] >= 0.5).astype(float)
        pr_fold["meta_labels"] = (meta_model.predict_proba(fold_Xm)[:, 1] >= 0.5).astype(float)

        r2 = tester(
            pr_fold,
            hyper_params["stop_loss"],
            hyper_params["take_profit"],
            idx.max(),
            idx.min(),
            hyper_params["markup"],
            False,
            hyper_params["symbol"],
            tag="wf",
        )
        if not math.isnan(r2):
            scores.append(r2)

    if not scores:
        return -1.0, 1.0
    return float(np.mean(scores)), float(np.std(scores))


def fit_final_models(dataset: pd.DataFrame):
    feature_cols = [
        c for c in dataset.columns
        if any(c.startswith(p) for p in ["ma_dev_", "mom_", "volratio_", "pos_", "rsi_like_"])
    ]
    meta_cols = [c for c in dataset.columns if c.endswith("_meta")]

    X_all = dataset[feature_cols]
    X_meta_all = dataset[meta_cols]
    y_main = dataset["labels"].astype("int16")
    y_meta = dataset["meta_target"].astype("int16")

    train_X, train_y, val_X, val_y, test_X, test_y = _time_split_3way(
        X_all, y_main, hyper_params["train_ratio"], hyper_params["val_ratio"]
    )
    train_Xm, train_ym, val_Xm, val_ym, test_Xm, test_ym = _time_split_3way(
        X_meta_all, y_meta, hyper_params["train_ratio"], hyper_params["val_ratio"]
    )

    train_X_s, val_X_s, test_X_s, train_Xm_s, val_Xm_s, test_Xm_s, scaler_main, scaler_meta = normalize_features(
        train_X, val_X, test_X, train_Xm, val_Xm, test_Xm
    )

    model = CatBoostClassifier(
        iterations=200,
        eval_metric="Accuracy",
        use_best_model=True,
        early_stopping_rounds=20,
        random_seed=SEED,
        task_type="CPU",
        verbose=False,
    )
    model.fit(train_X_s, train_y, eval_set=(val_X_s, val_y))

    meta_model = CatBoostClassifier(
        iterations=200,
        eval_metric="F1",
        use_best_model=True,
        early_stopping_rounds=20,
        random_seed=SEED,
        task_type="CPU",
        verbose=False,
    )
    meta_model.fit(train_Xm_s, train_ym, eval_set=(val_Xm_s, val_ym))

    wf_mean, wf_std = walk_forward_score(
        model,
        meta_model,
        test_X_s,
        test_Xm_s,
        test_X.index,
        dataset["close"],
    )

    return {
        "wf_mean": wf_mean,
        "wf_std": wf_std,
        "model": model,
        "meta_model": meta_model,
        "scaler_main": scaler_main,
        "scaler_meta": scaler_meta,
        "feature_cols": feature_cols,
        "meta_cols": meta_cols,
        "test_X": test_X_s,
        "test_Xm": test_Xm_s,
        "test_index": test_X.index,
    }


def run_pipeline():
    prices = get_prices()
    data = get_features(prices)

    labels_df = create_labels(data["close"], hyper_params["rolling"], markup=hyper_params["markup"], seed=SEED)
    data = data.join(labels_df, how="inner")

    # main model dùng label buy/sell thành công
    data = data[data["labels"].isin([0, 1])].copy()

    clusters = get_volatility_regime(data, hyper_params["n_clusters"], hyper_params["train_ratio"])
    data = data.join(clusters, how="inner")

    result = fit_final_models(data)
    print(f"Walk-forward R2 mean={result['wf_mean']:.4f}, std={result['wf_std']:.4f}")

    test_pred = pd.DataFrame(index=result["test_index"])
    test_pred["close"] = data.loc[result["test_index"], "close"]
    test_pred["labels"] = (result["model"].predict_proba(result["test_X"])[:, 1] >= 0.5).astype(float)
    test_pred["meta_labels"] = (result["meta_model"].predict_proba(result["test_Xm"])[:, 1] >= 0.5).astype(float)

    tester(
        test_pred,
        hyper_params["stop_loss"],
        hyper_params["take_profit"],
        hyper_params["forward"],
        hyper_params["backward"],
        hyper_params["markup"],
        True,
        hyper_params["symbol"],
        tag="best_model",
    )

    report = {
        "timeframe": "H1",
        "wf_r2": result["wf_mean"],
        "wf_std": result["wf_std"],
        "train_ratio": hyper_params["train_ratio"],
        "val_ratio": hyper_params["val_ratio"],
        "wf_splits": hyper_params["wf_splits"],
        "n_features_main": len(result["feature_cols"]),
        "n_features_meta": len(result["meta_cols"]),
        "main_model": {
            "iterations": result["model"].get_best_iteration(),
            "best_score": result["model"].get_best_score(),
        },
        "meta_model": {
            "iterations": result["meta_model"].get_best_iteration(),
            "best_score": result["meta_model"].get_best_score(),
        },
    }

    exported = export_artifacts(
        hyper_params["symbol"],
        hyper_params["model_number"],
        hyper_params["export_path"],
        result["model"],
        result["meta_model"],
        result["scaler_main"],
        result["scaler_meta"],
        report,
        hyper_params["periods"],
        hyper_params["periods_meta"],
    )
    print("Exported:")
    for k, v in exported.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_pipeline()

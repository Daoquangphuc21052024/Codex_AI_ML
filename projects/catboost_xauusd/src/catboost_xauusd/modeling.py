from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from .config import FeatureConfig, TrainConfig
from .validation import FoldIndices


@dataclass
class FoldResult:
    fold: int
    train_acc: float
    val_acc: float
    test_acc: float
    weighted_f1: float
    macro_f1: float
    balanced_acc: float
    mcc: float
    overfit_gap: float
    confusion: np.ndarray
    report: dict
    test_probs: np.ndarray
    test_pred: np.ndarray
    no_trade_threshold: float
    trade_threshold: float
    side_margin: float
    expected_edge_min: float
    selected_features: list[str]
    pred_distribution: dict[int, float]
    actual_distribution: dict[int, float]
    confidence_bucket_metrics: list[dict[str, float | str | int]]


def _compute_class_weights(y_train: pd.Series) -> list[float]:
    classes = [0, 1, 2]
    counts = y_train.value_counts().to_dict()
    total = float(len(y_train))
    return [total / (len(classes) * float(counts.get(cls, 1.0))) for cls in classes]


def _select_fold_features(x_train: pd.DataFrame, y_train: pd.Series, candidates: list[str], feat_cfg: FeatureConfig) -> list[str]:
    local = x_train[candidates].copy()
    valid = [c for c in candidates if local[c].std(ddof=0) > 1e-12]
    if not valid:
        raise ValueError("No valid features left after variance filter")

    mi = mutual_info_classif(local[valid], y_train.to_numpy(), random_state=42)
    ranked = sorted(zip(valid, mi), key=lambda x: x[1], reverse=True)

    corr = local[[f for f, _ in ranked]].corr().abs()
    selected: list[str] = []
    for feature, _ in ranked:
        if len(selected) >= feat_cfg.max_features:
            break
        if all(corr.loc[feature, kept] < feat_cfg.corr_threshold for kept in selected):
            selected.append(feature)

    if len(selected) < min(3, len(valid)):
        for feature, _ in ranked:
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= min(feat_cfg.max_features, len(valid)):
                break

    return selected[: feat_cfg.max_features]


def _decode_predictions(
    probs: np.ndarray,
    no_trade_threshold: float,
    trade_threshold: float,
    side_margin: float,
    expected_edge_min: float,
) -> np.ndarray:
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    side = np.where(p1 >= p2, 1, 2)
    side_prob = np.maximum(p1, p2)
    alt_prob = np.minimum(p1, p2)
    side_advantage = side_prob - alt_prob

    # Edge proxy encourages asymmetry towards directional class vs no-trade class.
    expected_edge = side_prob - p0
    should_trade = (
        (p0 < no_trade_threshold)
        & (side_prob >= trade_threshold)
        & (side_advantage >= side_margin)
        & (expected_edge >= expected_edge_min)
    )
    return np.where(should_trade, side, 0).astype(np.int32)


def _confidence_bucket_report(y_true: pd.Series, y_pred: np.ndarray, probs: np.ndarray) -> list[dict[str, float | str | int]]:
    confidence = np.maximum(probs[:, 1], probs[:, 2])
    bins = pd.cut(confidence, bins=[0.0, 0.40, 0.55, 0.70, 0.85, 1.0], include_lowest=True)
    rows: list[dict[str, float | str | int]] = []

    bucket_df = pd.DataFrame({"bucket": bins.astype(str), "y_true": y_true.to_numpy(), "y_pred": y_pred, "trade": y_pred != 0})
    for bucket, g in bucket_df.groupby("bucket", sort=False):
        if g.empty:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "samples": int(len(g)),
                "trade_count": int(g["trade"].sum()),
                "trade_ratio": float(g["trade"].mean()),
                "accuracy": float(accuracy_score(g["y_true"], g["y_pred"])),
                "macro_f1": float(f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0)),
            }
        )
    return rows


def _optimize_decision_thresholds(y_true: pd.Series, probs: np.ndarray, base_threshold: float) -> dict[str, float]:
    best = {
        "no_trade_threshold": float(np.clip(base_threshold, 0.30, 0.85)),
        "trade_threshold": 0.45,
        "side_margin": 0.04,
        "expected_edge_min": 0.02,
    }
    best_score = -1e9

    no_trade_grid = sorted(
        {
            float(np.clip(base_threshold - 0.10, 0.30, 0.85)),
            float(np.clip(base_threshold - 0.05, 0.30, 0.85)),
            float(np.clip(base_threshold, 0.30, 0.85)),
            float(np.clip(base_threshold + 0.05, 0.30, 0.85)),
            float(np.clip(base_threshold + 0.10, 0.30, 0.85)),
        }
    )

    for nt in no_trade_grid:
        for trade_th in [0.38, 0.44, 0.50, 0.56]:
            for margin in [0.00, 0.03, 0.06, 0.09]:
                for edge in [-0.02, 0.00, 0.03, 0.06]:
                    pred = _decode_predictions(probs, nt, trade_th, margin, edge)
                    macro = f1_score(y_true, pred, average="macro", zero_division=0)
                    bal = balanced_accuracy_score(y_true, pred)
                    mcc = matthews_corrcoef(y_true, pred)
                    pred_share = pd.Series(pred).value_counts(normalize=True)
                    collapse_penalty = sum(0.06 for cls in [1, 2] if pred_share.get(cls, 0.0) < 0.06)
                    trade_ratio = float((pred != 0).mean())
                    trade_ratio_penalty = 0.05 if trade_ratio < 0.10 else (0.03 if trade_ratio > 0.85 else 0.0)
                    score = 0.50 * macro + 0.30 * bal + 0.20 * mcc - collapse_penalty - trade_ratio_penalty
                    if score > best_score:
                        best_score = score
                        best = {
                            "no_trade_threshold": nt,
                            "trade_threshold": trade_th,
                            "side_margin": margin,
                            "expected_edge_min": edge,
                        }
    return best


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> tuple[dict, dict[str, float], list[float]]:
    class_weights = _compute_class_weights(y_train)
    candidates: list[dict] = []
    for d in [4, 5, 6]:
        for lr in [0.02, 0.05, 0.08]:
            for l2 in [3, 5, 7]:
                candidates.append(
                    {
                        "depth": d,
                        "learning_rate": lr,
                        "l2_leaf_reg": l2,
                        "iterations": cfg.iterations,
                        "loss_function": "MultiClass",
                        "random_seed": cfg.random_state,
                        "class_weights": class_weights,
                        "verbose": False,
                    }
                )

    scored: list[tuple[float, dict, dict[str, float]]] = []
    for params in candidates[: cfg.tuning_trials]:
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
        val_probs = model.predict_proba(x_val)
        decision_cfg = _optimize_decision_thresholds(y_val, val_probs, cfg.threshold_no_trade)
        pred = _decode_predictions(val_probs, **decision_cfg)
        score = (
            0.50 * f1_score(y_val, pred, average="macro", zero_division=0)
            + 0.30 * balanced_accuracy_score(y_val, pred)
            + 0.20 * matthews_corrcoef(y_val, pred)
        )
        scored.append((score, params, decision_cfg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][2], class_weights


def train_walk_forward(
    df: pd.DataFrame,
    candidate_features: list[str],
    folds: list[FoldIndices],
    cfg: TrainConfig,
    feat_cfg: FeatureConfig,
    logger: logging.Logger,
) -> tuple[CatBoostClassifier, list[FoldResult], pd.DataFrame, list[str]]:
    fold_results: list[FoldResult] = []
    all_predictions: list[pd.DataFrame] = []
    final_model: CatBoostClassifier | None = None
    final_features: list[str] = []

    for fold in folds:
        x_train_full = df.loc[fold.train_idx, candidate_features]
        y_train = df.loc[fold.train_idx, "label"]
        selected = _select_fold_features(x_train_full, y_train, candidate_features, feat_cfg)

        x_train = df.loc[fold.train_idx, selected]
        x_val = df.loc[fold.val_idx, selected]
        y_val = df.loc[fold.val_idx, "label"]
        x_test = df.loc[fold.test_idx, selected]
        y_test = df.loc[fold.test_idx, "label"]

        best_params, decision_cfg, class_weights = tune_hyperparameters(x_train, y_train, x_val, y_val, cfg)
        best_params["class_weights"] = class_weights

        model = CatBoostClassifier(**best_params)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

        train_pred = _decode_predictions(model.predict_proba(x_train), **decision_cfg)
        val_pred = _decode_predictions(model.predict_proba(x_val), **decision_cfg)
        test_probs = model.predict_proba(x_test)
        test_pred = _decode_predictions(test_probs, **decision_cfg)

        actual_distribution = {int(k): float(v) for k, v in y_test.value_counts(normalize=True).to_dict().items()}
        pred_distribution = {int(k): float(v) for k, v in pd.Series(test_pred).value_counts(normalize=True).to_dict().items()}
        conf_bucket_metrics = _confidence_bucket_report(y_test, test_pred, test_probs)

        result = FoldResult(
            fold=fold.fold,
            train_acc=accuracy_score(y_train, train_pred),
            val_acc=accuracy_score(y_val, val_pred),
            test_acc=accuracy_score(y_test, test_pred),
            weighted_f1=f1_score(y_test, test_pred, average="weighted", zero_division=0),
            macro_f1=f1_score(y_test, test_pred, average="macro", zero_division=0),
            balanced_acc=balanced_accuracy_score(y_test, test_pred),
            mcc=matthews_corrcoef(y_test, test_pred),
            overfit_gap=accuracy_score(y_train, train_pred) - accuracy_score(y_val, val_pred),
            confusion=confusion_matrix(y_test, test_pred, labels=[0, 1, 2]),
            report=classification_report(y_test, test_pred, output_dict=True, zero_division=0),
            test_probs=test_probs,
            test_pred=test_pred,
            no_trade_threshold=decision_cfg["no_trade_threshold"],
            trade_threshold=decision_cfg["trade_threshold"],
            side_margin=decision_cfg["side_margin"],
            expected_edge_min=decision_cfg["expected_edge_min"],
            selected_features=selected,
            pred_distribution=pred_distribution,
            actual_distribution=actual_distribution,
            confidence_bucket_metrics=conf_bucket_metrics,
        )
        fold_results.append(result)

        logger.info(
            "Fold %d | test_acc=%.4f macro_f1=%.4f bal_acc=%.4f mcc=%.4f | dec(nt=%.2f,tr=%.2f,margin=%.2f,edge=%.2f) | pred_dist=%s",
            result.fold,
            result.test_acc,
            result.macro_f1,
            result.balanced_acc,
            result.mcc,
            result.no_trade_threshold,
            result.trade_threshold,
            result.side_margin,
            result.expected_edge_min,
            {k: round(v, 3) for k, v in result.pred_distribution.items()},
        )

        fold_pred_df = df.loc[fold.test_idx, ["time", "open", "high", "low", "close", "label"]].copy()
        fold_pred_df["source_index"] = fold.test_idx
        fold_pred_df["pred"] = test_pred.astype(int)
        fold_pred_df["signal"] = test_pred.astype(int)
        fold_pred_df["prob_0"] = test_probs[:, 0]
        fold_pred_df["prob_1"] = test_probs[:, 1]
        fold_pred_df["prob_2"] = test_probs[:, 2]
        fold_pred_df["confidence"] = np.maximum(test_probs[:, 1], test_probs[:, 2])
        fold_pred_df["fold"] = fold.fold
        all_predictions.append(fold_pred_df)

        final_model = model
        final_features = selected

    if final_model is None:
        raise RuntimeError("Training failed: no fold model trained")

    pred_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)
    return final_model, fold_results, pred_df, final_features

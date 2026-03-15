from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import TrainConfig
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
    overfit_gap: float
    confusion: np.ndarray
    report: dict
    test_probs: np.ndarray
    test_pred: np.ndarray
    no_trade_threshold: float
    min_side_prob: float
    side_gap: float


def _compute_class_weights(y_train: pd.Series) -> list[float]:
    classes = [0, 1, 2]
    counts = y_train.value_counts().to_dict()
    total = float(len(y_train))
    weights: list[float] = []
    for cls in classes:
        count = float(counts.get(cls, 1.0))
        weights.append(total / (len(classes) * count))
    return weights


def _decode_predictions(
    probs: np.ndarray,
    no_trade_threshold: float,
    min_side_prob: float,
    side_gap: float,
) -> np.ndarray:
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    p2 = probs[:, 2]

    side = np.where(p1 >= p2, 1, 2)
    side_prob = np.maximum(p1, p2)
    gap = np.abs(p1 - p2)

    no_trade = (p0 >= no_trade_threshold) | (side_prob < min_side_prob) | (gap < side_gap)
    pred = np.where(no_trade, 0, side).astype(np.int32)
    return pred


def _optimize_decision_thresholds(y_true: pd.Series, probs: np.ndarray, base_threshold: float) -> dict[str, float]:
    best = {
        "no_trade_threshold": float(np.clip(base_threshold, 0.30, 0.80)),
        "min_side_prob": 0.38,
        "side_gap": 0.03,
    }
    best_score = -1.0

    no_trade_grid = [
        float(np.clip(base_threshold - 0.10, 0.30, 0.80)),
        float(np.clip(base_threshold - 0.05, 0.30, 0.80)),
        float(np.clip(base_threshold, 0.30, 0.80)),
        float(np.clip(base_threshold + 0.05, 0.30, 0.80)),
        float(np.clip(base_threshold + 0.10, 0.30, 0.80)),
    ]

    for nt in sorted(set(no_trade_grid)):
        for min_side in [0.34, 0.38, 0.42, 0.46]:
            for gap in [0.00, 0.03, 0.06]:
                pred = _decode_predictions(probs, nt, min_side, gap)
                score = f1_score(y_true, pred, average="macro", zero_division=0)
                if score > best_score:
                    best_score = score
                    best = {
                        "no_trade_threshold": nt,
                        "min_side_prob": min_side,
                        "side_gap": gap,
                    }

    return best


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> tuple[dict, dict[str, float], list[float]]:
    candidates: list[dict] = []
    depths = [4, 5, 6]
    learning_rates = [0.02, 0.05, 0.08]
    l2s = [3, 5, 7]
    class_weights = _compute_class_weights(y_train)

    for d in depths:
        for lr in learning_rates:
            for l2 in l2s:
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

    scores: list[tuple[float, dict, dict[str, float]]] = []
    for params in candidates[: cfg.tuning_trials]:
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

        val_probs = model.predict_proba(x_val)
        decision_cfg = _optimize_decision_thresholds(y_val, val_probs, cfg.threshold_no_trade)
        pred = _decode_predictions(
            val_probs,
            decision_cfg["no_trade_threshold"],
            decision_cfg["min_side_prob"],
            decision_cfg["side_gap"],
        )
        score = f1_score(y_val, pred, average="macro", zero_division=0)
        scores.append((score, params, decision_cfg))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1], scores[0][2], class_weights


def train_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[FoldIndices],
    cfg: TrainConfig,
    logger: logging.Logger,
) -> tuple[CatBoostClassifier, list[FoldResult], pd.DataFrame]:
    fold_results: list[FoldResult] = []
    all_predictions: list[pd.DataFrame] = []
    final_model: CatBoostClassifier | None = None

    for fold in folds:
        x_train = df.loc[fold.train_idx, feature_cols]
        y_train = df.loc[fold.train_idx, "label"]
        x_val = df.loc[fold.val_idx, feature_cols]
        y_val = df.loc[fold.val_idx, "label"]
        x_test = df.loc[fold.test_idx, feature_cols]
        y_test = df.loc[fold.test_idx, "label"]

        best_params, decision_cfg, class_weights = tune_hyperparameters(x_train, y_train, x_val, y_val, cfg)
        best_params["class_weights"] = class_weights

        model = CatBoostClassifier(**best_params)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

        train_probs = model.predict_proba(x_train)
        val_probs = model.predict_proba(x_val)
        test_probs = model.predict_proba(x_test)

        train_pred = _decode_predictions(
            train_probs,
            decision_cfg["no_trade_threshold"],
            decision_cfg["min_side_prob"],
            decision_cfg["side_gap"],
        )
        val_pred = _decode_predictions(
            val_probs,
            decision_cfg["no_trade_threshold"],
            decision_cfg["min_side_prob"],
            decision_cfg["side_gap"],
        )
        test_pred = _decode_predictions(
            test_probs,
            decision_cfg["no_trade_threshold"],
            decision_cfg["min_side_prob"],
            decision_cfg["side_gap"],
        )

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        overfit_gap = train_acc - val_acc

        result = FoldResult(
            fold=fold.fold,
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc,
            weighted_f1=f1_score(y_test, test_pred, average="weighted", zero_division=0),
            macro_f1=f1_score(y_test, test_pred, average="macro", zero_division=0),
            balanced_acc=balanced_accuracy_score(y_test, test_pred),
            overfit_gap=overfit_gap,
            confusion=confusion_matrix(y_test, test_pred, labels=[0, 1, 2]),
            report=classification_report(y_test, test_pred, output_dict=True, zero_division=0),
            test_probs=test_probs,
            test_pred=test_pred,
            no_trade_threshold=decision_cfg["no_trade_threshold"],
            min_side_prob=decision_cfg["min_side_prob"],
            side_gap=decision_cfg["side_gap"],
        )
        fold_results.append(result)

        class_mix = pd.Series(test_pred).value_counts(normalize=True).to_dict()
        logger.info(
            "Fold %d | train_acc=%.4f val_acc=%.4f test_acc=%.4f macro_f1=%.4f bal_acc=%.4f overfit_gap=%.4f | thr(no_trade=%.2f,min_side=%.2f,gap=%.2f) | pred_mix=%s",
            fold.fold,
            train_acc,
            val_acc,
            test_acc,
            result.macro_f1,
            result.balanced_acc,
            overfit_gap,
            result.no_trade_threshold,
            result.min_side_prob,
            result.side_gap,
            {int(k): round(float(v), 3) for k, v in class_mix.items()},
        )

        fold_pred_df = df.loc[fold.test_idx, ["time", "close", "label"]].copy()
        fold_pred_df["pred"] = test_pred.astype(int)
        fold_pred_df["signal"] = test_pred.astype(int)
        fold_pred_df["prob_0"] = test_probs[:, 0]
        fold_pred_df["prob_1"] = test_probs[:, 1]
        fold_pred_df["prob_2"] = test_probs[:, 2]
        fold_pred_df["fold"] = fold.fold
        all_predictions.append(fold_pred_df)

        final_model = model

    if final_model is None:
        raise RuntimeError("Training failed: no fold model trained")

    pred_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)
    return final_model, fold_results, pred_df

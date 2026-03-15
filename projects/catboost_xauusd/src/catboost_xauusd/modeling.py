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
    overfit_gap: float
    confusion: np.ndarray
    report: dict
    test_probs: np.ndarray
    test_pred: np.ndarray
    no_trade_threshold: float
    min_side_prob: float
    side_gap: float
    selected_features: list[str]


def _compute_class_weights(y_train: pd.Series) -> list[float]:
    classes = [0, 1, 2]
    counts = y_train.value_counts().to_dict()
    total = float(len(y_train))
    return [total / (len(classes) * float(counts.get(cls, 1.0))) for cls in classes]


def _select_fold_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: list[str],
    feat_cfg: FeatureConfig,
) -> list[str]:
    local = x_train[candidate_features].copy()
    valid = [col for col in candidate_features if local[col].std(ddof=0) > 1e-12]
    if not valid:
        raise ValueError("No valid features left after variance filter")

    mi = mutual_info_classif(local[valid], y_train.to_numpy(), random_state=42)
    ranked = sorted(zip(valid, mi), key=lambda x: x[1], reverse=True)

    corr = local[[f for f, _ in ranked]].corr().abs()
    selected: list[str] = []
    for feature, _ in ranked:
        if len(selected) >= feat_cfg.max_features:
            break
        if all(corr.loc[feature, picked] < feat_cfg.corr_threshold for picked in selected):
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
    min_side_prob: float,
    side_gap: float,
) -> np.ndarray:
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    side = np.where(p1 >= p2, 1, 2)
    side_prob = np.maximum(p1, p2)
    gap = np.abs(p1 - p2)
    no_trade = (p0 >= no_trade_threshold) | (side_prob < min_side_prob) | (gap < side_gap)
    return np.where(no_trade, 0, side).astype(np.int32)


def _optimize_decision_thresholds(y_true: pd.Series, probs: np.ndarray, base_threshold: float) -> dict[str, float]:
    best = {
        "no_trade_threshold": float(np.clip(base_threshold, 0.30, 0.80)),
        "min_side_prob": 0.38,
        "side_gap": 0.03,
    }
    best_score = -1e9

    no_trade_grid = sorted(
        {
            float(np.clip(base_threshold - 0.12, 0.30, 0.80)),
            float(np.clip(base_threshold - 0.06, 0.30, 0.80)),
            float(np.clip(base_threshold, 0.30, 0.80)),
            float(np.clip(base_threshold + 0.06, 0.30, 0.80)),
            float(np.clip(base_threshold + 0.12, 0.30, 0.80)),
        }
    )

    for nt in no_trade_grid:
        for min_side in [0.30, 0.36, 0.42, 0.48]:
            for gap in [0.00, 0.02, 0.04, 0.06]:
                pred = _decode_predictions(probs, nt, min_side, gap)
                macro = f1_score(y_true, pred, average="macro", zero_division=0)
                bal = balanced_accuracy_score(y_true, pred)
                pred_share = pd.Series(pred).value_counts(normalize=True)
                collapse_penalty = 0.0
                for cls in [0, 1, 2]:
                    if pred_share.get(cls, 0.0) < 0.05:
                        collapse_penalty += 0.05
                score = 0.65 * macro + 0.35 * bal - collapse_penalty
                if score > best_score:
                    best_score = score
                    best = {"no_trade_threshold": nt, "min_side_prob": min_side, "side_gap": gap}
    return best


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> tuple[dict, dict[str, float], list[float]]:
    candidates: list[dict] = []
    class_weights = _compute_class_weights(y_train)

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
        pred = _decode_predictions(
            val_probs,
            decision_cfg["no_trade_threshold"],
            decision_cfg["min_side_prob"],
            decision_cfg["side_gap"],
        )
        macro = f1_score(y_val, pred, average="macro", zero_division=0)
        bal = balanced_accuracy_score(y_val, pred)
        score = 0.65 * macro + 0.35 * bal
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

        train_probs = model.predict_proba(x_train)
        val_probs = model.predict_proba(x_val)
        test_probs = model.predict_proba(x_test)

        train_pred = _decode_predictions(train_probs, **decision_cfg)
        val_pred = _decode_predictions(val_probs, **decision_cfg)
        test_pred = _decode_predictions(test_probs, **decision_cfg)

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
            selected_features=selected,
        )
        fold_results.append(result)

        class_mix = pd.Series(test_pred).value_counts(normalize=True).to_dict()
        logger.info(
            "Fold %d | train_acc=%.4f val_acc=%.4f test_acc=%.4f macro_f1=%.4f bal_acc=%.4f overfit_gap=%.4f | thr(no_trade=%.2f,min_side=%.2f,gap=%.2f) | pred_mix=%s | n_feat=%d",
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
            len(selected),
        )

        fold_pred_df = df.loc[fold.test_idx, ["time", "open", "high", "low", "close", "label"]].copy()
        fold_pred_df["pred"] = test_pred.astype(int)
        fold_pred_df["signal"] = test_pred.astype(int)
        fold_pred_df["prob_0"] = test_probs[:, 0]
        fold_pred_df["prob_1"] = test_probs[:, 1]
        fold_pred_df["prob_2"] = test_probs[:, 2]
        fold_pred_df["fold"] = fold.fold
        all_predictions.append(fold_pred_df)

        final_model = model
        final_features = selected

    if final_model is None:
        raise RuntimeError("Training failed: no fold model trained")

    pred_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)
    return final_model, fold_results, pred_df, final_features

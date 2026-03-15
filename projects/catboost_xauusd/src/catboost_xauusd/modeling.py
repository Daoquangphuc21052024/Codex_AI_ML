from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import TrainConfig
from .validation import FoldIndices


@dataclass
class FoldResult:
    fold: int
    train_acc: float
    val_acc: float
    test_acc: float
    weighted_f1: float
    overfit_gap: float
    confusion: np.ndarray
    report: dict
    test_probs: np.ndarray
    test_pred: np.ndarray


def tune_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
) -> dict:
    candidates: list[dict] = []
    depths = [4, 5, 6]
    learning_rates = [0.02, 0.05, 0.08]
    l2s = [3, 5, 7]

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
                        "verbose": False,
                    }
                )

    scores: list[tuple[float, dict]] = []
    for params in candidates[: cfg.tuning_trials]:
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
        pred = model.predict(x_val).ravel()
        score = f1_score(y_val, pred, average="weighted")
        scores.append((score, params))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]


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

        best_params = tune_hyperparameters(x_train, y_train, x_val, y_val, cfg)
        model = CatBoostClassifier(**best_params)
        model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

        train_pred = model.predict(x_train).ravel()
        val_pred = model.predict(x_val).ravel()
        test_pred = model.predict(x_test).ravel()
        test_prob = model.predict_proba(x_test)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        overfit_gap = train_acc - val_acc

        result = FoldResult(
            fold=fold.fold,
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc,
            weighted_f1=f1_score(y_test, test_pred, average="weighted"),
            overfit_gap=overfit_gap,
            confusion=confusion_matrix(y_test, test_pred, labels=[0, 1, 2]),
            report=classification_report(y_test, test_pred, output_dict=True),
            test_probs=test_prob,
            test_pred=test_pred,
        )
        fold_results.append(result)
        logger.info(
            "Fold %d | train_acc=%.4f val_acc=%.4f test_acc=%.4f overfit_gap=%.4f",
            fold.fold,
            train_acc,
            val_acc,
            test_acc,
            overfit_gap,
        )

        fold_pred_df = df.loc[fold.test_idx, ["time", "close", "label"]].copy()
        fold_pred_df["pred"] = test_pred.astype(int)
        fold_pred_df["prob_0"] = test_prob[:, 0]
        fold_pred_df["prob_1"] = test_prob[:, 1]
        fold_pred_df["prob_2"] = test_prob[:, 2]
        fold_pred_df["fold"] = fold.fold
        all_predictions.append(fold_pred_df)

        final_model = model

    if final_model is None:
        raise RuntimeError("Training failed: no fold model trained")

    pred_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)
    return final_model, fold_results, pred_df

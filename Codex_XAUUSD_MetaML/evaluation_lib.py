from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> tuple[dict, np.ndarray, np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    return metrics, y_pred, confusion_matrix(y_true, y_pred)


def optimize_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
    min_positive_rate: float = 0.2,
    max_positive_rate: float = 0.8,
) -> tuple[float, dict]:
    """Threshold tuning on validation set.

    We optimize for macro F1 and constrain positive-rate range to avoid
    class-collapse (e.g. predict-all-sell).
    """
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.35, 0.66, 0.005)]

    best_t = 0.5
    best_score = -1.0
    best: dict = {"f1_macro": -1.0}
    rows = []

    for t in thresholds:
        m, y_pred, _ = classification_metrics(y_true, y_prob, t)
        pos_rate = float(np.mean(y_pred))
        penalty = 0.0
        if pos_rate < min_positive_rate or pos_rate > max_positive_rate:
            penalty = 0.2

        score = m["f1_macro"] - penalty
        row = {"threshold": t, "score": score, "penalty": penalty, **m}
        rows.append(row)

        if score > best_score:
            best_score = score
            best_t = t
            best = row

    return best_t, {"best": best, "all": rows}


def save_classification_reports(y_true: pd.Series, y_prob: np.ndarray, threshold: float, symbol: str, out_dir: str = "reports") -> dict[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _, y_pred, cm = classification_metrics(y_true, y_prob, threshold)

    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    p = f"{out_dir}/{symbol}_confusion_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["confusion_matrix_png"] = p

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="true buy(0)")
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="true sell(1)")
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1)
    ax.set_title("Probability Distribution")
    ax.legend()
    p = f"{out_dir}/{symbol}_probability_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["probability_distribution_png"] = p

    calib = pd.DataFrame({"prob": y_prob, "target": y_true.to_numpy()})
    calib["bin"] = pd.qcut(calib["prob"], q=10, duplicates="drop")
    grp = calib.groupby("bin", observed=True).agg(prob_mean=("prob", "mean"), target_rate=("target", "mean")).dropna()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(grp["prob_mean"], grp["target_rate"], marker="o")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("Calibration Plot")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    p = f"{out_dir}/{symbol}_calibration_plot.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["calibration_png"] = p
    return paths


def save_feature_importance(model, feature_names: list[str], symbol: str, out_dir: str = "reports", top_n: int = 20) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    imp = model.get_feature_importance()
    df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    df.to_csv(f"{out_dir}/{symbol}_feature_importance.csv", index=False)

    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#1f77b4")
    ax.set_title(f"Top {top_n} Feature Importance")
    p = f"{out_dir}/{symbol}_feature_importance_top{top_n}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

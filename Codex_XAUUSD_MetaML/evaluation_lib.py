from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0


def _safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0


def classification_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compatibility helper for legacy search module."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
    }
    return metrics, y_pred, confusion_matrix(y_true, y_pred)


def evaluate_dual_classification(
    y_buy: pd.Series,
    y_sell: pd.Series,
    prob_buy: np.ndarray,
    prob_sell: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
) -> dict:
    pred_buy = (prob_buy >= buy_threshold).astype(int)
    pred_sell = (prob_sell >= sell_threshold).astype(int)

    return {
        "auc_buy": _safe_auc(y_buy, prob_buy),
        "auc_sell": _safe_auc(y_sell, prob_sell),
        "pr_auc_buy": _safe_pr_auc(y_buy, prob_buy),
        "pr_auc_sell": _safe_pr_auc(y_sell, prob_sell),
        "brier_buy": float(brier_score_loss(y_buy, prob_buy)),
        "brier_sell": float(brier_score_loss(y_sell, prob_sell)),
        "confusion_buy": confusion_matrix(y_buy, pred_buy, labels=[0, 1]).tolist(),
        "confusion_sell": confusion_matrix(y_sell, pred_sell, labels=[0, 1]).tolist(),
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
    }


def action_semantic_diagnostics(actions: np.ndarray, y_buy: pd.Series, y_sell: pd.Series) -> dict:
    buy_actions = int(np.sum(actions == 0))
    sell_actions = int(np.sum(actions == 1))
    no_trade_actions = int(np.sum(actions == -1))

    idx_buy = actions == 0
    idx_sell = actions == 1
    return {
        "count_buy_actions": buy_actions,
        "count_sell_actions": sell_actions,
        "count_no_trade_actions": no_trade_actions,
        "actual_profitable_buy_rate_on_buy_actions": float(y_buy.to_numpy()[idx_buy].mean()) if buy_actions else 0.0,
        "actual_profitable_sell_rate_on_sell_actions": float(y_sell.to_numpy()[idx_sell].mean()) if sell_actions else 0.0,
    }


def optimize_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
    min_positive_rate: float = 0.2,
    max_positive_rate: float = 0.8,
) -> tuple[float, dict]:
    # compatibility helper kept for old callers
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.3, 0.8, 0.01)]
    rows = []
    best = {"score": -1e9, "threshold": 0.5}
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        pos_rate = float(pred.mean())
        penalty = 0.2 if pos_rate < min_positive_rate or pos_rate > max_positive_rate else 0.0
        score = _safe_pr_auc(y_true, y_prob) - penalty
        row = {"threshold": t, "score": score, "penalty": penalty, "predicted_positive_rate": pos_rate}
        rows.append(row)
        if score > best["score"]:
            best = row
    return float(best["threshold"]), {"best": best, "all": rows}


def save_dual_classification_reports(
    y_buy: pd.Series,
    y_sell: pd.Series,
    prob_buy: np.ndarray,
    prob_sell: np.ndarray,
    symbol: str,
    out_dir: str = "reports",
) -> dict[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(prob_buy[y_buy == 1], bins=40, alpha=0.6, label="buy positives")
    ax.hist(prob_buy[y_buy == 0], bins=40, alpha=0.5, label="buy negatives")
    ax.set_title("Buy probability distribution")
    ax.legend()
    p = f"{out_dir}/{symbol}_probability_buy_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["prob_buy_png"] = p

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(prob_sell[y_sell == 1], bins=40, alpha=0.6, label="sell positives")
    ax.hist(prob_sell[y_sell == 0], bins=40, alpha=0.5, label="sell negatives")
    ax.set_title("Sell probability distribution")
    ax.legend()
    p = f"{out_dir}/{symbol}_probability_sell_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["prob_sell_png"] = p

    for side, y, prob in [("buy", y_buy, prob_buy), ("sell", y_sell, prob_sell)]:
        calib = pd.DataFrame({"prob": prob, "target": y.to_numpy()})
        calib["bin"] = pd.qcut(calib["prob"], q=10, duplicates="drop")
        grp = calib.groupby("bin", observed=True).agg(prob_mean=("prob", "mean"), target_rate=("target", "mean")).dropna()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(grp["prob_mean"], grp["target_rate"], marker="o")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(f"Calibration {side}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        p = f"{out_dir}/{symbol}_calibration_{side}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[f"calibration_{side}_png"] = p

    return paths


def save_feature_importance(
    model,
    feature_names: list[str],
    symbol: str,
    out_dir: str = "reports",
    top_n: int = 20,
    tag: str = "",
    feature_groups: dict[str, str] | None = None,
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    imp = model.get_feature_importance()
    df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    if feature_groups:
        df["group"] = df["feature"].map(feature_groups).fillna("unknown")
        df.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False).to_csv(
            f"{out_dir}/{symbol}_feature_importance_group{tag}.csv", index=False
        )
    df.to_csv(f"{out_dir}/{symbol}_feature_importance{tag}.csv", index=False)

    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#1f77b4")
    ax.set_title(f"Top {top_n} Feature Importance {tag}")
    p = f"{out_dir}/{symbol}_feature_importance_top{top_n}{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

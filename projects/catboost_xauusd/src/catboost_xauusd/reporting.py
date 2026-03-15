from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .modeling import FoldResult


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_plots(
    labeled_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    fold_results: list[FoldResult],
    backtest_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    reports_dir: str,
) -> None:
    out_dir = Path(reports_dir)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(backtest_df["time"], backtest_df["equity"])
    ax.set_title("PnL Equity Curve")
    _save(fig, out_dir / "pnl.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    wr = backtest_df.groupby("fold")["is_win"].mean()
    wr.plot(kind="bar", ax=ax)
    ax.set_title("Winrate by Fold")
    _save(fig, out_dir / "winrate.png")

    confusion = sum(fr.confusion for fr in fold_results)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix (aggregated)")
    _save(fig, out_dir / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(feature_df[feature_cols].corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    _save(fig, out_dir / "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=feature_importance, x="importance", y="feature", ax=ax)
    ax.set_title("Feature Importance")
    _save(fig, out_dir / "feature_importance.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    labeled_df["label"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Label Distribution")
    _save(fig, out_dir / "label_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    backtest_df["pnl_r"].hist(bins=60, ax=ax)
    ax.set_title("Return Distribution")
    _save(fig, out_dir / "return_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(backtest_df["time"], backtest_df["drawdown"])
    ax.set_title("Drawdown")
    _save(fig, out_dir / "drawdown.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series({f"fold_{fr.fold}": fr.test_acc for fr in fold_results}).plot(kind="bar", ax=ax)
    ax.set_title("Accuracy by Fold")
    _save(fig, out_dir / "accuracy_by_fold.png")

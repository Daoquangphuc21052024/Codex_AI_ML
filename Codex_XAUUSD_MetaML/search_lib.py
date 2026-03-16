from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler

from evaluation_lib import classification_metrics


def _time_split_3way(X, y, train_ratio=0.6, val_ratio=0.2):
    n = len(X)
    i_val = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))
    return X.iloc[:i_val], y.iloc[:i_val], X.iloc[i_val:i_test], y.iloc[i_val:i_test], X.iloc[i_test:], y.iloc[i_test:]


def run_param_search(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float,
    val_ratio: float,
    search_space: dict,
    out_dir: str = "reports",
    max_trials: int = 20,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    grid_keys = ["depth", "learning_rate", "l2_leaf_reg", "iterations"]
    values = [search_space[k] for k in grid_keys]
    combos = list(itertools.product(*values))[:max_trials]

    train_X, train_y, val_X, val_y, _, _ = _time_split_3way(X, y, train_ratio, val_ratio)

    scaler = RobustScaler()
    train_X_s = scaler.fit_transform(train_X)
    val_X_s = scaler.transform(val_X)

    rows = []
    for d, lr, l2, it in combos:
        model = CatBoostClassifier(
            depth=d,
            learning_rate=lr,
            l2_leaf_reg=l2,
            iterations=it,
            random_seed=seed,
            eval_metric="F1",
            verbose=False,
            use_best_model=True,
            early_stopping_rounds=25,
        )
        model.fit(train_X_s, train_y, eval_set=(val_X_s, val_y))
        prob = model.predict_proba(val_X_s)[:, 1]
        metrics, _, _ = classification_metrics(val_y, prob, threshold=0.5)
        rows.append({"depth": d, "learning_rate": lr, "l2_leaf_reg": l2, "iterations": it, **metrics})

    res = pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False)
    res.to_csv(f"{out_dir}/grid_search_results.csv", index=False)
    res.head(1).to_json(f"{out_dir}/grid_search_best.json", orient="records", indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    top = res.head(15).copy()
    top["rank"] = range(1, len(top) + 1)
    ax.bar(top["rank"].astype(str), top["f1"], color="#2ca02c")
    ax.set_title("Grid Search Top F1")
    ax.set_xlabel("Rank")
    ax.set_ylabel("F1")
    fig.savefig(f"{out_dir}/grid_search_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    best = res.iloc[0].to_dict()
    return best, res

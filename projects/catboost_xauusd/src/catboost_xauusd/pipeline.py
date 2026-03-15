from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import load_config
from .exporter import export_artifacts
from .features import engineer_features
from .labeling import create_labels
from .logging_utils import setup_logging
from .modeling import train_walk_forward
from .mt5_connector import MT5Connector
from .preprocess import clean_ohlcv
from .reporting import generate_plots
from .validation import make_walk_forward_folds


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    logger = setup_logging(cfg.paths.logs_dir)

    if cfg.mt5.source.lower() == "mt5":
        connector = MT5Connector(cfg.mt5, logger)
        raw_df = connector.fetch_rates()
    elif cfg.mt5.source.lower() == "csv":
        if not cfg.mt5.csv_path:
            raise ValueError("mt5.csv_path is required when mt5.source=csv")
        raw_df = pd.read_csv(cfg.mt5.csv_path)
        logger.info("Loaded %d bars from CSV source: %s", len(raw_df), cfg.mt5.csv_path)
    else:
        raise ValueError(f"Unsupported mt5.source={cfg.mt5.source}")

    Path(cfg.paths.raw_data).parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(cfg.paths.raw_data, index=False)

    clean_df = clean_ohlcv(raw_df)
    feature_df, feature_cols = engineer_features(clean_df, cfg.features)
    labeled_df, diagnostics = create_labels(feature_df, cfg.labeling)
    folds = make_walk_forward_folds(labeled_df, cfg.train)

    model, fold_results, pred_df = train_walk_forward(labeled_df, feature_cols, folds, cfg.train, logger)
    backtest_df = run_backtest(pred_df, cfg.train.threshold_no_trade)

    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)

    generate_plots(
        labeled_df=labeled_df,
        feature_df=feature_df,
        feature_cols=feature_cols,
        fold_results=fold_results,
        backtest_df=backtest_df,
        feature_importance=fi,
        reports_dir=cfg.paths.reports_dir,
    )

    export_artifacts(model, feature_cols, labeled_df[feature_cols], cfg.paths.artifacts_dir)

    Path(cfg.paths.artifacts_dir).mkdir(parents=True, exist_ok=True)
    fi.to_csv(Path(cfg.paths.artifacts_dir) / "feature_importance.csv", index=False)
    backtest_df.to_csv(Path(cfg.paths.artifacts_dir) / "backtest_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": fr.fold,
                "train_acc": fr.train_acc,
                "val_acc": fr.val_acc,
                "test_acc": fr.test_acc,
                "weighted_f1": fr.weighted_f1,
                "overfit_gap": fr.overfit_gap,
            }
            for fr in fold_results
        ]
    ).to_csv(Path(cfg.paths.artifacts_dir) / "fold_metrics.csv", index=False)

    with (Path(cfg.paths.artifacts_dir) / "label_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label_counts": diagnostics.label_counts,
                "combos_tested": diagnostics.combos_tested,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Completed pipeline with %d selected features: %s", len(feature_cols), feature_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD H1 CatBoost training pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

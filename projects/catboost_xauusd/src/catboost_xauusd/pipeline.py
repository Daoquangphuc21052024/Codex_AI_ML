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
        raw_df = MT5Connector(cfg.mt5, logger).fetch_rates()
    elif cfg.mt5.source.lower() == "csv":
        if not cfg.mt5.csv_path:
            raise ValueError("mt5.csv_path is required when mt5.source=csv")
        raw_df = pd.read_csv(cfg.mt5.csv_path)
        logger.info("Loaded %d rows from CSV source: %s", len(raw_df), cfg.mt5.csv_path)
    else:
        raise ValueError(f"Unsupported mt5.source={cfg.mt5.source}")

    Path(cfg.paths.raw_data).parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(cfg.paths.raw_data, index=False)

    clean_df = clean_ohlcv(raw_df)
    feature_df, candidate_features, feature_families = engineer_features(clean_df, cfg.features)
    labeled_df, diagnostics = create_labels(feature_df, cfg.labeling)

    folds = make_walk_forward_folds(labeled_df, cfg.train)
    for fold in folds:
        dist = labeled_df.loc[fold.test_idx, "label"].value_counts().sort_index().to_dict()
        diagnostics.per_fold_label_distribution[f"fold_{fold.fold}"] = {str(int(k)): int(v) for k, v in dist.items()}

    model, fold_results, pred_df, final_features = train_walk_forward(
        labeled_df,
        candidate_features,
        folds,
        cfg.train,
        cfg.features,
        logger,
    )

    backtest_curve, trade_log, backtest_summary = run_backtest(pred_df, labeled_df, cfg.labeling, cfg.backtest)

    fi = pd.DataFrame(
        {"feature": final_features, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)

    generate_plots(
        labeled_df=labeled_df,
        feature_df=feature_df,
        feature_cols=final_features,
        fold_results=fold_results,
        backtest_df=backtest_curve,
        feature_importance=fi,
        reports_dir=cfg.paths.reports_dir,
    )

    onnx_verification = export_artifacts(model, final_features, labeled_df[final_features], cfg.paths.artifacts_dir)

    out_dir = Path(cfg.paths.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)
    backtest_curve.to_csv(out_dir / "backtest_results.csv", index=False)
    trade_log.to_csv(out_dir / "trade_log.csv", index=False)

    fold_metrics = []
    fold_features = {}
    fold_confusions = {}
    for fr in fold_results:
        fold_metrics.append(
            {
                "fold": fr.fold,
                "train_acc": fr.train_acc,
                "val_acc": fr.val_acc,
                "test_acc": fr.test_acc,
                "weighted_f1": fr.weighted_f1,
                "macro_f1": fr.macro_f1,
                "balanced_acc": fr.balanced_acc,
                "mcc": fr.mcc,
                "overfit_gap": fr.overfit_gap,
                "no_trade_threshold": fr.no_trade_threshold,
                "min_side_prob": fr.min_side_prob,
                "side_gap": fr.side_gap,
                "n_selected_features": len(fr.selected_features),
                "pred_dist": json.dumps(fr.pred_distribution),
                "actual_dist": json.dumps(fr.actual_distribution),
            }
        )
        fold_features[f"fold_{fr.fold}"] = fr.selected_features
        fold_confusions[f"fold_{fr.fold}"] = fr.confusion.tolist()

    pd.DataFrame(fold_metrics).to_csv(out_dir / "fold_metrics.csv", index=False)

    with (out_dir / "label_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics.__dict__, f, ensure_ascii=False, indent=2)

    with (out_dir / "fold_selected_features.json").open("w", encoding="utf-8") as f:
        json.dump(fold_features, f, ensure_ascii=False, indent=2)

    with (out_dir / "candidate_features.json").open("w", encoding="utf-8") as f:
        json.dump(candidate_features, f, ensure_ascii=False, indent=2)

    with (out_dir / "feature_families.json").open("w", encoding="utf-8") as f:
        json.dump(feature_families, f, ensure_ascii=False, indent=2)

    fold_stability: dict[str, int] = {}
    for features in fold_features.values():
        for feat in features:
            fold_stability[feat] = fold_stability.get(feat, 0) + 1

    with (out_dir / "feature_selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_feature_count": len(candidate_features),
                "final_selected_count": len(final_features),
                "max_features_config": cfg.features.max_features,
                "fold_stability": fold_stability,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (out_dir / "fold_confusion_matrices.json").open("w", encoding="utf-8") as f:
        json.dump(fold_confusions, f, ensure_ascii=False, indent=2)

    with (out_dir / "onnx_verification.json").open("w", encoding="utf-8") as f:
        json.dump(onnx_verification, f, ensure_ascii=False, indent=2)

    with (out_dir / "backtest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(backtest_summary, f, ensure_ascii=False, indent=2)

    logger.info("Completed pipeline with %d final selected features: %s", len(final_features), final_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD H1 CatBoost training pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

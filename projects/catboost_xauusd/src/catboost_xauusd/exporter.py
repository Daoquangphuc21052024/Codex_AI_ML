from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from . import __version__


def export_artifacts(
    model: CatBoostClassifier,
    feature_cols: list[str],
    sample_x: pd.DataFrame,
    artifacts_dir: str,
) -> None:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "catboost_xauusd.onnx"
    model.save_model(str(onnx_path), format="onnx")

    schema = {
        "model_name": "catboost_xauusd_h1_multiclass",
        "pipeline_version": __version__,
        "input_name": "features",
        "feature_order": feature_cols,
        "feature_count": len(feature_cols),
        "dtype": "float32",
        "class_mapping": {"0": "no_trade", "1": "buy_tp_first", "2": "sell_tp_first"},
        "example": sample_x[feature_cols].iloc[0].astype(float).to_dict(),
    }
    with (out_dir / "feature_schema.json").open("w", encoding="utf-8") as file:
        json.dump(schema, file, ensure_ascii=False, indent=2)

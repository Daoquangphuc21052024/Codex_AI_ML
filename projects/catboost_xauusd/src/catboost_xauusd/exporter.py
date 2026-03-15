from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from . import __version__


def _verify_onnx_inference(
    model: CatBoostClassifier,
    onnx_path: Path,
    sample_x: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    verification = {
        "onnxruntime_available": False,
        "verified": False,
        "checked_rows": 0,
        "match_ratio": None,
    }
    try:
        import onnxruntime as ort
    except Exception:
        return verification

    verification["onnxruntime_available"] = True
    n = min(32, len(sample_x))
    batch = sample_x[feature_cols].iloc[:n].to_numpy(dtype=np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: batch})

    onnx_pred = None
    for out in outputs:
        arr = np.array(out)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            onnx_pred = arr.astype(int)
            break
        if arr.ndim == 2 and arr.shape[1] >= 2:
            onnx_pred = np.argmax(arr, axis=1).astype(int)
            break
    if onnx_pred is None:
        return verification

    cb_pred = model.predict(sample_x[feature_cols].iloc[:n]).ravel().astype(int)
    match_ratio = float((onnx_pred == cb_pred).mean())

    verification.update(
        {
            "verified": bool(match_ratio >= 0.90),
            "checked_rows": int(n),
            "match_ratio": match_ratio,
        }
    )
    return verification


def _write_schema(path: Path, schema: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(schema, file, ensure_ascii=False, indent=2)


def export_artifacts(
    primary_model: CatBoostClassifier,
    primary_feature_cols: list[str],
    primary_sample_x: pd.DataFrame,
    artifacts_dir: str,
    meta_model: CatBoostClassifier | None = None,
    meta_feature_cols: list[str] | None = None,
    meta_sample_x: pd.DataFrame | None = None,
) -> dict:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_onnx_path = out_dir / "catboost_xauusd.onnx"
    primary_model.save_model(str(primary_onnx_path), format="onnx")

    primary_schema = {
        "model_name": "catboost_xauusd_h1_multiclass",
        "pipeline_version": __version__,
        "input_name": "features",
        "feature_order": primary_feature_cols,
        "feature_count": len(primary_feature_cols),
        "dtype": "float32",
        "class_mapping": {"0": "no_trade", "1": "buy_tp_first", "2": "sell_tp_first"},
        "example": primary_sample_x[primary_feature_cols].iloc[0].astype(float).to_dict(),
    }
    _write_schema(out_dir / "feature_schema.json", primary_schema)

    result = {
        "primary": _verify_onnx_inference(primary_model, primary_onnx_path, primary_sample_x, primary_feature_cols),
        "meta": None,
    }

    if meta_model is not None and meta_feature_cols and meta_sample_x is not None and not meta_sample_x.empty:
        meta_onnx_path = out_dir / "meta_filter.onnx"
        meta_model.save_model(str(meta_onnx_path), format="onnx")
        meta_schema = {
            "model_name": "catboost_xauusd_h1_meta_filter",
            "pipeline_version": __version__,
            "input_name": "features",
            "feature_order": meta_feature_cols,
            "feature_count": len(meta_feature_cols),
            "dtype": "float32",
            "target": "primary_trade_acceptance",
            "positive_class": "accept_primary_trade",
            "example": meta_sample_x[meta_feature_cols].iloc[0].astype(float).to_dict(),
        }
        _write_schema(out_dir / "meta_feature_schema.json", meta_schema)
        result["meta"] = _verify_onnx_inference(meta_model, meta_onnx_path, meta_sample_x, meta_feature_cols)

    return result

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


def _scaler_dict(scaler):
    return {"center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()}


def _validate_onnx_if_possible(onnx_path: str, sample: np.ndarray) -> dict:
    try:
        import onnxruntime as ort
    except Exception:
        return {"onnx_validation": "skipped (onnxruntime not installed)"}

    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        pred = sess.run(None, {inp_name: sample.astype(np.float32)})
        return {"onnx_validation": "ok", "onnx_output_shapes": [list(np.array(x).shape) for x in pred]}
    except Exception as e:
        return {"onnx_validation": f"failed: {e}"}


def export_artifacts(
    symbol: str,
    model_number: int,
    export_path: str,
    model,
    meta_model,
    scaler_main,
    scaler_meta,
    report: dict,
    periods,
    periods_meta,
    feature_names: list[str],
    feature_names_meta: list[str],
    decision_threshold: float,
    sample_main: np.ndarray,
    sample_meta: np.ndarray,
):
    Path(export_path).mkdir(parents=True, exist_ok=True)

    main_onnx = os.path.join(export_path, f"{symbol}_model_{model_number}.onnx")
    meta_onnx = os.path.join(export_path, f"{symbol}_meta_{model_number}.onnx")
    scaler_main_path = os.path.join(export_path, f"{symbol}_scaler_main_{model_number}.pkl")
    scaler_meta_path = os.path.join(export_path, f"{symbol}_scaler_meta_{model_number}.pkl")
    mqh_path = os.path.join(export_path, f"{symbol}_config_{model_number}.mqh")
    report_path = os.path.join(export_path, f"{symbol}_report_{model_number}.json")
    feature_main_path = os.path.join(export_path, f"{symbol}_feature_names_main_{model_number}.json")
    feature_meta_path = os.path.join(export_path, f"{symbol}_feature_names_meta_{model_number}.json")

    model.save_model(main_onnx, format="onnx")
    meta_model.save_model(meta_onnx, format="onnx")
    joblib.dump(scaler_main, scaler_main_path)
    joblib.dump(scaler_meta, scaler_meta_path)

    with open(feature_main_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    with open(feature_meta_path, "w", encoding="utf-8") as f:
        json.dump(feature_names_meta, f, ensure_ascii=False, indent=2)

    mqh = f'''#define MODEL_MAIN_PATH   "{os.path.basename(main_onnx)}"
#define MODEL_META_PATH   "{os.path.basename(meta_onnx)}"
#define N_FEATURES        {report["n_features_main"]}
#define N_FEATURES_META   {report["n_features_meta"]}
#define N_PERIODS         {len(periods)}
#define N_PERIODS_META    {len(periods_meta)}
#define DECISION_THRESHOLD {decision_threshold:.6f}

int Periods[{len(periods)}]     = {{{", ".join(map(str, periods))}}};
int PeriodsMeta[{len(periods_meta)}]  = {{{", ".join(map(str, periods_meta))}}};
'''
    with open(mqh_path, "w", encoding="utf-8") as f:
        f.write(mqh)

    onnx_check_main = _validate_onnx_if_possible(main_onnx, sample_main[:5])
    onnx_check_meta = _validate_onnx_if_possible(meta_onnx, sample_meta[:5])

    report_full = {
        **report,
        "symbol": symbol,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "periods": periods,
        "periods_meta": periods_meta,
        "decision_threshold": decision_threshold,
        "scaler_main": _scaler_dict(scaler_main),
        "scaler_meta": _scaler_dict(scaler_meta),
        "feature_names_main": feature_names,
        "feature_names_meta": feature_names_meta,
        "onnx_main_validation": onnx_check_main,
        "onnx_meta_validation": onnx_check_meta,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_full, f, ensure_ascii=False, indent=2)

    return {
        "main_onnx": main_onnx,
        "meta_onnx": meta_onnx,
        "scaler_main": scaler_main_path,
        "scaler_meta": scaler_meta_path,
        "mqh": mqh_path,
        "report": report_path,
        "feature_main": feature_main_path,
        "feature_meta": feature_meta_path,
    }

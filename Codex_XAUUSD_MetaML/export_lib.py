from __future__ import annotations

import json
import os
from datetime import datetime

import joblib


def _scaler_dict(scaler):
    return {
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def export_artifacts(symbol: str, model_number: int, export_path: str, model, meta_model, scaler_main, scaler_meta, report: dict, periods, periods_meta):
    os.makedirs(export_path, exist_ok=True)

    main_onnx = os.path.join(export_path, f"{symbol}_model_{model_number}.onnx")
    meta_onnx = os.path.join(export_path, f"{symbol}_meta_{model_number}.onnx")
    scaler_main_path = os.path.join(export_path, f"{symbol}_scaler_main_{model_number}.pkl")
    scaler_meta_path = os.path.join(export_path, f"{symbol}_scaler_meta_{model_number}.pkl")
    mqh_path = os.path.join(export_path, f"{symbol}_config_{model_number}.mqh")
    report_path = os.path.join(export_path, f"{symbol}_report_{model_number}.json")

    model.save_model(main_onnx, format="onnx")
    meta_model.save_model(meta_onnx, format="onnx")
    joblib.dump(scaler_main, scaler_main_path)
    joblib.dump(scaler_meta, scaler_meta_path)

    mqh = f'''#define MODEL_MAIN_PATH   "{os.path.basename(main_onnx)}"
#define MODEL_META_PATH   "{os.path.basename(meta_onnx)}"
#define N_FEATURES        {report["n_features_main"]}
#define N_FEATURES_META   {report["n_features_meta"]}
#define N_PERIODS         {len(periods)}
#define N_PERIODS_META    {len(periods_meta)}

int Periods[{len(periods)}]     = {{{", ".join(map(str, periods))}}};
int PeriodsMeta[{len(periods_meta)}]  = {{{", ".join(map(str, periods_meta))}}};
'''
    with open(mqh_path, "w", encoding="utf-8") as f:
        f.write(mqh)

    report_full = {
        **report,
        "symbol": symbol,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "periods": periods,
        "periods_meta": periods_meta,
        "scaler_main": _scaler_dict(scaler_main),
        "scaler_meta": _scaler_dict(scaler_meta),
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
    }

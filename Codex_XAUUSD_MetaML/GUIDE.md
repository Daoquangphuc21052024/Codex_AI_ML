# GUIDE.md — Verified MT5 Deployment for Codex_XAUUSD_MetaML

## 1) Verified audit from Python code (evidence-based)

### Feature vector
- Final inference features are **dynamic** and come from `selected` after train-only feature selection. In `train_pipeline`, `selected` is union of buy/sell selected features and then both models train on `X = X[selected]`. (`trend_following.py`).
- Therefore BUY and SELL use the **same final feature vector order** in runtime.

### Feature order
- Order is exactly `selected` list.
- Export stores this list as `feature_names_main` and `feature_names_meta` JSON (`export_lib.py`), and summary also has `feature_names_used`.
- MT5 must preserve this exact order.

### Feature scope
- Features are generated from H1 OHLC(+volume/spread) in `features_lib.py`.
- No H4/D1 features and no session-time features in current feature implementation.

### Scaler
- Scaler is `RobustScaler` (`trend_following.py::_normalize`).
- It is fit on train split and transformed on val/test.
- Export includes scaler params as:
  - `scaler_main.center` / `scaler_main.scale`
  - `scaler_meta.center` / `scaler_meta.scale`
  from `export_lib.py`.
- In current pipeline, same scaler object is passed for both exported slots in train mode.

### Models
- Two CatBoost binary classifiers are used:
  - `model_buy`
  - `model_sell`
- Exported as ONNX files `*_model_*.onnx` and `*_meta_*.onnx` (`export_lib.py`).
- Runtime probability in Python uses `predict_proba(...)[:, 1]` for both.

### Decision logic
- Decision is delta-edge model in `tester_lib.resolve_actions`:
  - `delta = prob_buy - prob_sell`
  - BUY if `delta > buy_edge_threshold` and `prob_buy >= edge_margin`
  - SELL if `delta < -sell_edge_threshold` and `prob_sell >= edge_margin`
  - else NO_TRADE

### Label/backtest alignment checks
- `trend_following.py::_verify_label_tester_alignment` verifies and writes:
  `reports/label_tester_alignment_check.json`
- Required aligned values include:
  - entry_mode = `next_open`
  - signal_shift = `0`
  - barrier_type = `atr`
  - same_bar_conflict = `sl_first`
  - same TP/SL ATR multipliers and max_hold

### Leakage controls
- Chronological splits only.
- Feature selection is train-only.
- Scaler fit is train-only.

---

## 2) File responsibilities

- `AI_FeaturePipeline.mqh`
  - Holds feature/scaler config and computes scaled input vector.
  - Enforces strict feature-name support (no guessing).

- `AI_ModelInference.mqh`
  - Loads both ONNX models, runs inference, applies delta-edge decision.

- `EA_AI_Example.mq5`
  - Minimal integration harness for closed-bar H1 inference.
  - Default dry-run mode for safe parity testing.

---

## 3) Where to place files in MT5

Copy to terminal data folder:

- `MQL5/Include/AI_FeaturePipeline.mqh`
- `MQL5/Include/AI_ModelInference.mqh`
- `MQL5/Experts/EA_AI_Example.mq5`

Compile `EA_AI_Example.mq5` in MetaEditor.

---

## 4) Required Python-exported artifacts

From `exports/<symbol>_report_<n>.json` and adjacent files:

1. ONNX paths for buy/sell models
2. `feature_names_main` (exact order)
3. `scaler_main.center`
4. `scaler_main.scale`
5. Threshold parameters from training report:
   - `buy_edge_threshold`
   - `sell_edge_threshold`
   - `edge_margin`
6. Alignment params:
   - entry_mode, signal_shift, barrier_type, same_bar_conflict, max_hold

---

## 5) Feature parity procedure

1. Open exported report JSON.
2. Set `FP_SetFeatureCount(n)` to exact selected feature count.
3. For each index i, call:
   `FP_SetFeatureDef(i, feature_name_i, center_i, scale_i)`
4. Ensure names/order exactly match Python.
5. If MT5 logs `Unsupported feature for strict parity`, implement that feature in `FP_ComputeFeature` before live use.

---

## 6) Scaler parity procedure

- Python uses RobustScaler: `x_scaled = (x - center) / scale`.
- Copy exact doubles; do not round aggressively.
- Validate no zero scale (MT5 code already guards very small scale).

---

## 7) Confirm ONNX output interpretation

Python uses class-1 probability (`predict_proba[:,1]`).
In MT5:
- verify `OnnxRun` output layout for your exported model.
- ensure `prob1` maps to class-1.
- if output tensor shape differs, adjust `MI_RunProbClass1` accordingly.

---

## 8) One-bar parity test against Python

For the same symbol/time and same closed bar:

1. Run Python inference and save:
   - raw feature vector
   - scaled vector
   - prob_buy/prob_sell
   - action
2. In MT5 dry-run (`InpDryRunOnly=true`), log same values.
3. Compare tolerance:
   - features/scaled: near-identical (floating tolerance)
   - probabilities: close
   - final action: identical

---

## 9) Deployment checklist

- [ ] Feature count equals Python selected feature count.
- [ ] Feature order equals Python `feature_names_main` exactly.
- [ ] Scaler center/scale copied exactly.
- [ ] Buy/sell ONNX paths correct and models load successfully.
- [ ] MT5 `prob_buy`/`prob_sell` close to Python on same bar.
- [ ] BUY/SELL/NO_TRADE decision matches Python.
- [ ] Alignment settings match (`next_open`, `signal_shift=0`, ATR setup).
- [ ] Dry-run logs stable before enabling trading.

---

## 10) Common failure modes

1. Wrong feature order -> probabilities meaningless.
2. Missing unsupported features in MT5 function map.
3. Scaler mismatch -> saturated probabilities.
4. Wrong ONNX output tensor interpretation.
5. Insufficient bars to compute long-window indicators.
6. Wrong symbol precision/spread assumptions in live execution.

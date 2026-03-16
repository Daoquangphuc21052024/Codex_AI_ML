# MT5 Deployment Guide (Dual-Edge AI: BUY + SELL)

This guide explains how to deploy the exported Python models into MetaTrader 5 using these files:

- `AI_FeaturePipeline.mqh`
- `AI_ModelInference.mqh`
- `EA_AI_Example.mq5`

---

## 1) What each file does

### `AI_FeaturePipeline.mqh`
- Builds the model input feature vector in a **fixed order**.
- Applies RobustScaler transformation: `(x - median) / IQR`.
- Implements decision rule using **delta probability**:
  - `delta = prob_buy - prob_sell`
  - BUY if `delta > buy_edge_threshold` and `prob_buy >= edge_margin`
  - SELL if `delta < -sell_edge_threshold` and `prob_sell >= edge_margin`
  - otherwise NO_TRADE.

### `AI_ModelInference.mqh`
- Loads ONNX buy/sell models.
- Runs inference and returns:
  - `prob_buy`
  - `prob_sell`
  - `delta`

### `EA_AI_Example.mq5`
- Example Expert Advisor integrating feature + inference modules.
- Triggers on new closed H1 bar.
- Supports **safe mode** (`InpSafeInferenceOnly=true`) to log inference without placing trades.

---

## 2) Where to place files in MT5

Copy files to:

- `MQL5/Include/AI_FeaturePipeline.mqh`
- `MQL5/Include/AI_ModelInference.mqh`
- `MQL5/Experts/EA_AI_Example.mq5`

Then open MetaEditor and compile `EA_AI_Example.mq5`.

---

## 3) Where to place ONNX model files

Place ONNX files in a location MT5 can access, then set in EA inputs:

- `InpBuyModelPath`
- `InpSellModelPath`

Typical approach:
- Put ONNX files under terminal data path and use relative path known by your terminal build.
- If path errors occur, print/log full path and verify file exists.

---

## 4) Python-exported values you MUST copy into MQL5

From Python reports/export JSON (and scaler export):

1. **Feature list + exact order**
   - Must match exactly in `FP_BuildRawFeatures()`.
   - Any mismatch invalidates inference.

2. **Scaler medians (`center`)**
   - Paste into `g_scaler_center[]`.

3. **Scaler IQR (`scale`)**
   - Paste into `g_scaler_scale[]`.

4. **Decision thresholds**
   - `InpBuyEdgeThreshold`
   - `InpSellEdgeThreshold`
   - `InpEdgeMargin`

5. **Regime settings (if used in Python)**
   - `InpUseRegimeAdjust`
   - `InpRegimeDelta`

6. **Model paths**
   - `InpBuyModelPath`
   - `InpSellModelPath`

---

## 5) How to compile the EA

1. Open MetaEditor.
2. Open `EA_AI_Example.mq5`.
3. Confirm include files are found in `MQL5/Include`.
4. Press **Compile**.
5. Fix any path/type warnings (especially ONNX input/output layout differences).

---

## 6) Safe inference test (NO real trades)

Before any live/paper trading:

1. Set `InpSafeInferenceOnly = true`.
2. Attach EA to **XAUUSD H1** chart.
3. Watch Experts log for lines like:
   - `prob_buy=... prob_sell=... delta=... action=...`
4. Validate outputs are stable and in [0,1].
5. Confirm action distribution is reasonable (BUY/SELL/NO_TRADE).

Only after validation, set `InpSafeInferenceOnly = false`.

---

## 7) Common failure cases

1. **Wrong feature order**
   - Symptom: probabilities random/flat, unstable actions.
   - Fix: ensure feature index mapping exactly matches Python `feature_names_used`.

2. **Wrong scaler values**
   - Symptom: probabilities saturate near 0 or 1.
   - Fix: copy exact `center`/`scale` arrays from export report.

3. **Wrong ONNX path**
   - Symptom: `OnnxCreate` fails.
   - Fix: verify file exists and path is valid for terminal.

4. **Insufficient bars**
   - Symptom: feature builder fails early.
   - Fix: load more history for H1.

5. **NaN / invalid feature values**
   - Symptom: inference fails or unstable output.
   - Fix: keep divide-by-zero guards and finite checks.

6. **ONNX IO shape mismatch**
   - Symptom: `OnnxRun` fails.
   - Fix: inspect exported ONNX graph input/output names and tensor shapes.

---

## 8) Deployment checklist

- [ ] Feature parity confirmed (same formulas, same order).
- [ ] Scaler parity confirmed (same center/scale).
- [ ] ONNX model paths valid.
- [ ] Probabilities are in valid range and behave reasonably.
- [ ] BUY/SELL/NO_TRADE logic matches Python decision layer.
- [ ] Safe mode logs reviewed over sufficient bars.
- [ ] Spread/slippage/commission assumptions reviewed.
- [ ] Final test on demo account before production.

---

## Notes

- This template is intentionally strict and explicit.
- Replace placeholder arrays and feature block with exact Python-exported config before real deployment.
- Keep the Python and MQL versions synchronized after every retrain/export.

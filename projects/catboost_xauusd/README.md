# CatBoost XAUUSD H1 MT5 Pipeline (v0.8.0)

Production pipeline for XAUUSD H1 using MT5 + CatBoost + ONNX with strict time-series controls.

## What is improved in v0.8.0
- **Label/backtest contract is now single-source**: both training labels and backtest execution use the same deterministic TP/SL profile (median TP/SL), same entry timing, tie-breaker, and horizon.
- **Target-proxy feature risk reduced**: direct TP/adverse/RR proxy features were removed from default pool and replaced by more general setup-quality signals.
- **Decision policy upgraded** from crude 3-threshold decoding to confidence-aware gating with:
  - `no_trade_threshold` (p0 gate)
  - `trade_threshold` (side confidence gate)
  - `side_margin` (buy/sell separation)
  - `expected_edge_min` (side-vs-no-trade edge gate)
- Added diagnostics: confidence bucket metrics and PnL by confidence bucket.

## Class definitions
- `0`: no-trade (low movement, conflict/ambiguous setup, or model gate says skip).
- `1`: buy TP-first expected.
- `2`: sell TP-first expected.

## Trade semantics (label == backtest)
- Entry: controlled by `labeling.entry_mode` (`signal_close` or `next_open`).
- Exit: TP-first / SL-first event check within `horizon_bars`.
- Tie on same bar: controlled by `labeling.tie_breaker`.
- Horizon expiry: exit at last bar close in horizon.
- Execution costs in backtest: spread + slippage + commission.

## Feature families (candidate pool)
1. price return
2. range/volatility
3. candle structure
4. momentum/velocity
5. trend/regime
6. microstructure-lite (from MT5 bar+tick volume)
7. session/time
8. normalized
9. context
10. setup quality (de-proxied)

## Config (schema v0.8.0)
`configs/config.yaml` includes:
- `features.max_features` (<= 60)
- label controls: `entry_mode`, `min_move_atr`, `dominance_threshold`
- backtest execution assumptions: spread/slippage/commission/risk/confidence

## Run
```bash
python run_pipeline.py --config configs/config.yaml
```

## Windows usage
```powershell
cd E:\GPT\catboost_xauusd
.\.venv\Scripts\Activate.ps1
python run_pipeline.py --config configs/config.yaml
```

## Offline smoke test
```bash
python scripts/smoke_test_csv.py
```

## Artifacts
- `artifacts/catboost_xauusd.onnx`
- `artifacts/feature_schema.json`
- `artifacts/onnx_verification.json`
- `artifacts/candidate_features.json`
- `artifacts/feature_families.json`
- `artifacts/fold_selected_features.json`
- `artifacts/feature_selection_summary.json`
- `artifacts/fold_metrics.csv`
- `artifacts/fold_confusion_matrices.json`
- `artifacts/label_diagnostics.json`
- `artifacts/trade_definition.json`
- `artifacts/confidence_bucket_metrics.csv`
- `artifacts/pnl_by_confidence_bucket.csv`
- `artifacts/backtest_results.csv`
- `artifacts/trade_log.csv`
- `artifacts/backtest_summary.json`
- `reports/*.png`

## ONNX safety
Use exact `feature_order` from `feature_schema.json` with float32 in MQL5 inference.

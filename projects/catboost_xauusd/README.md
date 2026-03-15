# CatBoost XAUUSD H1 MT5 Pipeline (v0.7.0)

Production pipeline for XAUUSD H1 using MT5 + CatBoost + ONNX with strict time-series controls.

## What is improved in v0.7.0
- Deep feature expansion to a **professional multi-family candidate space** (up to 60 final features).
- Label engine and backtest engine are explicitly aligned on entry mode, TP/SL, tie, and horizon.
- Backtest includes spread/slippage/commission and produces full trade log + summary.
- Fold-local feature selection and validation-only threshold tuning (anti-leakage).

## Class definitions
- `0`: no-trade (low movement, ambiguous/conflict setup, or model no-trade gate).
- `1`: buy TP-first expected.
- `2`: sell TP-first expected.

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
10. label-aligned setup quality

## Config (schema v0.7.0)
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
- `artifacts/backtest_results.csv`
- `artifacts/trade_log.csv`
- `artifacts/backtest_summary.json`
- `reports/*.png`

## ONNX safety
Use exact `feature_order` from `feature_schema.json` with float32 in MQL5 inference.

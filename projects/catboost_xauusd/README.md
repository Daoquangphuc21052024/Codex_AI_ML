# CatBoost XAUUSD H1 MT5 Pipeline (v0.9.0)

Production pipeline for XAUUSD H1 using MT5 + CatBoost + ONNX with strict time-series controls.

## What is improved in v0.9.0
- Keeps the primary 3-class model and adds a **parallel Meta confirmation model**.
- Meta model role: **false-positive rejection / trade-quality confirmation**.
- Final trade is opened only when:
  1) primary model predicts buy/sell, and
  2) meta acceptance probability is above fold-tuned threshold.
- Pipeline now outputs **before-vs-after filtering** backtest artifacts and summaries.

## Class definitions
- `0`: no-trade (low movement, conflict/ambiguous setup, or model gate says skip).
- `1`: buy TP-first expected.
- `2`: sell TP-first expected.

## Meta model contract
- Primary model target: multiclass (`0/1/2`).
- Meta model target: binary accept/reject for primary non-zero signals, where positive means primary side was correct on validation trade samples.
- Leakage safety:
  - primary model is trained on train fold,
  - meta model is trained on validation trade samples only,
  - meta threshold is tuned on a held-out validation tail,
  - final evaluation is on unseen fold test.

## Trade semantics (label == backtest)
- Entry: controlled by `labeling.entry_mode` (`signal_close` or `next_open`).
- Exit: TP-first / SL-first event check within `horizon_bars`.
- Tie on same bar: controlled by `labeling.tie_breaker`.
- Horizon expiry: exit at last bar close in horizon.
- Execution costs in backtest: spread + slippage + commission.

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
- Primary model: `artifacts/catboost_xauusd.onnx`, `artifacts/feature_schema.json`
- Meta model: `artifacts/meta_filter.onnx`, `artifacts/meta_feature_schema.json` (if enough meta training samples)
- ONNX checks: `artifacts/onnx_verification.json`
- Feature/selection: `candidate_features.json`, `feature_families.json`, `fold_selected_features.json`, `feature_selection_summary.json`, `feature_importance.csv`
- Metrics: `fold_metrics.csv`, `fold_confusion_matrices.json`, `confidence_bucket_metrics.csv`
- Labels and decision contract: `label_diagnostics.json`, `trade_definition.json`
- Backtests:
  - filtered/final: `backtest_results.csv`, `trade_log.csv`, `backtest_summary.json`
  - primary-only baseline: `backtest_results_primary.csv`, `trade_log_primary.csv`, `backtest_summary_primary.json`
  - comparison summary: `meta_filter_summary.json`, `pnl_by_confidence_bucket.csv`

## ONNX safety
Use exact `feature_order` from each schema JSON file with float32 at inference.

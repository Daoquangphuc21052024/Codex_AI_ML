# CatBoost XAUUSD H1 MT5 Pipeline (v0.6.0)

Production-style pipeline for XAUUSD H1 with strict time-series walk-forward, robust 3-class labeling, label-consistent backtest, and ONNX export.

## Class semantics (explicit)
- `0`: no-trade / rejected sample (low move, ambiguous, conflict, or model gate).
- `1`: buy TP-first expected.
- `2`: sell TP-first expected.

## v0.6.0 core upgrades
- Label redesign with explicit ambiguity handling (`dominance_threshold`, `min_move_atr`, conflict rejection).
- Entry assumption standardized (`entry_mode: next_open` or `signal_close`) and reused in both labeling + backtest.
- Backtest rebuilt to TP/SL/horizon first-hit event engine (same semantics as label), with spread/slippage/commission.
- Full trade log + backtest summary artifacts.
- Fold-local feature pruning only from train split (MI + correlation) to avoid leakage.
- Validation-only threshold tuning with class-collapse penalty.
- Per-fold class distribution / MCC / confusion artifacts.

## Project structure
```text
projects/catboost_xauusd/
├── configs/config.yaml
├── requirements.txt
├── run_pipeline.py
├── scripts/smoke_test_csv.py
└── src/catboost_xauusd/
    ├── backtest.py
    ├── config.py
    ├── exporter.py
    ├── features.py
    ├── labeling.py
    ├── logging_utils.py
    ├── modeling.py
    ├── mt5_connector.py
    ├── pipeline.py
    ├── preprocess.py
    ├── reporting.py
    └── validation.py
```

## Config highlights
`configs/config.yaml` (schema v0.6.0):
- `labeling.entry_mode`: `next_open` (recommended).
- `labeling.min_move_atr`: reject low movement noisy bars.
- `labeling.dominance_threshold`: reject weak side dominance.
- `backtest`: spread/slippage/commission/risk/confidence filter.

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

## Smoke test (offline CSV)
```bash
python scripts/smoke_test_csv.py
```

## Artifacts
- `artifacts/catboost_xauusd.onnx`
- `artifacts/feature_schema.json`
- `artifacts/onnx_verification.json`
- `artifacts/fold_metrics.csv`
- `artifacts/fold_confusion_matrices.json`
- `artifacts/fold_selected_features.json`
- `artifacts/label_diagnostics.json`
- `artifacts/backtest_results.csv`
- `artifacts/trade_log.csv`
- `artifacts/backtest_summary.json`
- `reports/*.png`

## ONNX safety
Feature order is pinned in `feature_schema.json`. MT5 inference must use exact same order and float32 dtype.

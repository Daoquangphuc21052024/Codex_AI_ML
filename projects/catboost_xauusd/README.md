# CatBoost XAUUSD H1 MT5 Pipeline (Production-style)

End-to-end Python project (v0.4.0) for training a **multi-class CatBoostClassifier** on **XAUUSD H1** from **MetaTrader5 broker API**, with strict anti-leakage time-series workflow and ONNX export for MT5 inference.

## Objectives

- Classes:
  - `0` = no trade
  - `1` = buy TP first
  - `2` = sell TP first
- Time-series walk-forward validation
- Strict anti-data-leakage processing
- TP/SL arrays supported in labeling
- Max 15 volatility/velocity/intensity features
- Artifacts: ONNX + `feature_schema.json`
- Reports: PnL, winrate, confusion matrix, heatmap, feature importance, label/return distribution, drawdown, accuracy by fold

## Structure

```text
projects/catboost_xauusd/
├── configs/config.yaml
├── requirements.txt
├── run_pipeline.py
├── README.md
├── src/catboost_xauusd/
│   ├── backtest.py
│   ├── config.py
│   ├── exporter.py
│   ├── features.py
│   ├── labeling.py
│   ├── logging_utils.py
│   ├── modeling.py
│   ├── mt5_connector.py
│   ├── pipeline.py
│   ├── preprocess.py
│   └── validation.py
├── data/
├── logs/
├── reports/
└── artifacts/
```

## Installation

```bash
cd /workspace/Codex_AI_ML/projects/catboost_xauusd
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Schema update (v0.4.0 - bắt buộc)

Pipeline hiện chỉ hỗ trợ schema mới theo **time-range time series**.
Các key cũ đã bị chặn cứng (không cho chạy):
- `mt5.bars`
- `train.min_train_size`, `train.val_size`, `train.test_size`

MT5 config mẫu chuẩn mới:

```yaml
mt5:
  source: "mt5"
  symbol: "XAUUSD"
  timeframe: "H1"
  timezone: "UTC"
  start_utc: "2021-01-01T00:00:00Z"
  end_utc: null
  login: null
  password: null
  server: null
  csv_path: null
```


## v0.4.0 improvements from live test feedback

- Added **class-weighted CatBoost** per fold to reduce class collapse.
- Added **validation-based threshold optimization** for final class decision (`no_trade_threshold`, `min_side_prob`, `side_gap`) to avoid always predicting one side.
- Added robust metrics per fold: `macro_f1`, `balanced_acc`, plus decision thresholds stored in `fold_metrics.csv`.
- Silenced sklearn undefined precision warnings correctly via `zero_division=0` in reports/F1 computation.

## Configuration

Edit `configs/config.yaml`:

- `mt5`: source (`mt5` or `csv`), symbol/timeframe + `start_utc/end_utc` time range (không fix cứng số bars), optional login credentials, and `csv_path` fallback
- `labeling`: `horizon_bars`, `tp_points[]`, `sl_points[]`, tie-breaker when same bar hits TP/SL
- `features`: rolling windows, max features, correlation threshold
- `train`: walk-forward theo thời gian (`min_train_days`, `val_days`, `test_days`, `step_days`), random seed, no-trade threshold
- `paths`: output locations


### Smoke test (không cần MT5)

```bash
python scripts/smoke_test_csv.py
```

Script sẽ tạo dữ liệu synthetic OHLCV, chạy full pipeline, và assert các artifact quan trọng (ONNX/schema/report) đã được sinh ra.

## Run

```bash
python run_pipeline.py --config configs/config.yaml
```

## Anti-leakage design

1. Features are only based on historical bars up to current index (rolling/pct_change/shift).
2. Labels use only **future horizon bars** and are truncated to avoid incomplete horizons.
3. Walk-forward split preserves chronology by **calendar time windows**: `train(min_train_days) -> val(val_days) -> test(test_days)` with rolling `step_days`.
4. Hyperparameter tuning uses train/val of each fold only.
5. Backtest evaluates only fold test predictions.

## Labeling logic

For each entry bar and each `(tp, sl)` from cartesian product of arrays:

- Buy TP: `high >= entry + tp`
- Buy SL: `low <= entry - sl`
- Sell TP: `low <= entry - tp`
- Sell SL: `high >= entry + sl`

Rules:
- if only buy reaches TP-first across configs => label `1`
- if only sell reaches TP-first across configs => label `2`
- else => label `0`
- same-bar TP+SL conflict handled by `tie_breaker` (`tp_priority` or `sl_priority`)

## Outputs

### Artifacts

- `artifacts/catboost_xauusd.onnx`
- `artifacts/feature_schema.json`
- `artifacts/feature_importance.csv`
- `artifacts/fold_metrics.csv`
- `artifacts/backtest_results.csv`
- `artifacts/label_diagnostics.json`

### Reports (PNG)

- `reports/pnl.png`
- `reports/winrate.png`
- `reports/confusion_matrix.png`
- `reports/correlation_heatmap.png`
- `reports/feature_importance.png`
- `reports/label_distribution.png`
- `reports/return_distribution.png`
- `reports/drawdown.png`
- `reports/accuracy_by_fold.png`

## MT5 inference integration note

`feature_schema.json` stores strict `feature_order` used by training and ONNX export. In MT5 inference code, compute and feed features in **exact same order** and dtype (`float32`) before ONNX model call.

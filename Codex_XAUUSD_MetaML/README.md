# Codex_XAUUSD_MetaML

Pipeline ML trading cho XAUUSD H1 với trọng tâm anti-leak, backtest, reporting và export deploy MQL5.

## Modules
- `download_mt5_data.py`: tải OHLCV từ MT5
- `data_lib.py`: đọc/chuẩn hóa dữ liệu giá, xử lý duplicate/missing, split visualization
- `features_lib.py`: feature engineering causal (main + meta)
- `labeling_lib.py`: triple-barrier labeling + label quality report
- `search_lib.py`: time-series grid search (không leak)
- `evaluation_lib.py`: metrics ML + confusion/prob/calibration + feature importance
- `tester_lib.py`: backtest có trade log, equity/drawdown/rolling stats
- `export_lib.py`: export ONNX/PKL/MQH/JSON + validate ONNX (nếu có onnxruntime)
- `trend_following.py`: entry point train/search end-to-end

## Quick start
```bash
cd Codex_XAUUSD_MetaML
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install --upgrade pip
pip install MetaTrader5 catboost scikit-learn pandas numpy numba scipy matplotlib joblib
```

## Run
```bash
# Download data from MT5
python download_mt5_data.py

# Train end-to-end using files/XAUUSD_H1.csv
python trend_following.py --mode train

# Run with hyper-parameter search
python trend_following.py --mode search

# Dev smoke run without MT5 data
python trend_following.py --mode train --synthetic
```

## Output
- `reports/*.png`, `reports/*.json`, `reports/*.csv`
- `exports/*.onnx`, `exports/*.pkl`, `exports/*.mqh`, `exports/*.json`

## Hướng dẫn chi tiết
Xem: `HUONG_DAN_CHAY_PIPELINE.md`

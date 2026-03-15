# Codex_XAUUSD_MetaML

Pipeline ML trading cho XAUUSD H1:
- Download data trực tiếp từ MT5 (`download_mt5_data.py`)
- Feature engineering causal + anti-leak
- CatBoost main model + meta model filter
- Walk-forward validation trên test set
- Export ONNX + scaler + config `.mqh`

## Cấu trúc

- `download_mt5_data.py`
- `labeling_lib.py`
- `tester_lib.py`
- `export_lib.py`
- `trend_following.py`
- `files/`
- `reports/`
- `exports/`

## Cài dependencies

```bash
pip install MetaTrader5 catboost scikit-learn pandas numpy numba scipy matplotlib joblib
```

## Chạy

```bash
cd Codex_XAUUSD_MetaML
python download_mt5_data.py
python trend_following.py
```

CSV đầu vào mặc định: `files/XAUUSD_H1.csv`.


## Hướng dẫn chi tiết

Xem file: `HUONG_DAN_CHAY_PIPELINE.md`

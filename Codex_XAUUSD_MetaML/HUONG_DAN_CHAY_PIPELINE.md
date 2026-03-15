# Hướng dẫn chạy pipeline `Codex_XAUUSD_MetaML`

Tài liệu này hướng dẫn từ **thiết lập môi trường ảo** đến **chạy full pipeline**:

1. Tải dữ liệu từ MT5
2. Train main model + meta model
3. Backtest + xuất báo cáo
4. Export ONNX + scaler + config cho MT5 EA

---

## 1) Chuẩn bị trước khi chạy

- Hệ điều hành: Windows (khuyến nghị khi dùng MetaTrader5 terminal)
- Đã cài MetaTrader 5 và đăng nhập tài khoản (demo hoặc real)
- Python 3.10+ (khuyên dùng 3.10/3.11)

> Lưu ý: package `MetaTrader5` cần MT5 terminal hoạt động cùng máy.

---

## 2) Tạo và kích hoạt virtual environment

### Windows (PowerShell)

```powershell
cd /path/to/Codex_AI_ML/Codex_XAUUSD_MetaML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu bị chặn execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS (bash/zsh)

```bash
cd /path/to/Codex_AI_ML/Codex_XAUUSD_MetaML
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3) Cài dependencies

```bash
pip install --upgrade pip
pip install MetaTrader5 catboost scikit-learn pandas numpy numba scipy matplotlib joblib
```

Kiểm tra nhanh:

```bash
python -c "import MetaTrader5, catboost, sklearn, pandas, numpy; print('OK')"
```

---

## 4) Thiết lập dữ liệu đầu vào

Pipeline mặc định đọc file:

- `files/XAUUSD_H1.csv`

Bạn có 2 cách:

### Cách A — Download trực tiếp từ MT5

```bash
python download_mt5_data.py
```

Script sẽ:
- Kết nối MT5
- `symbol_select()` cho `XAUUSD`
- Retry tải dữ liệu
- Lưu file tab-separated vào `files/`

### Cách B — Copy CSV có sẵn vào thư mục `files/`

Đặt tên file đúng:

```text
files/XAUUSD_H1.csv
```

Hỗ trợ cả format kiểu MT5 (`<DATE> <TIME> ...`) và format thường (`time,open,high,low,close,volume`).

---

## 5) Chỉnh cấu hình pipeline (nếu cần)

Mở file `trend_following.py` và sửa trong `hyper_params`:

- `symbol` (mặc định `XAUUSD_H1`)
- `train_ratio`, `val_ratio`, `wf_splits`
- `stop_loss`, `take_profit`, `markup`
- `periods`, `periods_meta`
- `export_path` (thư mục xuất artifacts)

Ví dụ đặt đường dẫn export vào MT5 Include:

```bash
export MT5_INCLUDE_PATH="/path/to/MT5/MQL5/Include/TrendFollowing"
```

(Windows PowerShell: `$env:MT5_INCLUDE_PATH="C:\...\TrendFollowing"`)

---

## 6) Chạy full pipeline

```bash
python trend_following.py
```

Pipeline sẽ chạy lần lượt:

1. Load giá (`get_prices`)
2. Tạo feature causal (`get_features`)
3. Labeling causal (`create_labels`)
4. Clustering regime (KMeans fit train-only)
5. Time split train/val/test
6. Normalize bằng RobustScaler (fit train-only)
7. Train CatBoost main + meta
8. Walk-forward validation trên test set
9. Backtest + lưu ảnh equity curve
10. Export ONNX/PKL/MQH/JSON

---

## 7) Kết quả đầu ra

### Báo cáo backtest

- Thư mục: `reports/`
- Ví dụ: `reports/XAUUSD_H1_best_model_YYYYMMDD_HHMM.png`

### Artifacts deploy MT5

- Thư mục: `exports/` (hoặc `MT5_INCLUDE_PATH` nếu đã set)
- Bao gồm:
  - `XAUUSD_H1_model_0.onnx`
  - `XAUUSD_H1_meta_0.onnx`
  - `XAUUSD_H1_scaler_main_0.pkl`
  - `XAUUSD_H1_scaler_meta_0.pkl`
  - `XAUUSD_H1_config_0.mqh`
  - `XAUUSD_H1_report_0.json`

---

## 8) Troubleshooting nhanh

### Lỗi không kết nối MT5

- Mở MT5 terminal trước
- Đăng nhập account
- Kiểm tra symbol `XAUUSD` có trong Market Watch
- Chạy lại `python download_mt5_data.py`

### Lỗi thiếu package

```bash
pip install -U MetaTrader5 catboost scikit-learn pandas numpy matplotlib joblib
```

### Lỗi không tìm thấy CSV

Đảm bảo tồn tại:

```text
Codex_XAUUSD_MetaML/files/XAUUSD_H1.csv
```

### Muốn chạy nhanh để test logic

- Giảm `iterations` của CatBoost trong `trend_following.py`
- Giảm số `periods`
- Giảm khoảng dữ liệu đầu vào

---

## 9) Lệnh chạy đầy đủ (copy nhanh)

### Linux/macOS

```bash
cd /path/to/Codex_AI_ML/Codex_XAUUSD_MetaML
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install MetaTrader5 catboost scikit-learn pandas numpy numba scipy matplotlib joblib
python download_mt5_data.py
python trend_following.py
```

### Windows PowerShell

```powershell
cd C:\path\to\Codex_AI_ML\Codex_XAUUSD_MetaML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install MetaTrader5 catboost scikit-learn pandas numpy numba scipy matplotlib joblib
python download_mt5_data.py
python trend_following.py
```

Chúc bạn chạy pipeline thuận lợi 🚀

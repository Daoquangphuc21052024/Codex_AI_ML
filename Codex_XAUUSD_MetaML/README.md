# Codex_XAUUSD_MetaML (Latest)

Pipeline ML Trading cho **XAUUSD H1** theo kiến trúc dual-edge:
- `model_buy` dự đoán xác suất BUY profitable
- `model_sell` dự đoán xác suất SELL profitable
- quyết định cuối cùng dùng **delta-edge** (`delta = prob_buy - prob_sell`)

Tài liệu này được cập nhật để người dùng có thể tự:
1) chạy pipeline end-to-end,
2) trích xuất artifacts,
3) import vào MT5,
4) build hàm MQL5 để đồng bộ inference giữa Python ↔ ONNX ↔ MT5.

---

## 1) Mục tiêu và phạm vi

Repository này cung cấp đầy đủ chu trình:
- Tải dữ liệu từ MT5 (`download_mt5_data.py`)
- Chuẩn hóa dữ liệu và tạo features causal
- Labeling dual-edge BUY/SELL (ATR barrier)
- Train CatBoost 2 model (BUY + SELL)
- Threshold search theo trading objective
- Backtest first-touch (OHLC) + báo cáo
- Walk-forward validation
- Export ONNX + scaler + metadata để deploy MT5

> **Lưu ý quan trọng**: tập feature cuối cùng là **dynamic** theo mỗi lần train (feature selection train-only), do đó MT5 phải đọc đúng feature list export của đúng run.

---

## 2) Cấu trúc project

```text
Codex_XAUUSD_MetaML/
├── trend_following.py           # entrypoint: train/search/walkforward
├── download_mt5_data.py         # tải dữ liệu từ MetaTrader5
├── data_lib.py                  # load/normalize/synthetic/split visualization
├── features_lib.py              # feature engineering + feature selection
├── labeling_lib.py              # dual-edge labeling + diagnostics
├── search_lib.py                # param search time-series-safe
├── evaluation_lib.py            # ML metrics, calibration, feature importance
├── tester_lib.py                # backtest engine + reports
├── export_lib.py                # export ONNX/PKL/MQH/JSON
│
├── AI_FeaturePipeline.mqh       # MQL5: feature+scaler parity scaffold
├── AI_ModelInference.mqh        # MQL5: ONNX inference + decision logic
├── EA_AI_Example.mq5            # MQL5 EA sample (dry-run safe)
├── GUIDE.md                     # MT5 deployment/parity checklist
├── HUONG_DAN_CHAY_PIPELINE.md   # hướng dẫn chi tiết (VN)
│
├── files/                       # input CSV data
├── reports/                     # png/json/csv reports
└── exports/                     # onnx/pkl/mqh/json artifacts
```

---

## 3) Chuẩn bị môi trường

### 3.1 Python

```bash
cd Codex_XAUUSD_MetaML
python -m venv .venv
```

Kích hoạt môi trường ảo:
- Linux/macOS:
```bash
source .venv/bin/activate
```
- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Cài dependencies:

```bash
pip install --upgrade pip
pip install MetaTrader5 catboost scikit-learn pandas numpy numba scipy matplotlib joblib
```

### 3.2 MetaTrader 5

- Cài MT5 terminal.
- Đăng nhập account có quyền load history symbol `XAUUSD`.
- Mở chart `XAUUSD` H1 để terminal cache dữ liệu.

---

## 4) Cách chạy pipeline

## 4.1 Tải dữ liệu từ MT5

```bash
python download_mt5_data.py
```

Kết quả kỳ vọng:
- File giá tại `files/XAUUSD_H1.csv`.

## 4.2 Train chuẩn

```bash
python trend_following.py --mode train
```

## 4.3 Train + hyperparameter search

```bash
python trend_following.py --mode search
```

## 4.4 Walk-forward validation

```bash
python trend_following.py --mode walkforward
```

Có thể điều chỉnh tham số WF:

```bash
python trend_following.py --mode walkforward \
  --wf-train-years 8 \
  --wf-val-years 2 \
  --wf-test-years 2 \
  --wf-step-years 1
```

## 4.5 Smoke run không cần data MT5

```bash
python trend_following.py --mode train --synthetic
```

---

## 5) Đầu ra sau khi train/export

### 5.1 Reports

Sinh trong `reports/`:
- Classification metrics (BUY/SELL)
- Backtest metrics (train/val/test)
- Equity/drawdown/rolling charts
- Feature importance
- Threshold candidates
- Label diagnostics
- Alignment diagnostics

### 5.2 Export artifacts

Sinh trong `exports/`:
- `*_model_*.onnx` (BUY model)
- `*_meta_*.onnx` (SELL model)
- `*_scaler_main_*.pkl`
- `*_scaler_meta_*.pkl`
- `*_feature_names_main_*.json`
- `*_feature_names_meta_*.json`
- `*_config_*.mqh`
- `*_report_*.json` (quan trọng nhất cho deploy parity)

---

## 6) Mapping Python -> MT5 (bắt buộc đúng)

Để đồng bộ Python và MT5, bắt buộc copy đúng từ `exports/*_report_*.json`:

1. `feature_names_main` (thứ tự feature)
2. `scaler_main.center`
3. `scaler_main.scale`
4. `buy_edge_threshold`
5. `sell_edge_threshold`
6. `edge_margin`
7. alignment params (`entry_mode`, `signal_shift`, `barrier_type`, `same_bar_conflict`, `max_hold`)
8. model file names/path

> Nếu sai thứ tự feature hoặc sai scaler => output ONNX trong MT5 sẽ lệch mạnh.

---

## 7) Triển khai MT5

Làm theo `GUIDE.md` (bắt buộc đọc).

### 7.1 Copy file MQL5

- `AI_FeaturePipeline.mqh` -> `MQL5/Include/`
- `AI_ModelInference.mqh` -> `MQL5/Include/`
- `EA_AI_Example.mq5` -> `MQL5/Experts/`

### 7.2 Copy model ONNX

Copy 2 file ONNX export vào path MT5 có thể đọc và set path đúng trong EA inputs.

### 7.3 Cập nhật config trong EA

Trong `EA_AI_Example.mq5`:
- set model paths
- set feature count + từng feature definition
- set scaler center/scale
- set thresholds + edge_margin

### 7.4 Compile và test an toàn

- Compile `EA_AI_Example.mq5` trong MetaEditor.
- Chạy trước với chế độ **dry-run/no-trade** để chỉ log inference.
- So sánh `prob_buy/prob_sell/action` giữa MT5 và Python trên cùng bar.

---

## 8) Checklist parity trước khi bật trade

- [ ] Feature count bằng đúng Python selected feature count
- [ ] Feature order khớp 100% `feature_names_main`
- [ ] Scaler center/scale khớp 100%
- [ ] ONNX path đúng
- [ ] `prob_buy/prob_sell` MT5 gần Python trên cùng bar
- [ ] Action BUY/SELL/NO_TRADE khớp Python
- [ ] Entry timing khớp (closed bar -> next open)

---

## 9) Lỗi thường gặp

1. `FileNotFoundError: files/XAUUSD_H1.csv`
   - Chưa chạy `download_mt5_data.py` hoặc đặt sai file name.

2. `ModuleNotFoundError`
   - Chưa activate venv/chưa cài package.

3. MT5 ONNX chạy nhưng output không hợp lý
   - Sai feature order / scaler / output tensor mapping.

4. Kết quả MT5 khác Python
   - Sai bar timing (dùng bar đang chạy thay vì bar đóng)
   - Sai threshold params
   - Sai symbol digits/spread assumptions

---

## 10) Tài liệu nên đọc tiếp

- `GUIDE.md`: triển khai MT5 + checklist parity chi tiết
- `HUONG_DAN_CHAY_PIPELINE.md`: hướng dẫn VN từng bước đầy đủ


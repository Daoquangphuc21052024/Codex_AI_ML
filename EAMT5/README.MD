# ICT SMC EA V3 - Hướng dẫn cài thông số input tối ưu

Tài liệu này hướng dẫn cách cấu hình **input tối ưu thực chiến** cho file:

- `EAMT5/ICT_SMC_EA_V3_step1.mq5`

> Lưu ý quan trọng:
> - Không có một bộ tham số “tối ưu tuyệt đối” cho mọi cặp, mọi broker, mọi giai đoạn thị trường.
> - Bộ dưới đây là **điểm khởi đầu đã cân bằng** (stability-first), sau đó bạn nên tối ưu thêm bằng backtest/forward test.

---

## 1) Bộ thông số khuyến nghị mặc định (Balanced / an toàn)

Áp dụng tốt để chạy ổn định ban đầu với major pairs (EURUSD, GBPUSD, XAUUSD cần chỉnh spread riêng).

### General
- `InpMagicNumber = 202501`
- `InpComment = "ICT_SMC"`
- `InpAllowBuy = true`
- `InpAllowSell = true`
- `InpOneTradePerSymbol = true`
- `InpDebugMode = true` (khi chạy thật lâu dài có thể tắt để nhẹ log)

### Timeframes
- `InpSignalTF = PERIOD_M15`
- `InpBiasTF = PERIOD_H1`

### Swing & Structure
- `InpSwingLeft = 3`
- `InpSwingRight = 3`
- `InpEqualTolerancePoints = 5`
- `InpMaxSwingsToKeep = 40`
- `InpMSSLookbackBars = 8`
- `InpSetupExpiryBars = 10`
- `InpSweepExpiryBars = 4`
- `InpDisplacementMaxBars = 3`

### Displacement
- `InpATRPeriod = 14`
- `InpDisplacementATRMult = 1.3`
- `InpMinBodyRangeRatio = 0.60`
- `InpImpulseMinATR = 1.8`

### FVG & OTE
- `InpUseFVG = true`
- `InpUseOTE = false`  
- `InpOTELevel1 = 0.62`
- `InpOTELevel2 = 0.79`
- `InpEntryMode = ENTRY_MID_FVG`
- `InpMinFVGSizePoints = 5`
- `InpAllowPending = true`
- `InpPendingExpiryBars = 6`

### Session
- `InpUseSessionFilter = true`
- `InpTradeSession = SESSION_BOTH`
- `InpLondonStart = 8`
- `InpLondonEnd = 12`
- `InpNewYorkStart = 13`
- `InpNewYorkEnd = 18`

### Risk Management
- `InpRiskMode = RISK_PERCENT`
- `InpFixedLot = 0.01`
- `InpRiskPercent = 1.0`  
- `InpSLBufferPoints = 10`
- `InpRR_TP1 = 2.0`
- `InpRR_TP2 = 4.0`
- `InpUsePartialClose = true`
- `InpPartialPercent = 50.0`
- `InpMoveToBEAfterTP1 = true`
- `InpBESafetyPoints = 2`
- `InpUseTrailing = false`
- `InpTrailingMode = TRAIL_ATR`
- `InpTrailingATRMult = 1.5`

### Filters
- `InpMaxSpreadPoints = 30` *(với vàng hoặc broker spread cao, tăng lên 40–80 tùy thực tế)*
- `InpUsePremiumDiscountFilter = false`

### Visualization
- Bật toàn bộ khi debug/đánh giá setup.
- `InpShowStateComment = true`: hiển thị comment trạng thái state-machine trực tiếp trên chart (BUY/SELL state, bias, timeout/filter, mode entry).
- Khi chạy VPS/live, có thể tắt bớt để giảm tải chart.

---

## 2) Bộ thông số theo phong cách giao dịch

## 2.1) Conservative (ít lệnh, chất lượng cao)
- `InpDisplacementATRMult = 1.5`
- `InpMinBodyRangeRatio = 0.65`
- `InpMinFVGSizePoints = 8`
- `InpSweepExpiryBars = 3`
- `InpSetupExpiryBars = 8`
- `InpRiskPercent = 0.5`
- `InpUseOTE = true`
- `InpEntryMode = ENTRY_MID_FVG`

Phù hợp khi muốn lọc nhiễu mạnh, giảm drawdown.

## 2.2) Balanced (khuyến nghị)
Giữ đúng bộ mặc định mục (1).

## 2.3) Aggressive (nhiều cơ hội hơn)
- `InpDisplacementATRMult = 1.15`
- `InpMinBodyRangeRatio = 0.55`
- `InpMinFVGSizePoints = 3`
- `InpSweepExpiryBars = 5`
- `InpSetupExpiryBars = 12`
- `InpRiskPercent = 1.0` (không nên vượt 1.5%/lệnh)
- `InpUseOTE = false`
- `InpEntryMode = ENTRY_NEAR_FVG`

Phù hợp giai đoạn thị trường có trend rõ, nhưng cần giám sát DD kỹ hơn.

---

## 3) Thiết lập nhanh theo sản phẩm

### EURUSD / GBPUSD (5-digit)
- Bắt đầu với bộ Balanced.
- `InpMaxSpreadPoints`: 20–35.
- Nếu thị trường nhiễu: tăng `InpDisplacementATRMult` lên 1.4.

### XAUUSD
- Giữ cấu trúc như Balanced nhưng thường cần:
  - `InpMaxSpreadPoints`: 40–120 (tùy broker/time)
  - `InpSLBufferPoints`: 20–60
  - `InpMinFVGSizePoints`: 10–30
- Nên backtest riêng theo session London/NY.

### Chỉ số (US30/NAS100...)
- Tăng `InpSLBufferPoints`, `InpMinFVGSizePoints`, `InpMaxSpreadPoints` theo đặc tính sản phẩm.
- Ưu tiên `InpUseSessionFilter = true`.

---

## 4) Quy trình tối ưu chuẩn (khuyên dùng)

1. **Bước 1 - Baseline:** chạy Balanced trong 6–12 tháng dữ liệu.
2. **Bước 2 - Tối ưu nhẹ:** chỉ tối ưu nhóm:
   - `InpDisplacementATRMult`
   - `InpMinBodyRangeRatio`
   - `InpMinFVGSizePoints`
   - `InpSweepExpiryBars`
   - `InpSetupExpiryBars`
3. **Bước 3 - Kiểm tra out-of-sample:** chia dữ liệu 70/30 hoặc theo năm.
4. **Bước 4 - Forward test:** chạy demo/live nhỏ ít nhất 2–4 tuần.
5. **Bước 5 - Khóa rủi ro:** giữ `InpRiskPercent <= 1.0%` để ổn định dài hạn.

---

## 5) Mẫu preset đề xuất để copy nhanh

### Preset A - Safe Start
- Giữ bộ Balanced.
- Chỉ đổi: `InpRiskPercent = 0.5`.

### Preset B - London Focus
- `InpTradeSession = SESSION_LONDON`
- `InpDisplacementATRMult = 1.35`
- `InpMinFVGSizePoints = 6`

### Preset C - NewYork Momentum
- `InpTradeSession = SESSION_NEWYORK`
- `InpEntryMode = ENTRY_NEAR_FVG`
- `InpDisplacementATRMult = 1.2`
- `InpMinBodyRangeRatio = 0.58`

---

## 6) Cảnh báo quản trị rủi ro

- Không tăng lot theo cảm tính sau chuỗi thua.
- Tránh chạy đồng thời nhiều symbol tương quan cao nếu cùng logic (dễ cộng dồn rủi ro).
- Nếu drawdown vượt ngưỡng kế hoạch (ví dụ 8–12%), giảm `InpRiskPercent` ngay.
- Luôn kiểm tra lại timezone server để session filter hoạt động đúng giờ.

---

## 7) Ghi chú về “tối ưu nhất”

“Tối ưu nhất” luôn phụ thuộc:
- broker (spread, execution, stop/freeze level),
- sản phẩm giao dịch,
- session,
- giai đoạn thị trường.

Vì vậy, tài liệu này cung cấp **bộ tối ưu thực dụng, an toàn để bắt đầu**; bạn nên tinh chỉnh theo dữ liệu thực tế của chính tài khoản/broker bạn.

# ICT_SMC_EA_V3_step1.mq5 — Tài liệu cơ chế hoạt động (Instruction)

Tài liệu này mô tả **rõ luồng hoạt động**, **từng module**, và **logic xử lý** của EA:

- `EAMT5/ICT_SMC_EA_V3_step1.mq5`

Mục tiêu: giúp bạn đọc/duy trì/mở rộng code mà không bị mất narrative ICT/SMC cốt lõi.

---

## 1) Tổng quan kiến trúc

EA được thiết kế theo dạng **single-file procedural architecture** (không tách `.mqh`) nhưng chia section rõ ràng:

1. Enums
2. Structs
3. Inputs
4. Globals
5. Helpers / Safety
6. Init / Deinit / Reset
7. Signal Engines
8. State Machine
9. Trade Builder + Validation + Execution
10. Runtime Management
11. Visualization + Debug

Logic chiến lược chính vẫn giữ chuỗi ICT/SMC:

**Liquidity Sweep → Shift cấu trúc (BOS/CHoCH) → Displacement → FVG/OTE retrace → Entry → Execution → Management**

---

## 2) Các thành phần dữ liệu chính

## 2.1) Enum quan trọng

- `ENUM_SETUP_STATE`: vòng đời setup (`WAIT_SWEEP`, `SWEEP_CONFIRMED`, `SHIFT_CONFIRMED`, `DISPLACEMENT_READY`, `ENTRY_READY`, `ORDER_PLACED`, `INVALID`).
- `ENUM_STRUCTURE_TYPE`: `STRUCT_BOS`, `STRUCT_CHOCH`.
- `ENUM_BIAS`: `BIAS_BULLISH`, `BIAS_BEARISH`, `BIAS_NONE`.
- `ENUM_ORDER_EXECUTION`: `EXEC_MARKET` hoặc `EXEC_PENDING_LIMIT`.
- `ENUM_INVALID_REASON`: mã hóa nguyên nhân setup invalid/timeout.

## 2.2) Struct lõi

- `SwingPoint`: lưu swing internal/external.
- `MarketStructure`: sự kiện break cấu trúc gần nhất.
- `LiquiditySweep`: tín hiệu quét thanh khoản gần nhất.
- `DisplacementInfo`: thông tin nến displacement + impulse leg.
- `FVGZone`, `OTEZone`: vùng retracement để vào lệnh.
- `TradeSetup`: setup giao dịch hoàn chỉnh (entry/sl/tp/lot/expiry/reason).
- `SetupContext`: context riêng cho **BUY** hoặc **SELL** để tránh lẫn narrative.
- `PositionRuntime`: trạng thái quản lý lệnh đang mở (partial, BE, trailing...).

---

## 3) Luồng thực thi theo thời gian (runtime flow)

## 3.1) OnInit

- Cài magic/deviation/filling mode.
- Tạo ATR handle cho signal TF và bias TF.
- Resize các mảng swings.
- Reset toàn bộ state + init 2 context:
  - `g_buyCtx.direction = 1`
  - `g_sellCtx.direction = -1`

## 3.2) OnTick (trung tâm)

Mỗi tick, EA chạy thứ tự:

1. Refresh giá.
2. `ManageOrdersAndPositions()` (luôn chạy mỗi tick).
3. Nếu có nến mới bias TF: cập nhật HTF bias.
4. Nếu **không có nến mới signal TF**: dừng xử lý signal.
5. Nếu có nến mới signal TF:
   - Kiểm tra đủ bars.
   - Chạy engines phát hiện: swing → structure → sweep → displacement → FVG/OTE.
   - Cập nhật state machine cho BUY context và SELL context.
   - Build setup tổng hợp `g_currentSetup` theo bias + context hợp lệ.
   - Validate + tính lot + execute nếu đạt điều kiện.
   - Vẽ chart objects.

---

## 4) Module Signal Engines

## 4.1) Swing Engine

- `DetectSwingPoints()` gọi:
  - `DetectTierSwing(SWING_INTERNAL, ...)`
  - `DetectTierSwing(SWING_EXTERNAL, ...)`
- Pivot kiểm theo cặp `leftBars/rightBars`.
- Tránh thêm swing trùng bằng `SwingExists`.
- Dùng `AddSwingToArray` để giữ số lượng tối đa (`InpMaxSwingsToKeep`).

## 4.2) Market Structure Engine

- `DetermineInternalTrend()` dùng 2 swing high + 2 swing low gần nhất để xác định xu hướng internal.
- `DetectMarketStructure()` so close nến #1 với swing chưa broken:
  - Break high → bullish BOS/CHoCH.
  - Break low → bearish BOS/CHoCH.
- Khi break, đánh dấu swing tương ứng là `broken=true`.

## 4.3) Liquidity Sweep Engine

- `DetectLiquiditySweep()` dựa trên **external swing**:
  - Bullish setup: wick quét dưới SSL và close quay lại trên level.
  - Bearish setup: wick quét trên BSL và close quay lại dưới level.
- Có đánh giá rejection strength từ tỉ lệ body/range.
- Đã có comment extension point cho `LiquidityPool` (equal highs/lows cluster).

## 4.4) Displacement + Impulse Engine

- `DetectDisplacement()` duyệt vài nến gần nhất (`InpDisplacementMaxBars`), dùng:
  - body >= ATR * multiplier,
  - body/range >= threshold,
  - close gần cực trị nến.
- Nếu đạt, gọi `BuildImpulseLeg()` để xác định điểm bắt đầu/kết thúc leg theo direction.

## 4.5) FVG / OTE Engine

- `DetectFVG()` kiểm tra gap 3 nến quanh displacement candle.
- `CalculateOTEZone()` lấy fib retrace của impulse leg theo `InpOTELevel1/2`.
- Có placeholder fields freshness/mitigation cho mở rộng sau.

## 4.6) HTF Bias Engine

- `DetermineHTFBias()` detect swing trên bias TF.
- Lấy 2 đỉnh + 2 đáy gần nhất:
  - HH + HL → bullish bias.
  - LH + LL → bearish bias.
  - còn lại → none.

---

## 5) State Machine (trọng tâm)

Mỗi direction (BUY/SELL) có một `SetupContext` riêng, cập nhật bằng `UpdateSetupContext(ctx)`.

### Trạng thái và chuyển tiếp

1. `SETUP_WAIT_SWEEP`
   - Chờ sweep cùng chiều context.
   - Có sweep → `SETUP_SWEEP_CONFIRMED`.

2. `SETUP_SWEEP_CONFIRMED`
   - Chờ structure shift cùng chiều, xảy ra sau thời điểm sweep.
   - Timeout theo `InpSweepExpiryBars` → invalid/reset.

3. `SETUP_SHIFT_CONFIRMED`
   - Chờ displacement cùng chiều, xảy ra sau structure time.
   - Nếu bật FVG/OTE: bắt buộc zone hợp lệ cùng chiều.
   - Đạt điều kiện → `SETUP_DISPLACEMENT_READY`.

4. `SETUP_DISPLACEMENT_READY`
   - Build trade từ context.
   - Nếu trade hợp lệ → `SETUP_ENTRY_READY`.
   - Hết hạn → invalid.

5. `SETUP_ENTRY_READY`
   - Rebuild trade mỗi nến mới để kiểm tra retracement còn hợp lệ.
   - Hết hạn hoặc mất hợp lệ → invalid.

6. `SETUP_ORDER_PLACED`
   - Chờ đến khi không còn exposure theo chiều đó thì reset về wait sweep.

7. `SETUP_INVALID`
   - Reset context có lưu metadata reason để debug.

### Helper state-machine

- `MoveSetupState`: đổi state + log + cập nhật thời gian.
- `InvalidateSetupContext`: đánh dấu invalid có reason code/text.
- `ResetContextToWaitSweep`: reset mềm, vẫn giữ reason cuối để truy vết.

---

## 6) Trade Builder Logic

`BuildTradeFromContext(ctx)` tạo setup với nguyên tắc:

1. Bắt buộc có `sweep + structure + displacement`.
2. Nếu bật FVG:
   - Chọn entry theo `InpEntryMode`:
     - `ENTRY_MID_FVG`
     - `ENTRY_NEAR_FVG`
     - `ENTRY_CANDLE_CONFIRM`
   - Quyết định market hay pending theo vị trí giá hiện tại so với vùng FVG.
3. Nếu bật OTE: entry phải nằm trong vùng OTE.
4. SL lấy theo min/max giữa sweep extreme và impulse start, có `InpSLBufferPoints`.
5. TP1/TP2 theo RR (`InpRR_TP1`, `InpRR_TP2`).
6. Gán `EXEC_MARKET` hoặc `EXEC_PENDING_LIMIT` + expiry pending.

---

## 7) Validation trước khi vào lệnh

`ValidateTradeSetup()` kiểm tra theo thứ tự:

1. Session filter.
2. Spread max.
3. One-trade-per-symbol theo direction exposure.
4. Premium/Discount filter (nếu bật).
5. Broker stop/freeze constraints (`ValidateBrokerLevels`).

Nếu pass toàn bộ mới cho phép execute.

---

## 8) Execution Engine

`ExecuteTradeSetup(setup)`:

- Cấu hình filling mode phù hợp symbol.
- Gửi market order hoặc pending limit.
- Khi thành công:
  - Cập nhật đúng global context (`g_buyCtx`/`g_sellCtx`) sang `SETUP_ORDER_PLACED`.
  - Lưu `orderTicket`.
  - Reset runtime partial flags.
- Khi thất bại: log retcode + mô tả.

---

## 9) Position & Pending Management

## 9.1) Pending orders

- `ManagePendingOrders()` duyệt pending cùng symbol+magic.
- Nếu quá expiry thì gửi `TRADE_ACTION_REMOVE` để xóa.

## 9.2) Open positions

`ManagePositions()` xử lý:

1. TP1 touch → partial close theo `InpPartialPercent` (nếu bật).
2. Sau partial, có thể dời SL về BE + safety points.
3. Trailing (nếu bật):
   - ATR trailing (`ApplyATRTrailing`), hoặc
   - Swing trailing (`ApplySwingTrailing`).

---

## 10) Visualization + Debug

- Vẽ swings internal/external.
- Vẽ line/text cho BOS/CHoCH.
- Đánh dấu sweep.
- Vẽ rectangle FVG/OTE.
- Vẽ line Entry/SL/TP1/TP2 khi setup active.
- Prefix object thống nhất `ICTV3_<Magic>_`.
- Log debug có prefix `[ICT_V3]`.

---

## 11) Input nhóm nào ảnh hưởng gì

- **Swing & Structure:** độ nhạy nhận diện cấu trúc.
- **Displacement:** độ mạnh xung lực cần thiết.
- **FVG & OTE:** mức chọn lọc retracement entry.
- **Session:** giới hạn giờ giao dịch.
- **Risk:** khối lượng, RR, partial, BE, trailing.
- **Filters:** spread và premium/discount constraints.

---

## 12) Nguyên tắc khi chỉnh sửa/mở rộng code

1. Không phá vỡ narrative chain.
2. Ưu tiên giữ behavior cũ nếu chưa có bằng chứng cải thiện.
3. Khi thêm module mới (liquidity pool, MSS nâng cao...), phải gắn chặt theo timeline event trong context.
4. Mọi state transition nên đi qua helper (`MoveSetupState` / `InvalidateSetupContext`).
5. Luôn thêm guard trước khi đọc dữ liệu series theo shift.

---

## 13) Checklist debug nhanh khi EA “không vào lệnh”

1. Có đang trong session cho phép không?
2. Spread có vượt `InpMaxSpreadPoints`?
3. Đã có chain đủ: sweep → shift → displacement → FVG/OTE?
4. Setup bị timeout ở phase nào (xem invalid reason)?
5. Broker có chặn vì stop/freeze level?
6. Đã có exposure cùng chiều (one-trade-per-symbol) chưa?

---

## 14) Tài liệu liên quan

- Cấu hình input/preset: `EAMT5/README.MD`
- Mã nguồn EA: `EAMT5/ICT_SMC_EA_V3_step1.mq5`

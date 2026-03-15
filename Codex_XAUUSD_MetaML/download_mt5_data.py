from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - environment may not have MT5
    mt5 = None

SYMBOLS = ["XAUUSD"]
TIMEFRAME = "H1"
DATE_FROM = datetime(2010, 1, 1)
DATE_TO = datetime.now()
OUTPUT_DIR = "files"

TIMEFRAME_MAP = {
    "M1": getattr(mt5, "TIMEFRAME_M1", None) if mt5 else None,
    "M5": getattr(mt5, "TIMEFRAME_M5", None) if mt5 else None,
    "M15": getattr(mt5, "TIMEFRAME_M15", None) if mt5 else None,
    "H1": getattr(mt5, "TIMEFRAME_H1", None) if mt5 else None,
    "H4": getattr(mt5, "TIMEFRAME_H4", None) if mt5 else None,
    "D1": getattr(mt5, "TIMEFRAME_D1", None) if mt5 else None,
}


def connect_mt5() -> bool:
    if mt5 is None:
        print("MetaTrader5 package chưa được cài đặt.")
        return False
    if not mt5.initialize():
        print(f"Lỗi kết nối MT5: {mt5.last_error()}")
        return False
    return True


def _fetch_rates(symbol: str, timeframe_code: int, date_from: datetime, date_to: datetime):
    rates = None
    for _ in range(3):
        time.sleep(1)
        rates = mt5.copy_rates_range(symbol, timeframe_code, date_from, date_to)
        if rates is not None and len(rates) > 0:
            return rates

    recent_from = max(date_from, date_to - timedelta(days=365 * 3))
    for _ in range(3):
        time.sleep(1)
        rates = mt5.copy_rates_range(symbol, timeframe_code, recent_from, date_to)
        if rates is not None and len(rates) > 0:
            return rates
    return rates


def download_symbol(symbol: str, timeframe_name: str = TIMEFRAME) -> str:
    timeframe_code = TIMEFRAME_MAP.get(timeframe_name)
    if timeframe_code is None:
        raise ValueError(f"Unsupported timeframe: {timeframe_name}")

    if not connect_mt5():
        raise RuntimeError("Không thể kết nối MT5")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Không thể thêm symbol vào Market Watch: {symbol}")
        time.sleep(1)

        rates = _fetch_rates(symbol, timeframe_code, DATE_FROM, DATE_TO)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Không tải được dữ liệu cho {symbol}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        out = pd.DataFrame(
            {
                "<DATE>": df["time"].dt.strftime("%Y.%m.%d"),
                "<TIME>": df["time"].dt.strftime("%H:%M"),
                "<OPEN>": df["open"],
                "<HIGH>": df["high"],
                "<LOW>": df["low"],
                "<CLOSE>": df["close"],
                "<TICKVOL>": df.get("tick_volume", 0),
                "<VOL>": df.get("real_volume", 0),
                "<SPREAD>": df.get("spread", 0),
            }
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{symbol}_{timeframe_name}.csv"
        path = os.path.join(OUTPUT_DIR, filename)
        out.to_csv(path, sep="\t", index=False)
        print(f"Saved {path} ({len(out)} rows)")
        return path
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    for s in SYMBOLS:
        download_symbol(s, TIMEFRAME)

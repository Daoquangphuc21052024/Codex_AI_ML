from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import MetaTrader5 as mt5
import pandas as pd

from .config import MT5Config


@dataclass
class MT5Connector:
    config: MT5Config
    logger: logging.Logger

    def _resolve_timeframe(self) -> int:
        attr = f"TIMEFRAME_{self.config.timeframe}"
        if not hasattr(mt5, attr):
            raise ValueError(f"Unsupported timeframe: {self.config.timeframe}")
        return int(getattr(mt5, attr))

    def fetch_rates(self) -> pd.DataFrame:
        self.logger.info("Initializing MT5...")
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        try:
            if self.config.login and self.config.password and self.config.server:
                authorized = mt5.login(
                    login=self.config.login,
                    password=self.config.password,
                    server=self.config.server,
                )
                if not authorized:
                    raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

            timeframe = self._resolve_timeframe()
            utc_to = datetime.now(timezone.utc)
            rates = mt5.copy_rates_from(
                self.config.symbol,
                timeframe,
                utc_to,
                self.config.bars,
            )
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"No rates returned: {mt5.last_error()}")

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            self.logger.info("Fetched %d bars for %s", len(df), self.config.symbol)
            return df
        finally:
            mt5.shutdown()

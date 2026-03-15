from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from catboost_xauusd.pipeline import run


def _synthetic_ohlcv(n: int = 2200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 1.2, n)
    close = 2300 + np.cumsum(steps)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(1.4, 0.5, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.integers(120, 1200, size=n)
    time = pd.date_range("2021-01-01", periods=n, freq="H", tz="UTC")
    return pd.DataFrame(
        {
            "time": time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": vol,
        }
    )


def main() -> None:
    base_cfg_path = Path("projects/catboost_xauusd/configs/config.yaml")
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="catboost_xauusd_smoke_") as td:
        tdir = Path(td)
        csv_path = tdir / "synthetic_xauusd_h1.csv"
        _synthetic_ohlcv().to_csv(csv_path, index=False)

        cfg["mt5"]["source"] = "csv"
        cfg["mt5"]["csv_path"] = str(csv_path)
        cfg["train"].update(
            {
                "n_splits": 2,
                "min_train_size": 800,
                "val_size": 200,
                "test_size": 200,
                "tuning_trials": 2,
                "iterations": 60,
            }
        )
        cfg["paths"] = {
            "raw_data": str(tdir / "raw_data.csv"),
            "reports_dir": str(tdir / "reports"),
            "artifacts_dir": str(tdir / "artifacts"),
            "logs_dir": str(tdir / "logs"),
        }

        cfg_path = tdir / "config_smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        run(str(cfg_path))

        required = [
            tdir / "artifacts" / "catboost_xauusd.onnx",
            tdir / "artifacts" / "feature_schema.json",
            tdir / "reports" / "pnl.png",
            tdir / "reports" / "confusion_matrix.png",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"Smoke test failed, missing outputs: {missing}")

        print("Smoke test passed. Outputs generated at:", tdir)


if __name__ == "__main__":
    main()

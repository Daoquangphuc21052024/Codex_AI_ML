from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(logs_dir: str) -> logging.Logger:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("catboost_xauusd")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    logfile = logging.FileHandler(Path(logs_dir) / "pipeline.log", encoding="utf-8")
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    return logger

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

LOGGER = logging.getLogger(__name__)

LABEL_LONG = 0
LABEL_SHORT = 1
LABEL_NO_TRADE = 2


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class LabelConfig:
    horizon_mode: Literal["fixed", "sampled"] = "fixed"
    horizon_bars: int = 15
    horizon_min: int = 10
    horizon_max: int = 30
    random_state: int = 42


@dataclass(frozen=True)
class CatBoostConfig:
    iterations: int = 700
    depth: int = 6
    learning_rate: float = 0.03
    loss_function: str = "MultiClass"
    eval_metric: str = "TotalF1"
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 100
    verbose: int = 100


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path = Path("exports")
    cpp_model_filename: str = "catboost_model.cpp"
    mqh_filename: str = "strategy_model.mqh"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "EURUSD"
    timeframe: int = 16385
    transaction_cost: float = 0.00010
    spread_cost: float = 0.0
    markup: float = 0.0
    ma_periods: tuple[int, ...] = (5, 25, 55, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500)
    grid_size: int = 10
    grid_distances: tuple[float, ...] = tuple(np.linspace(0.003, 0.008, num=10).tolist())
    grid_coefficients: tuple[float, ...] = tuple(np.linspace(1, 5, num=10).tolist())
    random_seed: int = 42
    include_no_trade_class: bool = True
    class_weights: dict[int, float] | None = None
    train_range: DateRange = field(default_factory=lambda: DateRange(datetime(2018, 1, 1), datetime(2020, 1, 1)))
    validation_range: DateRange = field(default_factory=lambda: DateRange(datetime(2020, 1, 1), datetime(2021, 1, 1)))
    test_range: DateRange = field(default_factory=lambda: DateRange(datetime(2021, 1, 1), datetime(2022, 1, 1)))
    label: LabelConfig = field(default_factory=LabelConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


@dataclass
class ModelArtifacts:
    model: CatBoostClassifier
    feature_names: list[str]
    train_summary: dict[str, Any]
    validation_metrics: dict[str, Any]


@dataclass
class BacktestResult:
    trade_log: pd.DataFrame
    equity_curve: pd.DataFrame
    trading_metrics: dict[str, float]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def initialize_mt5() -> None:
    """Initialize MT5 with explicit error handling."""
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed.")
    if not mt5.initialize():
        error = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: code={error[0]}, message={error[1]}")


def shutdown_mt5() -> None:
    if mt5 is not None:
        mt5.shutdown()


def fetch_prices_from_mt5(
    symbol: str,
    timeframe: int,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Fetch OHLCV from MT5 in [start_date, end_date)."""
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not installed.")
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        raise RuntimeError(f"MT5 returned None for symbol={symbol} timeframe={timeframe}")

    df = pd.DataFrame(rates)
    if df.empty:
        raise ValueError(f"No bars returned for {symbol} from {start_date} to {end_date}")

    required_cols = {"time", "open", "high", "low", "close", "tick_volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing MT5 columns: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.empty:
        raise ValueError("OHLCV data became empty after dropna().")
    return df


def build_ma_distance_features(df: pd.DataFrame, ma_periods: tuple[int, ...]) -> pd.DataFrame:
    """Build distance-to-MA features using only current and past bars."""
    out = df.copy()
    for period in ma_periods:
        ma_col = f"ma_{period}"
        dist_col = f"dist_ma_{period}"
        out[ma_col] = out["close"].rolling(window=period, min_periods=period).mean()
        out[dist_col] = out["close"] - out[ma_col]
    return out


def _resolve_horizon(index: int, n_rows: int, config: LabelConfig, rng: random.Random) -> int:
    if config.horizon_mode == "fixed":
        horizon = config.horizon_bars
    else:
        horizon = rng.randint(config.horizon_min, config.horizon_max)
    max_allowed = n_rows - index - 1
    return min(horizon, max_allowed)


def _simulate_grid_payoff(
    path: pd.Series,
    direction: Literal["long", "short"],
    transaction_cost: float,
    grid_distances: tuple[float, ...],
    grid_coefficients: tuple[float, ...],
) -> tuple[float, int]:
    """Simulate one directional grid trade on future path with explicit fills."""
    entry_price = float(path.iloc[0])
    exit_price = float(path.iloc[-1])
    if direction == "long":
        adverse_range = entry_price - float(path.min())
        base_pnl = exit_price - entry_price - transaction_cost
    else:
        adverse_range = float(path.max()) - entry_price
        base_pnl = entry_price - exit_price - transaction_cost

    realized_pnl = base_pnl
    accumulated_distance = 0.0
    additional_orders = 0
    for level_distance, coeff in zip(grid_distances, grid_coefficients):
        if accumulated_distance + float(level_distance) <= adverse_range:
            accumulated_distance += float(level_distance)
            additional_orders += 1
            if direction == "long":
                level_pnl = (exit_price - entry_price + accumulated_distance) * float(coeff)
            else:
                level_pnl = (entry_price - exit_price + accumulated_distance) * float(coeff)
            realized_pnl += level_pnl - transaction_cost * float(coeff)
        else:
            break
    return realized_pnl, additional_orders


def generate_labels(
    df: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Create labels: 0=long, 1=short, 2=no_trade, with controlled horizon."""
    labeled = df.copy()
    rng = random.Random(config.label.random_state)
    labels: list[int] = []
    horizons: list[int] = []

    n_rows = len(labeled)
    for idx in range(n_rows):
        horizon = _resolve_horizon(idx, n_rows, config.label, rng)
        if horizon <= 0:
            labels.append(LABEL_NO_TRADE)
            horizons.append(0)
            continue

        future_path = labeled["close"].iloc[idx : idx + horizon + 1]
        long_pnl, _ = _simulate_grid_payoff(
            future_path,
            direction="long",
            transaction_cost=config.transaction_cost + config.spread_cost + config.markup,
            grid_distances=config.grid_distances,
            grid_coefficients=config.grid_coefficients,
        )
        short_pnl, _ = _simulate_grid_payoff(
            future_path,
            direction="short",
            transaction_cost=config.transaction_cost + config.spread_cost + config.markup,
            grid_distances=config.grid_distances,
            grid_coefficients=config.grid_coefficients,
        )

        if long_pnl > short_pnl and long_pnl > 0:
            labels.append(LABEL_LONG)
        elif short_pnl > long_pnl and short_pnl > 0:
            labels.append(LABEL_SHORT)
        else:
            labels.append(LABEL_NO_TRADE)
        horizons.append(horizon)

    labeled["label"] = labels
    labeled["label_horizon"] = horizons
    if not config.include_no_trade_class:
        labeled = labeled[labeled["label"] != LABEL_NO_TRADE].copy()
    return labeled


def split_by_time(df: pd.DataFrame, config: StrategyConfig) -> dict[str, pd.DataFrame]:
    """Strict out-of-sample split by configured date ranges."""
    train = df[(df.index >= config.train_range.start) & (df.index < config.train_range.end)]
    valid = df[(df.index >= config.validation_range.start) & (df.index < config.validation_range.end)]
    test = df[(df.index >= config.test_range.start) & (df.index < config.test_range.end)]

    if train.empty or valid.empty or test.empty:
        raise ValueError("One of train/validation/test splits is empty. Check date ranges and data availability.")
    if not (train.index.max() < valid.index.min() and valid.index.max() < test.index.min()):
        raise ValueError("Train/validation/test overlap detected.")
    return {"train": train, "validation": valid, "test": test}


def _build_xy(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    clean = df.dropna(subset=feature_names + ["label"]).copy()
    return clean[feature_names], clean["label"].astype(int)


def train_catboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    config: StrategyConfig,
) -> ModelArtifacts:
    """Train CatBoost on real time-series data only."""
    X_train, y_train = _build_xy(train_df, feature_names)
    X_valid, y_valid = _build_xy(valid_df, feature_names)

    unique_labels = sorted(y_train.unique().tolist())
    is_multiclass = len(unique_labels) > 2
    params = asdict(config.catboost)
    params["loss_function"] = "MultiClass" if is_multiclass else "Logloss"
    params["eval_metric"] = "TotalF1" if is_multiclass else "AUC"
    params["random_seed"] = config.random_seed
    params["class_weights"] = config.class_weights

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=config.catboost.early_stopping_rounds,
    )

    valid_probs = model.predict_proba(X_valid)
    valid_pred = model.predict(X_valid).astype(int).ravel()
    valid_metrics = classification_metrics(y_valid, valid_pred, valid_probs)

    summary = {
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "feature_count": len(feature_names),
        "classes": unique_labels,
        "best_iteration": int(model.get_best_iteration()),
    }
    return ModelArtifacts(model=model, feature_names=feature_names, train_summary=summary, validation_metrics=valid_metrics)


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    labels = sorted(pd.Series(y_true).unique().tolist())
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    precision, recall, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics["per_class"] = {
        int(label): {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_each[idx]),
        }
        for idx, label in enumerate(labels)
    }

    if len(labels) == 2 and y_prob.shape[1] == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    if y_prob.shape[1] >= 2:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
    return metrics


def simulate_long_grid_trade(
    entry_time: pd.Timestamp,
    future_prices: pd.Series,
    config: StrategyConfig,
) -> dict[str, Any]:
    pnl, additional_orders = _simulate_grid_payoff(
        path=future_prices,
        direction="long",
        transaction_cost=config.transaction_cost + config.spread_cost + config.markup,
        grid_distances=config.grid_distances,
        grid_coefficients=config.grid_coefficients,
    )
    return {
        "entry_time": entry_time,
        "exit_time": future_prices.index[-1],
        "direction": "long",
        "entry": float(future_prices.iloc[0]),
        "exit": float(future_prices.iloc[-1]),
        "transaction_cost": config.transaction_cost + config.spread_cost + config.markup,
        "additional_grid_orders": int(additional_orders),
        "realized_pnl": float(pnl),
    }


def simulate_short_grid_trade(
    entry_time: pd.Timestamp,
    future_prices: pd.Series,
    config: StrategyConfig,
) -> dict[str, Any]:
    pnl, additional_orders = _simulate_grid_payoff(
        path=future_prices,
        direction="short",
        transaction_cost=config.transaction_cost + config.spread_cost + config.markup,
        grid_distances=config.grid_distances,
        grid_coefficients=config.grid_coefficients,
    )
    return {
        "entry_time": entry_time,
        "exit_time": future_prices.index[-1],
        "direction": "short",
        "entry": float(future_prices.iloc[0]),
        "exit": float(future_prices.iloc[-1]),
        "transaction_cost": config.transaction_cost + config.spread_cost + config.markup,
        "additional_grid_orders": int(additional_orders),
        "realized_pnl": float(pnl),
    }


def backtest_predictions(
    df: pd.DataFrame,
    predictions: pd.Series,
    config: StrategyConfig,
) -> BacktestResult:
    """Backtest point-in-time predictions with deterministic grid payoff execution."""
    trade_events: list[dict[str, Any]] = []
    horizon = config.label.horizon_bars

    for idx in range(len(df) - horizon):
        signal = int(predictions.iloc[idx])
        if signal == LABEL_NO_TRADE:
            continue
        future_prices = df["close"].iloc[idx : idx + horizon + 1]
        entry_time = df.index[idx]
        if signal == LABEL_LONG:
            event = simulate_long_grid_trade(entry_time, future_prices, config)
        elif signal == LABEL_SHORT:
            event = simulate_short_grid_trade(entry_time, future_prices, config)
        else:
            continue
        trade_events.append(event)

    trade_log = pd.DataFrame(trade_events)
    if trade_log.empty:
        equity_curve = pd.DataFrame({"equity": [0.0]}, index=[df.index.min()])
        return BacktestResult(trade_log=trade_log, equity_curve=equity_curve, trading_metrics=_empty_trading_metrics())

    trade_log = trade_log.sort_values("exit_time").reset_index(drop=True)
    trade_log["equity"] = trade_log["realized_pnl"].cumsum()
    equity_curve = trade_log[["exit_time", "equity"]].set_index("exit_time")
    metrics = _trading_metrics(trade_log)
    return BacktestResult(trade_log=trade_log, equity_curve=equity_curve, trading_metrics=metrics)


def _empty_trading_metrics() -> dict[str, float]:
    return {
        "total_pnl": 0.0,
        "average_pnl_per_trade": 0.0,
        "number_of_trades": 0.0,
        "hit_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "sharpe_like": 0.0,
        "expectancy": 0.0,
    }


def _trading_metrics(trade_log: pd.DataFrame) -> dict[str, float]:
    pnl = trade_log["realized_pnl"].to_numpy(dtype=float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    equity = trade_log["equity"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    std = float(np.std(pnl))
    mean = float(np.mean(pnl))
    return {
        "total_pnl": float(np.sum(pnl)),
        "average_pnl_per_trade": mean,
        "number_of_trades": float(len(pnl)),
        "hit_rate": float(np.mean(pnl > 0)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": float(np.max(drawdown)) if len(drawdown) else 0.0,
        "sharpe_like": float(mean / std) if std > 0 else 0.0,
        "expectancy": float(np.mean(pnl)),
    }


def save_equity_plot(equity_curve: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_split(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
    split_name: str,
    config: StrategyConfig,
) -> dict[str, Any]:
    feature_names = artifacts.feature_names
    X, y_true = _build_xy(df, feature_names)
    y_prob = artifacts.model.predict_proba(X)
    y_pred = artifacts.model.predict(X).astype(int).ravel()

    class_metrics = classification_metrics(y_true, y_pred, y_prob)
    backtest = backtest_predictions(df.loc[X.index], pd.Series(y_pred, index=X.index), config)

    return {
        "split": split_name,
        "classification": class_metrics,
        "trading": backtest.trading_metrics,
        "trade_log": backtest.trade_log,
        "equity_curve": backtest.equity_curve,
    }


def choose_best_model(results: list[dict[str, Any]], key: str = "profit_factor") -> dict[str, Any]:
    """Select best model using explicit metric key and deterministic tie-breakers."""
    if not results:
        raise ValueError("Empty result list.")

    def rank_value(item: dict[str, Any]) -> tuple[float, float, float]:
        trading = item["validation"]["trading"]
        classification = item["validation"]["classification"]
        return (
            float(trading.get(key, -np.inf)),
            float(trading.get("sharpe_like", -np.inf)),
            float(classification.get("balanced_accuracy", -np.inf)),
        )

    return max(results, key=rank_value)


def export_to_mql5(
    artifacts: ModelArtifacts,
    config: StrategyConfig,
    symbol_suffix: str | None = None,
) -> dict[str, Path]:
    """Export CatBoost C++ and generate MQL5 include with strict feature ordering."""
    output_dir = config.export.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = output_dir / config.export.cpp_model_filename
    artifacts.model.save_model(str(cpp_path), format="cpp")
    cpp_code = cpp_path.read_text(encoding="utf-8")

    feature_order = artifacts.feature_names
    header_name = config.export.mqh_filename
    if symbol_suffix:
        header_name = f"{Path(header_name).stem}_{symbol_suffix}.mqh"
    mqh_path = output_dir / header_name

    config_block = {
        "ma_periods": list(config.ma_periods),
        "grid_distances": list(config.grid_distances),
        "grid_coefficients": list(config.grid_coefficients),
        "feature_order": feature_order,
    }

    mqh = []
    mqh.append("// AUTO-GENERATED. DO NOT EDIT MANUALLY.")
    mqh.append("// Feature order must match Python training order 100%.")
    mqh.append("#include <Math\\Stat\\Math.mqh>")
    feature_order_mql = ','.join('\"' + name + '\"' for name in feature_order)
    mqh.append(f"string FEATURE_ORDER[{len(feature_order)}] = {{{feature_order_mql}}};")
    mqh.append(f"int MA_PERIODS[{len(config.ma_periods)}] = {{{','.join(map(str, config.ma_periods))}}};")
    mqh.append(
        f"double GRID_DISTANCES[{len(config.grid_distances)}] = {{{','.join(f'{x:.10f}' for x in config.grid_distances)}}};"
    )
    mqh.append(
        f"double GRID_COEFFICIENTS[{len(config.grid_coefficients)}] = {{{','.join(f'{x:.10f}' for x in config.grid_coefficients)}}};"
    )
    mqh.append(
        "void FillFeatures(double &features[]) { ArrayResize(features, ArraySize(MA_PERIODS)); "
        "for(int i=0;i<ArraySize(MA_PERIODS);i++){double pr[];CopyClose(NULL, PERIOD_CURRENT, 1, MA_PERIODS[i], pr);"
        "double mean=MathMean(pr);features[i]=pr[MA_PERIODS[i]-1]-mean;} }"
    )
    mqh.append("// CatBoost C++ inference code below.")
    mqh.append(cpp_code)
    mqh_path.write_text("\n".join(mqh), encoding="utf-8")

    meta_path = output_dir / "export_metadata.json"
    meta_path.write_text(json.dumps(config_block, indent=2), encoding="utf-8")
    return {"cpp": cpp_path, "mqh": mqh_path, "metadata": meta_path}


def run_pipeline(config: StrategyConfig) -> dict[str, Any]:
    set_global_seed(config.random_seed)

    initialize_mt5()
    try:
        start_date = min(config.train_range.start, config.validation_range.start, config.test_range.start)
        end_date = max(config.train_range.end, config.validation_range.end, config.test_range.end)
        prices = fetch_prices_from_mt5(config.symbol, config.timeframe, start_date, end_date)
    finally:
        shutdown_mt5()

    featured = build_ma_distance_features(prices, config.ma_periods)
    labeled = generate_labels(featured, config)
    feature_names = [f"dist_ma_{period}" for period in config.ma_periods]
    splits = split_by_time(labeled, config)

    artifacts = train_catboost(splits["train"], splits["validation"], feature_names, config)
    eval_validation = evaluate_split(splits["validation"], artifacts, "validation", config)
    eval_test = evaluate_split(splits["test"], artifacts, "test", config)

    return {
        "artifacts": artifacts,
        "validation": eval_validation,
        "test": eval_test,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    config = StrategyConfig()
    LOGGER.info("Running pipeline for symbol=%s", config.symbol)
    results = run_pipeline(config)
    export_paths = export_to_mql5(results["artifacts"], config, symbol_suffix=config.symbol)

    LOGGER.info("Validation trading metrics: %s", results["validation"]["trading"])
    LOGGER.info("Test trading metrics: %s", results["test"]["trading"])
    LOGGER.info("Model exported: %s", export_paths)


if __name__ == "__main__":
    main()

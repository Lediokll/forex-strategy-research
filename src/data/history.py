"""Load saved MT5 history for the backtest (no MT5 connection needed)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config as cfg_module
from src.config import Config, display_path
from src.data.broker_time import server_to_utc
from src.data.symbol_spec import DEFAULT_EURUSD_SPEC, SymbolSpec

logger = logging.getLogger(__name__)


def history_path(cfg: Config) -> Path:
    return cfg_module.DATA_RAW_DIR / f"{cfg.symbol}_{cfg.strategy.timeframe}.csv"


def spec_path(cfg: Config) -> Path:
    return cfg_module.DATA_RAW_DIR / f"{cfg.symbol}_spec.json"


def add_utc_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Adds `time_utc` next to the broker-server `datetime` column."""

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["time_utc"] = server_to_utc(
        df["datetime"],
        cfg.broker_time.base_utc_offset_hours,
        cfg.broker_time.dst_rule,
    )
    return df


def load_history(cfg: Config, path: Path | None = None) -> pd.DataFrame:
    path = path or history_path(cfg)

    if not path.exists():
        raise FileNotFoundError(
            f"History not found: {display_path(path)}. "
            "Run `py main.py --download` with the MT5 terminal open."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"History file is empty: {display_path(path)}")

    return add_utc_time(df, cfg)


def load_symbol_spec(cfg: Config) -> SymbolSpec:
    path = spec_path(cfg)

    if path.exists():
        return SymbolSpec.load(path)

    if cfg.symbol != DEFAULT_EURUSD_SPEC.symbol:
        raise FileNotFoundError(f"No symbol spec snapshot for {cfg.symbol}: {display_path(path)}")

    logger.warning("No broker symbol snapshot at %s; using standard EURUSD/USD defaults.", display_path(path))
    return DEFAULT_EURUSD_SPEC

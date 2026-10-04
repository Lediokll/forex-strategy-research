"""MT5 connection helpers and candle downloads (needs the terminal running)."""

from __future__ import annotations

import logging
import time

import MetaTrader5 as mt5
import pandas as pd

from src.config import Config, display_path
from src.data.history import history_path, spec_path
from src.data.symbol_spec import SymbolSpec

logger = logging.getLogger(__name__)

RATE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume", "spread", "real_volume"]


def mt5_timeframe(name: str) -> int:
    return getattr(mt5, f"TIMEFRAME_{name}")


def initialize_mt5(max_retries: int = 3, retry_delay: int = 5) -> bool:
    """
    Initializes connection to the MetaTrader 5 terminal.

    mt5.initialize() connects Python to the locally running MT5 terminal.
    If the terminal is closed, not logged in, or disconnected, initialization
    can fail. This retry loop protects against temporary connection issues.
    """

    for attempt in range(1, max_retries + 1):
        try:
            if mt5.initialize():
                logger.info("MT5 connection initialized successfully.")
                return True

            error_code, error_msg = mt5.last_error()
            logger.warning(
                "MT5 initialization failed. Attempt %s/%s. Error %s: %s",
                attempt, max_retries, error_code, error_msg,
            )

        except Exception as e:
            logger.warning("Unexpected MT5 connection error: %s", e)

        if attempt < max_retries:
            time.sleep(retry_delay)

    return False


def rates_to_frame(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
    return df[RATE_COLUMNS]


def fetch_closed_bars(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Most recent `count` CLOSED bars (start_pos=1 skips the forming bar).
    Times are broker server time."""

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol}: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe(timeframe), 1, count)

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No candle data returned for {symbol}: {mt5.last_error()}")

    return rates_to_frame(rates)


def get_symbol_spec(symbol: str) -> SymbolSpec:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol}: {mt5.last_error()}")

    info = mt5.symbol_info(symbol)

    if info is None:
        raise RuntimeError(f"symbol_info({symbol}) failed: {mt5.last_error()}")

    return SymbolSpec.from_mt5(info)


def download_history(cfg: Config) -> tuple[pd.DataFrame, SymbolSpec]:
    """Downloads closed history bars plus a symbol-spec snapshot to data/raw/."""

    if not initialize_mt5():
        raise ConnectionError(f"Could not connect to MetaTrader 5: {mt5.last_error()}")

    try:
        terminal = mt5.terminal_info()
        count = cfg.backtest.history_bars

        if terminal is not None and terminal.maxbars <= count:
            logger.warning(
                "Terminal 'Max bars in chart' is %s; requesting %s instead of %s. "
                "Raise it in Tools > Options > Charts for more history.",
                terminal.maxbars, terminal.maxbars - 1, count,
            )
            count = terminal.maxbars - 1

        df = fetch_closed_bars(cfg.symbol, cfg.strategy.timeframe, count)
        spec = get_symbol_spec(cfg.symbol)

        path = history_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        spec.save(spec_path(cfg))

        logger.info(
            "Saved %s %s %s bars (%s -> %s, server time) to %s",
            len(df), cfg.symbol, cfg.strategy.timeframe,
            df["datetime"].iloc[0], df["datetime"].iloc[-1], display_path(path),
        )

        if len(df) < count:
            logger.warning("Broker returned only %s of %s requested bars.", len(df), count)

        return df, spec

    finally:
        mt5.shutdown()

import logging

import numpy as np
import pandas as pd

from src.config import RegimeConfig, StrategyConfig

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Converts raw OHLCV candle data into the features used by the strategy
    and the regime model. Shared by the backtest and the live loop so both
    see identical numbers.

    Every feature looks only backward (ewm/rolling/shift), so computing them
    over a whole history file does not leak future bars into past rows.
    This module stays independent from MT5.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: StrategyConfig | None = None,
        regime: RegimeConfig | None = None,
    ):
        self.df = df.copy()
        self.strategy = strategy or StrategyConfig()
        self.regime = regime or RegimeConfig()

    def validate_columns(self) -> None:
        required_columns = {
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def clean_data(self) -> None:
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.df = self.df.sort_values("datetime").reset_index(drop=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "spread"]

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df = self.df.dropna(subset=numeric_cols).reset_index(drop=True)

    def add_returns(self) -> None:
        self.df["log_returns"] = np.log(self.df["close"] / self.df["close"].shift(1))

    def add_moving_averages(self) -> None:
        for period in (
            self.strategy.ema_pullback,
            self.strategy.ema_trend_fast,
            self.strategy.ema_trend_slow,
        ):
            self.df[f"ema_{period}"] = self.df["close"].ewm(span=period, adjust=False).mean()

    def add_atr(self) -> None:
        """Wilder's ATR (the standard MT5/TradingView definition)."""

        period = self.strategy.atr_period
        prev_close = self.df["close"].shift(1)

        true_range = pd.concat(
            [
                self.df["high"] - self.df["low"],
                (self.df["high"] - prev_close).abs(),
                (self.df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        self.df["atr"] = true_range.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()

    def add_regime_features(self) -> None:
        """Inputs to the HMM: recent drift and realized volatility."""

        window = self.regime.feature_window

        self.df["regime_drift"] = np.log(self.df["close"] / self.df["close"].shift(window))
        self.df["regime_vol"] = self.df["log_returns"].rolling(window).std()

    def run(self) -> pd.DataFrame:
        self.validate_columns()
        self.clean_data()

        self.add_returns()
        self.add_moving_averages()
        self.add_atr()
        self.add_regime_features()

        # Division by a zero close/std can produce +/-inf, which dropna()
        # would otherwise let through.
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Only the warm-up rows at the start have NaNs (rolling windows).
        feature_cols = ["log_returns", "atr", "regime_drift", "regime_vol"]
        self.df = self.df.dropna(subset=feature_cols).reset_index(drop=True)

        return self.df

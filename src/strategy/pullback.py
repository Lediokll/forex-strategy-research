"""
EMA20 pullback in the EMA50/EMA200 trend direction.

The single source of truth for entry logic: the backtest and the live loop
both call these functions, so they can't drift apart.

Long setup on a closed bar (shorts are the mirror image):
  1. trend: EMA50 > EMA200
  2. pullback: low touched/crossed EMA20 on this bar or the previous
     `pullback_bars - 1` bars
  3. confirmation: this bar closed above EMA20 and closed bullish (close > open)

The trade is entered at the next bar's open (live: at market as soon as the
bar closes). Stop = sl_atr_mult x ATR, target = tp_r_mult x stop distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import StrategyConfig, session_minutes
from src.regime.regime_detector import TREND

BUY = "buy"
SELL = "sell"


def compute_setups(df: pd.DataFrame, cfg: StrategyConfig) -> np.ndarray:
    """+1 (buy setup), -1 (sell setup) or 0 for every row. Causal."""

    ema_pb = df[f"ema_{cfg.ema_pullback}"]
    ema_fast = df[f"ema_{cfg.ema_trend_fast}"]
    ema_slow = df[f"ema_{cfg.ema_trend_slow}"]

    window = cfg.pullback_bars
    touched_from_above = (df["low"] - ema_pb).rolling(window, min_periods=1).min() <= 0
    touched_from_below = (df["high"] - ema_pb).rolling(window, min_periods=1).max() >= 0

    long_setup = (
        (ema_fast > ema_slow)
        & touched_from_above
        & (df["close"] > ema_pb)
        & (df["close"] > df["open"])
    )
    short_setup = (
        (ema_fast < ema_slow)
        & touched_from_below
        & (df["close"] < ema_pb)
        & (df["close"] < df["open"])
    )

    setups = np.zeros(len(df), dtype=int)
    setups[long_setup.to_numpy()] = 1
    setups[short_setup.to_numpy()] = -1

    return setups


def direction_name(setup: int) -> str:
    return BUY if setup > 0 else SELL


@dataclass(frozen=True)
class TradePlan:
    direction: str
    sl_distance: float   # price units
    tp_distance: float   # price units
    atr: float


def plan_trade(direction: str, atr: float, cfg: StrategyConfig, min_stop_distance: float = 0.0) -> TradePlan:
    """ATR stop (widened to the broker minimum if needed) and R-multiple target."""

    if direction not in (BUY, SELL):
        raise ValueError(f"direction must be '{BUY}' or '{SELL}', got {direction!r}")

    if not atr > 0:
        raise ValueError(f"ATR must be positive, got {atr!r}")

    sl_distance = max(atr * cfg.sl_atr_mult, min_stop_distance)
    tp_distance = max(sl_distance * cfg.tp_r_mult, min_stop_distance)

    return TradePlan(direction, sl_distance, tp_distance, atr)


def sl_tp_prices(direction: str, entry_price: float, plan: TradePlan) -> tuple[float, float]:
    if direction == BUY:
        return entry_price - plan.sl_distance, entry_price + plan.tp_distance
    return entry_price + plan.sl_distance, entry_price - plan.tp_distance


def in_session(entry_time_utc: datetime, cfg: StrategyConfig) -> bool:
    start, end = session_minutes(cfg)
    minute_of_day = entry_time_utc.hour * 60 + entry_time_utc.minute
    return start <= minute_of_day < end


def strategy_skip_reasons(
    regime_label: str | None,
    entry_time_utc: datetime,
    spread_pips: float,
    cfg: StrategyConfig,
    regime_enabled: bool = True,
) -> list[str]:
    """Strategy-level filters. Empty list = setup passes. Risk-level checks
    (limits, $ risk, kill switch) are handled by src.risk."""

    reasons = []

    if regime_enabled and regime_label != TREND:
        reasons.append(f"regime_{regime_label or 'none'}")

    if not in_session(entry_time_utc, cfg):
        reasons.append("outside_session")

    if spread_pips > cfg.max_spread_pips:
        reasons.append(f"spread_{spread_pips:.1f}pips")

    return reasons

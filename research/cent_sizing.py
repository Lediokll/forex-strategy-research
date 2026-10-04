"""
Can the portfolio be traded on a $10 cent account?

A cent account's lot is the standard contract x `contract_size_multiplier`
(0.01: 0.01 lot of EURUSD = 10 EUR). Every position must be a whole number
of `lot_step` lots of at least `min_lot`. A vol-targeted position smaller
than one minimum lot can't be opened at its intended size: it is either
skipped or opened too big.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.portfolio import LotRounding, MarketData, PortfolioResult
from research.settings import CentAccountConfig


def lot_units(md: MarketData, cfg: CentAccountConfig) -> np.ndarray:
    """Base-currency units in one lot step on the cent account."""

    if abs(cfg.min_lot - cfg.lot_step) > 1e-12:
        raise ValueError("Only min_lot == lot_step is modelled.")

    return (md.contract_size * cfg.contract_size_multiplier * cfg.lot_step).to_numpy(float)


def rounding(md: MarketData, cfg: CentAccountConfig, policy: str) -> LotRounding:
    return LotRounding(lot_units=lot_units(md, cfg), policy=policy)


def feasibility_table(result: PortfolioResult, md: MarketData, cfg: CentAccountConfig) -> pd.DataFrame:
    """Per symbol: typical intended position at $10 vs the minimum lot."""

    weights = result.weights_at_rebalance.abs()
    last = md.dates[-1]
    notional_per_unit = md.close.loc[last] * md.conv.loc[last]
    min_units = pd.Series(lot_units(md, cfg), index=md.symbols)
    min_notional = min_units * notional_per_unit

    rows = {}
    for symbol in md.symbols:
        active = weights[symbol][weights[symbol] > 0]
        if active.empty:
            continue
        typical_weight = active.median()
        target_notional = typical_weight * cfg.start_equity_usd
        rows[symbol] = {
            "typical_weight": typical_weight,
            "target_notional_usd": target_notional,
            "min_lot_notional_usd": min_notional[symbol],
            "target_in_min_lots": target_notional / min_notional[symbol],
            "min_lot_risk_multiple": min_notional[symbol] / target_notional,
            "equity_for_1_min_lot": min_notional[symbol] / typical_weight,
            "min_lot_margin_usd": min_notional[symbol] / cfg.leverage,
        }

    return pd.DataFrame(rows).T.sort_values("equity_for_1_min_lot", ascending=False)


def intended_lots(result: PortfolioResult, md: MarketData, cfg: CentAccountConfig) -> pd.DataFrame:
    """Intended position size in minimum lots at every rebalance of `result`
    (before rounding), using that run's own equity at the time."""

    weights = result.weights_at_rebalance
    dates = weights.index
    equity = result.equity.reindex(dates)
    notional_per_unit = (md.close.loc[dates] * md.conv.loc[dates]).replace(0, np.nan)
    units = weights.mul(equity, axis=0) / notional_per_unit
    return units.abs() / lot_units(md, cfg)


def share_below(lots: pd.DataFrame, threshold: float) -> float:
    """Share of intended non-zero positions smaller than `threshold` lots."""

    values = lots.to_numpy().ravel()
    values = values[np.isfinite(values) & (values > 0)]
    return float((values < threshold).mean()) if len(values) else np.nan

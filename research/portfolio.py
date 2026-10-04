"""
Multi-symbol daily portfolio backtester.

Accounting is in units of each instrument (not weights), so P&L, swaps and
costs are what a broker statement would show:

  - Targets are decided at the close of a rebalance day and filled at the
    next day's open. Between rebalances the units are held unchanged.
  - Every fill pays half the spread plus slippage (a fraction of the spread)
    plus commission, per unit traded.
  - Swap is charged/credited every calendar night a position is held
    (Friday -> Monday counts 3 nights, the same weekly total as the broker's
    triple-swap day), at the broker's long/short swap rate.
  - P&L in the quote currency is converted to USD with that day's rate.

Sizing: each position gets |signal| x position_vol_target of annualized
volatility (equal risk per position), using trailing realized vol. If the
sum of position vols exceeds max_gross_risk, every position is scaled down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.settings import CostsConfig, PortfolioConfig


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    symbols: list[str]
    open: pd.DataFrame           # NaN-free: missing bars filled with the previous close
    close: pd.DataFrame
    has_bar: pd.DataFrame        # True where the symbol actually printed a bar
    conv: pd.DataFrame           # USD per 1 unit of quote currency
    spread_price: pd.Series      # typical spread in price units, per symbol
    contract_size: pd.Series
    swap_long: pd.Series         # annual fraction of notional (+ = earn)
    swap_short: pd.Series

    def window(self, end_exclusive: pd.Timestamp) -> "MarketData":
        keep = self.dates < end_exclusive
        return MarketData(
            dates=self.dates[keep],
            symbols=self.symbols,
            open=self.open[keep],
            close=self.close[keep],
            has_bar=self.has_bar[keep],
            conv=self.conv[keep],
            spread_price=self.spread_price,
            contract_size=self.contract_size,
            swap_long=self.swap_long,
            swap_short=self.swap_short,
        )


def realized_vol(md: MarketData, lookback: int, trading_days: int = 252) -> pd.DataFrame:
    """Annualized vol of daily log returns, using only days with a real bar."""

    returns = np.log(md.close / md.close.shift(1)).where(md.has_bar)
    return returns.rolling(lookback, min_periods=int(lookback * 0.8)).std() * np.sqrt(trading_days)


def target_weights(signal_row: np.ndarray, vol_row: np.ndarray, cfg: PortfolioConfig) -> np.ndarray:
    """Signed notional / equity per symbol for one rebalance."""

    valid = np.isfinite(signal_row) & np.isfinite(vol_row) & (vol_row > 0)
    weights = np.zeros_like(signal_row, dtype=float)
    weights[valid] = signal_row[valid] * cfg.position_vol_target / vol_row[valid]

    gross_risk = np.abs(signal_row[valid]).sum() * cfg.position_vol_target
    if gross_risk > cfg.max_gross_risk:
        weights *= cfg.max_gross_risk / gross_risk

    return weights


@dataclass
class LotRounding:
    """Round units to a broker lot grid (cent-account feasibility runs)."""

    lot_units: np.ndarray        # units per minimum lot, per symbol
    policy: str = "nearest"      # nearest | at_least_min

    def apply(self, units: np.ndarray) -> np.ndarray:
        lots = units / self.lot_units
        rounded = np.round(lots)
        if self.policy == "at_least_min":
            rounded = np.where((units != 0) & (rounded == 0), np.sign(units), rounded)
        elif self.policy != "nearest":
            raise ValueError(f"Unknown rounding policy {self.policy!r}")
        return rounded * self.lot_units


@dataclass
class PortfolioResult:
    equity: pd.Series
    symbol_pnl: pd.DataFrame     # USD per day per symbol (after costs and swap)
    symbol_cost: pd.Series       # total spread/slippage/commission per symbol
    symbol_swap: pd.Series       # total swap per symbol (+ = earned)
    units: pd.DataFrame          # units held at each close
    weights_at_rebalance: pd.DataFrame
    turnover_usd: float

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def run_portfolio(
    md: MarketData,
    signal: pd.DataFrame,
    rebalance: np.ndarray,
    vol: pd.DataFrame,
    pcfg: PortfolioConfig,
    costs: CostsConfig,
    start: pd.Timestamp,
    spread_multiplier: float | None = None,
    start_equity: float = 1.0,
    rounding: LotRounding | None = None,
) -> PortfolioResult:
    dates = md.dates
    n_sym = len(md.symbols)
    k0 = int(dates.searchsorted(start))
    if k0 >= len(dates) - 1:
        raise ValueError("Start date leaves no data to trade.")

    # Before a symbol's history starts its prices are NaN; it can't be held
    # then (no signal/vol), so zeros keep the arithmetic clean.
    o = np.nan_to_num(md.open.to_numpy(float))
    c = np.nan_to_num(md.close.to_numpy(float))
    conv = np.nan_to_num(md.conv.to_numpy(float))
    sig = signal.reindex(index=dates, columns=md.symbols).to_numpy(float)
    vol_arr = vol.reindex(index=dates, columns=md.symbols).to_numpy(float)

    mult = costs.spread_multiplier if spread_multiplier is None else spread_multiplier
    cost_price = md.spread_price.to_numpy(float) * mult * (0.5 + costs.slippage_spread_fraction)
    commission_unit = costs.commission_per_lot_usd / 2 / md.contract_size.to_numpy(float)
    swap_long = md.swap_long.to_numpy(float)
    swap_short = md.swap_short.to_numpy(float)
    nights = np.diff(dates.as_unit("ns").asi8) / (86_400 * 10**9)

    equity = start_equity
    units = np.zeros(n_sym)
    pending = None

    n_days = len(dates) - k0
    equity_out = np.empty(n_days)
    pnl_out = np.zeros((n_days, n_sym))
    units_out = np.zeros((n_days, n_sym))
    cost_total = np.zeros(n_sym)
    swap_total = np.zeros(n_sym)
    turnover = 0.0
    rebalance_rows = {}

    for step, k in enumerate(range(k0, len(dates))):
        if step > 0:
            if pending is not None:
                pnl = units * (o[k] - c[k - 1]) * conv[k] + pending * (c[k] - o[k]) * conv[k]
                traded = np.abs(pending - units)
                cost = traded * (cost_price * conv[k] + commission_unit)
                turnover += float((traded * o[k] * conv[k]).sum())
                units = pending
                pending = None
            else:
                pnl = units * (c[k] - c[k - 1]) * conv[k]
                cost = np.zeros(n_sym)

            rate = np.where(units > 0, swap_long, np.where(units < 0, swap_short, 0.0))
            swap = np.abs(units) * c[k] * conv[k] * rate / 365 * nights[k - 1]

            day_pnl = pnl - cost + swap
            pnl_out[step] = day_pnl
            cost_total += cost
            swap_total += swap
            equity += day_pnl.sum()

        equity_out[step] = equity
        units_out[step] = units

        if rebalance[k] and k < len(dates) - 1 and equity > 0:
            weights = target_weights(sig[k], vol_arr[k], pcfg)
            notional_per_unit = c[k] * conv[k]
            target = np.divide(weights * equity, notional_per_unit,
                               out=np.zeros(n_sym), where=notional_per_unit > 0)
            if rounding is not None:
                target = rounding.apply(target)
            pending = target
            rebalance_rows[dates[k]] = weights

    index = dates[k0:]
    return PortfolioResult(
        equity=pd.Series(equity_out, index=index),
        symbol_pnl=pd.DataFrame(pnl_out, index=index, columns=md.symbols),
        symbol_cost=pd.Series(cost_total, index=md.symbols),
        symbol_swap=pd.Series(swap_total, index=md.symbols),
        units=pd.DataFrame(units_out, index=index, columns=md.symbols),
        weights_at_rebalance=pd.DataFrame.from_dict(rebalance_rows, orient="index", columns=md.symbols),
        turnover_usd=turnover,
    )

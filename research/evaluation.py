"""Performance statistics, drawdown analysis and Monte Carlo reshuffling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def longest_recovery_days(equity: pd.Series) -> tuple[int, bool]:
    """Longest stretch (calendar days) from a peak until equity regained it.
    Returns (days, recovered); recovered=False if the longest one is still open."""

    peak_time = equity.index[0]
    peak_value = equity.iloc[0]
    longest, recovered = 0, True

    for time, value in equity.items():
        if value >= peak_value:
            longest = max(longest, (time - peak_time).days)
            peak_time, peak_value = time, value

    open_days = (equity.index[-1] - peak_time).days
    if open_days > longest:
        longest, recovered = open_days, False

    return longest, recovered


def yearly_returns(equity: pd.Series) -> pd.Series:
    year_end = equity.groupby(equity.index.year).last()
    previous = year_end.shift(1)
    previous.iloc[0] = equity.iloc[0]
    return year_end / previous - 1.0


def perf_stats(equity: pd.Series, trading_days: int = 252) -> dict:
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    final_ratio = equity.iloc[-1] / equity.iloc[0]

    vol = returns.std() * np.sqrt(trading_days)
    sharpe = returns.mean() / returns.std() * np.sqrt(trading_days) if returns.std() > 0 else np.nan
    cagr = final_ratio ** (1 / years) - 1 if final_ratio > 0 and years > 0 else -1.0
    yearly = yearly_returns(equity)
    recovery_days, recovered = longest_recovery_days(equity)

    return {
        "start": equity.index[0],
        "end": equity.index[-1],
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "vol": float(vol),
        "max_dd": float(-drawdown_series(equity).min()),
        "worst_year": float(yearly.min()),
        "worst_year_label": int(yearly.idxmin()),
        "best_year": float(yearly.max()),
        "longest_dd_days": recovery_days,
        "longest_dd_recovered": recovered,
        "total_return": float(final_ratio - 1),
    }


def period_returns(equity: pd.Series, frequency: str = "W") -> np.ndarray:
    """Compounded returns per calendar period: the 'trades' of a portfolio
    that rebalances on that schedule."""

    period_end = equity.groupby(equity.index.to_period(frequency)).last()
    start = pd.concat([pd.Series([equity.iloc[0]]), period_end.iloc[:-1]], ignore_index=True)
    return (period_end.to_numpy() / start.to_numpy()) - 1.0


def monte_carlo_max_drawdowns(returns: np.ndarray, runs: int, seed: int) -> np.ndarray:
    """Max drawdown of `runs` random reorderings of the same period returns.
    Same returns, same final equity, different paths: shows how much of the
    historical drawdown was luck of the ordering."""

    rng = np.random.default_rng(seed)
    out = np.empty(runs)

    for i in range(runs):
        path = np.cumprod(1.0 + rng.permutation(returns))
        peak = np.maximum.accumulate(np.concatenate([[1.0], path]))
        out[i] = 1.0 - (np.concatenate([[1.0], path]) / peak).min()

    return out


def per_symbol_stats(result, trading_days: int = 252) -> pd.DataFrame:
    """Contribution of each symbol: P&L as a fraction of prior-day equity."""

    prior_equity = result.equity.shift(1).bfill()
    contrib = result.symbol_pnl.div(prior_equity, axis=0)
    years = (result.equity.index[-1] - result.equity.index[0]).days / 365.25
    held = (result.units != 0).mean()

    # Swap and cost totals are in USD; express them per year relative to
    # the average equity (approximate under compounding).
    scale = result.equity.mean() * years if years > 0 else np.nan

    rows = {}
    for symbol in contrib.columns:
        series = contrib[symbol]
        std = series.std()
        rows[symbol] = {
            "ann_contribution": series.sum() / years if years > 0 else np.nan,
            "sharpe": series.mean() / std * np.sqrt(trading_days) if std > 0 else np.nan,
            "ann_swap": result.symbol_swap[symbol] / scale,
            "ann_cost": -result.symbol_cost[symbol] / scale,
            "time_in_market": held[symbol],
        }

    return pd.DataFrame(rows).T

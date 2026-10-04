"""Performance statistics for a list of backtest trades."""

from __future__ import annotations

import numpy as np
import pandas as pd


def longest_losing_streak(pnl: np.ndarray) -> int:
    longest = current = 0
    for value in pnl:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def max_drawdown(cumulative: np.ndarray, start: float = 0.0) -> float:
    curve = np.concatenate([[start], start + cumulative])
    return float((np.maximum.accumulate(curve) - curve).max())


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
            "expectancy_r": np.nan, "expectancy_usd": np.nan, "net_usd": 0.0,
            "net_r": 0.0, "max_dd_usd": 0.0, "max_dd_r": 0.0, "longest_losing_streak": 0,
        }

    pnl = trades["pnl_usd"].to_numpy(float)
    r = trades["r_multiple"].to_numpy(float)

    gross_win = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    return {
        "trades": len(trades),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "expectancy_r": float(r.mean()),
        "expectancy_usd": float(pnl.mean()),
        "net_usd": float(pnl.sum()),
        "net_r": float(r.sum()),
        "max_dd_usd": max_drawdown(np.cumsum(pnl)),
        "max_dd_r": max_drawdown(np.cumsum(r)),
        "longest_losing_streak": longest_losing_streak(pnl),
    }


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    years = pd.DatetimeIndex(trades["exit_time_utc"]).year
    rows = {year: summarize(group) for year, group in trades.groupby(years)}
    return pd.DataFrame(rows).T


def daily_summary(signals: pd.DataFrame, trades: pd.DataFrame, start_equity: float) -> pd.DataFrame:
    """One row per UTC day with any signal or closed trade (backtest journal)."""

    frames = []

    if not signals.empty:
        sig = signals.assign(day_utc=pd.DatetimeIndex(signals["time_utc"]).date)
        frames.append(sig.groupby("day_utc").agg(
            signals=("taken", "size"), signals_taken=("taken", "sum"),
        ))

    if not trades.empty:
        tr = trades.assign(day_utc=pd.DatetimeIndex(trades["exit_time_utc"]).date)
        frames.append(tr.groupby("day_utc").agg(
            trades_closed=("pnl_usd", "size"),
            wins=("pnl_usd", lambda s: int((s > 0).sum())),
            losses=("pnl_usd", lambda s: int((s < 0).sum())),
            realized_pnl_usd=("pnl_usd", "sum"),
        ))

    if not frames:
        return pd.DataFrame()

    daily = pd.concat(frames, axis=1).fillna(0).sort_index()
    daily["end_equity"] = start_equity + daily["realized_pnl_usd"].cumsum()
    return daily.reset_index().rename(columns={"index": "day_utc"})

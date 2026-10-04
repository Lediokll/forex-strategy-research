"""
Strategy signals. Each returns a (dates x symbols) DataFrame of signed
position scales in [-1, 1] (NaN = no view), read only on rebalance dates.
Position sizing (vol targeting, risk cap) is done by the portfolio engine.

All inputs at date t use closes up to and including t; the engine executes
at the next bar's open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

USD = "USD"


def rebalance_mask(dates: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """True on the first trading day of each week/month. Known at that day's
    close without needing tomorrow's date (no calendar look-ahead)."""

    if frequency == "weekly":
        period = dates.to_period("W")
    elif frequency == "monthly":
        period = dates.to_period("M")
    else:
        raise ValueError(f"Unknown rebalance frequency: {frequency!r}")

    codes = period.asi8
    mask = np.ones(len(dates), dtype=bool)
    mask[1:] = codes[1:] != codes[:-1]
    return mask


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
def trend_signal(closes: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    """Average of sign(return over each lookback). Values in {-1, -1/3, 1/3, 1}
    for three lookbacks. NaN until the longest lookback is available."""

    signs = [np.sign(closes / closes.shift(lb) - 1.0) for lb in lookbacks]
    total = sum(signs)
    return total / len(lookbacks)


# ---------------------------------------------------------------------------
# Carry
# ---------------------------------------------------------------------------
def annual_swap_rates(info: dict) -> tuple[float, float]:
    """(long, short) swap as an annual fraction of position notional.
    Positive = the position earns swap, negative = it pays.

    MT5 swap modes: 0 disabled, 1 points, 2 base-currency money,
    3 margin-currency money, 4 deposit-currency money, 5/6 annual interest %,
    7/8 reopen (expressed in points).
    """

    mode = info["swap_mode"]
    long_, short_ = info["swap_long"], info["swap_short"]
    price = info["price"]

    if mode == 0:
        return 0.0, 0.0

    if mode in (1, 7, 8):
        # points x point size = price change per night; as a share of price.
        factor = info["point"] / price * 365
    elif mode in (5, 6):
        factor = 1 / 100
    elif mode == 4:
        notional_usd_per_lot = price * info["tick_value"] / info["tick_size"]
        factor = 365 / notional_usd_per_lot
    elif mode in (2, 3):
        # Money in the base (2) or margin (3) currency per lot per night.
        # For FX that currency is the base, whose notional is contract_size
        # units; for a CFD quoted in its own currency (US500: USD/USD) the
        # notional in that currency is contract_size x price.
        money_ccy = info["currency_base"] if mode == 2 else info["currency_margin"]
        notional = info["contract_size"] * (price if money_ccy == info["currency_profit"] else 1.0)
        factor = 365 / notional
    else:
        raise ValueError(f"Unsupported swap_mode {mode} for {info.get('name')}")

    return long_ * factor, short_ * factor


def currency_rates_from_swaps(meta: dict[str, dict], fx_symbols: list[str]) -> pd.Series:
    """
    Relative annual interest rate per currency, inferred from the broker's
    swaps by least squares: for each pair B/Q, (long - short) / 2 ~ r_B - r_Q
    (the broker's markup appears on both sides and cancels). Rates are only
    defined up to a constant, so they are normalized to mean zero.
    """

    currencies = sorted({meta[s]["currency_base"] for s in fx_symbols} | {meta[s]["currency_profit"] for s in fx_symbols})
    index = {c: i for i, c in enumerate(currencies)}

    rows, targets = [], []
    for symbol in fx_symbols:
        rate_long, rate_short = annual_swap_rates(meta[symbol])
        row = np.zeros(len(currencies))
        row[index[meta[symbol]["currency_base"]]] = 1.0
        row[index[meta[symbol]["currency_profit"]]] = -1.0
        rows.append(row)
        targets.append((rate_long - rate_short) / 2)

    # Sum-to-zero constraint row pins down the free constant.
    rows.append(np.ones(len(currencies)))
    targets.append(0.0)

    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)
    return pd.Series(solution, index=currencies).sort_values(ascending=False)


def carry_currency_weights(rates: pd.Series, n_long: int, n_short: int) -> pd.Series:
    ranked = rates.sort_values(ascending=False)
    weights = pd.Series(0.0, index=ranked.index)
    weights.iloc[:n_long] = 1.0
    weights.iloc[-n_short:] = -1.0
    return weights


def carry_pair_directions(currency_weights: pd.Series, meta: dict[str, dict], fx_symbols: list[str]) -> pd.Series:
    """
    Expresses 'long the top currencies, short the bottom ones' with one USD
    pair per non-USD currency (XXXUSD long = long XXX; USDXXX long = short XXX).
    USD itself needs no leg: its exposure is the net of the other legs.
    """

    directions = {}
    for currency, weight in currency_weights.items():
        if weight == 0 or currency == USD:
            continue

        leg = None
        for symbol in fx_symbols:
            base, quote = meta[symbol]["currency_base"], meta[symbol]["currency_profit"]
            if base == currency and quote == USD:
                leg = (symbol, weight)
            elif base == USD and quote == currency:
                leg = (symbol, -weight)

        if leg is None:
            raise ValueError(f"No USD pair in the universe to trade {currency}")

        directions[leg[0]] = leg[1]

    return pd.Series(directions, dtype=float)


def carry_signal(dates: pd.DatetimeIndex, symbols: list[str], directions: pd.Series) -> pd.DataFrame:
    """Static carry book (the broker only exposes today's swap rates)."""

    signal = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for symbol, direction in directions.items():
        signal[symbol] = direction
    return signal


# ---------------------------------------------------------------------------
# Combination and HMM risk filter
# ---------------------------------------------------------------------------
def combine(signals: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    total = None
    for signal, weight in zip(signals, weights):
        part = signal.fillna(0.0) * weight
        total = part if total is None else total.add(part, fill_value=0.0)

    any_view = sum(s.notna().astype(int) for s in signals) > 0
    return total.where(any_view)


def apply_high_vol_filter(signal: pd.DataFrame, high_vol: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Scale positions down (e.g. halve them) where the symbol's HMM says high-vol."""

    flags = high_vol.reindex_like(signal).fillna(False).astype(bool)
    return signal.where(~flags, signal * scale)

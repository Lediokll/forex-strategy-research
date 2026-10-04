import numpy as np
import pandas as pd
import pytest

import research.data as rdata
import research.regime_filter as rfilter
from research.evaluation import longest_recovery_days, monte_carlo_max_drawdowns, perf_stats, yearly_returns
from research.portfolio import LotRounding, MarketData, run_portfolio, target_weights
from research.settings import CostsConfig, HmmFilterConfig, PortfolioConfig
from research.strategies import (
    annual_swap_rates,
    apply_high_vol_filter,
    carry_currency_weights,
    carry_pair_directions,
    combine,
    currency_rates_from_swaps,
    rebalance_mask,
    trend_signal,
)

PAIRS = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"), "USDJPY": ("USD", "JPY"), "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"), "EURJPY": ("EUR", "JPY"),
}
TRUE_RATES = {"USD": 0.045, "EUR": 0.02, "GBP": 0.04, "AUD": 0.036, "NZD": 0.03,
              "JPY": 0.005, "CHF": 0.0, "CAD": 0.025}


def fx_meta(markup=0.004):
    """Swaps (interest mode, % per year) = rate differential minus a broker markup."""
    meta = {}
    for symbol, (base, quote) in PAIRS.items():
        diff = TRUE_RATES[base] - TRUE_RATES[quote]
        meta[symbol] = {
            "name": symbol, "currency_base": base, "currency_profit": quote, "swap_mode": 5,
            "swap_long": (diff - markup) * 100, "swap_short": (-diff - markup) * 100,
            "price": 1.0, "point": 1e-5, "contract_size": 100_000, "tick_value": 1.0, "tick_size": 1e-5,
        }
    return meta


# --- swaps and carry -----------------------------------------------------

def test_annual_swap_rate_points_mode():
    info = {"swap_mode": 1, "swap_long": -5.0, "swap_short": 2.0, "price": 1.1, "point": 1e-5}
    long_, short_ = annual_swap_rates(info)
    assert long_ == pytest.approx(-5e-5 / 1.1 * 365)
    assert short_ == pytest.approx(2e-5 / 1.1 * 365)


def test_annual_swap_rate_money_mode_for_cfd_uses_price_notional():
    # US500: -0.31 USD per 1-unit lot per night on a 6,000 USD notional.
    info = {"swap_mode": 2, "swap_long": -0.31, "swap_short": -0.26, "price": 6000.0, "contract_size": 1.0,
            "currency_base": "USD", "currency_profit": "USD", "currency_margin": "USD"}
    assert annual_swap_rates(info)[0] == pytest.approx(-0.31 * 365 / 6000)


def test_annual_swap_rate_money_mode_for_fx_uses_contract_units():
    info = {"swap_mode": 2, "swap_long": -2.0, "swap_short": 1.0, "price": 1.1, "contract_size": 100_000.0,
            "currency_base": "EUR", "currency_profit": "USD", "currency_margin": "EUR"}
    assert annual_swap_rates(info)[0] == pytest.approx(-2.0 * 365 / 100_000)


def test_annual_swap_rate_interest_mode():
    assert annual_swap_rates({"swap_mode": 5, "swap_long": -1.5, "swap_short": 0.5, "price": 1.0}) == (-0.015, 0.005)


def test_currency_rates_recovered_from_swaps_despite_markup():
    meta = fx_meta()
    rates = currency_rates_from_swaps(meta, list(meta))
    true = pd.Series(TRUE_RATES)

    # Recovered up to a constant: compare demeaned values.
    assert (rates - (true - true.mean())).abs().max() < 1e-9
    assert list(rates.index[:3]) == ["USD", "GBP", "AUD"]
    assert set(rates.index[-3:]) == {"JPY", "CHF", "EUR"}


def test_carry_legs_long_top_three_short_bottom_three():
    meta = fx_meta()
    weights = carry_currency_weights(currency_rates_from_swaps(meta, list(meta)), 3, 3)
    directions = carry_pair_directions(weights, meta, list(meta))

    # Long GBP, AUD (USD is long implicitly); short JPY, CHF, EUR.
    assert directions.to_dict() == {
        "GBPUSD": 1.0, "AUDUSD": 1.0, "USDJPY": 1.0, "USDCHF": 1.0, "EURUSD": -1.0,
    }


# --- trend and rebalancing --------------------------------------------------

def test_trend_signal_averages_signs():
    closes = pd.DataFrame({"A": [1.0, 2.0, 3.0, 2.5, 2.9]})
    signal = trend_signal(closes, (1, 2, 4))
    assert np.isnan(signal["A"].iloc[3])
    # At t=4: 2.9 vs 2.5 (+), vs 3.0 (-), vs 1.0 (+) -> 1/3
    assert signal["A"].iloc[4] == pytest.approx(1 / 3)


def test_rebalance_mask_first_trading_day():
    dates = pd.bdate_range("2026-03-02", "2026-03-20")       # Mon 2 Mar .. Fri 20 Mar
    weekly = rebalance_mask(dates, "weekly")
    assert list(dates[weekly].day) == [2, 9, 16]
    monthly = rebalance_mask(pd.bdate_range("2026-03-25", "2026-04-07"), "monthly")
    assert monthly.sum() == 2


def test_hmm_filter_halves_only_high_vol():
    signal = pd.DataFrame({"A": [1.0, 1.0], "B": [-1.0, -1.0]})
    flags = pd.DataFrame({"A": [True, False], "B": [False, True]})
    out = apply_high_vol_filter(signal, flags, 0.5)
    assert out.to_dict("list") == {"A": [0.5, 1.0], "B": [-1.0, -0.5]}


def test_combine_keeps_symbols_with_one_view():
    a = pd.DataFrame({"X": [1.0], "Y": [np.nan]})
    b = pd.DataFrame({"X": [-1.0], "Y": [1.0]})
    assert combine([a, b], [0.5, 0.5]).to_dict("list") == {"X": [0.0], "Y": [0.5]}


# --- portfolio engine ----------------------------------------------------

def one_symbol_market(closes, opens=None, spread=0.0002):
    dates = pd.bdate_range("2026-03-02", periods=len(closes))
    close = pd.DataFrame({"X": closes}, index=dates)
    open_ = pd.DataFrame({"X": opens if opens is not None else closes}, index=dates)
    return MarketData(
        dates=dates, symbols=["X"], open=open_, close=close, has_bar=close.notna(),
        conv=pd.DataFrame({"X": 1.0}, index=dates), spread_price=pd.Series({"X": spread}),
        contract_size=pd.Series({"X": 100_000.0}),
        swap_long=pd.Series({"X": -0.0365}), swap_short=pd.Series({"X": 0.0}),
    )


def test_portfolio_accounting_by_hand():
    # Signal +1, vol 10% -> weight 0.2 -> 200 USD notional on 1,000 equity.
    md = one_symbol_market(closes=[1.0, 1.0, 1.1, 1.1], opens=[1.0, 1.0, 1.05, 1.1])
    signal = pd.DataFrame({"X": 1.0}, index=md.dates)
    vol = pd.DataFrame({"X": 0.10}, index=md.dates)
    rebalance = np.array([True, False, False, False])
    pcfg = PortfolioConfig(position_vol_target=0.02, max_gross_risk=0.2)
    costs = CostsConfig(spread_multiplier=1.0, slippage_spread_fraction=0.25, commission_per_lot_usd=7.0)

    res = run_portfolio(md, signal, rebalance, vol, pcfg, costs, md.dates[0], start_equity=1000.0)

    units = 200.0
    cost = units * (0.0002 * 0.75 + 7.0 / 2 / 100_000)            # half spread + slippage + half commission
    swap_day1 = -units * 1.0 * 0.0365 / 365 * 1                    # Mon -> Tue, 1 night
    day1 = units * (1.0 - 1.0) - cost + swap_day1
    swap_day2 = -units * 1.1 * 0.0365 / 365 * 1
    day2 = units * (1.1 - 1.0) + swap_day2

    assert res.units["X"].iloc[1] == pytest.approx(units)
    assert res.equity.iloc[1] == pytest.approx(1000 + day1)
    assert res.equity.iloc[2] == pytest.approx(1000 + day1 + day2)
    assert res.symbol_cost["X"] == pytest.approx(cost)


def test_weekend_swap_counts_three_nights():
    dates = pd.DatetimeIndex(["2026-03-05", "2026-03-06", "2026-03-09"])  # Thu, Fri, Mon
    md = one_symbol_market([1.0, 1.0, 1.0])
    md = MarketData(dates, md.symbols, md.open.set_axis(dates), md.close.set_axis(dates),
                    md.has_bar.set_axis(dates), md.conv.set_axis(dates), md.spread_price,
                    md.contract_size, md.swap_long, md.swap_short)
    res = run_portfolio(md, pd.DataFrame({"X": 1.0}, index=dates), np.array([True, False, False]),
                        pd.DataFrame({"X": 0.10}, index=dates), PortfolioConfig(),
                        CostsConfig(spread_multiplier=0.0), dates[0], start_equity=1000.0)
    assert res.symbol_swap["X"] == pytest.approx(-200 * 0.0365 / 365 * 4)   # 1 + 3 nights


def test_gross_risk_cap_scales_everything():
    signal = np.ones(20)
    vol = np.full(20, 0.10)
    weights = target_weights(signal, vol, PortfolioConfig(position_vol_target=0.02, max_gross_risk=0.20))
    # Uncapped: 20 x 2% = 40% gross risk -> halved.
    assert weights == pytest.approx(np.full(20, 0.1))


def test_target_weights_ignore_missing_inputs():
    weights = target_weights(np.array([1.0, np.nan, 1.0]), np.array([0.1, 0.1, np.nan]), PortfolioConfig())
    assert list(weights) == [pytest.approx(0.2), 0.0, 0.0]


def test_lot_rounding_policies():
    lots = LotRounding(lot_units=np.array([1000.0, 1000.0, 1000.0]), policy="nearest")
    assert list(lots.apply(np.array([400.0, 600.0, -1600.0]))) == [0.0, 1000.0, -2000.0]
    lots.policy = "at_least_min"
    assert list(lots.apply(np.array([400.0, 0.0, -100.0]))) == [1000.0, 0.0, -1000.0]


# --- evaluation ------------------------------------------------------------

def test_longest_recovery_and_yearly_returns():
    idx = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-06-01", "2025-01-01", "2025-03-01"])
    equity = pd.Series([100, 110, 90, 111, 105], index=idx)

    days, recovered = longest_recovery_days(equity)
    assert days == (idx[3] - idx[1]).days and recovered

    yearly = yearly_returns(equity)
    assert yearly[2024] == pytest.approx(90 / 100 - 1)
    assert yearly[2025] == pytest.approx(105 / 90 - 1)

    stats = perf_stats(equity)
    assert stats["max_dd"] == pytest.approx(20 / 110)


def test_monte_carlo_reshuffle_keeps_final_equity():
    returns = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    dds = monte_carlo_max_drawdowns(returns, runs=200, seed=1)
    assert dds.min() >= 0.2 - 1e-12          # the -20% week alone is a 20% drawdown
    assert dds.max() <= 1 - 0.8 * 0.9 + 1e-12  # worst: -20% then -10% back to back


# --- holdout and look-ahead -----------------------------------------------

def test_holdout_start_is_twelve_months_before_last_bar():
    meta = {"symbols": {"A": {"d1_last": "2026-10-02"}, "B": {"d1_last": "2026-09-30"}}}
    assert rdata.holdout_start(meta, 12) == pd.Timestamp("2025-10-02")


def test_development_data_excludes_the_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(rdata, "DATA_DIR", tmp_path)
    dates = pd.bdate_range("2024-01-01", "2026-10-02")
    pd.DataFrame({"datetime": dates, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                  "volume": 1, "spread": 1, "real_volume": 0}).to_csv(tmp_path / "EURUSD_D1.csv", index=False)
    meta = {"EURUSD": {"currency_base": "EUR", "currency_profit": "USD", "point": 1e-5, "contract_size": 1e5,
                       "spread_snapshot_points": 5, "swap_mode": 0, "swap_long": 0, "swap_short": 0,
                       "price": 1.0, "tick_value": 1.0, "tick_size": 1e-5}}

    cutoff = pd.Timestamp("2025-10-02")
    md = rdata.build_market_data(meta, ["EURUSD"], cutoff)
    assert md.dates.max() < cutoff


def test_h4_history_starts_where_bars_are_really_intraday():
    daily_era = pd.bdate_range("1997-01-01", "1998-12-31")                       # ~260 bars/year
    h4_era = pd.date_range("1999-01-01", "1999-12-31", freq="4h")               # ~2,190 bars
    bars = pd.DataFrame({"datetime": daily_era.append(h4_era)})

    trimmed = rfilter.intraday_h4_only(bars)
    assert trimmed["datetime"].min() == pd.Timestamp("1999-01-01")


def test_coverage_counts_history_that_stops_early_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rdata, "DATA_DIR", tmp_path)
    dates = pd.bdate_range("2020-01-01", "2020-12-31")          # data stops after one year
    pd.DataFrame({"datetime": dates, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
                  "spread": 1, "real_volume": 0}).to_csv(tmp_path / "UK100_D1.csv", index=False)

    assert rdata.coverage("UK100", None, pd.Timestamp("2020-12-31")) == pytest.approx(1.0)
    assert rdata.coverage("UK100", None, pd.Timestamp("2023-12-29")) < 0.4


def test_daily_hmm_flag_uses_only_closed_h4_bars(monkeypatch):
    # H4 bar opening 2026-03-02 20:00 closes at 03-03 00:00 = close of server day 03-02.
    labels = pd.DataFrame({
        "close_time": pd.to_datetime(["2026-03-02 20:00", "2026-03-03 00:00", "2026-03-03 04:00"]),
        "label": ["trend", "high_vol", "trend"],
    })
    monkeypatch.setattr(rfilter, "h4_labels", lambda *a, **k: labels)
    dates = pd.DatetimeIndex(["2026-03-02", "2026-03-03"])

    flags, ready = rfilter.daily_high_vol_flags(["X"], dates, HmmFilterConfig(), None)

    # Day 03-02 decides at 03-03 00:00: last closed bar says high_vol.
    # Day 03-03 decides at 03-04 00:00: last closed bar (04:00) says trend.
    assert flags["X"].tolist() == [True, False]
    assert ready["X"] == pd.Timestamp("2026-03-02")

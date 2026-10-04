"""Hand-checked fills and risk rules in the backtest simulator."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import PreparedData, SimParams, Simulator
from src.backtest.metrics import longest_losing_streak, max_drawdown, summarize
from src.config import CostConfig, RiskConfig, StrategyConfig
from src.data.symbol_spec import DEFAULT_EURUSD_SPEC

COSTS = CostConfig(spread_floor_pips=1.0, slippage_pips=0.2, commission_per_lot_usd=0.0)


def make_sim(bars, setups, atr=0.0006, start_balance=10.0, risk=None):
    """bars: (open, high, low, close); setups: {bar index: +1/-1}."""

    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"])
    df["atr"] = atr
    df["spread"] = 0
    df["time_utc"] = pd.date_range("2026-05-04 08:00", periods=len(df), freq="15min")
    df["ema_20"] = df["ema_50"] = df["ema_200"] = 1.0     # flat: no organic setups

    prep = PreparedData(df=df, regimes=np.array(["trend"] * len(df), dtype=object), first_idx=0)
    params = SimParams(StrategyConfig(), risk or RiskConfig(), COSTS, True, start_balance, 500.0)
    sim = Simulator(prep, DEFAULT_EURUSD_SPEC, params)

    sim.setups[:] = 0
    for i, direction in setups.items():
        sim.setups[i] = direction
    sim.setup_idx = np.flatnonzero(sim.setups)
    sim.strategy_reasons = {int(i): [] for i in sim.setup_idx}
    return sim


def test_long_take_profit_pays_two_r():
    # Entry: ask = 1.1000 + 1.0 pip spread + 0.2 pip slip = 1.10012
    # SL = 1.10012 - 9 pips = 1.09922, TP = 1.10012 + 18 pips = 1.10192
    sim = make_sim([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.1025, 1.0995, 1.1020)], {0: 1})
    trade = sim.run().trades.iloc[0]

    assert trade["entry_price"] == pytest.approx(1.10012)
    assert trade["exit_reason"] == "tp"
    assert trade["pnl_usd"] == pytest.approx(1.80)
    assert trade["r_multiple"] == pytest.approx(2.0)
    assert trade["equity_after"] == pytest.approx(11.80)


def test_long_stop_loss_pays_slippage():
    sim = make_sim([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.1005, 1.0980, 1.0985)], {0: 1})
    trade = sim.run().trades.iloc[0]

    assert trade["exit_reason"] == "sl"
    assert trade["exit_price"] == pytest.approx(1.09920)     # SL 1.09922 - 0.2 pip
    assert trade["pnl_usd"] == pytest.approx(-0.92)
    assert trade["r_multiple"] == pytest.approx(-0.92 / 0.90)


def test_sl_and_tp_in_the_same_bar_counts_as_loss():
    sim = make_sim([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.1030, 1.0980, 1.1000)], {0: 1})
    assert sim.run().trades.iloc[0]["exit_reason"] == "sl"


def test_short_stop_triggers_on_the_ask():
    # Short at bid 1.1000 - 0.2 slip = 1.09998, SL = 1.10088.
    # Bid high 1.10080 never reaches the SL, but ask = bid + 1 pip = 1.10090 does.
    sim = make_sim([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.10080, 1.0995, 1.1000)], {0: -1})
    trade = sim.run().trades.iloc[0]

    assert trade["exit_reason"] == "sl"
    assert trade["exit_price"] == pytest.approx(1.10088 + 0.00002)


def losing_day_bars(n):
    # Every bar opens at 1.1000 and dips to 1.0980: each long entry is stopped out.
    return [(1.1000, 1.1001, 1.0980, 1.0990)] * n


def test_kill_switch_stops_the_account():
    sim = make_sim(losing_day_bars(4), {0: 1, 1: 1, 2: 1}, start_balance=6.5)
    result = sim.run(equity_rules=True)

    assert result.killed_at is not None
    assert len(result.trades) == 1
    assert result.final_equity <= 6.0


def test_without_equity_rules_the_strategy_keeps_trading():
    sim = make_sim(losing_day_bars(4), {0: 1, 1: 1, 2: 1}, start_balance=6.5)
    result = sim.run(equity_rules=False)

    assert result.killed_at is None
    assert len(result.trades) == 3


def test_daily_loss_cap_blocks_third_trade():
    # ATR 0.00066 -> 9.9 pip stop, $0.99 risk, each loss $1.01 -> -$2.02 after two.
    sim = make_sim(losing_day_bars(4), {0: 1, 1: 1, 2: 1}, atr=0.00066)
    result = sim.run(equity_rules=False)

    assert len(result.trades) == 2
    assert result.signals["reason"].tolist() == ["taken", "taken", "daily_loss_cap"]


def test_risk_over_one_dollar_is_skipped_and_logged():
    # ATR 0.0008 -> 12 pip stop -> $1.20 at 0.01 lot.
    sim = make_sim(losing_day_bars(3), {0: 1}, atr=0.0008)
    result = sim.run()

    assert result.trades.empty
    assert result.signals.iloc[0]["reason"].startswith("risk_$1.200_over_$1.00")


def test_max_trades_per_day():
    risk = RiskConfig(daily_loss_cap_usd=100.0, max_trades_per_day=3)
    sim = make_sim(losing_day_bars(6), {0: 1, 1: 1, 2: 1, 3: 1}, risk=risk)
    result = sim.run(equity_rules=False)

    assert len(result.trades) == 3
    assert result.signals.iloc[-1]["reason"] == "max_trades_per_day"


def test_metrics():
    pnl = np.array([1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0])
    trades = pd.DataFrame({"pnl_usd": pnl, "r_multiple": pnl})
    stats = summarize(trades)

    assert stats["trades"] == 7
    assert stats["profit_factor"] == pytest.approx(3.0 / 5.0)
    assert stats["longest_losing_streak"] == longest_losing_streak(pnl) == 3
    assert stats["max_dd_usd"] == pytest.approx(3.0)
    assert max_drawdown(np.cumsum(pnl)) == pytest.approx(3.0)

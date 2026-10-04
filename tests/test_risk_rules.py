from dataclasses import replace
from datetime import date

import pytest

from src.config import RiskConfig
from src.data.symbol_spec import DEFAULT_EURUSD_SPEC
from src.risk.rules import (
    AccountSnapshot,
    DailyLedger,
    ProfitAlert,
    RiskManager,
    dollar_risk,
    estimate_margin,
    lot_size,
    margin_warning,
    scale_margin_to_leverage,
)

SPEC = DEFAULT_EURUSD_SPEC
PIP = 0.0001


def healthy_snapshot(**overrides):
    values = dict(equity=10.0, open_positions=0, trades_today=0, realized_pnl_today=0.0)
    values.update(overrides)
    return AccountSnapshot(**values)


@pytest.fixture
def risk():
    return RiskManager(RiskConfig())


# --- lot size -------------------------------------------------------------

def test_lot_size_uses_configured_lot_when_allowed():
    assert lot_size(SPEC, 0.01) == 0.01


def test_lot_size_raised_to_symbol_minimum():
    spec = replace(SPEC, volume_min=0.1, volume_step=0.1)
    assert lot_size(spec, 0.01) == pytest.approx(0.1)


def test_lot_size_snaps_up_to_volume_step():
    spec = replace(SPEC, volume_min=0.02, volume_step=0.02)
    assert lot_size(spec, 0.03) == pytest.approx(0.04)


def test_lot_size_above_symbol_maximum_raises():
    with pytest.raises(ValueError):
        lot_size(replace(SPEC, volume_max=0.005, volume_min=0.001, volume_step=0.001), 0.01)


# --- $ risk ---------------------------------------------------------------

def test_dollar_risk_ten_pips_at_001_lot_is_one_dollar():
    assert dollar_risk(10 * PIP, 0.01, SPEC) == pytest.approx(1.00)


def test_dollar_risk_reads_pip_value_from_symbol_spec():
    # A non-USD account where one tick is worth 0.85 instead of 1.0.
    spec = replace(SPEC, tick_value=0.85)
    assert dollar_risk(10 * PIP, 0.01, spec) == pytest.approx(0.85)
    assert spec.pip_value(0.01) == pytest.approx(0.085)


def test_dollar_risk_rejects_non_positive_stop():
    with pytest.raises(ValueError):
        dollar_risk(0.0, 0.01, SPEC)


def test_entry_allowed_within_all_limits(risk):
    decision = risk.check_entry(healthy_snapshot(), risk_usd=0.75)
    assert decision.allowed
    assert decision.reasons == []


def test_entry_skipped_when_dollar_risk_over_one_dollar(risk):
    decision = risk.check_entry(healthy_snapshot(), risk_usd=1.01)
    assert not decision.allowed
    assert decision.reason.startswith("risk_$1.010_over_$1.00")


def test_entry_allowed_at_exactly_one_dollar(risk):
    assert risk.check_entry(healthy_snapshot(), risk_usd=1.00).allowed


def test_entry_skipped_when_risk_exceeds_equity():
    manager = RiskManager(RiskConfig(kill_switch_equity=0.1))
    decision = manager.check_entry(healthy_snapshot(equity=0.5), risk_usd=0.75)
    assert "risk_exceeds_equity" in decision.reasons


# --- position and trade limits --------------------------------------------

def test_max_one_open_position(risk):
    decision = risk.check_entry(healthy_snapshot(open_positions=1), risk_usd=0.5)
    assert not decision.allowed
    assert "max_open_positions" in decision.reasons


def test_max_three_trades_per_day(risk):
    assert risk.check_entry(healthy_snapshot(trades_today=2), 0.5).allowed
    decision = risk.check_entry(healthy_snapshot(trades_today=3), 0.5)
    assert not decision.allowed
    assert "max_trades_per_day" in decision.reasons


# --- daily loss cap -------------------------------------------------------

def test_daily_loss_cap_blocks_at_minus_two_dollars(risk):
    decision = risk.check_entry(healthy_snapshot(realized_pnl_today=-2.00), 0.5)
    assert not decision.allowed
    assert "daily_loss_cap" in decision.reasons


def test_daily_loss_cap_not_hit_just_above(risk):
    assert risk.check_entry(healthy_snapshot(realized_pnl_today=-1.99), 0.5).allowed


def test_daily_ledger_resets_on_next_day(risk):
    ledger = DailyLedger()
    ledger.record_entry(date(2026, 3, 2))
    ledger.record_exit(date(2026, 3, 2), -1.10)
    ledger.record_entry(date(2026, 3, 2))
    ledger.record_exit(date(2026, 3, 2), -1.05)

    assert ledger.trades == 2
    assert risk.daily_loss_cap_hit(ledger.realized_pnl)

    assert ledger.roll(date(2026, 3, 3)) is True
    assert ledger.trades == 0
    assert ledger.realized_pnl == 0.0
    assert not risk.daily_loss_cap_hit(ledger.realized_pnl)


def test_daily_ledger_books_pnl_on_the_exit_day():
    ledger = DailyLedger()
    ledger.record_entry(date(2026, 3, 2))           # opened Monday
    ledger.record_exit(date(2026, 3, 3), -2.5)      # stopped out Tuesday

    assert ledger.day == date(2026, 3, 3)
    assert ledger.realized_pnl == -2.5
    assert ledger.trades == 0                       # Tuesday's entries


# --- kill switch ----------------------------------------------------------

def test_kill_switch_triggers_at_six_dollars(risk):
    assert risk.kill_switch_triggered(6.00)
    assert risk.kill_switch_triggered(5.20)
    assert not risk.kill_switch_triggered(6.01)


def test_entry_blocked_once_kill_level_reached(risk):
    decision = risk.check_entry(healthy_snapshot(equity=6.0), 0.5)
    assert not decision.allowed
    assert "kill_switch" in decision.reasons


# --- profit alert ---------------------------------------------------------

def test_profit_alert_fires_once_per_day_at_thirty():
    alert = ProfitAlert(30.0)
    day = date(2026, 5, 1)

    assert not alert.check(29.99, day)
    assert alert.check(30.00, day)
    assert not alert.check(31.00, day)
    assert alert.check(30.50, date(2026, 5, 2))


# --- margin ---------------------------------------------------------------

def test_margin_estimate_for_001_lot_at_500_leverage():
    # 0.01 x 100,000 x 1.15 / 500 = $2.30
    assert estimate_margin(0.01, 100_000, 1.15, 500) == pytest.approx(2.30)


def test_margin_estimate_at_eu_retail_leverage_exceeds_ten_dollars():
    margin = estimate_margin(0.01, 100_000, 1.15, 30)
    assert margin == pytest.approx(38.33, abs=0.01)
    assert margin_warning(margin, free_margin=10.0) is not None


def test_margin_warning_none_when_affordable():
    assert margin_warning(2.30, free_margin=10.0) is None


def test_scale_broker_margin_to_virtual_leverage():
    # Broker reports $11.50 at the real 1:100; at 1:500 that is $2.30.
    assert scale_margin_to_leverage(11.50, 100, 500) == pytest.approx(2.30)

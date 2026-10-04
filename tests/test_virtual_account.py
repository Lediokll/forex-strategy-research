import json
import logging
from datetime import datetime, timezone

import pytest

from src.config import RiskConfig
from src.risk.kill_switch import KillSwitch
from src.risk.rules import RiskManager
from src.risk.virtual_account import VirtualAccount, VirtualAccountError

T0 = datetime(2026, 10, 5, 7, 0, tzinfo=timezone.utc)
LOGIN = 12345678          # fake placeholder, not a real account
MAGIC = 20260519


def create(path, balance=10.0, login=LOGIN, symbol="EURUSD", magic=MAGIC):
    return VirtualAccount.load_or_create(path, balance, login, symbol, magic, now_utc=T0)


def test_new_account_starts_at_configured_balance(tmp_path):
    account = create(tmp_path / "va.json")

    assert account.start_balance == 10.0
    assert account.started_at == T0
    assert (tmp_path / "va.json").exists()


def test_equity_is_virtual_balance_plus_realized_plus_open(tmp_path):
    account = create(tmp_path / "va.json")
    assert account.equity(realized_pnl=-1.25, open_pnl=0.40) == pytest.approx(9.15)


def test_real_balance_never_enters_equity(tmp_path):
    # A $100k demo account: decisions see only the virtual equity.
    account = create(tmp_path / "va.json")
    risk = RiskManager(RiskConfig())

    virtual_equity = account.equity(realized_pnl=-4.0, open_pnl=0.0)

    assert virtual_equity == pytest.approx(6.0)
    assert risk.kill_switch_triggered(virtual_equity)


def test_restart_keeps_start_time_and_balance(tmp_path):
    path = tmp_path / "va.json"
    first = create(path)
    first.update_realized(-1.5)

    later = datetime(2026, 10, 9, tzinfo=timezone.utc)
    reloaded = VirtualAccount.load_or_create(path, 10.0, LOGIN, "EURUSD", MAGIC, now_utc=later)

    assert reloaded.started_at == T0
    assert reloaded.start_balance == 10.0
    assert reloaded.state.cached_realized_pnl == pytest.approx(-1.5)


def test_changed_config_balance_warns_and_keeps_saved_value(tmp_path, caplog):
    path = tmp_path / "va.json"
    create(path, balance=10.0)

    with caplog.at_level(logging.WARNING):
        reloaded = create(path, balance=50.0)

    assert reloaded.start_balance == 10.0
    assert "Keeping the saved value" in caplog.text


def test_other_account_login_refuses(tmp_path):
    path = tmp_path / "va.json"
    create(path)

    with pytest.raises(VirtualAccountError, match="belongs to account") as error:
        create(path, login=87654321)

    # Logins are masked in messages (they end up in logs).
    assert str(LOGIN) not in str(error.value)
    assert "87654321" not in str(error.value)
    assert "*****678" in str(error.value)


def test_other_magic_or_symbol_refuses(tmp_path):
    path = tmp_path / "va.json"
    create(path)

    with pytest.raises(VirtualAccountError):
        create(path, magic=1)
    with pytest.raises(VirtualAccountError):
        create(path, symbol="GBPUSD")


def test_corrupt_state_file_refuses(tmp_path):
    path = tmp_path / "va.json"
    path.write_text("{not json")

    with pytest.raises(VirtualAccountError, match="unreadable"):
        create(path)


def test_reset_moves_state_aside_and_next_start_is_fresh(tmp_path):
    path = tmp_path / "va.json"
    create(path).update_realized(-3.0)

    backup = VirtualAccount.reset(path, now_utc=T0)

    assert backup is not None and backup.exists()
    assert not path.exists()
    assert json.loads(backup.read_text())["cached_realized_pnl"] == -3.0
    assert create(path).state.cached_realized_pnl == 0.0


def test_reset_without_state_file_is_a_noop(tmp_path):
    assert VirtualAccount.reset(tmp_path / "missing.json") is None


# --- persistent kill switch ----------------------------------------------

def test_kill_switch_file_persists_across_restarts(tmp_path):
    path = tmp_path / "KILLED"
    KillSwitch(path).trip("equity $5.90 <= $6.00", equity=5.9, now_utc=T0)

    restarted = KillSwitch(path)

    assert restarted.is_tripped()
    assert restarted.details()["equity"] == 5.9
    assert "5.90" in restarted.details()["reason"]


def test_kill_switch_not_tripped_by_default(tmp_path):
    assert not KillSwitch(tmp_path / "KILLED").is_tripped()

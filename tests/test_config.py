import pytest

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, ConfigError, display_path, load_config, mask_login


def write(tmp_path, text, name="config.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_project_config_loads_with_requested_defaults():
    # Explicit path: a developer's private config.local.yaml must not affect this.
    cfg = load_config(DEFAULT_CONFIG_PATH)

    assert cfg.mode == "demo"
    assert cfg.strategy.timeframe == "M15"
    assert cfg.strategy.sl_atr_mult == 1.5
    assert cfg.strategy.tp_r_mult == 2.0
    assert cfg.strategy.max_spread_pips == 1.5
    assert cfg.risk.lot_size == 0.01
    assert cfg.risk.max_risk_usd == 1.0
    assert cfg.risk.max_open_positions == 1
    assert cfg.risk.max_trades_per_day == 3
    assert cfg.risk.daily_loss_cap_usd == 2.0
    assert cfg.risk.kill_switch_equity == 6.0
    assert cfg.risk.profit_alert_equity == 30.0
    assert cfg.virtual_account.virtual_balance == 10.0
    assert cfg.virtual_account.leverage == 500
    assert cfg.live.enabled is False


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(write(tmp_path, "risk:\n  max_risk_ust: 5\n"))


def test_kill_level_must_be_below_alert_level(tmp_path):
    text = "risk:\n  kill_switch_equity: 30\n  profit_alert_equity: 30\n"
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_virtual_balance_can_be_disabled(tmp_path):
    cfg = load_config(write(tmp_path, "virtual_account:\n  virtual_balance: null\n"))
    assert not cfg.virtual_account.enabled


def test_mode_override_and_validation(tmp_path):
    path = write(tmp_path, "mode: demo\n")
    assert load_config(path, {"mode": "backtest"}).mode == "backtest"

    with pytest.raises(ConfigError):
        load_config(path, {"mode": "yolo"})


def test_local_config_overrides_nested_values(tmp_path):
    base = write(tmp_path, "live:\n  enabled: false\n  account_number: null\n")
    local = write(tmp_path, "live:\n  account_number: 12345678\n", "config.local.yaml")

    cfg = load_config(base, local_path=local)

    assert cfg.live.account_number == 12345678
    assert cfg.live.enabled is False          # untouched keys survive the merge


def test_local_config_example_is_valid_and_has_only_placeholders():
    example = PROJECT_ROOT / "config.local.example.yaml"
    cfg = load_config(DEFAULT_CONFIG_PATH, local_path=example)
    assert cfg.live.account_number == 12345678


def test_mask_login_keeps_last_three_digits():
    assert mask_login(12345678) == "*****678"
    assert mask_login(42) == "42"


def test_display_path_is_project_relative():
    shown = display_path(PROJECT_ROOT / "logs" / "bot.log")
    assert shown in ("logs/bot.log", "logs\\bot.log")
    assert str(PROJECT_ROOT) not in display_path(PROJECT_ROOT / "config.yaml")


def test_wrong_type_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "live:\n  enabled: 'yes'\n"))

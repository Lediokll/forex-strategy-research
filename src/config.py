"""
Central configuration for the forex bot.

All tunable values live in config.yaml at the project root. This module
loads them into frozen, typed dataclasses, rejects unknown keys (so a typo
can't silently fall back to a default), and validates the values that
matter for safety. Paths and logging setup also live here so the rest of
the codebase can `from src import config`.
"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths (absolute, independent of the current working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# Optional, gitignored overrides for values that must never be published
# (e.g. live.account_number). See config.local.example.yaml.
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "state"
RESULTS_DIR = PROJECT_ROOT / "results"

LOG_FILE = LOGS_DIR / "bot.log"
KILL_SWITCH_FILE = STATE_DIR / "KILLED"

MODES = ("backtest", "demo", "live")
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
DST_RULES = ("us", "eu", "none")


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    timeframe: str = "M15"
    ema_pullback: int = 20
    ema_trend_fast: int = 50
    ema_trend_slow: int = 200
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_r_mult: float = 2.0
    pullback_bars: int = 2
    session_start_utc: str = "07:00"
    session_end_utc: str = "17:00"
    max_spread_pips: float = 1.5

    @property
    def bar_minutes(self) -> int:
        return TIMEFRAME_MINUTES[self.timeframe]


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = True
    n_states: int = 3
    feature_window: int = 16
    train_bars: int = 25000
    refit_every_bars: int = 2000
    filter_burn_in_bars: int = 500
    n_iter: int = 200
    random_state: int = 42


@dataclass(frozen=True)
class RiskConfig:
    lot_size: float = 0.01
    max_risk_usd: float = 1.0
    max_open_positions: int = 1
    max_trades_per_day: int = 3
    daily_loss_cap_usd: float = 2.0
    kill_switch_equity: float = 6.0
    profit_alert_equity: float = 30.0


@dataclass(frozen=True)
class VirtualAccountConfig:
    virtual_balance: typing.Optional[float] = 10.0
    leverage: float = 500.0
    state_file: str = "state/virtual_account.json"

    @property
    def enabled(self) -> bool:
        return self.virtual_balance is not None

    @property
    def state_path(self) -> Path:
        path = Path(self.state_file)
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class BrokerTimeConfig:
    base_utc_offset_hours: int = 2
    dst_rule: str = "us"


@dataclass(frozen=True)
class CostConfig:
    spread_floor_pips: float = 1.0
    slippage_pips: float = 0.2
    commission_per_lot_usd: float = 0.0


@dataclass(frozen=True)
class BacktestConfig:
    history_bars: int = 99000
    start_balance: float = 10.0
    costs: CostConfig = field(default_factory=CostConfig)
    pessimistic_costs: CostConfig = field(
        default_factory=lambda: CostConfig(1.5, 0.2, 7.0)
    )
    grid_sl_atr_mult: tuple[float, ...] = (1.0, 1.5, 2.0)
    grid_tp_r_mult: tuple[float, ...] = (1.5, 2.0)


@dataclass(frozen=True)
class LiveConfig:
    enabled: bool = False
    account_number: typing.Optional[int] = None
    deviation_points: int = 20
    max_order_retries: int = 3
    loop_seconds: int = 10


@dataclass(frozen=True)
class Config:
    mode: str = "demo"
    symbol: str = "EURUSD"
    magic_number: int = 20260519
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    virtual_account: VirtualAccountConfig = field(default_factory=VirtualAccountConfig)
    broker_time: BrokerTimeConfig = field(default_factory=BrokerTimeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _coerce(value, type_hint, key: str):
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    if origin is typing.Union:
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{key}: null is not allowed")
        non_none = [a for a in args if a is not type(None)]
        return _coerce(value, non_none[0], key)

    if value is None:
        raise ConfigError(f"{key}: null is not allowed")

    if is_dataclass(type_hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{key}: expected a mapping")
        return _build(type_hint, value, key)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{key}: expected a list")
        return tuple(_coerce(v, args[0], key) for v in value)

    if type_hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: expected true/false, got {value!r}")
        return value

    if type_hint in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key}: expected a number, got {value!r}")
        if type_hint is int and float(value) != int(value):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        return type_hint(value)

    if type_hint is str:
        return str(value)

    return value


def _build(cls, data: dict, prefix: str = ""):
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known

    if unknown:
        raise ConfigError(f"Unknown config key(s) under '{prefix or 'root'}': {sorted(unknown)}")

    kwargs = {}
    for name, value in data.items():
        key = f"{prefix}.{name}" if prefix else name
        kwargs[name] = _coerce(value, hints[name], key)

    return cls(**kwargs)


def build_dataclass(cls, data: dict, prefix: str = ""):
    """Public entry point to the typed loader (unknown keys are rejected).
    Lets other config files (e.g. research/config.yaml) reuse it."""

    return _build(cls, data, prefix)


def _parse_hhmm(value: str, key: str) -> int:
    try:
        hours, minutes = value.split(":")
        total = int(hours) * 60 + int(minutes)
    except ValueError as error:
        raise ConfigError(f"{key}: expected HH:MM, got {value!r}") from error

    if not 0 <= total <= 24 * 60:
        raise ConfigError(f"{key}: out of range: {value!r}")

    return total


def validate(cfg: Config) -> Config:
    s, r, k, v = cfg.strategy, cfg.regime, cfg.risk, cfg.virtual_account

    if cfg.mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {cfg.mode!r}")
    if s.timeframe not in TIMEFRAME_MINUTES:
        raise ConfigError(f"strategy.timeframe must be one of {list(TIMEFRAME_MINUTES)}")
    if not s.ema_pullback < s.ema_trend_fast < s.ema_trend_slow:
        raise ConfigError("strategy EMAs must satisfy ema_pullback < ema_trend_fast < ema_trend_slow")
    if s.sl_atr_mult <= 0 or s.tp_r_mult <= 0 or s.pullback_bars < 1:
        raise ConfigError("strategy sl_atr_mult/tp_r_mult must be > 0 and pullback_bars >= 1")
    if _parse_hhmm(s.session_start_utc, "session_start_utc") >= _parse_hhmm(s.session_end_utc, "session_end_utc"):
        raise ConfigError("strategy.session_start_utc must be before session_end_utc")
    if r.n_states < 2:
        raise ConfigError("regime.n_states must be >= 2")
    if r.train_bars < 1000 or r.refit_every_bars < 1 or r.filter_burn_in_bars < 0:
        raise ConfigError("regime.train_bars must be >= 1000 and refit_every_bars >= 1")
    if k.lot_size <= 0 or k.max_risk_usd <= 0:
        raise ConfigError("risk.lot_size and risk.max_risk_usd must be > 0")
    if k.max_open_positions < 1 or k.max_trades_per_day < 1 or k.daily_loss_cap_usd <= 0:
        raise ConfigError("risk position/trade/daily-loss limits must be positive")
    if not 0 < k.kill_switch_equity < k.profit_alert_equity:
        raise ConfigError("risk: need 0 < kill_switch_equity < profit_alert_equity")
    if v.virtual_balance is not None and v.virtual_balance <= k.kill_switch_equity:
        raise ConfigError("virtual_account.virtual_balance must be above risk.kill_switch_equity")
    if v.leverage <= 0:
        raise ConfigError("virtual_account.leverage must be > 0")
    if cfg.broker_time.dst_rule not in DST_RULES:
        raise ConfigError(f"broker_time.dst_rule must be one of {DST_RULES}")

    return cfg


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"{display_path(path)}: top level must be a mapping")

    return data


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: Path | str | None = None,
    overrides: dict | None = None,
    local_path: Path | None = None,
) -> Config:
    """Load config.yaml (or `path`) into a validated Config.

    When loading the default config, config.local.yaml (gitignored) is
    merged on top if it exists, so private values never live in a committed
    file. `overrides` is a flat {"mode": "backtest"}-style dict applied last,
    used for CLI flags.
    """

    explicit = path is not None
    path = Path(path) if explicit else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise ConfigError(f"Config file not found: {display_path(path)}")

    data = _read_yaml(path)

    local_path = local_path or (None if explicit else LOCAL_CONFIG_PATH)
    if local_path is not None and Path(local_path).exists():
        data = _deep_merge(data, _read_yaml(Path(local_path)))

    data.update(overrides or {})

    return validate(_build(Config, data))


def session_minutes(strategy: StrategyConfig) -> tuple[int, int]:
    """Session window as (start, end) minutes after 00:00 UTC."""

    return (
        _parse_hhmm(strategy.session_start_utc, "session_start_utc"),
        _parse_hhmm(strategy.session_end_utc, "session_end_utc"),
    )


# ---------------------------------------------------------------------------
# Privacy helpers for log and error messages
# ---------------------------------------------------------------------------
def display_path(path: Path | str) -> str:
    """Path relative to the project root when possible, so logs and error
    messages don't contain the user's home directory."""

    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def mask_login(login) -> str:
    """Account login with all but the last 3 digits hidden: ****110."""

    text = str(login)
    return "*" * max(len(text) - 3, 0) + text[-3:]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once: console + rotating file handler."""

    root = logging.getLogger()

    if root.handlers:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

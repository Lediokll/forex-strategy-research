"""Typed research settings, loaded with the same strict loader as the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.config import RegimeConfig, build_dataclass

RESEARCH_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = RESEARCH_DIR / "config.yaml"
DATA_DIR = RESEARCH_DIR / "data"
RESULTS_DIR = RESEARCH_DIR / "results"
HOLDOUT_MARKER = RESULTS_DIR / "HOLDOUT_USED.json"


@dataclass(frozen=True)
class UniverseConfig:
    fx: tuple[str, ...] = ()
    other: tuple[str, ...] = ()
    indices: tuple[str, ...] = ()

    @property
    def symbols(self) -> list[str]:
        return list(self.fx) + list(self.other) + list(self.indices)


@dataclass(frozen=True)
class DataConfig:
    daily_bars: int = 99999
    h4_bars: int = 99999
    min_coverage: float = 0.90


@dataclass(frozen=True)
class PortfolioConfig:
    vol_lookback_days: int = 60
    position_vol_target: float = 0.02
    max_gross_risk: float = 0.20
    trading_days_per_year: int = 252


@dataclass(frozen=True)
class CostsConfig:
    spread_floor: dict = field(default_factory=dict)   # symbol -> price units
    spread_multiplier: float = 1.0
    slippage_spread_fraction: float = 0.25
    commission_per_lot_usd: float = 0.0
    pessimistic_spread_multiplier: float = 2.0


@dataclass(frozen=True)
class TrendConfig:
    lookbacks_days: tuple[int, ...] = (63, 126, 252)
    rebalance: str = "weekly"


@dataclass(frozen=True)
class CarryConfig:
    n_long: int = 3
    n_short: int = 3
    rebalance: str = "monthly"


@dataclass(frozen=True)
class ComboConfig:
    trend_weight: float = 0.5
    carry_weight: float = 0.5


@dataclass(frozen=True)
class HmmFilterConfig:
    high_vol_scale: float = 0.5
    n_states: int = 3
    feature_window: int = 30
    train_bars: int = 4500
    refit_every_bars: int = 500
    filter_burn_in_bars: int = 300
    n_iter: int = 200
    random_state: int = 42

    def regime_config(self) -> RegimeConfig:
        return RegimeConfig(
            enabled=True,
            n_states=self.n_states,
            feature_window=self.feature_window,
            train_bars=self.train_bars,
            refit_every_bars=self.refit_every_bars,
            filter_burn_in_bars=self.filter_burn_in_bars,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    neighborhood_scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    monte_carlo_runs: int = 5000
    monte_carlo_seed: int = 7


@dataclass(frozen=True)
class CentAccountConfig:
    start_equity_usd: float = 10.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    contract_size_multiplier: float = 0.01
    leverage: float = 500.0


@dataclass(frozen=True)
class ResearchConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    holdout_months: int = 12
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    carry: CarryConfig = field(default_factory=CarryConfig)
    combo: ComboConfig = field(default_factory=ComboConfig)
    hmm_filter: HmmFilterConfig = field(default_factory=HmmFilterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cent_account: CentAccountConfig = field(default_factory=CentAccountConfig)


def load_research_config(path: Path | None = None) -> ResearchConfig:
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = build_dataclass(ResearchConfig, data)

    if cfg.trend.rebalance != "weekly" or cfg.carry.rebalance != "monthly":
        raise ValueError("Only weekly trend / monthly carry rebalancing is implemented.")

    return cfg

"""Broker contract specification for a symbol, independent of MT5.

Live code builds it from mt5.symbol_info(); the backtest loads the snapshot
saved next to the downloaded history, so both price risk the same way.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    digits: int
    point: float
    tick_size: float
    tick_value: float          # account currency per tick per 1.0 lot
    contract_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    stops_level: int = 0       # points
    freeze_level: int = 0      # points

    @property
    def pip_size(self) -> float:
        """One pip in price units (0.0001 on a 5-digit EURUSD quote)."""

        return self.point * 10 if self.digits in (3, 5) else self.point

    @property
    def min_stop_distance(self) -> float:
        """Smallest SL/TP distance the broker accepts, in price units."""

        return max(self.stops_level, self.freeze_level) * self.point

    def pip_value(self, lot: float) -> float:
        """Account-currency value of one pip for `lot` lots."""

        return self.pip_size / self.tick_size * self.tick_value * lot

    @classmethod
    def from_mt5(cls, info) -> "SymbolSpec":
        return cls(
            symbol=info.name,
            digits=int(info.digits),
            point=float(info.point),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_step=float(info.volume_step),
            volume_max=float(info.volume_max),
            stops_level=int(info.trade_stops_level),
            freeze_level=int(info.trade_freeze_level),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SymbolSpec":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# Standard 5-digit EURUSD on a USD account; only used when no broker
# snapshot has been downloaded yet.
DEFAULT_EURUSD_SPEC = SymbolSpec(
    symbol="EURUSD",
    digits=5,
    point=0.00001,
    tick_size=0.00001,
    tick_value=1.0,
    contract_size=100_000.0,
    volume_min=0.01,
    volume_step=0.01,
    volume_max=100.0,
)

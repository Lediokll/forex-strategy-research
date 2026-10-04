"""
Pure risk rules. No MT5, no I/O: the backtest and the live bot call these
same functions, and the unit tests pin their behaviour down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from src.config import RiskConfig
from src.data.symbol_spec import SymbolSpec


def lot_size(spec: SymbolSpec, configured_lot: float) -> float:
    """Configured lot, raised to the symbol minimum and snapped UP to the
    volume step (so it is never below what was configured)."""

    lot = max(configured_lot, spec.volume_min)
    steps = math.ceil(round(lot / spec.volume_step, 8))
    lot = round(steps * spec.volume_step, 8)

    if lot > spec.volume_max:
        raise ValueError(f"Lot {lot} exceeds symbol maximum {spec.volume_max}")

    return lot


def dollar_risk(sl_distance: float, lot: float, spec: SymbolSpec) -> float:
    """Account-currency loss if the stop is hit: stop distance x pip value."""

    if sl_distance <= 0:
        raise ValueError(f"sl_distance must be positive, got {sl_distance}")

    return sl_distance / spec.tick_size * spec.tick_value * lot


def estimate_margin(lot: float, contract_size: float, price: float, leverage: float) -> float:
    """Margin for a base-currency-quoted pair when the account currency is
    the quote currency (EURUSD on a USD account): notional / leverage."""

    return lot * contract_size * price / leverage


def scale_margin_to_leverage(broker_margin: float, broker_leverage: float, target_leverage: float) -> float:
    """Re-express the broker's own margin figure (from order_calc_margin, at
    the real account's leverage) at a different leverage. Keeps the broker's
    contract size and currency conversion."""

    return broker_margin * broker_leverage / target_leverage


def margin_warning(estimated_margin: float, free_margin: float) -> str | None:
    if estimated_margin > free_margin:
        return (
            f"estimated margin ${estimated_margin:.2f} exceeds virtual free margin "
            f"${free_margin:.2f} (a real account of this size would reject the order)"
        )
    return None


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    open_positions: int
    trades_today: int
    realized_pnl_today: float


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return ";".join(self.reasons)


class RiskManager:
    """Account-level rules applied before every entry and on every equity update."""

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def kill_switch_triggered(self, equity: float) -> bool:
        return equity <= self.cfg.kill_switch_equity

    def daily_loss_cap_hit(self, realized_pnl_today: float) -> bool:
        return realized_pnl_today <= -self.cfg.daily_loss_cap_usd

    def check_entry(self, snap: AccountSnapshot, risk_usd: float) -> Decision:
        reasons = []

        if self.kill_switch_triggered(snap.equity):
            reasons.append("kill_switch")

        if snap.open_positions >= self.cfg.max_open_positions:
            reasons.append("max_open_positions")

        if snap.trades_today >= self.cfg.max_trades_per_day:
            reasons.append("max_trades_per_day")

        if self.daily_loss_cap_hit(snap.realized_pnl_today):
            reasons.append("daily_loss_cap")

        if risk_usd > self.cfg.max_risk_usd:
            reasons.append(f"risk_${risk_usd:.3f}_over_${self.cfg.max_risk_usd:.2f}")

        if risk_usd > snap.equity:
            reasons.append("risk_exceeds_equity")

        return Decision(allowed=not reasons, reasons=reasons)


class DailyLedger:
    """Trades opened and P&L realized per UTC day.

    Entries count toward the day they were opened; P&L toward the day the
    trade closed (a loss realized today blocks today, whenever it opened).
    """

    def __init__(self):
        self.day: date | None = None
        self.trades = 0
        self.realized_pnl = 0.0

    def roll(self, day: date) -> bool:
        """Switch to `day`, resetting counters. Returns True on a new day."""

        if day == self.day:
            return False

        self.day = day
        self.trades = 0
        self.realized_pnl = 0.0
        return True

    def record_entry(self, day: date) -> None:
        self.roll(day)
        self.trades += 1

    def record_exit(self, day: date, pnl: float) -> None:
        self.roll(day)
        self.realized_pnl += pnl


class ProfitAlert:
    """Fires at most once per UTC day while equity is at/above the threshold."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.last_alert_day: date | None = None

    def check(self, equity: float, day: date) -> bool:
        if equity >= self.threshold and day != self.last_alert_day:
            self.last_alert_day = day
            return True
        return False

"""
Virtual account: lets a large demo account behave like a small one.

virtual equity = start balance + realized P&L + open P&L, counting only this
bot's trades (by magic number) since the virtual account was created. All
equity-based rules (kill switch, profit alert, $ risk vs equity, margin
warning) use this number instead of the real balance.

The start balance, start time and account login are persisted so restarts
don't reset the account. Realized P&L is rebuilt from MT5 deal history by
the caller; the cached value here is only a cross-check.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import display_path, mask_login

logger = logging.getLogger(__name__)


class VirtualAccountError(RuntimeError):
    pass


@dataclass
class VirtualAccountState:
    start_balance: float
    started_at_utc: str          # ISO 8601, UTC
    login: int
    symbol: str
    magic: int
    cached_realized_pnl: float = 0.0
    updated_at_utc: str = ""


class VirtualAccount:
    def __init__(self, path: Path, state: VirtualAccountState):
        self.path = Path(path)
        self.state = state

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        configured_balance: float,
        login: int,
        symbol: str,
        magic: int,
        now_utc: datetime | None = None,
    ) -> "VirtualAccount":
        path = Path(path)
        now_utc = now_utc or datetime.now(timezone.utc)

        if not path.exists():
            state = VirtualAccountState(
                start_balance=float(configured_balance),
                started_at_utc=now_utc.isoformat(),
                login=int(login),
                symbol=symbol,
                magic=int(magic),
            )
            account = cls(path, state)
            account.save(now_utc)
            logger.info(
                "Created virtual account: $%.2f starting %s (state file %s)",
                state.start_balance, state.started_at_utc, display_path(path),
            )
            return account

        try:
            state = VirtualAccountState(**json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError) as error:
            raise VirtualAccountError(
                f"Virtual account file {display_path(path)} is unreadable ({error}). "
                "Fix it or run with --reset-virtual."
            ) from error

        if state.login != int(login):
            raise VirtualAccountError(
                f"Virtual account file belongs to account {mask_login(state.login)}, but the terminal "
                f"is logged in to {mask_login(login)}. Refusing to mix accounts; use --reset-virtual "
                "to start a new virtual account."
            )

        if state.symbol != symbol or state.magic != int(magic):
            raise VirtualAccountError(
                f"Virtual account file is for {state.symbol}/magic {state.magic}, config is "
                f"{symbol}/magic {magic}. Use --reset-virtual to start over."
            )

        if abs(state.start_balance - float(configured_balance)) > 1e-9:
            logger.warning(
                "config virtual_balance is $%.2f but the saved virtual account started with "
                "$%.2f. Keeping the saved value; use --reset-virtual to apply the new one.",
                configured_balance, state.start_balance,
            )

        logger.info(
            "Loaded virtual account: started $%.2f at %s, cached realized P&L $%.2f",
            state.start_balance, state.started_at_utc, state.cached_realized_pnl,
        )
        return cls(path, state)

    @property
    def start_balance(self) -> float:
        return self.state.start_balance

    @property
    def started_at(self) -> datetime:
        return datetime.fromisoformat(self.state.started_at_utc)

    def equity(self, realized_pnl: float, open_pnl: float) -> float:
        return self.state.start_balance + realized_pnl + open_pnl

    def update_realized(self, realized_pnl: float, now_utc: datetime | None = None) -> None:
        """Persist the latest realized P&L (rebuilt from deal history)."""

        cached = self.state.cached_realized_pnl

        if abs(realized_pnl - cached) > 0.005:
            logger.info(
                "Virtual realized P&L changed: $%.2f -> $%.2f", cached, realized_pnl,
            )
            self.state.cached_realized_pnl = round(realized_pnl, 2)
            self.save(now_utc)

    def save(self, now_utc: datetime | None = None) -> None:
        now_utc = now_utc or datetime.now(timezone.utc)
        self.state.updated_at_utc = now_utc.isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def reset(path: Path, now_utc: datetime | None = None) -> Path | None:
        """Moves the state file aside (kept as a backup). Returns the backup path."""

        path = Path(path)

        if not path.exists():
            return None

        now_utc = now_utc or datetime.now(timezone.utc)
        backup = path.with_name(f"{path.stem}.{now_utc:%Y%m%dT%H%M%SZ}.bak{path.suffix}")
        os.replace(path, backup)
        logger.warning("Virtual account reset; previous state saved to %s", display_path(backup))
        return backup

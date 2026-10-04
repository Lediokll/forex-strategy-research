"""
Persistent kill switch.

Once tripped, a file records why, and the bot refuses to start again until
a human deletes it. A restart can never quietly resume trading after the
equity floor was hit.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import display_path

logger = logging.getLogger(__name__)


class KillSwitch:
    def __init__(self, path: Path):
        self.path = Path(path)

    def is_tripped(self) -> bool:
        return self.path.exists()

    def details(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"reason": "unreadable kill-switch file"}

    def trip(self, reason: str, equity: float, now_utc: datetime | None = None) -> None:
        now_utc = now_utc or datetime.now(timezone.utc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"tripped_at_utc": now_utc.isoformat(), "reason": reason, "equity": round(equity, 2)},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.critical("KILL SWITCH TRIPPED: %s (equity $%.2f). File: %s", reason, equity, display_path(self.path))

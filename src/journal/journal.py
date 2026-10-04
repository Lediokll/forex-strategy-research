"""
CSV trade journal: signals, trades and daily summaries.

The backtest and the live bot write the same columns, so their files can be
compared side by side.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

SIGNAL_COLUMNS = [
    "time_utc", "direction", "taken", "reason", "regime", "close", "atr",
    "sl_distance_pips", "spread_pips", "risk_usd", "equity", "margin_warning",
]

TRADE_COLUMNS = [
    "entry_time_utc", "exit_time_utc", "direction", "regime", "lot",
    "requested_price", "entry_price", "sl", "tp", "exit_price", "exit_reason",
    "spread_pips", "entry_slippage_pips", "exit_slippage_pips", "commission_usd",
    "risk_usd", "pnl_usd", "r_multiple", "equity_after", "real_equity_after",
    "margin_warning", "ticket",
]

DAILY_COLUMNS = [
    "day_utc", "signals", "signals_taken", "trades_closed", "wins", "losses",
    "realized_pnl_usd", "end_equity", "real_equity", "events",
]


class Journal:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.signals_path = self.out_dir / "signals.csv"
        self.trades_path = self.out_dir / "trades.csv"
        self.daily_path = self.out_dir / "daily_summary.csv"

    def _append(self, path: Path, columns: list[str], row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in columns})

    def log_signal(self, row: dict) -> None:
        self._append(self.signals_path, SIGNAL_COLUMNS, row)

    def log_trade(self, row: dict) -> None:
        self._append(self.trades_path, TRADE_COLUMNS, row)

    def log_daily(self, row: dict) -> None:
        self._append(self.daily_path, DAILY_COLUMNS, row)

    def write_frames(self, signals: pd.DataFrame, trades: pd.DataFrame, daily: pd.DataFrame) -> None:
        """Bulk write (backtest). Overwrites existing files."""

        self.out_dir.mkdir(parents=True, exist_ok=True)
        signals.reindex(columns=SIGNAL_COLUMNS).to_csv(self.signals_path, index=False)
        trades.reindex(columns=TRADE_COLUMNS).to_csv(self.trades_path, index=False)
        daily.reindex(columns=DAILY_COLUMNS).to_csv(self.daily_path, index=False)

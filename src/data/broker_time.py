"""
Broker server time <-> UTC.

MT5 bar and tick times are broker server time, not UTC. Most MT5 brokers
run UTC+2, switching to UTC+3 on the same dates as US daylight saving, so
that 00:00 server time is always 17:00 New York (the FX day roll). Some
follow EU daylight saving instead.

DST switches happen on Sundays, when FX is closed, so checking the rule at
date granularity is exact for every bar the market actually prints.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd


def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def _last_sunday(year: int, month: int) -> date:
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    last_day = next_month - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() + 1) % 7)


def dst_active(day: date, rule: str) -> bool:
    if rule == "none":
        return False

    if rule == "us":
        # Second Sunday of March to first Sunday of November.
        return _nth_sunday(day.year, 3, 2) <= day < _nth_sunday(day.year, 11, 1)

    if rule == "eu":
        # Last Sunday of March to last Sunday of October.
        return _last_sunday(day.year, 3) <= day < _last_sunday(day.year, 10)

    raise ValueError(f"Unknown DST rule: {rule!r}")


def server_offset_hours(server_time: datetime, base_offset: int, rule: str) -> int:
    return base_offset + (1 if dst_active(server_time.date(), rule) else 0)


def server_to_utc(server_times: pd.Series, base_offset: int, rule: str) -> pd.Series:
    """Vectorized server-time -> naive UTC conversion for a datetime Series."""

    days = server_times.dt.normalize()
    offsets = {
        d: server_offset_hours(d.to_pydatetime(), base_offset, rule)
        for d in days.unique()
    }
    hours = days.map(offsets).astype("int64")

    return server_times - pd.to_timedelta(hours, unit="h")


def detect_server_offset(tick_epoch: float, now_epoch: float, tolerance_seconds: float = 180) -> int | None:
    """Infer the server's UTC offset from a live tick.

    MT5 tick timestamps are server-local wall-clock time encoded as if it
    were UTC epoch seconds, so (tick - now) is the offset plus the tick's
    age. Returns None when the tick is stale (market closed, no quotes),
    because then the age swamps the offset.
    """

    diff = tick_epoch - now_epoch
    hours = round(diff / 3600)

    if abs(diff - hours * 3600) > tolerance_seconds:
        return None

    return int(hours)

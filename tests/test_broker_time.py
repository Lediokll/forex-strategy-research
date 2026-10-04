from datetime import date, datetime

import pandas as pd

from src.data.broker_time import detect_server_offset, dst_active, server_to_utc


def test_us_dst_boundaries_2026():
    # 2026: second Sunday of March = Mar 8, first Sunday of November = Nov 1.
    assert not dst_active(date(2026, 3, 7), "us")
    assert dst_active(date(2026, 3, 8), "us")
    assert dst_active(date(2026, 10, 31), "us")
    assert not dst_active(date(2026, 11, 1), "us")


def test_eu_dst_boundaries_2026():
    # 2026: last Sunday of March = Mar 29, last Sunday of October = Oct 25.
    assert not dst_active(date(2026, 3, 28), "eu")
    assert dst_active(date(2026, 3, 29), "eu")
    assert not dst_active(date(2026, 10, 25), "eu")


def test_server_to_utc_summer_and_winter():
    server = pd.Series(pd.to_datetime(["2026-07-01 10:00", "2026-01-15 10:00"]))
    utc = server_to_utc(server, base_offset=2, rule="us")

    assert utc.iloc[0] == pd.Timestamp("2026-07-01 07:00")   # UTC+3 in summer
    assert utc.iloc[1] == pd.Timestamp("2026-01-15 08:00")   # UTC+2 in winter


def test_session_window_in_server_time():
    # 07:00 UTC session open is 10:00 server in summer, 09:00 in winter.
    server = pd.Series(pd.to_datetime(["2026-06-10 10:00", "2026-12-10 09:00"]))
    utc = server_to_utc(server, 2, "us")
    assert list(utc.dt.hour) == [7, 7]


def test_detect_offset_from_fresh_tick():
    now = datetime(2026, 7, 1, 7, 0).timestamp()
    assert detect_server_offset(now + 3 * 3600 - 2, now) == 3


def test_detect_offset_ignores_stale_tick():
    now = datetime(2026, 10, 3, 12, 0).timestamp()
    stale = now + 3 * 3600 - 20 * 60     # last tick 20 minutes ago
    assert detect_server_offset(stale, now) is None

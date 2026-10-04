from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.config import StrategyConfig
from src.features.feature_pipeline import FeaturePipeline
from src.strategy.pullback import (
    BUY,
    SELL,
    compute_setups,
    in_session,
    plan_trade,
    sl_tp_prices,
    strategy_skip_reasons,
)
from tests.test_feature_pipeline import make_synthetic_ohlcv

CFG = StrategyConfig()


def bars(rows):
    """rows: (open, high, low, close, ema20, ema50, ema200)"""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "ema_20", "ema_50", "ema_200"])


def test_long_setup_pullback_touch_then_bullish_close_above_ema20():
    df = bars([
        (1.1010, 1.1015, 1.1005, 1.1012, 1.1000, 1.0990, 1.0950),  # above EMA20, no touch
        (1.1001, 1.1010, 1.0998, 1.1006, 1.1000, 1.0990, 1.0950),  # touches EMA20, bullish close above
    ])
    assert list(compute_setups(df, CFG)) == [0, 1]


def test_touch_on_previous_bar_counts():
    df = bars([
        (1.1005, 1.1006, 1.0997, 1.0999, 1.1000, 1.0990, 1.0950),  # dips below EMA20, bearish
        (1.1001, 1.1009, 1.1001, 1.1008, 1.1000, 1.0990, 1.0950),  # no touch itself, bullish above
    ])
    assert list(compute_setups(df, CFG)) == [0, 1]


def test_no_long_when_trend_is_down_or_candle_bearish():
    down_trend = bars([(1.1008, 1.1010, 1.0998, 1.1003, 1.1000, 1.0940, 1.0950)])
    bearish = bars([(1.1006, 1.1010, 1.0998, 1.1003, 1.1000, 1.0990, 1.0950)])
    assert compute_setups(down_trend, CFG)[0] == 0
    assert compute_setups(bearish, CFG)[0] == 0


def test_short_setup_mirror():
    # Downtrend, high touches EMA20, bearish close below it.
    df = bars([(1.0998, 1.1002, 1.0990, 1.0993, 1.1000, 1.1010, 1.1050)])
    assert compute_setups(df, CFG)[0] == -1


def test_setups_have_no_lookahead():
    features = FeaturePipeline(make_synthetic_ohlcv(n=600, seed=3)).run()
    full = compute_setups(features, CFG)

    for k in (250, 400, 599):
        prefix = FeaturePipeline(make_synthetic_ohlcv(n=600, seed=3).iloc[:k]).run()
        assert np.array_equal(compute_setups(prefix, CFG), full[: len(prefix)])


def test_plan_trade_stop_and_target():
    plan = plan_trade(BUY, atr=0.0006, cfg=CFG)
    assert plan.sl_distance == pytest.approx(0.0009)     # 1.5 x ATR
    assert plan.tp_distance == pytest.approx(0.0018)     # 2 x stop

    sl, tp = sl_tp_prices(BUY, 1.1000, plan)
    assert sl == pytest.approx(1.0991) and tp == pytest.approx(1.1018)

    sl, tp = sl_tp_prices(SELL, 1.1000, plan)
    assert sl == pytest.approx(1.1009) and tp == pytest.approx(1.0982)


def test_plan_trade_widens_to_broker_minimum():
    plan = plan_trade(SELL, atr=0.0002, cfg=CFG, min_stop_distance=0.0005)
    assert plan.sl_distance == pytest.approx(0.0005)
    assert plan.tp_distance == pytest.approx(0.0010)


def test_plan_trade_rejects_bad_input():
    with pytest.raises(ValueError):
        plan_trade("hold", 0.0005, CFG)
    with pytest.raises(ValueError):
        plan_trade(BUY, 0.0, CFG)


def test_session_is_0700_to_1700_utc():
    assert not in_session(datetime(2026, 5, 4, 6, 45), CFG)
    assert in_session(datetime(2026, 5, 4, 7, 0), CFG)
    assert in_session(datetime(2026, 5, 4, 16, 45), CFG)
    assert not in_session(datetime(2026, 5, 4, 17, 0), CFG)


def test_strategy_skip_reasons():
    t = datetime(2026, 5, 4, 9, 0)
    assert strategy_skip_reasons("trend", t, 1.0, CFG) == []
    assert strategy_skip_reasons("chop", t, 1.0, CFG) == ["regime_chop"]
    assert strategy_skip_reasons("high_vol", t, 1.0, CFG) == ["regime_high_vol"]
    assert strategy_skip_reasons("chop", t, 1.0, CFG, regime_enabled=False) == []
    assert strategy_skip_reasons("trend", t, 1.6, CFG) == ["spread_1.6pips"]
    assert strategy_skip_reasons("trend", datetime(2026, 5, 4, 20, 0), 1.0, CFG) == ["outside_session"]

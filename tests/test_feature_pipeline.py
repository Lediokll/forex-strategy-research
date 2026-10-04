import numpy as np
import pandas as pd

from src.features.feature_pipeline import FeaturePipeline


def make_synthetic_ohlcv(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    datetimes = pd.date_range("2024-01-01", periods=n, freq="h")

    steps = rng.normal(loc=0.0, scale=0.0006, size=n)
    close = 1.1000 + np.cumsum(steps)

    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.0001, 0.0004, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.0001, 0.0004, size=n)
    volume = rng.integers(50, 500, size=n)
    spread = rng.integers(1, 5, size=n)

    return pd.DataFrame({
        "datetime": datetimes,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "spread": spread,
    })


def test_run_produces_no_nan_or_inf():
    raw_df = make_synthetic_ohlcv()

    features_df = FeaturePipeline(raw_df).run()

    assert not features_df.empty

    numeric_df = features_df.select_dtypes(include=[np.number])

    assert not numeric_df.isna().any().any()
    assert not np.isinf(numeric_df.to_numpy()).any()


def test_run_survives_a_flat_price_run():
    # A perfectly flat close makes RSI's avg_loss (and Bollinger std) zero,
    # which previously produced +/-inf that slipped through dropna().
    raw_df = make_synthetic_ohlcv(n=120)
    raw_df.loc[20:80, "close"] = 1.1000
    raw_df.loc[20:80, "open"] = 1.1000
    raw_df.loc[20:80, "high"] = 1.1000
    raw_df.loc[20:80, "low"] = 1.1000

    features_df = FeaturePipeline(raw_df).run()

    numeric_df = features_df.select_dtypes(include=[np.number])

    assert not numeric_df.isna().any().any()
    assert not np.isinf(numeric_df.to_numpy()).any()

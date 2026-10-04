import numpy as np
import pandas as pd

from src.config import RegimeConfig
from src.regime.regime_detector import (
    CHOP,
    HIGH_VOL,
    TREND,
    RegimeModel,
    label_states,
    walk_forward_labels,
)


def regime_frame(n_per_state=400, seed=1):
    """Three blocks: quiet chop, steady trend, violent high-vol."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([
        rng.normal(0.0, 0.0003, n_per_state),
        rng.normal(0.0030, 0.0004, n_per_state),
        rng.normal(0.0, 0.0030, n_per_state),
    ])
    vol = np.concatenate([
        rng.normal(0.0004, 0.00003, n_per_state),
        rng.normal(0.0006, 0.00003, n_per_state),
        rng.normal(0.0020, 0.0001, n_per_state),
    ])
    return pd.DataFrame({"regime_drift": drift, "regime_vol": vol})


def test_label_states_assigns_trend_chop_high_vol():
    df = regime_frame()
    states = np.repeat([0, 1, 2], 400)

    mapping, stats = label_states(df, states)

    assert mapping == {0: CHOP, 1: TREND, 2: HIGH_VOL}
    assert set(stats["label"]) == {CHOP, TREND, HIGH_VOL}


def test_label_states_two_states():
    df = regime_frame().iloc[400:]
    mapping, _ = label_states(df, np.repeat([0, 1], 400))
    assert mapping == {0: TREND, 1: HIGH_VOL}


def test_fitted_model_recovers_regimes():
    df = regime_frame()
    model = RegimeModel.fit(df, RegimeConfig(n_iter=50))
    labels = model.filter_labels(df)

    # Majority label inside each block (skip the first bars after a switch).
    assert pd.Series(labels[50:400]).mode()[0] == CHOP
    assert pd.Series(labels[450:800]).mode()[0] == TREND
    assert pd.Series(labels[850:]).mode()[0] == HIGH_VOL


def test_forward_filter_has_no_lookahead():
    df = regime_frame()
    model = RegimeModel.fit(df, RegimeConfig(n_iter=50))
    full = model.filter_states(df)

    for k in (100, 450, 900):
        assert np.array_equal(model.filter_states(df.iloc[:k]), full[:k])


def test_walk_forward_labels_only_after_training_window_and_causal():
    df = pd.concat([regime_frame(seed=s) for s in (1, 2)], ignore_index=True)  # 2400 rows
    cfg = RegimeConfig(train_bars=1000, refit_every_bars=500, filter_burn_in_bars=100, n_iter=30)

    labels = walk_forward_labels(df, cfg)
    assert all(label is None for label in labels[:1000])
    assert all(label in (TREND, CHOP, HIGH_VOL) for label in labels[1000:])

    # Appending future rows must not change any earlier label.
    shorter = walk_forward_labels(df.iloc[:1800], cfg)
    assert list(shorter[1000:]) == list(labels[1000:1800])

"""
HMM market-regime model.

A GaussianHMM on two features (recent drift and realized volatility) splits
the market into `n_states` hidden states, which are then labelled from
their average statistics:

  - the highest-volatility state            -> "high_vol"  (skip)
  - of the rest, highest |drift| / volatility -> "trend"     (trade)
  - everything else                          -> "chop"      (skip)

No look-ahead: the current regime is computed with the HMM *forward filter*
(each state probability uses only observations up to and including that
bar). Viterbi decoding, which the previous version used, revises earlier
states once later bars arrive, so it can't be used bar-by-bar.

The same RegimeModel class and walk-forward schedule are used by the
backtest and the live bot.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from src import config as cfg_module
from src.config import RegimeConfig

logger = logging.getLogger(__name__)

REGIME_FEATURES = ["regime_drift", "regime_vol"]

TREND = "trend"
CHOP = "chop"
HIGH_VOL = "high_vol"


def label_states(df_clean: pd.DataFrame, states: np.ndarray) -> tuple[dict[int, str], pd.DataFrame]:
    """
    Assigns meaning to arbitrary HMM state numbers (which shuffle on every
    refit) from each state's mean volatility and directional efficiency.
    """

    labeled = df_clean[REGIME_FEATURES].copy()
    labeled["state"] = states
    labeled["abs_drift"] = labeled["regime_drift"].abs()

    stats = labeled.groupby("state").agg(
        mean_vol=("regime_vol", "mean"),
        mean_abs_drift=("abs_drift", "mean"),
        count=("regime_vol", "size"),
    )
    stats["efficiency"] = stats["mean_abs_drift"] / stats["mean_vol"]

    high_vol_state = int(stats["mean_vol"].idxmax())
    mapping: dict[int, str] = {high_vol_state: HIGH_VOL}

    remaining = stats.drop(index=high_vol_state)

    if not remaining.empty:
        trend_state = int(remaining["efficiency"].idxmax())
        mapping[trend_state] = TREND

        for state in remaining.index:
            mapping.setdefault(int(state), CHOP)

    stats["label"] = stats.index.map(mapping)

    return mapping, stats


def forward_filter(model: GaussianHMM, X_scaled: np.ndarray) -> np.ndarray:
    """Most likely state at each row given rows [0..t] only (filtering)."""

    n_states = model.n_components
    covars = model.covars_

    log_emission = np.column_stack([
        multivariate_normal.logpdf(X_scaled, mean=model.means_[k], cov=covars[k], allow_singular=True)
        for k in range(n_states)
    ]).reshape(len(X_scaled), n_states)

    # Work in probability space with per-row rescaling for stability.
    emission = np.exp(log_emission - log_emission.max(axis=1, keepdims=True))
    transmat = model.transmat_

    states = np.empty(len(X_scaled), dtype=int)
    alpha = model.startprob_ * emission[0]
    alpha /= alpha.sum()
    states[0] = alpha.argmax()

    for t in range(1, len(X_scaled)):
        alpha = (alpha @ transmat) * emission[t]
        total = alpha.sum()

        if total <= 0 or not np.isfinite(total):
            # Observation is impossible under every state (numerical
            # underflow); restart from the stationary prior.
            alpha = model.startprob_ * emission[t]
            total = alpha.sum()

        alpha /= total
        states[t] = alpha.argmax()

    return states


@dataclass
class RegimeModel:
    model: GaussianHMM
    scaler: StandardScaler
    labels: dict[int, str]
    stats: pd.DataFrame
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None

    @classmethod
    def fit(cls, features_df: pd.DataFrame, cfg: RegimeConfig) -> "RegimeModel":
        X = features_df[REGIME_FEATURES].to_numpy()

        if not np.isfinite(X).all():
            raise ValueError("Regime features contain NaN/inf.")

        if len(X) < 100:
            raise ValueError("Not enough rows to train the HMM.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type="full",
            n_iter=cfg.n_iter,
            random_state=cfg.random_state,
        )
        model.fit(X_scaled)

        if not model.monitor_.converged:
            logger.warning("HMM did not fully converge.")

        # Viterbi on the *training* window only, to characterize the states.
        train_states = model.predict(X_scaled)
        labels, stats = label_states(features_df, train_states)

        time_col = "time_utc" if "time_utc" in features_df else "datetime"
        return cls(
            model=model,
            scaler=scaler,
            labels=labels,
            stats=stats,
            train_start=features_df[time_col].iloc[0] if time_col in features_df else None,
            train_end=features_df[time_col].iloc[-1] if time_col in features_df else None,
        )

    def filter_states(self, features_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(features_df[REGIME_FEATURES].to_numpy())
        return forward_filter(self.model, X)

    def filter_labels(self, features_df: pd.DataFrame) -> np.ndarray:
        states = self.filter_states(features_df)
        return np.array([self.labels.get(int(s), CHOP) for s in states], dtype=object)

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RegimeModel":
        return joblib.load(path)


def walk_forward_labels(features_df: pd.DataFrame, cfg: RegimeConfig) -> np.ndarray:
    """
    Regime label for every row, computed the way the live bot would have:
    the model for rows [k, k + refit_every) is trained only on the
    `train_bars` rows before k, and labels come from the forward filter
    started `filter_burn_in_bars` before k. Rows inside the first training
    window get None (no model exists yet).
    """

    n = len(features_df)
    labels = np.full(n, None, dtype=object)

    for block_start in range(cfg.train_bars, n, cfg.refit_every_bars):
        train = features_df.iloc[block_start - cfg.train_bars:block_start]
        model = RegimeModel.fit(train, cfg)

        lo = max(0, block_start - cfg.filter_burn_in_bars)
        hi = min(n, block_start + cfg.refit_every_bars)
        block_labels = model.filter_labels(features_df.iloc[lo:hi])
        labels[block_start:hi] = block_labels[block_start - lo:]

        logger.debug(
            "Regime refit at row %s: labels %s", block_start, model.labels,
        )

    return labels


def main() -> None:
    """Fit on the latest downloaded history and print the state summary."""

    from src.data.history import load_history
    from src.features.feature_pipeline import FeaturePipeline

    cfg_module.setup_logging()

    try:
        cfg = cfg_module.load_config()
        features = FeaturePipeline(load_history(cfg), cfg.strategy, cfg.regime).run()
        train = features.iloc[-cfg.regime.train_bars:]

        model = RegimeModel.fit(train, cfg.regime)
        current = model.filter_labels(features.iloc[-cfg.regime.filter_burn_in_bars - 1:])[-1]

        logger.info("Trained on %s rows (%s -> %s UTC)", len(train), model.train_start, model.train_end)
        logger.info("Per-state stats:\n%s", model.stats)
        logger.info("Current regime: %s", current)

    except Exception as error:
        logger.error("Regime detector failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()

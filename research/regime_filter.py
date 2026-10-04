"""
Per-symbol HMM high-volatility flag for the daily portfolio.

Reuses the bot's FeaturePipeline and walk-forward RegimeModel on H4 bars.
A daily decision taken at the close of server day D (= 00:00 of D+1) may
only use H4 bars that closed by then, so each day gets the label of the last
H4 bar whose close time is <= D + 1 day.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict

import pandas as pd

from research.data import load_bars
from research.settings import DATA_DIR, HmmFilterConfig
from src.config import StrategyConfig
from src.features.feature_pipeline import FeaturePipeline
from src.regime.regime_detector import HIGH_VOL, walk_forward_labels

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache"
CACHE_VERSION = 2
H4 = pd.Timedelta(hours=4)
MIN_H4_BARS_PER_YEAR = 1000      # real H4 data is ~1,560 bars/year


def intraday_h4_only(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop the early part of the history where the broker's 'H4' bars are
    really one bar per day (MetaQuotes data before ~1999), so the HMM's
    feature window means the same thing throughout."""

    per_year = bars.groupby(bars["datetime"].dt.year).size()
    dense = per_year[per_year >= MIN_H4_BARS_PER_YEAR]
    if dense.empty:
        return bars.iloc[0:0]
    start = pd.Timestamp(year=int(dense.index[0]), month=1, day=1)
    return bars[bars["datetime"] >= start].reset_index(drop=True)


def h4_labels(symbol: str, cfg: HmmFilterConfig, end_exclusive: pd.Timestamp | None) -> pd.DataFrame | None:
    """Walk-forward labels per H4 bar, indexed by bar close time. Cached."""

    key = json.dumps({"symbol": symbol, "end": str(end_exclusive), "cfg": asdict(cfg), "v": CACHE_VERSION},
                     sort_keys=True)
    cache_path = CACHE_DIR / f"{symbol}_hmm_{hashlib.sha1(key.encode()).hexdigest()[:12]}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["close_time"])

    bars = load_bars(symbol, "H4", end_exclusive)
    if bars is not None:
        bars = intraday_h4_only(bars)
    regime_cfg = cfg.regime_config()

    if bars is None or len(bars) < regime_cfg.train_bars + regime_cfg.refit_every_bars:
        logger.warning("%s: not enough H4 history for the HMM; filter disabled for it.", symbol)
        return None

    features = FeaturePipeline(bars, StrategyConfig(), regime_cfg).run()
    labels = walk_forward_labels(features, regime_cfg)

    out = pd.DataFrame({
        "close_time": features["datetime"] + H4,
        "label": pd.Series(labels).fillna("none").to_numpy(),
    })
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def daily_high_vol_flags(symbols: list[str], dates: pd.DatetimeIndex, cfg: HmmFilterConfig,
                         end_exclusive: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.Series]:
    """(dates x symbols) True where the symbol is in its high-vol state at
    that day's close, plus the first date each symbol has a label."""

    flags = pd.DataFrame(False, index=dates, columns=symbols)
    ready = {}
    decision_times = pd.DataFrame({"decision_time": dates + pd.Timedelta(days=1), "date": dates})

    for symbol in symbols:
        labels = h4_labels(symbol, cfg, end_exclusive)
        if labels is None:
            ready[symbol] = pd.NaT
            continue

        merged = pd.merge_asof(
            decision_times, labels.sort_values("close_time"),
            left_on="decision_time", right_on="close_time", direction="backward",
        )
        flags[symbol] = (merged["label"] == HIGH_VOL).to_numpy()
        labelled = merged["label"].notna() & (merged["label"] != "none")
        ready[symbol] = merged.loc[labelled, "date"].min() if labelled.any() else pd.NaT

    return flags, pd.Series(ready)

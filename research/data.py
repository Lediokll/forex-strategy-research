"""
Research data: download (MT5) and loading (offline).

Downloads D1 and H4 bars plus a symbol_info snapshot (contract, tick value,
swap_long/swap_short, spread) for the research universe into research/data/.
Loading applies the holdout cut-off BEFORE anything else touches the data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.portfolio import MarketData
from src.config import display_path
from research.settings import DATA_DIR, ResearchConfig
from research.strategies import USD, annual_swap_rates

logger = logging.getLogger(__name__)

META_PATH = DATA_DIR / "symbols.json"
INDEX_FOLDERS = ("indexes", "indices", "index")
SPREAD_LOOKBACK = pd.DateOffset(years=2)


def bars_path(symbol: str, timeframe: str):
    return DATA_DIR / f"{symbol}_{timeframe}.csv"


# ---------------------------------------------------------------------------
# Download (needs the MT5 terminal)
# ---------------------------------------------------------------------------
def discover_index_symbols(mt5) -> list[str]:
    """Symbols in the broker's top-level index folder (e.g. 'Indexes\\US500')."""

    return sorted(
        s.name for s in mt5.symbols_get()
        if s.path.split("\\")[0].lower() in INDEX_FOLDERS
    )


def download_research_data(cfg: ResearchConfig) -> dict:
    import MetaTrader5 as mt5

    from src.data.mt5_data import initialize_mt5, rates_to_frame

    if not initialize_mt5():
        raise ConnectionError(f"Could not connect to MetaTrader 5: {mt5.last_error()}")

    try:
        symbols = cfg.universe.symbols
        offered_indices = discover_index_symbols(mt5)
        logger.info("Index CFDs offered by the broker (%s): %s", len(offered_indices), offered_indices)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        meta = {}

        for symbol in symbols:
            if not mt5.symbol_select(symbol, True):
                logger.warning("Skipping %s: not available (%s)", symbol, mt5.last_error())
                continue

            info = mt5.symbol_info(symbol)
            frames = {}
            for tf, count in (("D1", cfg.data.daily_bars), ("H4", cfg.data.h4_bars)):
                rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{tf}"), 1, count)
                if rates is None or len(rates) == 0:
                    logger.warning("No %s bars for %s: %s", tf, symbol, mt5.last_error())
                    continue
                frames[tf] = rates_to_frame(rates)
                frames[tf].to_csv(bars_path(symbol, tf), index=False)

            if "D1" not in frames:
                continue

            h4 = frames.get("H4")

            meta[symbol] = {
                "name": symbol,
                "path": info.path,
                "description": info.description,
                "currency_base": info.currency_base,
                "currency_profit": info.currency_profit,
                "currency_margin": info.currency_margin,
                "digits": info.digits,
                "point": info.point,
                "tick_size": info.trade_tick_size,
                "tick_value": info.trade_tick_value,
                "contract_size": info.trade_contract_size,
                "volume_min": info.volume_min,
                "volume_step": info.volume_step,
                "swap_mode": info.swap_mode,
                "swap_long": info.swap_long,
                "swap_short": info.swap_short,
                "swap_rollover3days": info.swap_rollover3days,
                "spread_snapshot_points": info.spread,
                "price": float(frames["D1"]["close"].iloc[-1]),
                "d1_first": str(frames["D1"]["datetime"].iloc[0]),
                "d1_last": str(frames["D1"]["datetime"].iloc[-1]),
                "h4_first": str(h4["datetime"].iloc[0]) if h4 is not None else None,
                "d1_bars": len(frames["D1"]),
                "h4_bars": len(h4) if h4 is not None else 0,
            }
            logger.info(
                "%-8s D1 %s bars from %s, H4 %s bars from %s, swap L/S %s/%s (mode %s)",
                symbol, meta[symbol]["d1_bars"], meta[symbol]["d1_first"][:10],
                meta[symbol]["h4_bars"], (meta[symbol]["h4_first"] or "")[:10],
                info.swap_long, info.swap_short, info.swap_mode,
            )

        snapshot = {
            "taken_at_utc": datetime.now(timezone.utc).isoformat(),
            "offered_indices": offered_indices,
            "symbols": meta,
        }
        META_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return snapshot

    finally:
        mt5.shutdown()


# ---------------------------------------------------------------------------
# Loading (offline)
# ---------------------------------------------------------------------------
def load_meta() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError(f"{display_path(META_PATH)} missing. Run `py -m research.run --download` first.")
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def load_bars(symbol: str, timeframe: str, end_exclusive: pd.Timestamp | None = None) -> pd.DataFrame | None:
    path = bars_path(symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    if end_exclusive is not None:
        df = df[df["datetime"] < end_exclusive]
    return df.reset_index(drop=True)


def coverage(symbol: str, end_exclusive: pd.Timestamp | None, window_end: pd.Timestamp) -> float:
    """Share of weekdays from the symbol's first bar to the end of the
    evaluation window that have a D1 bar. Measured to the window end, not
    the symbol's last bar, so history that simply stops counts as missing."""

    bars = load_bars(symbol, "D1", end_exclusive)
    if bars is None or bars.empty:
        return 0.0
    weekdays = pd.bdate_range(bars["datetime"].iloc[0], window_end)
    return len(bars) / len(weekdays)


def holdout_start(meta: dict, months: int) -> pd.Timestamp:
    """First day of the locked holdout: `months` before the latest FX bar."""

    last = max(pd.Timestamp(m["d1_last"]) for m in meta["symbols"].values())
    return (last - pd.DateOffset(months=months)).normalize()


def spread_estimate_price(symbol: str, m: dict, floor_price: float | None,
                          end_exclusive: pd.Timestamp | None) -> float:
    """Typical spread in price units: the larger of the configured floor and
    the median H4 bar spread over the 2 years before the cut-off. Bar
    spreads are per-bar minimums, so alone they would understate cost."""

    h4 = load_bars(symbol, "H4", end_exclusive)
    median_price = 0.0
    if h4 is not None and not h4.empty:
        recent = h4[h4["datetime"] >= h4["datetime"].iloc[-1] - SPREAD_LOOKBACK]
        median_price = float(recent["spread"].median()) * m["point"]

    if floor_price is None:
        logger.warning("%s: no spread floor configured; using broker bar spreads only.", symbol)
        floor_price = 0.0

    return max(floor_price, median_price)


def usd_conversion(currency: str, closes: pd.DataFrame, meta: dict[str, dict]) -> pd.Series | None:
    """USD per 1 unit of `currency`, from a downloaded USD pair if possible."""

    if currency == USD:
        return pd.Series(1.0, index=closes.index)

    for symbol, m in meta.items():
        if symbol not in closes:
            continue
        if m["currency_base"] == currency and m["currency_profit"] == USD:
            return closes[symbol]
        if m["currency_base"] == USD and m["currency_profit"] == currency:
            return 1.0 / closes[symbol]

    return None


def build_market_data(meta: dict[str, dict], symbols: list[str], end_exclusive: pd.Timestamp | None,
                      spread_floor: dict[str, float] | None = None) -> MarketData:
    frames = {s: load_bars(s, "D1", end_exclusive) for s in symbols}
    frames = {s: f for s, f in frames.items() if f is not None and not f.empty}
    symbols = list(frames)

    dates = pd.DatetimeIndex(sorted(set().union(*[set(f["datetime"]) for f in frames.values()])))

    def panel(column):
        return pd.DataFrame({s: f.set_index("datetime")[column] for s, f in frames.items()}).reindex(dates)

    raw_open, raw_close = panel("open"), panel("close")
    has_bar = raw_close.notna()
    close = raw_close.ffill()
    open_ = raw_open.where(has_bar, close.shift(1))

    conv = {}
    for symbol in symbols:
        series = usd_conversion(meta[symbol]["currency_profit"], close, meta)
        if series is None:
            # No USD pair for this currency: use the snapshot's tick value (constant).
            m = meta[symbol]
            constant = m["tick_value"] / (m["tick_size"] * m["contract_size"])
            logger.warning("%s: no USD pair for %s, using constant conversion %.6f",
                           symbol, m["currency_profit"], constant)
            series = pd.Series(constant, index=dates)
        conv[symbol] = series.ffill()

    spread_floor = spread_floor or {}
    spread_price = pd.Series({
        s: spread_estimate_price(s, meta[s], spread_floor.get(s), end_exclusive) for s in symbols
    })
    swaps = pd.DataFrame({s: annual_swap_rates(meta[s]) for s in symbols}, index=["long", "short"]).T

    return MarketData(
        dates=dates,
        symbols=symbols,
        open=open_,
        close=close,
        has_bar=has_bar,
        conv=pd.DataFrame(conv).reindex(columns=symbols),
        spread_price=spread_price,
        contract_size=pd.Series({s: meta[s]["contract_size"] for s in symbols}),
        swap_long=swaps["long"],
        swap_short=swaps["short"],
    )

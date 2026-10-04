"""
Research runner.

    py -m research.run --download        # fetch D1/H4 + symbol snapshot from MT5
    py -m research.run                   # development run (holdout excluded)
    py -m research.run --final-holdout   # ONE-TIME run on the locked holdout

The development run never loads bars on or after the holdout start: prices,
the HMM and the spread estimates are all cut off before it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research import cent_sizing
from research.data import build_market_data, coverage, download_research_data, holdout_start, load_meta
from research.evaluation import (
    monte_carlo_max_drawdowns,
    per_symbol_stats,
    perf_stats,
    period_returns,
    yearly_returns,
)
from research.portfolio import MarketData, realized_vol, run_portfolio
from research.regime_filter import daily_high_vol_flags
from research.settings import HOLDOUT_MARKER, RESULTS_DIR, ResearchConfig, load_research_config
from research.strategies import (
    apply_high_vol_filter,
    carry_currency_weights,
    carry_pair_directions,
    carry_signal,
    combine,
    currency_rates_from_swaps,
    rebalance_mask,
    trend_signal,
)
from src.config import display_path, setup_logging

logger = logging.getLogger(__name__)

STRATEGIES = ["Trend", "Carry", "Trend+Carry", "Trend +HMM", "Carry +HMM", "Trend+Carry +HMM"]


@dataclass
class Context:
    cfg: ResearchConfig
    md: MarketData
    fx: list[str]
    carry_directions: pd.Series
    currency_rates: pd.Series
    high_vol: pd.DataFrame
    hmm_ready: pd.Series
    weekly: np.ndarray
    monthly: np.ndarray
    excluded: dict


def build_context(cfg: ResearchConfig, meta: dict, end_exclusive: pd.Timestamp | None) -> Context:
    window_end = (end_exclusive - pd.Timedelta(days=1)) if end_exclusive is not None else \
        max(pd.Timestamp(m["d1_last"]) for m in meta.values())

    symbols, excluded = [], {}
    for s in cfg.universe.symbols:
        if s not in meta:
            excluded[s] = "not downloaded"
            continue
        cov = coverage(s, end_exclusive, window_end)
        if cov < cfg.data.min_coverage:
            excluded[s] = f"only {cov:.0%} of weekday bars present"
            continue
        symbols.append(s)
    for s, why in excluded.items():
        logger.warning("Excluded %s: %s", s, why)

    md = build_market_data(meta, symbols, end_exclusive, cfg.costs.spread_floor)
    fx = [s for s in cfg.universe.fx if s in md.symbols]

    rates = currency_rates_from_swaps(meta, fx)
    directions = carry_pair_directions(carry_currency_weights(rates, cfg.carry.n_long, cfg.carry.n_short), meta, fx)

    logger.info("Computing walk-forward HMM labels on H4 for %s symbols (cached after the first run)...", len(md.symbols))
    high_vol, ready = daily_high_vol_flags(md.symbols, md.dates, cfg.hmm_filter, end_exclusive)

    return Context(cfg, md, fx, directions, rates, high_vol, ready,
                   rebalance_mask(md.dates, "weekly"), rebalance_mask(md.dates, "monthly"), excluded)


def strategy_inputs(ctx: Context, name: str, scale: float = 1.0) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(signal, rebalance mask, vol) for a strategy, with lookbacks x scale."""

    cfg = ctx.cfg
    lookbacks = tuple(max(2, round(lb * scale)) for lb in cfg.trend.lookbacks_days)
    vol = realized_vol(ctx.md, max(5, round(cfg.portfolio.vol_lookback_days * scale)), cfg.portfolio.trading_days_per_year)

    trend = trend_signal(ctx.md.close, lookbacks)
    carry = carry_signal(ctx.md.dates, ctx.md.symbols, ctx.carry_directions)
    # Carry as decided at the latest monthly rebalance (for the weekly combo).
    carry_held = carry.loc[ctx.monthly].reindex(ctx.md.dates).ffill()

    base = name.replace(" +HMM", "")
    if base == "Trend":
        signal, mask = trend, ctx.weekly
    elif base == "Carry":
        signal, mask = carry, ctx.monthly
    elif base == "Trend+Carry":
        signal = combine([trend, carry_held], [cfg.combo.trend_weight, cfg.combo.carry_weight])
        mask = ctx.weekly
    else:
        raise ValueError(name)

    if name.endswith("+HMM"):
        signal = apply_high_vol_filter(signal, ctx.high_vol, cfg.hmm_filter.high_vol_scale)

    return signal, mask, vol


def evaluation_start(ctx: Context) -> pd.Timestamp:
    """One common start for every strategy and every neighbourhood scale:
    longest trend lookback at x1.25, vol at x1.25, and the HMM ready on all FX."""

    max_scale = max(ctx.cfg.evaluation.neighborhood_scales)
    trend, _, vol = strategy_inputs(ctx, "Trend", max_scale)
    trend_ready = trend[ctx.fx].notna().all(axis=1)
    vol_ready = vol[ctx.fx].notna().all(axis=1)
    hmm_ready = ctx.hmm_ready[ctx.fx]

    if hmm_ready.isna().any():
        raise RuntimeError(f"HMM labels missing for: {list(hmm_ready[hmm_ready.isna()].index)}")

    candidates = [
        trend_ready[trend_ready].index.min(),
        vol_ready[vol_ready].index.min(),
        hmm_ready.max(),
    ]
    return max(candidates)


def run_strategy(ctx: Context, name: str, start: pd.Timestamp) -> dict:
    cfg = ctx.cfg
    signal, mask, vol = strategy_inputs(ctx, name)
    base = run_portfolio(ctx.md, signal, mask, vol, cfg.portfolio, cfg.costs, start)
    stats = perf_stats(base.equity, cfg.portfolio.trading_days_per_year)

    pess = run_portfolio(ctx.md, signal, mask, vol, cfg.portfolio, cfg.costs, start,
                         spread_multiplier=cfg.costs.pessimistic_spread_multiplier)
    stats["sharpe_pessimistic"] = perf_stats(pess.equity)["sharpe"]
    stats["cagr_pessimistic"] = perf_stats(pess.equity)["cagr"]

    for scale in cfg.evaluation.neighborhood_scales:
        if scale == 1.0:
            continue
        s_sig, s_mask, s_vol = strategy_inputs(ctx, name, scale)
        s_res = run_portfolio(ctx.md, s_sig, s_mask, s_vol, cfg.portfolio, cfg.costs, start)
        s_stats = perf_stats(s_res.equity)
        stats[f"sharpe_x{scale:g}"] = s_stats["sharpe"]
        stats[f"cagr_x{scale:g}"] = s_stats["cagr"]

    thirds = np.array_split(base.equity.index, 3)
    for i, idx in enumerate(thirds, 1):
        stats[f"sharpe_third{i}"] = perf_stats(base.equity.loc[idx[0]:idx[-1]])["sharpe"]

    weekly = period_returns(base.equity, "W")
    mc = monte_carlo_max_drawdowns(weekly, cfg.evaluation.monte_carlo_runs, cfg.evaluation.monte_carlo_seed)
    stats["mc_dd_p5"], stats["mc_dd_p50"], stats["mc_dd_p95"] = np.percentile(mc, [5, 50, 95])

    stats["turnover_x_per_year"] = base.turnover_usd / base.equity.mean() / ((base.equity.index[-1] - start).days / 365.25)
    stats["avg_gross_leverage"] = float(
        (base.units.abs() * ctx.md.close.loc[base.units.index] * ctx.md.conv.loc[base.units.index]).sum(axis=1).div(base.equity).mean()
    )

    cent_cfg = cfg.cent_account
    cent = {}
    for policy in ("nearest", "at_least_min"):
        res = run_portfolio(ctx.md, signal, mask, vol, cfg.portfolio, cfg.costs, start,
                            start_equity=cent_cfg.start_equity_usd,
                            rounding=cent_sizing.rounding(ctx.md, cent_cfg, policy))
        cs = perf_stats(res.equity) if (res.equity > 0).all() else {"sharpe": np.nan, "cagr": -1.0, "vol": np.nan,
                                                                    "max_dd": 1.0}
        lots = cent_sizing.intended_lots(res, ctx.md, cent_cfg)
        cent[policy] = {
            **{k: cs[k] for k in ("sharpe", "cagr", "vol", "max_dd")},
            "final_equity": float(res.equity.iloc[-1]),
            "min_equity": float(res.equity.min()),
            "share_below_half_lot": cent_sizing.share_below(lots, 0.5),
            "share_below_one_lot": cent_sizing.share_below(lots, 1.0),
        }

    return {
        "name": name,
        "stats": stats,
        "yearly": yearly_returns(base.equity),
        "per_symbol": per_symbol_stats(base, cfg.portfolio.trading_days_per_year),
        "equity": base.equity,
        "cent": cent,
        "feasibility": cent_sizing.feasibility_table(base, ctx.md, cent_cfg),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def pct(x, digits=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def num(x, digits=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(ctx: Context, start: pd.Timestamp, results: list[dict], label: str, meta_snapshot_time: str) -> str:
    cfg, md = ctx.cfg, ctx.md
    L = [f"# Research report: {label}\n"]

    L.append("## Setup\n")
    L.append(f"- Evaluation period: **{start:%Y-%m-%d} to {md.dates[-1]:%Y-%m-%d}** "
             f"({(md.dates[-1] - start).days / 365.25:.1f} years); same period for every strategy")
    L.append(f"- Symbols ({len(md.symbols)}): {', '.join(md.symbols)}")
    for s, why in ctx.excluded.items():
        L.append(f"- Excluded {s}: {why}")
    L.append("- Start of each symbol's data: " + ", ".join(
        f"{s} {md.close[s].first_valid_index():%Y}" for s in md.symbols))
    hmm_off = [s for s in md.symbols if pd.isna(ctx.hmm_ready.get(s))]
    if hmm_off:
        L.append(f"- Not enough H4 history for the HMM (filter has no effect on them): {', '.join(hmm_off)}")
    L.append(f"- Sizing: {pct(cfg.portfolio.position_vol_target, 0)} annualized vol per position "
             f"({cfg.portfolio.vol_lookback_days}-day realized vol), gross risk cap {pct(cfg.portfolio.max_gross_risk, 0)}")
    L.append(f"- Costs: broker spread x{cfg.costs.spread_multiplier:g} plus {pct(cfg.costs.slippage_spread_fraction, 0)} "
             f"of spread slippage per fill, commission ${cfg.costs.commission_per_lot_usd}/lot, daily swap; "
             f"pessimistic: spread x{cfg.costs.pessimistic_spread_multiplier:g}")
    L.append(f"- Carry book (from broker swaps, snapshot {meta_snapshot_time[:10]}): "
             + ", ".join(f"{'long' if d > 0 else 'short'} {s}" for s, d in ctx.carry_directions.items()))
    L.append("- Currency rates implied by swaps (relative, annual): "
             + ", ".join(f"{c} {r * 100:+.2f}%" for c, r in ctx.currency_rates.items()))

    L.append("\n## All strategies\n")
    rows = []
    for r in results:
        s = r["stats"]
        rows.append([
            r["name"], num(s["sharpe"]), pct(s["cagr"]), pct(s["vol"]), pct(s["max_dd"]),
            f"{pct(s['worst_year'])} ({s['worst_year_label']})",
            f"{s['longest_dd_days']}{'' if s['longest_dd_recovered'] else ' (open)'}",
            f"{num(s['sharpe_third1'])} / {num(s['sharpe_third2'])} / {num(s['sharpe_third3'])}",
            num(s["sharpe_pessimistic"]),
            f"{num(s['sharpe_x0.75'])} / {num(s['sharpe_x1.25'])}",
            f"{pct(s['mc_dd_p5'], 0)} / {pct(s['mc_dd_p50'], 0)} / {pct(s['mc_dd_p95'], 0)}",
            pct(r["cent"]["nearest"]["share_below_half_lot"], 0),
        ])
    L.append(table(
        ["Strategy", "Sharpe", "CAGR", "Vol", "Max DD", "Worst year", "Longest DD (days)",
         "Sharpe by third", "Sharpe, 2x spread", "Sharpe, lookbacks x0.75 / x1.25",
         "MC max DD p5/p50/p95", "$10: positions < 0.5 lot"],
        rows,
    ))
    L.append("\nSharpe by third: the evaluation period split into three equal consecutive parts "
             "(walk-forward consistency). MC: 5,000 reshuffles of weekly returns.")

    years = sorted(set().union(*[set(r["yearly"].index) for r in results]))
    L.append("\n## Calendar-year returns\n")
    L.append(table(["Strategy"] + [str(y) for y in years],
                   [[r["name"]] + [pct(r["yearly"].get(y, np.nan), 0) for y in years] for r in results]))

    L.append("\n## Per-symbol contribution (annualized, share of equity)\n")
    symbols = md.symbols
    L.append(table(["Symbol"] + [r["name"] for r in results],
                   [[s] + [f"{pct(r['per_symbol'].loc[s, 'ann_contribution'])} (SR {num(r['per_symbol'].loc[s, 'sharpe'], 1)})"
                           if s in r["per_symbol"].index else "-" for r in results] for s in symbols]))
    swap_rows = [[r["name"], pct(r["per_symbol"]["ann_swap"].sum()), pct(r["per_symbol"]["ann_cost"].sum()),
                  num(r["stats"]["turnover_x_per_year"], 1), num(r["stats"]["avg_gross_leverage"], 2)] for r in results]
    L.append("\n" + table(["Strategy", "Swap per year", "Spread+slippage per year", "Turnover (x equity/yr)",
                           "Avg gross leverage"], swap_rows))

    L.append(f"\n## $10 cent account (min lot {cfg.cent_account.min_lot}, contract x{cfg.cent_account.contract_size_multiplier:g}, "
             f"1:{cfg.cent_account.leverage:g})\n")
    L.append(table(
        ["Strategy", "Policy", "Positions < 0.5 lot (skipped)", "Positions < 1 lot", "CAGR", "Vol", "Max DD",
         "Lowest equity", "Final equity"],
        [[r["name"], policy, pct(c["share_below_half_lot"], 0), pct(c["share_below_one_lot"], 0), pct(c["cagr"]),
          pct(c["vol"]), pct(c["max_dd"]), f"${c['min_equity']:.2f}", f"${c['final_equity']:.2f}"]
         for r in results for policy, c in r["cent"].items()],
    ))
    L.append("\nnearest: round to the nearest lot (small positions become 0). "
             "at_least_min: any non-zero position gets at least 1 min lot (oversized).")

    feas = results[0]["feasibility"]
    L.append(f"\n**Minimum lot vs intended size at $10 ({results[0]['name']})**\n")
    L.append(table(
        ["Symbol", "Intended notional", "Min-lot notional", "Intended / min lot", "Equity for 1 min lot",
         "Min-lot margin"],
        [[s, f"${row['target_notional_usd']:.2f}", f"${row['min_lot_notional_usd']:.2f}",
          num(row["target_in_min_lots"]), f"${row['equity_for_1_min_lot']:.0f}", f"${row['min_lot_margin_usd']:.2f}"]
         for s, row in feas.iterrows()],
    ))

    L.append("\n## Caveats\n")
    L.append("- **Carry uses today's swap rates for the whole history.** MT5 only exposes current swaps, "
             "so the carry book is the one that looks attractive *now*, applied to years when rates were "
             "very different. Treat carry (and every swap cost) as biased; the holdout is the fairest test.")
    L.append("- **The swaps are this broker's.** Check that the implied currency rates in Setup resemble real "
             "policy-rate differentials before trusting carry. Demo servers often quote swaps that don't "
             "(both sides negative, or a low-rate currency ranked above a high-rate one).")
    L.append("- The $10 section uses one min lot / contract multiplier for every symbol; real cent accounts "
             "differ per symbol (this broker's index CFDs have min volume 0.1-1.0).")
    L.append("- Spreads are modelled from the current spread and recent bar spreads, not historical tick data.")
    L.append("- Monte Carlo reshuffling ignores autocorrelation in trend returns, so it understates "
             "clustered drawdowns.")
    return "\n".join(L) + "\n"


def run_suite(cfg: ResearchConfig, meta_all: dict, end_exclusive, eval_start, label: str):
    t0 = time.time()
    ctx = build_context(cfg, meta_all["symbols"], end_exclusive)
    start = eval_start or evaluation_start(ctx)
    logger.info("Evaluation period %s -> %s", start.date(), ctx.md.dates[-1].date())

    results = []
    for name in STRATEGIES:
        results.append(run_strategy(ctx, name, start))
        s = results[-1]["stats"]
        logger.info("%-18s Sharpe %5.2f CAGR %6.1f%% MaxDD %5.1f%%", name, s["sharpe"], s["cagr"] * 100, s["max_dd"] * 100)

    out_dir = RESULTS_DIR / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = render(ctx, start, results, label, meta_all["taken_at_utc"])
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    pd.DataFrame({r["name"]: r["stats"] for r in results}).T.to_csv(out_dir / "summary.csv")
    pd.DataFrame({r["name"]: r["equity"] for r in results}).to_csv(out_dir / "equity.csv")
    for r in results:
        r["per_symbol"].to_csv(out_dir / f"per_symbol_{r['name'].replace(' ', '').replace('+', '_')}.csv")
    logger.info("Done in %.0fs: %s", time.time() - t0, display_path(out_dir / "REPORT.md"))
    print(report)
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Portfolio research (trend / carry / HMM filter)")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--final-holdout", action="store_true", help="one-time run on the locked holdout")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_research_config()

    if args.download:
        download_research_data(cfg)
        return 0

    meta_all = load_meta()
    cutoff = holdout_start(meta_all, cfg.holdout_months)

    if not args.final_holdout:
        logger.info("Development run: data before %s only (holdout locked).", cutoff.date())
        run_suite(cfg, meta_all, end_exclusive=cutoff, eval_start=None, label="dev")
        return 0

    if HOLDOUT_MARKER.exists():
        logger.error("The holdout was already used (%s). It cannot be re-run.", HOLDOUT_MARKER.read_text())
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_MARKER.write_text(json.dumps({"used_at_utc": datetime.now(timezone.utc).isoformat(),
                                          "holdout_start": str(cutoff.date())}))
    run_suite(cfg, meta_all, end_exclusive=None, eval_start=cutoff, label="HOLDOUT")
    return 0


if __name__ == "__main__":
    sys.exit(main())

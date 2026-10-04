"""
Runs the full backtest suite and writes a report to results/<timestamp>/.

Every pre-declared variant is reported, not just the best one:
  - SL x TP grid from config (the live bot uses the config defaults regardless)
  - defaults with the HMM regime filter OFF (does the HMM add anything?)
  - defaults with pessimistic costs
For each variant: strategy statistics (per-trade/per-day rules only) and a
$10-account simulation with the exact risk rules, restarted at the start of
every month to measure how often the kill switch fires.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg_module
from src.config import Config, CostConfig, display_path
from src.backtest.engine import PreparedData, SimParams, Simulator, prepare_data
from src.backtest.metrics import by_year, daily_summary, summarize
from src.data.history import load_history, load_symbol_spec
from src.journal.journal import Journal
from src.risk.rules import estimate_margin

logger = logging.getLogger(__name__)

HORIZONS_MONTHS = (3, 6, 12)


def rolling_account_runs(sim: Simulator) -> pd.DataFrame:
    """A fresh $10 account started on the first bar of every month."""

    times = sim.time_utc
    first = sim.prep.first_idx
    month_id = times.year.to_numpy() * 12 + times.month.to_numpy()
    starts = [first] + [int(i) for i in np.flatnonzero(np.diff(month_id)) + 1 if i > first]
    data_end = times[-1]
    start_balance = sim.params.start_balance

    rows = []
    for s in starts:
        res = sim.run(start_idx=s, record_signals=False, equity_rules=True)
        start_time = times[s]
        first_alert = res.alerts[0][0] if res.alerts else None
        row = {
            "start": start_time,
            "trades": len(res.trades),
            "killed_at": res.killed_at,
            "days_to_kill": (res.killed_at - start_time).days if res.killed_at is not None else np.nan,
            "first_30_alert": first_alert,
            "final_equity": res.final_equity,
        }

        for h in HORIZONS_MONTHS:
            horizon_end = start_time + pd.DateOffset(months=h)
            complete = horizon_end <= data_end
            row[f"complete_{h}m"] = complete
            row[f"killed_{h}m"] = res.killed_at is not None and res.killed_at <= horizon_end
            row[f"hit30_{h}m"] = first_alert is not None and first_alert <= horizon_end

            if res.trades.empty:
                equity_h = start_balance
            else:
                closed = res.trades[res.trades["exit_time_utc"] <= horizon_end]
                equity_h = closed["equity_after"].iloc[-1] if not closed.empty else start_balance
            row[f"equity_{h}m"] = equity_h

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_rolling(rolling: pd.DataFrame) -> dict:
    out = {"starts": len(rolling)}

    for h in HORIZONS_MONTHS:
        done = rolling[rolling[f"complete_{h}m"]]
        out[f"n_{h}m"] = len(done)
        out[f"killed_{h}m"] = done[f"killed_{h}m"].mean() if len(done) else np.nan
        out[f"hit30_{h}m"] = done[f"hit30_{h}m"].mean() if len(done) else np.nan
        out[f"median_equity_{h}m"] = done[f"equity_{h}m"].median() if len(done) else np.nan

    out["killed_by_end"] = rolling["killed_at"].notna().mean()
    out["hit30_by_end"] = rolling["first_30_alert"].notna().mean()
    out["median_days_to_kill"] = rolling["days_to_kill"].median()
    return out


def build_variants(cfg: Config) -> list[tuple[str, SimParams]]:
    base = SimParams(
        strategy=cfg.strategy,
        risk=cfg.risk,
        costs=cfg.backtest.costs,
        regime_enabled=cfg.regime.enabled,
        start_balance=cfg.backtest.start_balance,
        leverage=cfg.virtual_account.leverage,
    )

    variants = []
    for sl in cfg.backtest.grid_sl_atr_mult:
        for tp in cfg.backtest.grid_tp_r_mult:
            is_default = sl == cfg.strategy.sl_atr_mult and tp == cfg.strategy.tp_r_mult
            name = f"SL {sl:g}xATR / TP {tp:g}R" + (" (DEFAULT)" if is_default else "")
            variants.append((name, replace(base, strategy=replace(cfg.strategy, sl_atr_mult=sl, tp_r_mult=tp))))

    hmm_toggle = "OFF" if cfg.regime.enabled else "ON"
    variants.append((f"Default, HMM filter {hmm_toggle}", replace(base, regime_enabled=not cfg.regime.enabled)))
    variants.append(("Default, pessimistic costs", replace(base, costs=cfg.backtest.pessimistic_costs)))
    variants.append((ZERO_COST_NAME, replace(base, costs=CostConfig(0.0, 0.0, 0.0))))
    return variants


ZERO_COST_NAME = "Default, zero costs (diagnostic, not tradeable)"


def random_entry_win_rate(trades: pd.DataFrame, costs: CostConfig, pip: float) -> float:
    """Win rate a driftless random walk would give with the same stops and
    entry costs: the entry pays spread + slippage, which moves the bid
    closer to the stop and further from the target."""

    if trades.empty:
        return np.nan

    sl_pips = (trades["entry_price"] - trades["sl"]).abs() / pip
    tp_pips = (trades["tp"] - trades["entry_price"]).abs() / pip
    entry_cost = trades["spread_pips"] + costs.slippage_pips
    return float(((sl_pips - entry_cost) / (sl_pips + tp_pips)).mean())


def _fmt(value, kind="f2"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "usd":
        return f"${value:.2f}"
    if kind == "int":
        return f"{int(value)}"
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return f"{value:.2f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def run_backtest(cfg: Config, data_path: Path | None = None) -> Path:
    t0 = time.time()
    raw = load_history(cfg, data_path)
    spec = load_symbol_spec(cfg)

    logger.info("Preparing %s bars (features + walk-forward HMM)...", len(raw))
    prep: PreparedData = prepare_data(raw, cfg)
    df = prep.df
    oos_start, data_end = df["time_utc"].iloc[prep.first_idx], df["time_utc"].iloc[-1]
    n_refits = len(range(cfg.regime.train_bars, len(df), cfg.regime.refit_every_bars))
    logger.info("Prepared in %.0fs. Out-of-sample: %s -> %s", time.time() - t0, oos_start, data_end)

    out_dir = cfg_module.RESULTS_DIR / f"backtest_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_rows, default_detail = [], None

    for name, params in build_variants(cfg):
        sim = Simulator(prep, spec, params)
        strat = sim.run(equity_rules=False)
        stats = summarize(strat.trades)
        rolling = rolling_account_runs(sim)
        roll = summarize_rolling(rolling)
        variant_rows.append({"variant": name, **stats, **{f"roll_{k}": v for k, v in roll.items()}})
        logger.info(
            "%-32s trades=%4d win=%s PF=%s exp=%sR net=$%.2f | killed(6m)=%s",
            name, stats["trades"], _fmt(stats["win_rate"], "pct"), _fmt(stats["profit_factor"]),
            _fmt(stats["expectancy_r"]), stats["net_usd"], _fmt(roll["killed_6m"], "pct"),
        )

        if "(DEFAULT)" in name:
            full = sim.run(equity_rules=True)
            default_detail = (name, sim, strat, stats, rolling, roll, full)

    if default_detail is None:
        raise RuntimeError("The config defaults are not part of the SL/TP grid.")

    name, sim, strat, stats, rolling, roll, full = default_detail

    # Journals: strategy run at the top level, the $10 path in ten_dollar/.
    start_balance = cfg.backtest.start_balance
    Journal(out_dir).write_frames(
        strat.signals, strat.trades, daily_summary(strat.signals, strat.trades, start_balance)
    )
    Journal(out_dir / "ten_dollar").write_frames(
        pd.DataFrame(), full.trades, daily_summary(pd.DataFrame(), full.trades, start_balance)
    )
    pd.DataFrame(variant_rows).to_csv(out_dir / "variants.csv", index=False)
    rolling.to_csv(out_dir / "rolling_10usd_starts.csv", index=False)

    report = render_report(cfg, spec, prep, n_refits, oos_start, data_end, variant_rows, default_detail)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    logger.info("Backtest finished in %.0fs. Report: %s", time.time() - t0, display_path(out_dir / "report.md"))
    print(report)
    return out_dir


def render_report(cfg, spec, prep, n_refits, oos_start, data_end, variant_rows, default_detail) -> str:
    name, sim, strat, stats, rolling, roll, full = default_detail
    costs, pess = cfg.backtest.costs, cfg.backtest.pessimistic_costs
    oos_regimes = pd.Series(prep.regimes[prep.first_idx:]).value_counts(normalize=True)

    L = []
    L.append(f"# Backtest report: {cfg.symbol} {cfg.strategy.timeframe}\n")
    L.append("## Data and assumptions\n")
    L.append(f"- Bars: {len(prep.df):,} ({prep.df['time_utc'].iloc[0]} to {data_end} UTC)")
    L.append(f"- HMM warm-up (no trading): first {cfg.regime.train_bars:,} bars. "
             f"**Out-of-sample period: {oos_start:%Y-%m-%d} to {data_end:%Y-%m-%d}**")
    L.append(f"- HMM: {cfg.regime.n_states} states on drift + volatility ({cfg.regime.feature_window} bars), "
             f"refit every {cfg.regime.refit_every_bars:,} bars on the trailing {cfg.regime.train_bars:,} "
             f"({n_refits} refits), forward-filtered (no look-ahead)")
    L.append("- OOS regime share: " + ", ".join(f"{k} {v:.0%}" for k, v in oos_regimes.items()))
    L.append(f"- Symbol: tick value ${spec.tick_value} per {spec.tick_size} per lot, "
             f"pip value at {sim.lot} lot = ${spec.pip_value(sim.lot):.2f}")
    L.append(f"- Costs: spread max(bar spread, {costs.spread_floor_pips} pip), slippage "
             f"{costs.slippage_pips} pip per market fill, commission ${costs.commission_per_lot_usd}/lot. "
             f"Pessimistic: {pess.spread_floor_pips} pip floor, ${pess.commission_per_lot_usd}/lot")
    L.append("- Fills: next bar open, SL before TP when both touched in one bar, gaps through SL fill at the open")
    L.append(f"- Risk: {cfg.risk.lot_size} lot, max ${cfg.risk.max_risk_usd:.2f} risk, "
             f"{cfg.risk.max_open_positions} position, {cfg.risk.max_trades_per_day} trades/day, "
             f"${cfg.risk.daily_loss_cap_usd:.2f} daily loss cap, kill at ${cfg.risk.kill_switch_equity:.2f}, "
             f"alert at ${cfg.risk.profit_alert_equity:.2f}\n")

    L.append(f"## Default strategy ({name}), strategy statistics\n")
    L.append("Per-trade and per-day rules applied; no equity floor, so the whole period is measured.\n")
    L.append(_md_table(
        ["Trades", "Win rate", "Profit factor", "Expectancy", "Net", "Max DD", "Longest losing streak"],
        [[str(stats["trades"]), _fmt(stats["win_rate"], "pct"), _fmt(stats["profit_factor"]),
          f"{_fmt(stats['expectancy_r'])}R / {_fmt(stats['expectancy_usd'], 'usd')}",
          f"{_fmt(stats['net_r'])}R / {_fmt(stats['net_usd'], 'usd')}",
          f"{_fmt(stats['max_dd_r'])}R / {_fmt(stats['max_dd_usd'], 'usd')}",
          str(stats["longest_losing_streak"])]],
    ))

    yearly = by_year(strat.trades)
    if not yearly.empty:
        L.append("\n**By year (exit year)**\n")
        L.append(_md_table(
            ["Year", "Trades", "Win rate", "PF", "Exp. R", "Net $"],
            [[str(y), _fmt(r["trades"], "int"), _fmt(r["win_rate"], "pct"), _fmt(r["profit_factor"]),
              _fmt(r["expectancy_r"]), _fmt(r["net_usd"], "usd")] for y, r in yearly.iterrows()],
        ))

    if not strat.signals.empty:
        primary = strat.signals["reason"].str.split(";").str[0]
        primary = primary.str.replace(r"^risk_\$.*", "risk_over_max_usd", regex=True)
        primary = primary.str.replace(r"^spread_.*", "spread_too_wide", regex=True)
        counts = primary.value_counts()
        L.append(f"\n**What happened to the {len(strat.signals):,} pullback setups** (first failing rule)\n")
        L.append(_md_table(["Outcome", "Count", "Share"],
                           [[k, str(v), f"{v / len(primary):.0%}"] for k, v in counts.items()]))

    if not strat.trades.empty:
        exits = strat.trades["exit_reason"].value_counts()
        L.append("\nExit reasons: " + ", ".join(f"{k} {v}" for k, v in exits.items()))
        baseline = random_entry_win_rate(strat.trades, cfg.backtest.costs, spec.pip_size)
        L.append(f"\n**Random-entry benchmark:** with these stops and entry costs, a driftless random walk "
                 f"wins {_fmt(baseline, 'pct')} of trades; the strategy won {_fmt(stats['win_rate'], 'pct')}. "
                 "A strategy with an edge must beat this number clearly.")

    L.append("\n## All variants (every one tested is reported)\n")
    L.append("Kill/hit-$30 columns: share of fresh $10 accounts started on the first of each month "
             "(only starts with the full horizon inside the data are counted).\n")
    L.append(_md_table(
        ["Variant", "Trades", "Win", "PF", "Exp R", "Exp $", "Net $", "Max DD $", "Lose streak",
         "Killed 3m", "Killed 6m", "Killed 12m", "Hit $30 12m", "Median eq. 6m"],
        [[v["variant"], str(v["trades"]), _fmt(v["win_rate"], "pct"), _fmt(v["profit_factor"]),
          _fmt(v["expectancy_r"]), _fmt(v["expectancy_usd"], "usd"), _fmt(v["net_usd"], "usd"),
          _fmt(v["max_dd_usd"], "usd"), str(v["longest_losing_streak"]),
          _fmt(v["roll_killed_3m"], "pct"), _fmt(v["roll_killed_6m"], "pct"), _fmt(v["roll_killed_12m"], "pct"),
          _fmt(v["roll_hit30_12m"], "pct"), _fmt(v["roll_median_equity_6m"], "usd")]
         for v in variant_rows],
    ))

    L.append("\n## $10 account simulation (default strategy, exact risk rules)\n")
    if full.killed_at is not None:
        L.append(f"- One account from the start of the out-of-sample period: **kill switch at "
                 f"{full.killed_at:%Y-%m-%d}** after {len(full.trades)} trades, final equity "
                 f"{_fmt(full.final_equity, 'usd')}")
    else:
        L.append(f"- One account from the start of the out-of-sample period: survived to {data_end:%Y-%m-%d}, "
                 f"{len(full.trades)} trades, final equity {_fmt(full.final_equity, 'usd')}")
    if full.alerts:
        L.append(f"- First $30 profit alert: {full.alerts[0][0]:%Y-%m-%d}")
    if not full.trades.empty:
        L.append(f"- Margin warnings at {cfg.virtual_account.leverage:g}:1 (0.01 lot needs about "
                 f"${estimate_margin(sim.lot, spec.contract_size, full.trades['entry_price'].mean(), cfg.virtual_account.leverage):.2f}): "
                 f"{int(full.trades['margin_warning'].sum())} of {len(full.trades)} trades")
    L.append(f"- Fresh $10 account each month ({roll['starts']} starts): killed by end of data "
             f"{_fmt(roll['killed_by_end'], 'pct')}, median days to kill {_fmt(roll['median_days_to_kill'])}, "
             f"reached $30 {_fmt(roll['hit30_by_end'], 'pct')}")
    L.append(_md_table(
        ["Horizon", "Starts counted", "Kill switch hit", "Reached $30", "Median equity"],
        [[f"{h} months", str(roll[f"n_{h}m"]), _fmt(roll[f"killed_{h}m"], "pct"),
          _fmt(roll[f"hit30_{h}m"], "pct"), _fmt(roll[f"median_equity_{h}m"], "usd")]
         for h in HORIZONS_MONTHS],
    ))

    L.append("\n## Caveats\n")
    L.append("- Spread is modelled, not measured: MT5 history only stores each bar's minimum spread.")
    L.append("- The kill switch is checked at bar closes here; live checks every few seconds.")
    L.append("- About 3 years of out-of-sample data is a small sample for a strategy trading a few times "
             "a week. Treat differences between variants within ~0.1 PF as noise.")
    return "\n".join(L) + "\n"

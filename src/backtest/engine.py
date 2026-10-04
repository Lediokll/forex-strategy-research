"""
Bar-by-bar backtest simulator.

Uses the same strategy (src.strategy.pullback) and risk code (src.risk.rules)
as the live bot. Assumptions, all deliberately conservative:

  - MT5 bars are BID prices. Longs buy at the ask (bid + spread) and exit on
    the bid; shorts sell at the bid and exit at the ask, so their SL/TP
    trigger when bid + spread touches the level.
  - Spread = max(bar spread, spread floor). MT5's bar spread is the bar's
    minimum, often 0, hence the floor.
  - Market fills (entries, stop-loss exits, kill-switch closes) pay
    `slippage_pips`. A stop gapped through fills at the bar open (worse).
    Take-profits fill at the TP price, never better.
  - If SL and TP are both touched within one bar, it counts as the SL.
  - Signals use bar i's close; entry is at bar i+1's open.
  - Commission is charged per round turn.
  - The kill switch and profit alert are checked at each bar close (live
    checks every few seconds, so live can only stop earlier, never later).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config, CostConfig, RiskConfig, StrategyConfig
from src.data.symbol_spec import SymbolSpec
from src.features.feature_pipeline import FeaturePipeline
from src.regime.regime_detector import walk_forward_labels
from src.risk.rules import (
    AccountSnapshot,
    DailyLedger,
    ProfitAlert,
    RiskManager,
    dollar_risk,
    estimate_margin,
    lot_size,
    margin_warning,
)
from src.strategy.pullback import (
    BUY,
    compute_setups,
    direction_name,
    plan_trade,
    sl_tp_prices,
    strategy_skip_reasons,
)

NS_PER_DAY = 86_400 * 10**9


@dataclass
class PreparedData:
    df: pd.DataFrame
    regimes: np.ndarray
    first_idx: int            # first row with a walk-forward regime label


def prepare_data(raw_df: pd.DataFrame, cfg: Config) -> PreparedData:
    """Features + walk-forward regimes. `raw_df` must already have time_utc."""

    df = FeaturePipeline(raw_df, cfg.strategy, cfg.regime).run()

    if len(df) <= cfg.regime.train_bars + 1000:
        raise ValueError(
            f"Only {len(df)} usable bars; need more than regime.train_bars "
            f"({cfg.regime.train_bars}) + 1000 for an out-of-sample period."
        )

    regimes = walk_forward_labels(df, cfg.regime)
    return PreparedData(df=df, regimes=regimes, first_idx=cfg.regime.train_bars)


@dataclass(frozen=True)
class SimParams:
    strategy: StrategyConfig
    risk: RiskConfig
    costs: CostConfig
    regime_enabled: bool = True
    start_balance: float = 10.0
    leverage: float = 500.0


@dataclass
class SimResult:
    trades: pd.DataFrame
    signals: pd.DataFrame
    killed_at: pd.Timestamp | None = None
    alerts: list = field(default_factory=list)
    end_time: pd.Timestamp | None = None
    final_equity: float = 0.0


class Simulator:
    """Precomputes everything that doesn't depend on account state, so the
    many runs ($10 restarts every month) only replay the account logic."""

    def __init__(self, prep: PreparedData, spec: SymbolSpec, params: SimParams):
        self.prep = prep
        self.spec = spec
        self.params = params
        df = prep.df

        self.n = len(df)
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.atr = df["atr"].to_numpy(float)
        # pandas >= 3 infers the datetime unit; pin ns so the day arithmetic below holds.
        self.time_utc = pd.DatetimeIndex(df["time_utc"]).as_unit("ns")
        self.bar_delta = pd.Timedelta(minutes=params.strategy.bar_minutes)

        pip = spec.pip_size
        bar_spread_pips = df["spread"].to_numpy(float) * spec.point / pip
        self.spread_pips = np.maximum(bar_spread_pips, params.costs.spread_floor_pips)
        self.spread_price = self.spread_pips * pip
        self.slip_price = params.costs.slippage_pips * pip

        time_ns = self.time_utc.asi8
        self.bar_day = time_ns // NS_PER_DAY
        self.entry_day = (time_ns + self.bar_delta.value) // NS_PER_DAY

        self.lot = lot_size(spec, params.risk.lot_size)
        self.commission = params.costs.commission_per_lot_usd * self.lot
        self.usd_per_price = spec.tick_value / spec.tick_size * self.lot

        setups = compute_setups(df, params.strategy)
        setups[: prep.first_idx] = 0
        setups[-1] = 0                      # no next bar to enter on
        self.setups = setups
        self.setup_idx = np.flatnonzero(setups)

        # Strategy-level filters depend only on bar data: evaluate once.
        self.strategy_reasons: dict[int, list[str]] = {}
        for i in self.setup_idx:
            self.strategy_reasons[int(i)] = strategy_skip_reasons(
                prep.regimes[i],
                (self.time_utc[i] + self.bar_delta).to_pydatetime(),
                float(self.spread_pips[i]),
                params.strategy,
                params.regime_enabled,
            )

    def day_date(self, day_number: int):
        return (pd.Timestamp(0) + pd.Timedelta(days=int(day_number))).date()

    def run(self, start_idx: int | None = None, record_signals: bool = True, equity_rules: bool = True) -> SimResult:
        """Replay from `start_idx` (default: first out-of-sample bar).

        equity_rules=True is the "$10 account": the kill switch ends the run
        and equity-based checks apply. False measures the strategy itself
        with only the per-trade and per-day rules (no equity floor).
        """

        p = self.params
        risk = RiskManager(p.risk)
        ledger = DailyLedger()
        alert = ProfitAlert(p.risk.profit_alert_equity)

        start_idx = self.prep.first_idx if start_idx is None else start_idx
        equity = p.start_balance
        pos = None
        trades, signals, alerts = [], [], []
        killed_at = None
        last_i = start_idx

        def close_position(j: int, exit_price: float, reason: str, exit_slip_pips: float):
            nonlocal equity, pos
            sign = 1 if pos["direction"] == BUY else -1
            pnl = sign * (exit_price - pos["entry_price"]) * self.usd_per_price - self.commission
            equity += pnl
            ledger.record_exit(self.day_date(self.bar_day[j]), pnl)
            trades.append({
                **pos["record"],
                "exit_time_utc": self.time_utc[j],
                "exit_price": round(exit_price, 6),
                "exit_reason": reason,
                "exit_slippage_pips": exit_slip_pips,
                "commission_usd": self.commission,
                "pnl_usd": pnl,
                "r_multiple": pnl / pos["risk_usd"],
                "equity_after": equity,
            })
            pos = None

        def open_pnl_at_close(j: int) -> float:
            if pos["direction"] == BUY:
                diff = self.c[j] - pos["entry_price"]
            else:
                diff = pos["entry_price"] - (self.c[j] + self.spread_price[j])
            return diff * self.usd_per_price - self.commission

        i = start_idx
        while i < self.n:
            last_i = i

            # 1. Manage the open position during bar i.
            if pos is not None:
                sl, tp, spr = pos["sl"], pos["tp"], self.spread_price[i]

                if pos["direction"] == BUY:
                    hit_sl, hit_tp = self.l[i] <= sl, self.h[i] >= tp
                    sl_fill = min(self.o[i], sl) - self.slip_price
                else:
                    hit_sl, hit_tp = self.h[i] + spr >= sl, self.l[i] + spr <= tp
                    sl_fill = max(self.o[i] + spr, sl) + self.slip_price

                if hit_sl:
                    close_position(i, sl_fill, "sl", p.costs.slippage_pips)
                elif hit_tp:
                    close_position(i, tp, "tp", 0.0)

            # 2. Equity checks at bar close.
            mark_equity = equity + (open_pnl_at_close(i) if pos is not None else 0.0)

            if equity_rules and risk.kill_switch_triggered(mark_equity):
                if pos is not None:
                    if pos["direction"] == BUY:
                        fill = self.c[i] - self.slip_price
                    else:
                        fill = self.c[i] + self.spread_price[i] + self.slip_price
                    close_position(i, fill, "kill_switch", p.costs.slippage_pips)
                killed_at = self.time_utc[i]
                break

            if alert.check(mark_equity, self.day_date(self.bar_day[i])):
                alerts.append((self.time_utc[i], mark_equity))

            # 3. Evaluate a setup at bar i's close.
            if self.setups[i] != 0 and i + 1 < self.n:
                row, new_pos = self._evaluate_signal(i, pos, mark_equity, risk, ledger, equity_rules)
                if record_signals:
                    signals.append(row)
                if new_pos is not None:
                    pos = new_pos
                    ledger.record_entry(self.day_date(self.entry_day[i]))

            # 4. Next bar: step while a position is open, else jump to the next setup.
            if pos is not None:
                i += 1
            else:
                k = np.searchsorted(self.setup_idx, i + 1)
                if k >= len(self.setup_idx):
                    break
                i = int(self.setup_idx[k])

        if pos is not None:
            j = self.n - 1
            fill = self.c[j] if pos["direction"] == BUY else self.c[j] + self.spread_price[j]
            close_position(j, fill, "end_of_data", 0.0)

        return SimResult(
            trades=pd.DataFrame(trades),
            signals=pd.DataFrame(signals),
            killed_at=killed_at,
            alerts=alerts,
            end_time=self.time_utc[last_i] if killed_at is None else killed_at,
            final_equity=equity,
        )

    def _evaluate_signal(self, i, pos, mark_equity, risk, ledger, equity_rules) -> tuple[dict, dict | None]:
        """Signal-log row for the setup at bar i, plus the new position if taken."""

        p = self.params
        direction = direction_name(self.setups[i])
        plan = plan_trade(direction, float(self.atr[i]), p.strategy, self.spec.min_stop_distance)
        risk_usd = dollar_risk(plan.sl_distance, self.lot, self.spec)

        ledger.roll(self.day_date(self.entry_day[i]))
        snap = AccountSnapshot(
            equity=mark_equity if equity_rules else float("inf"),
            open_positions=0 if pos is None else 1,
            trades_today=ledger.trades,
            realized_pnl_today=ledger.realized_pnl,
        )

        reasons = list(self.strategy_reasons[i])
        decision = risk.check_entry(snap, risk_usd)
        reasons += [r for r in decision.reasons if r not in reasons]

        e = i + 1
        entry_bid = self.o[e]
        margin_note = margin_warning(
            estimate_margin(self.lot, self.spec.contract_size, entry_bid, p.leverage),
            mark_equity,
        )

        row = {
            "time_utc": self.time_utc[i],
            "direction": direction,
            "taken": not reasons,
            "reason": ";".join(reasons) if reasons else "taken",
            "regime": self.prep.regimes[i],
            "close": self.c[i],
            "atr": self.atr[i],
            "sl_distance_pips": plan.sl_distance / self.spec.pip_size,
            "spread_pips": self.spread_pips[i],
            "risk_usd": risk_usd,
            "equity": mark_equity,
            "margin_warning": bool(margin_note),
        }

        if reasons:
            return row, None

        spr = self.spread_price[e]
        if direction == BUY:
            requested = entry_bid + spr
            entry_price = requested + self.slip_price
        else:
            requested = entry_bid
            entry_price = requested - self.slip_price
        sl, tp = sl_tp_prices(direction, entry_price, plan)

        new_pos = {
            "direction": direction,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "risk_usd": risk_usd,
            "record": {
                "entry_time_utc": self.time_utc[e],
                "direction": direction,
                "regime": self.prep.regimes[i],
                "lot": self.lot,
                "requested_price": round(requested, 6),
                "entry_price": round(entry_price, 6),
                "sl": round(sl, 6),
                "tp": round(tp, 6),
                "spread_pips": self.spread_pips[e],
                "entry_slippage_pips": p.costs.slippage_pips,
                "risk_usd": risk_usd,
                "margin_warning": bool(margin_note),
            },
        }
        return row, new_pos

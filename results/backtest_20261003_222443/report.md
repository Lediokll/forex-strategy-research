# Backtest report: EURUSD M15

## Data and assumptions

- Bars: 98,984 (2022-10-07 20:15:00 to 2026-10-02 20:30:00 UTC)
- HMM warm-up (no trading): first 25,000 bars. **Out-of-sample period: 2023-10-10 to 2026-10-02**
- HMM: 3 states on drift + volatility (16 bars), refit every 2,000 bars on the trailing 25,000 (37 refits), forward-filtered (no look-ahead)
- OOS regime share: trend 37%, high_vol 32%, chop 31%
- Symbol: tick value $1.0 per 1e-05 per lot, pip value at 0.01 lot = $0.10
- Costs: spread max(bar spread, 1.0 pip), slippage 0.2 pip per market fill, commission $0.0/lot. Pessimistic: 1.5 pip floor, $7.0/lot
- Fills: next bar open, SL before TP when both touched in one bar, gaps through SL fill at the open
- Risk: 0.01 lot, max $1.00 risk, 1 position, 3 trades/day, $2.00 daily loss cap, kill at $6.00, alert at $30.00

## Default strategy (SL 1.5xATR / TP 2R (DEFAULT)), strategy statistics

Per-trade and per-day rules applied; no equity floor, so the whole period is measured.

| Trades | Win rate | Profit factor | Expectancy | Net | Max DD | Longest losing streak |
|---|---|---|---|---|---|---|
| 690 | 28% | 0.78 | -0.17R / $-0.12 | -119.48R / $-85.05 | 134.01R / $97.49 | 14 |

**By year (exit year)**

| Year | Trades | Win rate | PF | Exp. R | Net $ |
|---|---|---|---|---|---|
| 2023 | 88 | 28% | 0.80 | -0.17 | $-10.09 |
| 2024 | 291 | 29% | 0.77 | -0.17 | $-33.36 |
| 2025 | 128 | 29% | 0.77 | -0.15 | $-18.05 |
| 2026 | 183 | 27% | 0.78 | -0.20 | $-23.55 |

**What happened to the 11,216 pullback setups** (first failing rule)

| Outcome | Count | Share |
|---|---|---|
| regime_chop | 4094 | 37% |
| regime_high_vol | 2734 | 24% |
| outside_session | 2450 | 22% |
| taken | 690 | 6% |
| max_open_positions | 652 | 6% |
| risk_over_max_usd | 459 | 4% |
| max_trades_per_day | 132 | 1% |
| daily_loss_cap | 4 | 0% |
| spread_too_wide | 1 | 0% |

Exit reasons: sl 495, tp 195

**Random-entry benchmark:** with these stops and entry costs, a driftless random walk wins 28% of trades; the strategy won 28%. A strategy with an edge must beat this number clearly.

## All variants (every one tested is reported)

Kill/hit-$30 columns: share of fresh $10 accounts started on the first of each month (only starts with the full horizon inside the data are counted).

| Variant | Trades | Win | PF | Exp R | Exp $ | Net $ | Max DD $ | Lose streak | Killed 3m | Killed 6m | Killed 12m | Hit $30 12m | Median eq. 6m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SL 1xATR / TP 1.5R | 1093 | 30% | 0.62 | -0.29 | $-0.16 | $-171.95 | $175.76 | 12 | 94% | 100% | 100% | 0% | $5.80 |
| SL 1xATR / TP 2R | 1060 | 25% | 0.64 | -0.29 | $-0.16 | $-169.24 | $173.67 | 16 | 94% | 100% | 100% | 0% | $5.74 |
| SL 1.5xATR / TP 1.5R | 724 | 33% | 0.74 | -0.18 | $-0.13 | $-97.06 | $103.88 | 12 | 82% | 100% | 100% | 0% | $5.83 |
| SL 1.5xATR / TP 2R (DEFAULT) | 690 | 28% | 0.78 | -0.17 | $-0.12 | $-85.05 | $97.49 | 14 | 74% | 100% | 100% | 0% | $5.84 |
| SL 2xATR / TP 1.5R | 314 | 36% | 0.81 | -0.12 | $-0.11 | $-33.21 | $37.44 | 15 | 47% | 55% | 72% | 0% | $5.97 |
| SL 2xATR / TP 2R | 300 | 30% | 0.86 | -0.11 | $-0.08 | $-25.23 | $35.19 | 15 | 50% | 52% | 56% | 0% | $5.95 |
| Default, HMM filter OFF | 1148 | 27% | 0.74 | -0.21 | $-0.14 | $-164.26 | $176.82 | 18 | 88% | 100% | 100% | 0% | $5.74 |
| Default, pessimistic costs | 702 | 26% | 0.63 | -0.32 | $-0.23 | $-159.68 | $168.86 | 15 | 91% | 100% | 100% | 0% | $5.83 |
| Default, zero costs (diagnostic, not tradeable) | 682 | 33% | 0.97 | -0.02 | $-0.02 | $-10.77 | $33.55 | 12 | 56% | 65% | 92% | 8% | $5.89 |

## $10 account simulation (default strategy, exact risk rules)

- One account from the start of the out-of-sample period: **kill switch at 2023-12-08** after 59 trades, final equity $5.96
- Margin warnings at 500:1 (0.01 lot needs about $2.15): 0 of 59 trades
- Fresh $10 account each month (37 starts): killed by end of data 95%, median days to kill 36.00, reached $30 0%
| Horizon | Starts counted | Kill switch hit | Reached $30 | Median equity |
|---|---|---|---|---|
| 3 months | 34 | 74% | 0% | $5.90 |
| 6 months | 31 | 100% | 0% | $5.84 |
| 12 months | 25 | 100% | 0% | $5.84 |

## Caveats

- Spread is modelled, not measured: MT5 history only stores each bar's minimum spread.
- The kill switch is checked at bar closes here; live checks every few seconds.
- About 3 years of out-of-sample data is a small sample for a strategy trading a few times a week. Treat differences between variants within ~0.1 PF as noise.

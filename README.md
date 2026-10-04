# Forex Strategy Research & Backtesting Framework

Python tooling for MetaTrader 5 (Windows) built around one goal: **controlled risk and
honest measurement** for a very small (e.g. $10) account. It contains:

1. **An EURUSD M15 strategy backtester** (`src/`): EMA pullback in the EMA50/200 trend,
   HMM regime filter, session and spread filters, and hard account-level risk rules
   (fixed lot, max $ risk per trade, daily loss cap, kill switch, profit alert), plus a
   "virtual balance" so a large demo account can behave like a $10 one.
2. **A multi-symbol portfolio research suite** (`research/`): trend, carry and their
   combination, with an HMM risk filter, volatility-targeted sizing, swap and spread costs,
   a locked holdout, parameter-neighbourhood and Monte Carlo checks, and a cent-account
   feasibility analysis. See [research/README.md](research/README.md).

> **Results so far are negative.** The M15 strategy has no edge after costs (profit factor
> 0.78; win rate equal to a random-entry benchmark). The portfolio strategies are weak and
> not robust (best Sharpe 0.28, fading in recent years). Reports are in `results/` and
> `research/results/`. Nothing here is investment advice.

## Results

| Strategy | Sharpe / profit factor | Verdict |
|---|---|---|
| EURUSD M15 EMA pullback (OOS 2023-10 to 2026-10) | PF 0.78, win rate 28% (random entries: 28%) | No edge after costs |
| Trend (15 symbols, 2002-2025) | Sharpe 0.28 | Weak; fading, last third negative (-0.09) |
| Carry | Sharpe -0.01 | No edge |
| Trend + Carry | Sharpe 0.25 | Lower drawdown, still weak |
| Trend + HMM filter | Sharpe 0.23 | Filter did not help |
| Carry + HMM filter | Sharpe -0.07 | Filter did not help |
| Trend + Carry + HMM filter | Sharpe 0.21 | Not worth deploying |

## What I learned

1. **Costs decide small-timeframe strategies.** The M15 pullback looked plausible until spread and slippage were applied; its win rate matched a random-entry benchmark.
2. **A regime filter is not a free lunch.** The HMM filter lowered Sharpe for every strategy it was applied to.
3. **Weak edges are not stable.** Trend's Sharpe was 0.46 / 0.47 / -0.09 across the three equal thirds of the sample, and shifting lookbacks by +/-25% cut it to 0.10-0.19.
4. **A tiny account cannot follow a portfolio strategy.** On $10, 93-99% of the sized positions are below the minimum lot, so the backtested portfolio is not the one you would actually trade.

## Setup

```
pip install -r requirements.txt
```

The `MetaTrader5` package needs Windows and a running, logged-in MT5 terminal for downloads
(backtests and tests run offline). **Market data is not included in this repository**;
download it from your own broker:

```
py main.py --download                # EURUSD M15 history + symbol spec -> data/raw/
py -m research.run --download        # D1/H4 bars + swap snapshot for the research universe
```

## Run

```
py main.py --mode backtest           # walk-forward backtest of the M15 strategy -> results/
py -m research.run                   # portfolio research (holdout excluded) -> research/results/
py -m pytest                         # unit tests (no MT5 needed)
```

The live/demo trading loop is **not implemented** (`--mode demo|live` exits).

## Configuration

- `config.yaml`: all strategy, risk and backtest settings. Unknown keys are rejected.
- `config.local.yaml` (gitignored, optional): private values merged on top of `config.yaml`,
  such as `live.account_number`. Start from `config.local.example.yaml`.
- `research/config.yaml`: research parameters, fixed before any result was seen.

Never commit `config.local.yaml`, `state/` or `logs/`; they hold account details.

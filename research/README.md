# Portfolio research: trend, carry, HMM risk filter

A separate experiment from the M15 bot. It reuses the bot's feature pipeline,
walk-forward HMM (`src/regime`) and strict config loader.

## Run

```
py -m research.run --download        # D1 + H4 bars and a symbol_info snapshot (swaps, spreads) from MT5
py -m research.run                   # development run: everything except the locked holdout
py -m research.run --final-holdout   # ONE time only, after the development results are final
```

Reports go to `research/results/<label>_<timestamp>/REPORT.md` (plus CSVs).

## Rules of the experiment

- All parameters are in `research/config.yaml` and were fixed before any result was seen.
  The only variations are the pre-declared ones: lookbacks x0.75 / x1.25 and 2x spread.
- The most recent 12 months are a holdout. The development run never loads those bars
  (prices, HMM and spread estimates are all cut off). `--final-holdout` writes
  `research/results/HOLDOUT_USED.json` and refuses to run a second time.
- Every strategy is evaluated over the same period.

## What the strategies do

| Strategy | Signal | Rebalance |
|---|---|---|
| Trend | mean of sign(3, 6, 12-month return) per symbol | weekly |
| Carry | long the 3 highest-rate currencies vs the 3 lowest (rates implied by broker swaps), one USD pair per currency | monthly |
| Trend+Carry | 50/50 of the two signals | weekly |
| +HMM | same, positions halved while the symbol's H4 HMM is in its high-vol state | same |

Each position is sized to the same annualized volatility (2% of equity, 60-day realized
vol); if the positions add up to more than 20%, all are scaled down. Costs: spread +
slippage on every fill, commission, and swap every night held.

## Known limitations

- MT5 only gives today's swap rates, so carry and swap costs use them for the whole history.
- Spreads are estimated, not taken from historical ticks.
- The $10 cent-account section assumes min lot = lot step and a lot of standard contract x
  `contract_size_multiplier`; check your broker's actual cent contract specs.

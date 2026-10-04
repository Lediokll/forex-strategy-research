# Research report: dev

## Setup

- Evaluation period: **2002-11-26 to 2025-09-30** (22.8 years); same period for every strategy
- Symbols (15): EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURJPY, GBPJPY, AUDJPY, EURGBP, XAUUSD, US500, DE40, JPN225
- Excluded UK100: only 43% of weekday bars present
- Start of each symbol's data: EURUSD 1971, GBPUSD 1993, USDJPY 1971, AUDUSD 1993, NZDUSD 1994, USDCAD 1993, USDCHF 1971, EURJPY 1993, GBPJPY 1993, AUDJPY 1993, EURGBP 1993, XAUUSD 2004, US500 2012, DE40 2012, JPN225 2019
- Sizing: 2% annualized vol per position (60-day realized vol), gross risk cap 20%
- Costs: broker spread x1 plus 25% of spread slippage per fill, commission $0.0/lot, daily swap; pessimistic: spread x2
- Carry book (from broker swaps, snapshot 2026-10-03): long AUDUSD, long NZDUSD, long GBPUSD, long USDCAD, long USDCHF
- Currency rates implied by swaps (relative, annual): AUD +3.46%, NZD +2.00%, GBP -0.61%, JPY -0.62%, EUR -0.78%, CAD -1.04%, USD -1.09%, CHF -1.33%

## All strategies

| Strategy | Sharpe | CAGR | Vol | Max DD | Worst year | Longest DD (days) | Sharpe by third | Sharpe, 2x spread | Sharpe, lookbacks x0.75 / x1.25 | MC max DD p5/p50/p95 | $10: positions < 0.5 lot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Trend | 0.28 | 2.5% | 10.7% | 27.7% | -10.9% (2018) | 2804 (open) | 0.46 / 0.47 / -0.09 | 0.24 | 0.10 / 0.19 | 25% / 34% / 48% | 93% |
| Carry | -0.01 | -0.1% | 4.0% | 20.4% | -10.2% (2008) | 6644 (open) | 0.11 / 0.11 / -0.25 | -0.01 | -0.03 / -0.00 | 14% / 19% / 27% | 99% |
| Trend+Carry | 0.25 | 1.4% | 6.2% | 18.8% | -6.8% (2018) | 2806 (open) | 0.45 / 0.48 / -0.18 | 0.21 | 0.09 / 0.16 | 15% / 22% / 32% | 95% |
| Trend +HMM | 0.23 | 1.8% | 9.5% | 27.9% | -11.6% (2018) | 2804 (open) | 0.45 / 0.43 / -0.16 | 0.18 | 0.07 / 0.12 | 23% / 33% / 46% | 94% |
| Carry +HMM | -0.07 | -0.3% | 3.6% | 23.0% | -7.8% (2008) | 6644 (open) | 0.12 / 0.00 / -0.33 | -0.08 | -0.09 / -0.06 | 14% / 20% / 27% | 99% |
| Trend+Carry +HMM | 0.21 | 1.1% | 5.4% | 17.9% | -7.0% (2018) | 2806 (open) | 0.42 / 0.43 / -0.19 | 0.17 | 0.07 / 0.11 | 14% / 20% / 29% | 95% |

Sharpe by third: the evaluation period split into three equal consecutive parts (walk-forward consistency). MC: 5,000 reshuffles of weekly returns.

## Calendar-year returns

| Strategy | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trend | 4% | 25% | 1% | -10% | -1% | 3% | 21% | -2% | 5% | 1% | -0% | 15% | 9% | -1% | -0% | 6% | -11% | -8% | 6% | -1% | 5% | -1% | -2% | -1% |
| Carry | 1% | 7% | 2% | -3% | 5% | -2% | -10% | 5% | -1% | 0% | 2% | -1% | 2% | -1% | -3% | 2% | -2% | -2% | -1% | -1% | -3% | -1% | 2% | 0% |
| Trend+Carry | 3% | 16% | 1% | -6% | 3% | 0% | 6% | 2% | 3% | 0% | 1% | 8% | 6% | -2% | -1% | 5% | -7% | -4% | 2% | 0% | 1% | -0% | -2% | -1% |
| Trend +HMM | 3% | 21% | 2% | -6% | 0% | 3% | 11% | -2% | 5% | 2% | -1% | 13% | 7% | -1% | -4% | 7% | -12% | -8% | 5% | -1% | 2% | -2% | -1% | -1% |
| Carry +HMM | 0% | 7% | 3% | -2% | 4% | -2% | -8% | 2% | -0% | -2% | 2% | -2% | 2% | -3% | -3% | 3% | -3% | -1% | -1% | -1% | -2% | -2% | 1% | 0% |
| Trend+Carry +HMM | 2% | 14% | 2% | -5% | 3% | 0% | 2% | 0% | 2% | 0% | 1% | 7% | 4% | -1% | -3% | 5% | -7% | -4% | 3% | 1% | 1% | -1% | 0% | -1% |

## Per-symbol contribution (annualized, share of equity)

| Symbol | Trend | Carry | Trend+Carry | Trend +HMM | Carry +HMM | Trend+Carry +HMM |
|---|---|---|---|---|---|---|
| EURUSD | 0.6% (SR 0.3) | 0.0% (SR n/a) | 0.3% (SR 0.3) | 0.5% (SR 0.3) | 0.0% (SR n/a) | 0.2% (SR 0.3) |
| GBPUSD | 0.4% (SR 0.3) | -0.1% (SR -0.1) | 0.2% (SR 0.1) | 0.3% (SR 0.2) | -0.1% (SR -0.1) | 0.1% (SR 0.1) |
| USDJPY | 0.4% (SR 0.2) | 0.0% (SR n/a) | 0.2% (SR 0.2) | 0.4% (SR 0.3) | 0.0% (SR n/a) | 0.2% (SR 0.2) |
| AUDUSD | 0.1% (SR 0.0) | 0.6% (SR 0.3) | 0.5% (SR 0.3) | 0.0% (SR 0.0) | 0.4% (SR 0.2) | 0.4% (SR 0.3) |
| NZDUSD | 0.0% (SR 0.0) | 0.4% (SR 0.2) | 0.2% (SR 0.2) | -0.0% (SR -0.0) | 0.2% (SR 0.1) | 0.2% (SR 0.1) |
| USDCAD | 0.2% (SR 0.1) | -0.1% (SR -0.0) | 0.1% (SR 0.0) | -0.0% (SR -0.0) | 0.0% (SR 0.0) | 0.0% (SR 0.0) |
| USDCHF | -0.5% (SR -0.3) | -0.8% (SR -0.3) | -0.7% (SR -0.5) | -0.4% (SR -0.3) | -0.8% (SR -0.4) | -0.6% (SR -0.5) |
| EURJPY | 0.2% (SR 0.1) | 0.0% (SR n/a) | 0.1% (SR 0.1) | 0.2% (SR 0.1) | 0.0% (SR n/a) | 0.1% (SR 0.1) |
| GBPJPY | 0.2% (SR 0.1) | 0.0% (SR n/a) | 0.1% (SR 0.1) | 0.1% (SR 0.0) | 0.0% (SR n/a) | 0.0% (SR 0.0) |
| AUDJPY | 0.1% (SR 0.1) | 0.0% (SR n/a) | 0.0% (SR 0.0) | 0.0% (SR 0.0) | 0.0% (SR n/a) | 0.0% (SR 0.0) |
| EURGBP | -0.1% (SR -0.0) | 0.0% (SR n/a) | -0.1% (SR -0.1) | -0.1% (SR -0.1) | 0.0% (SR n/a) | -0.1% (SR -0.1) |
| XAUUSD | 1.0% (SR 0.6) | 0.0% (SR n/a) | 0.5% (SR 0.6) | 1.0% (SR 0.7) | 0.0% (SR n/a) | 0.5% (SR 0.6) |
| US500 | 0.3% (SR 0.2) | 0.0% (SR n/a) | 0.1% (SR 0.2) | 0.3% (SR 0.2) | 0.0% (SR n/a) | 0.1% (SR 0.2) |
| DE40 | 0.0% (SR 0.0) | 0.0% (SR n/a) | 0.0% (SR 0.0) | -0.1% (SR -0.0) | 0.0% (SR n/a) | -0.0% (SR -0.0) |
| JPN225 | 0.1% (SR 0.1) | 0.0% (SR n/a) | 0.0% (SR 0.1) | 0.1% (SR 0.1) | 0.0% (SR n/a) | 0.1% (SR 0.1) |

| Strategy | Swap per year | Spread+slippage per year | Turnover (x equity/yr) | Avg gross leverage |
|---|---|---|---|---|
| Trend | -1.0% | -0.4% | 31.5 | 1.98 |
| Carry | 1.0% | -0.0% | 1.4 | 1.17 |
| Trend+Carry | 0.3% | -0.2% | 16.1 | 1.24 |
| Trend +HMM | -0.8% | -0.5% | 36.8 | 1.82 |
| Carry +HMM | 0.9% | -0.0% | 2.7 | 1.03 |
| Trend+Carry +HMM | 0.3% | -0.3% | 19.1 | 1.11 |

## $10 cent account (min lot 0.01, contract x0.01, 1:500)

| Strategy | Policy | Positions < 0.5 lot (skipped) | Positions < 1 lot | CAGR | Vol | Max DD | Lowest equity | Final equity |
|---|---|---|---|---|---|---|---|---|
| Trend | nearest | 93% | 96% | 0.7% | 2.4% | 8.1% | $9.76 | $11.68 |
| Trend | at_least_min | 52% | 75% | 7.7% | 32.1% | 61.4% | $8.61 | $54.27 |
| Carry | nearest | 99% | 100% | 0.2% | 1.4% | 5.1% | $9.99 | $10.49 |
| Carry | at_least_min | 99% | 100% | -0.7% | 20.1% | 63.2% | $6.07 | $8.57 |
| Trend+Carry | nearest | 95% | 97% | 0.2% | 1.1% | 3.6% | $9.96 | $10.55 |
| Trend+Carry | at_least_min | 67% | 86% | 7.5% | 29.1% | 59.5% | $7.41 | $52.07 |
| Trend +HMM | nearest | 94% | 96% | 0.3% | 2.5% | 9.1% | $9.74 | $10.63 |
| Trend +HMM | at_least_min | 56% | 77% | 7.6% | 32.1% | 61.4% | $8.61 | $53.15 |
| Carry +HMM | nearest | 99% | 100% | -0.1% | 1.3% | 11.3% | $9.76 | $9.80 |
| Carry +HMM | at_least_min | 99% | 100% | -0.7% | 20.1% | 63.2% | $6.07 | $8.57 |
| Trend+Carry +HMM | nearest | 95% | 97% | 0.3% | 1.1% | 3.7% | $9.96 | $10.59 |
| Trend+Carry +HMM | at_least_min | 70% | 87% | 7.5% | 29.1% | 59.5% | $7.41 | $52.10 |

nearest: round to the nearest lot (small positions become 0). at_least_min: any non-zero position gets at least 1 min lot (oversized).

**Minimum lot vs intended size at $10 (Trend)**

| Symbol | Intended notional | Min-lot notional | Intended / min lot | Equity for 1 min lot | Min-lot margin |
|---|---|---|---|---|---|
| XAUUSD | $0.88 | $38.58 | 0.02 | $440 | $0.08 |
| GBPJPY | $1.25 | $13.44 | 0.09 | $108 | $0.03 |
| EURJPY | $1.48 | $11.73 | 0.13 | $79 | $0.02 |
| GBPUSD | $1.76 | $13.44 | 0.13 | $76 | $0.03 |
| USDJPY | $1.45 | $10.00 | 0.15 | $69 | $0.02 |
| EURUSD | $1.79 | $11.73 | 0.15 | $66 | $0.02 |
| USDCHF | $1.55 | $10.00 | 0.15 | $65 | $0.02 |
| AUDJPY | $1.03 | $6.61 | 0.16 | $64 | $0.01 |
| EURGBP | $1.94 | $11.73 | 0.17 | $61 | $0.02 |
| USDCAD | $1.66 | $10.00 | 0.17 | $60 | $0.02 |
| AUDUSD | $1.32 | $6.61 | 0.20 | $50 | $0.01 |
| NZDUSD | $1.20 | $5.79 | 0.21 | $48 | $0.01 |
| JPN225 | $0.68 | $3.03 | 0.23 | $44 | $0.01 |
| DE40 | $0.80 | $2.81 | 0.28 | $35 | $0.01 |
| US500 | $1.19 | $0.67 | 1.78 | $6 | $0.00 |

## Caveats

- **Carry uses today's swap rates for the whole history.** MT5 only exposes current swaps, so the carry book is the one that looks attractive *now*, applied to years when rates were very different. Treat carry (and every swap cost) as biased; the holdout is the fairest test.
- **The swaps are this broker's.** Check that the implied currency rates in Setup resemble real policy-rate differentials before trusting carry. Demo servers often quote swaps that don't (both sides negative, or a low-rate currency ranked above a high-rate one).
- The $10 section uses one min lot / contract multiplier for every symbol; real cent accounts differ per symbol (this broker's index CFDs have min volume 0.1-1.0).
- Spreads are modelled from the current spread and recent bar spreads, not historical tick data.
- Monte Carlo reshuffling ignores autocorrelation in trend returns, so it understates clustered drawdowns.

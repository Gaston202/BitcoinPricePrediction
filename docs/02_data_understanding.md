# CRISP-DM Phase 2 — Data Understanding

Full analysis with code and figures: `notebooks/01_data_collection.ipynb`, `02_data_quality.ipynb`, `03_eda.ipynb`.
Figures for the report: `reports/figures/`.

## 1. Initial data collection

| | |
|---|---|
| Source | Binance spot REST API, `GET /api/v3/klines`, through `python-binance` 1.0.37 |
| Symbol / interval | BTCUSDT, 1h |
| Range | 2017-08-17 04:00 → 2026-09-30 23:00 UTC |
| Rows | 79,836 |
| File | `data/raw/btcusdt_1h_20170817_20260930.parquet` (+ `_metadata.json`) |
| Reproduce | `python -m src.data.collect` |

## 2. Data description

11 fields per hourly candle: OHLC prices (USDT), base volume (BTC), quote volume (USDT), number of trades,
taker-buy base/quote volume, and open/close time. The full data dictionary is in notebook 01.

## 3. Data quality

| Dimension | Result |
|---|---|
| Completeness | 99.84%: 128 missing hours in 28 gaps (longest 33h, 8–9 Feb 2018) |
| Uniqueness | 0 duplicate timestamps |
| Validity | 0 non-positive prices, 0 OHLC violations |
| Consistency | 4 zero-volume candles (valid "no trade" hours, kept) |
| Outliers | 49 daily returns (1.5% of days) with robust z-score > 5; all match real market events, kept |

The gaps are concentrated in 2018–2021, and 13 of the 28 start at exactly 02:00 UTC. This points to scheduled
exchange maintenance, i.e. missingness close to **MCAR**.
Policy applied: 28 hours forward-filled (gaps ≤ 3h), 100 hours left NaN, plus 34 incomplete 4h and 13 incomplete daily buckets set to NaN.

**Outliers.** Screened on daily log returns and high–low ranges with a robust z-score (median/MAD, threshold 5),
because a mean/std z-score is distorted by fat tails. The largest flagged days are documented events
(COVID crash 2020-03-12: −39.5%; FTX collapse 2022-11-09: −14.1%; Tesla purchase 2021-02-08: +19.5%).
They are **kept**: removing them would drop excess kurtosis from 16.2 to 1.7 and make any backtest look safer
than reality. Robustness comes from the pipeline instead (return/ratio features, 3-class label, rank-based
statistics, tree models, scaling inside the model pipeline). Figure: `reports/figures/02_outliers.png`.

## 4. Exploration: key findings

| # | Finding | Evidence | Implication |
|---|---|---|---|
| 3 | Drift ≪ noise | Daily mean +0.09% vs. std 3.5% | Direction prediction is inherently hard |
| 4 | Fat tails, negative skew | Excess kurtosis 16–37. 3σ moves happen 6–7× more often than under normality. JB/D'Agostino p ≈ 0. | Robust methods, tree models, classification target |
| 6 | Price non-stationary, returns stationary | Log price ADF p = 0.56 / KPSS p ≤ 0.01. Returns ADF p ≈ 0 / KPSS p ≥ 0.1. | Features from returns and ratios only |
| 7 | Weak reversal, strong volatility clustering | Lag-1 ACF ≈ −0.05. Squared-return Ljung-Box p ≈ 0. 4h lag-6 (24h) ACF ≈ −0.07. | Lagged-return and volatility features |
| 10 | Nothing linearly predicts direction; range and volume predict magnitude | \|ρ\| < 0.07 vs. sign. Spearman ≈ 0.35 (range) and 0.17 (volume) vs. \|next move\| | Nonlinear models; a volatility signal for position sizing |
| 11 | 1D moves are ≈ 7× the round-trip fee; 4h moves only ≈ 2.5× | Median \|move\|: 1.47% (1D) vs. 0.50% (4h) | **Recommended label: 1D, θ = 1%** (≈ 30 / 38 / 33% down/flat/up) |
| 12 | Baselines | Majority macro-F1 0.18. Persistence macro-F1 0.36. Buy-and-hold Sharpe 0.82, max drawdown −83%. | The bars from the success criteria |
| 13 | High-activity days precede bigger moves | Mann–Whitney U (one-sided), p ≈ 2×10⁻⁷; median next-day move 1.67% vs. 1.33%; rank-biserial r = 0.12 (small) | Keep a scale-free volume feature; volume is a risk signal, not a direction signal |

## 5. Open questions for Data Preparation

1. **Confirm the label:** 1D/θ = 1% (recommended) or 4h/θ = 0.5%.
2. **Chronological split:** which window is the test set? Report its regime and its buy-and-hold Sharpe.
3. **Start date:** keep 2017–2018 (thin early market, most of the gaps) or start in 2019?
4. **Optional EDA items** not done yet: price/volume regimes (2), hour/weekday seasonality (8, motivated by the 12h volatility spike), regime tests (9). Outliers (5) are done in notebook 02.

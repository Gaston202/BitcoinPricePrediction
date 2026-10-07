# AI-Driven Bitcoin Portfolio Simulation System

An ML model trained on historical Binance OHLCV data predicts the direction of Bitcoin's next price move.
An AI agent combines those predictions with news sentiment and market indicators to make
allocation decisions, and the strategy is evaluated through a historical backtest.

The project follows **CRISP-DM**. Current status: **Phase 1 (Business Understanding) and Phase 2 (Data Understanding) are complete.**

**Team:** Ghassen Siala · Baya Chabbene · Amine Neffati · Marwen Wali · Mohamed Aziz Souissi

## Quick start

Requires Python ≥ 3.11.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

pytest                              # 14 tests: gap policy, resampling, labels, baselines
jupyter notebook notebooks/         # run 01 → 02 → 03 in order
```

The raw data snapshot is already in `data/raw/`, so **you do not need to download anything**.
Notebook 02 regenerates the cleaned files in `data/interim/`.

### Re-collecting data (optional)

```bash
python -m src.data.collect                 # rebuild the frozen snapshot (skips if it exists)
python -m src.data.collect --end now       # fetch up to the latest closed candle
```

No API key is needed. If you get **HTTP 451** (location blocked), use Binance's market-data mirror:
`BINANCE_USE_DATA_API=1 python -m src.data.collect`.

## Project structure

```
data/
  raw/          frozen Binance snapshot (immutable, committed)
  interim/      cleaned 1h / 4h / 1D data (generated, gitignored)
  processed/    model-ready features (next phase, gitignored)
docs/
  01_business_understanding.md   objectives, success criteria, decisions log
  02_data_understanding.md       data quality and EDA findings
notebooks/
  01_data_collection.ipynb       source, metadata, data dictionary
  02_data_quality.ipynb          gaps, gap policy, resampling (EDA item 1)
  03_eda.ipynb                   returns, normality, stationarity, ACF, correlation, target, baselines
reports/figures/                 PNGs exported by the notebooks for the report
src/
  config.py                      all constants (symbol, dates, gap threshold, fees)
  data/collect.py                Binance REST collector (pagination, retries, 451 fallback)
  data/clean.py                  quality report and gap policy
  features/returns.py            log returns, resampling, 3-class labels
  evaluation/baselines.py        majority, persistence, buy-and-hold metrics
  visualization/style.py         shared chart style
tests/                           pytest suite
```

## Data

| | |
|---|---|
| Source | Binance spot REST API `/api/v3/klines` |
| Pair / interval | BTCUSDT, 1h (resampled to 4h, 1D) |
| Range | 2017-08-17 → 2026-09-30 23:00 UTC |
| Rows | 79,836 (99.84% complete) |

## Key EDA findings

* Returns have **fat tails** (excess kurtosis 16–37) and are **stationary**. Prices are not.
* **Direction is barely predictable** from past returns (lag-1 autocorrelation ≈ −0.05).
  **Volatility is strongly predictable** (volatility clustering).
* Recommended target: **1-day direction, θ = 1%** (typical daily move ≈ 7× the round-trip fee).
* Bars to beat: persistence macro-F1 ≈ 0.36 (model). Buy-and-hold Sharpe ≈ 0.82 over the full period (agent).

Details: [`docs/02_data_understanding.md`](docs/02_data_understanding.md).

## Next phase: Data Preparation

Confirm the label, define the chronological train/validation/test split, and engineer the features
(lagged returns, volatility, volume, taker-buy ratio).

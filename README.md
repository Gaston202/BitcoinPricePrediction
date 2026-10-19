# AI-Driven Bitcoin Portfolio Simulation System

An ML model trained on historical Binance OHLCV data predicts the direction of Bitcoin's next price move.
An AI agent combines those predictions with news sentiment and market indicators to make
allocation decisions, and the strategy is evaluated through a historical backtest.

The project follows **CRISP-DM**. Current status: **Phases 1-4 are complete** (Business Understanding,
Data Understanding incl. EDA, Data Preparation / Feature Engineering, and Modeling). Next: Evaluation.


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
  processed/    model-ready features and train/val/test splits (generated, gitignored)
docs/
  01_business_understanding.md   objectives, success criteria, decisions log
  02_data_understanding.md       data quality and EDA findings
  04_data_preparation.md         label, features, chronological split, test-window regime
  05_modeling.md                 candidate models, val/test results, business-read caveat
notebooks/
  01_data_collection.ipynb       source, metadata, data dictionary
  02_data_quality.ipynb          gaps, gap policy, resampling (EDA item 1)
  03_eda.ipynb                   returns, normality, stationarity, ACF, correlation, target, baselines
  04_feature_engineering.ipynb   label, features, chronological split, processed parquet output
  05_modeling.ipynb              model selection, test evaluation, strategy backtest vs. buy-and-hold
reports/figures/                 PNGs exported by the notebooks for the report
src/
  config.py                      all constants (symbol, dates, gap threshold, fees)
  data/collect.py                Binance REST collector (pagination, retries, 451 fallback)
  data/clean.py                  quality report and gap policy
  features/returns.py            log returns, resampling, 3-class labels
  features/engineering.py        lagged returns, volatility, ATR, volume features, chronological split
  evaluation/baselines.py        majority, persistence, buy-and-hold metrics
  evaluation/strategy.py         predicted-label -> position backtest vs. buy-and-hold
  modeling/train.py              candidate classifiers, feature/label split, evaluation
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

## Data Preparation

Label confirmed (1D direction, θ = 1%), features engineered (lagged returns, volatility/ATR, volume,
taker-buy ratio), and a chronological, purged train/validation/test split applied. The test window
(2025-07 → 2026-09) is a bear market (buy-and-hold Sharpe ≈ −0.33), which matters for how the final
backtest should be read.

Details: [`docs/04_data_preparation.md`](docs/04_data_preparation.md).

## Modeling

Three classifiers (logistic regression, random forest, gradient boosting) trained on `train` and selected
on `val`. **None beat the persistence baseline on macro-F1** — consistent with the EDA finding that 1D
direction carries little learnable signal from price/volume features alone. The selected model's test-window
backtest does outperform buy-and-hold (+9.5% vs. −24.9% return), but mostly by predicting "flat" 89% of the
time against an actual 45% flat rate — i.e. sitting out of a bear market, not genuine directional skill.

Details: [`docs/05_modeling.md`](docs/05_modeling.md).

## Next phase: Evaluation

Decide, given the honest modeling result above, whether to proceed to the full agent (news sentiment +
market indicators) with this as the ML leg's baseline, or iterate on the label/features first.

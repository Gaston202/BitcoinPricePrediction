# AI-Driven Bitcoin Portfolio Simulation System

An ML model trained on historical Binance OHLCV data predicts the direction of Bitcoin's next price move.
An AI agent combines those predictions with news sentiment and market indicators to make
allocation decisions, and the strategy is evaluated through a historical backtest.

The project follows **CRISP-DM**. Current status: **Phases 1-5 are complete** (Business Understanding,
Data Understanding incl. EDA, Data Preparation / Feature Engineering, Modeling, and Evaluation), plus one
modeling iteration. The model has **not** cleared the success criteria across multiple regimes — see
"Evaluation" and "Modeling Iteration" below before assuming this is deployment-ready.


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
  06_evaluation.md               formal check against the Phase 1 success criteria, go/no-go verdict
  07_modeling_iteration.md       calibration fix + multi-regime walk-forward, revised verdict
notebooks/
  01_data_collection.ipynb       source, metadata, data dictionary
  02_data_quality.ipynb          gaps, gap policy, resampling (EDA item 1)
  03_eda.ipynb                   returns, normality, stationarity, ACF, correlation, target, baselines
  04_feature_engineering.ipynb   label, features, chronological split, processed parquet output
  05_modeling.ipynb              model selection, test evaluation, strategy backtest vs. buy-and-hold
  06_evaluation.ipynb            data-mining and business success criteria checked against docs/01
  07_modeling_iteration.ipynb    calibrated threshold + walk-forward test across bear/bull/range regimes
reports/figures/                 PNGs exported by the notebooks for the report
src/
  config.py                      all constants (symbol, dates, gap threshold, fees)
  data/collect.py                Binance REST collector (pagination, retries, 451 fallback)
  data/clean.py                  quality report and gap policy
  features/returns.py            log returns, resampling, 3-class labels
  features/engineering.py        lagged returns, volatility, ATR, volume features, chronological split
  evaluation/baselines.py        majority, persistence, buy-and-hold metrics
  evaluation/strategy.py         predicted-label -> position backtest vs. buy-and-hold
  evaluation/criteria.py         formal yes/no checks against the docs/01 success criteria
  modeling/train.py              candidate classifiers, feature/label split, evaluation,
                                  calibrated decision-threshold tuning
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

## Evaluation

Checked against the two criteria fixed in advance in `docs/01_business_understanding.md`:

| Criterion | Status |
|---|---|
| Data-mining success (beat both baselines on macro-F1 **and** ROC-AUC) | **Not met** — wins on ROC-AUC (0.58 vs. 0.52/0.50), loses to persistence on macro-F1 |
| Business success (beat buy-and-hold on Sharpe, net of fees) | **Technically met**, but driven by the model sitting in cash ~89% of the time during a bear-market test window, not demonstrated skill |

**Recommendation: not ready for Deployment (Phase 6) as-is.** A business-criterion pass on a single
bear-market window is exactly the false positive flagged as a risk in `docs/01`. Next: re-test across a
bull and a range-bound window, and address the AUC/F1 mismatch (likely a decision-threshold/calibration
issue) before feeding this into the full agent.

Details: [`docs/06_evaluation.md`](docs/06_evaluation.md).

## Modeling Iteration

Both issues above were addressed directly — and neither turned out to be fixable this way:

* **Calibrated decision threshold** (`tune_class_weights`/`weighted_predict`, tuned on validation only):
  closes the macro-F1 gap in the original window only (1 of 4 tested windows) — not a reliable fix.
* **Walk-forward test across regimes** (2022 bear, 2023 bull, 2025 range, plus the original window, all
  retrained fresh per window): the earlier business-criterion "win" **disappears** once the model is
  forced to take more directional positions — the original window now *underperforms* buy-and-hold
  (Sharpe −0.59 vs −0.33). Across regimes the strategy wins in bear/range markets but **loses badly in the
  bull-market window** (2023: strategy −27.7% vs. buy-and-hold +154.0%).

**Still not ready for Deployment.** This is a stronger confirmation of the EDA's original warning than
the first-pass Phase 4/5 result suggested: price/volume features alone do not produce a regime-robust
directional edge. The project's own design already treats the ML model as one input to an agent that also
uses news sentiment and market indicators — this result suggests those are not optional extras.

Details: [`docs/07_modeling_iteration.md`](docs/07_modeling_iteration.md).

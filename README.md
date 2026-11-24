# AI-Driven Bitcoin Portfolio Simulation System

The project aims to predict cryptocurrency price movements; **Bitcoin (BTCUSDT) is the case study**.
An ML model trained on historical Binance OHLCV data predicts the direction of Bitcoin's next price move.
An AI agent combines those predictions with news sentiment and market indicators to make
allocation decisions, and the strategy is evaluated through a historical backtest.

The project follows **CRISP-DM**. Current status (2026-11-23): **Phases 1–5 are complete**, with four modeling
iterations and a fully evaluated trading agent. The selected direction model is **1h, binary (up/down), LightGBM, Platt-calibrated**. It beats the
majority, persistence **and reversal** baselines in 4/4 walk-forward windows, but its edge is small: about 54% hourly
accuracy, 61% on the most confident 5%. Price alone gives **no daily direction edge**. Trading performance (the
business criterion) is the agent's job: a **hybrid multi-agent team** modelled on
[TradingAgents](https://github.com/TauricResearch/TradingAgents) trades a $1,000 paper wallet on data kept up to date
from Binance. An open-weights LLM only reads the daily news; every trade and risk limit is an explicit rule.

**Business criterion verdict:** on the unseen test window the agent's Sharpe beats buy-and-hold on 4h (−0.04 vs −0.32)
and 1D candles (+0.18 vs −0.34), not on 1h, and its max drawdown is −16 to −20% instead of −53%. The difference is
not statistically significant, and a simple 200-candle moving-average filter does as well or better. Details:
[`docs/13_agent_evaluation.md`](docs/13_agent_evaluation.md); design: [`docs/11`](docs/11_rule_based_agents.md);
news: [`docs/12`](docs/12_news_sentiment.md).


> **Just want to run it?** Step-by-step commands and the full list of tools: [`HOW_TO_RUN.md`](HOW_TO_RUN.md).
> In short: `pip install -r requirements.txt`, then `python -m dashboard`. Tests: `make test`; the whole agent
> evaluation: `make agent`.
>
> **New to the project?** Read [`docs/project_guide.pdf`](docs/project_guide.pdf): a 16-page beginner-friendly guide
> from feature engineering to the final models, with every key figure explained.

## Quick start

Requires Python ≥ 3.11.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt

pytest                              # unit tests: gap policy, outliers, resampling, labels, features, baselines
jupyter notebook notebooks/         # run the notebooks in order, 01 → 07
```

The raw data snapshot is already in `data/raw/`, so **you do not need to download anything**.
Notebook 02 regenerates the cleaned files in `data/interim/`, and notebook 04 the model-ready files in
`data/processed/`, so run them before the later notebooks. To rerun everything headless:

```bash
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

If Jupyter picks up the system Python instead of `.venv`, register the venv as a kernel and pass it:
`python -m ipykernel install --user --name btc-venv`, then add `--ExecutePreprocessor.kernel_name=btc-venv`.

Notebooks 05–07 read saved results from `reports/results/`. To regenerate them (about 5.7 h on 28 cores, one
process per horizon × target):

```bash
BTC_N_JOBS=3 python scripts/run_multi_horizon_experiment.py --horizon 1h --targets binary   # etc. for 4h/1D, vol3/triple_barrier
python scripts/run_multi_horizon_experiment.py --horizon 4h --smoke                         # quick check, 1 window
python scripts/add_reversal_baseline.py                                                     # only for results from before 2026-11-22
```

**Midterm scope** (CRISP-DM Business Understanding → Feature Engineering): notebooks 01–04 and
`docs/01`, `02`, `04`, `08`.

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
  external/     frozen funding, Fear & Greed, news headlines and their sentiment scores (committed)
docs/
  01_business_understanding.md   objectives, success criteria, decisions log
  02_data_understanding.md       data quality and EDA findings
  04_data_preparation.md         label, split, 43 candidates, selection (16 kept) and why, PCA
  05_modeling.md                 walk-forward protocol, targets, models, tuning, calibration
  06_evaluation.md               baselines incl. reversal, revised criterion, calibration, stability, verdict
  07_modeling_iteration.md       four iterations and the final model
  08_conclusions.md              research questions, modeling results, limitations, roadmap
  09_literature_review.md        41 papers: models, labels, intraday horizons, validation, leakage flags
  10_modeling_handoff.md         2026-11-18 handoff (historical)
  11_rule_based_agents.md        hybrid multi-agent trader (design, rules, bug fixes), data updater
  12_news_sentiment.md           headlines, LLM vs lexicon scoring, predictive tests, look-ahead check
  13_agent_evaluation.md         business criterion: baselines, bootstrap tests, ablation, sensitivity
notebooks/
  01_data_collection.ipynb       source, metadata, data dictionary
  02_data_quality.ipynb          gaps, gap policy, outliers, resampling (EDA items 1, 5)
  03_eda.ipynb                   returns, normality, stationarity, ACF, correlation, target, baselines,
                                  Mann-Whitney test (volume vs. next-move size)
  04_feature_engineering.ipynb   label, split, 43 candidates (22 original + 21 short-term), selection, PCA
  05_modeling.ipynb              walk-forward protocol, targets, per-window features, 9 models, calibration, live rerun
  06_evaluation.ipynb            baselines (incl. reversal), revised criterion, sign-AUC, calibration, stability
  07_modeling_iteration.ipynb    four iterations, what changed and why, saved final model
  08_agent_evaluation.ipynb      the agent vs. seven baselines on every window, verdict, ablation, robustness
  09_news_sentiment.ipynb        news coverage, scoring, predictive value, look-ahead check
reports/figures/                 PNGs exported by the notebooks for the report
src/
  config.py                      all constants (symbol, dates, gap threshold, fees)
  data/collect.py                Binance REST collector (pagination, retries, 451 fallback)
  data/clean.py                  quality report, gap policy, robust outlier flags
  data/binance.py                public Binance spot/futures + Fear & Greed data over HTTPS
  data/update.py                 incremental sync of new candles -> data/live + *_latest.parquet
  data/external.py               frozen copies of funding and Fear & Greed for reproducible backtests
  data/news.py                   daily Bitcoin headlines (Hacker News or GDELT), resumable, frozen snapshot
  agents/                        hybrid multi-agent trader (analysts, debate, trader, risk team,
                                  portfolio manager, memory, quant analyst, replay/live sessions)
  agents/news_sentiment.py       LLM (OpenRouter, Nemotron 3 Ultra) and lexicon news scorers, cached per day
  features/returns.py            log returns, resampling, 3-class labels
  features/engineering.py        candidate features (lags, momentum, volatility, trend, volume, calendar,
                                  interactions), chronological split
  features/selection.py          train-only feature selection (ANOVA, MI, correlation) and PCA
  evaluation/baselines.py        majority, persistence, buy-and-hold metrics
  evaluation/strategy.py         predicted-label -> position backtest vs. buy-and-hold
  evaluation/criteria.py         formal yes/no checks against the docs/01 success criteria
  evaluation/portfolio.py        paper-wallet simulation, 7 baseline strategies, Sharpe/Sortino/drawdown,
                                  paired block-bootstrap Sharpe test
  evaluation/agent_backtest.py   windows, frozen market data, running the agent on a window
  evaluation/agent_experiments.py preset tuning, team selection, evaluation, ablation, sensitivity
  evaluation/news_analysis.py    does news predict returns? (Spearman, Granger, event study)
  features/short_term.py         fast indicators, candle shape, intraday-from-1h features, candidate_pool
  features/targets.py            binary / vol-scaled 3-class / triple-barrier targets
  modeling/train.py              9 candidate classifiers, ensembles, model-choice manifest
  modeling/tuning.py             Optuna search spaces
  modeling/multi_horizon.py      walk-forward: per-window selection, calibration, metrics, bootstrap, baselines
  modeling/range_model.py        next-day high/low regression (HAR + ML), 1D
  modeling/price_targets.py      next-bar high/low price levels with conformal bands, 5-day return (1h/4h)
scripts/                         runnable experiments (direction, multi-horizon, range, agent evaluation)
reports/results/                 saved experiment outputs read by notebooks 05-07
dashboard/                       admin dashboard (python -m dashboard): FastAPI + Chart.js
  visualization/style.py         shared chart style
tests/                           pytest suite (runs in CI: .github/workflows/tests.yml)
Makefile                         one-command pipeline: make test / data / news / sentiment / agent / notebooks
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
* 49 extreme days (robust z-score > 5, e.g. the COVID crash) are real market events and are **kept**.
* High-activity days are followed by bigger moves (Mann–Whitney p ≈ 2×10⁻⁷, small effect): volume
  predicts the **size** of the next move, not its direction.
* **Direction is barely predictable** from past returns (lag-1 autocorrelation ≈ −0.05).
  **Volatility is strongly predictable** (volatility clustering).
* Recommended target: **1-day direction, θ = 1%** (typical daily move ≈ 7× the round-trip fee).
* Bars to beat: persistence macro-F1 ≈ 0.36 (model). Buy-and-hold Sharpe ≈ 0.82 over the full period (agent).

Details: [`docs/02_data_understanding.md`](docs/02_data_understanding.md).

## Data Preparation

Label kept at 1D direction, θ = 1%. The 22 original candidate features (lags, momentum, volatility/ATR, MA
distance, RSI, relative volume, calendar, interactions) are joined by **21 new short-term indicators** (fast
indicators, candle shape, intraday statistics from the 1h candles), giving 43 candidates. A chronological, purged
train/validation/test split comes first. Selection on the training split then keeps **16 of 43**: 10 original
and 6 new (`rv_168h`, `range_pct`, `rv_4h`, `ret_last_4h`, `close_location`, `mom_3`). It drops noise that fails
both ANOVA and a shuffled-label MI test, and one of each pair with |ρ| > 0.85. A PCA variant (17 components, 91.2%
of variance) is built for comparison. The multi-horizon models re-select features inside every walk-forward window
instead (`notebooks/05`).

Details: [`docs/04_data_preparation.md`](docs/04_data_preparation.md).

## Modeling

9 models (LR, RF, GB, XGBoost, LightGBM, SVM, kNN, soft voting, stacking) × 3 horizons × 3 targets (binary,
volatility-scaled 3-class, triple barrier) × 4 walk-forward windows (2022 bear, 2023 bull, 2025 range, 2025-07 →
2026-09). Optuna tuning, Platt calibration and class weights all use pre-window data only. A live rerun in notebook 05
reproduces the saved metrics exactly.

Details: [`docs/05_modeling.md`](docs/05_modeling.md).

## Evaluation

| Horizon | Direction skill (sign-AUC) | Beats majority + persistence + reversal | Lift over reversal significant |
|---|---|---|---|
| **1h** | 0.558 | 4/4 windows (8 of 9 models) | 3/4 windows |
| 4h | 0.555 | 4/4 (best model) | 1–2/4 |
| 1D | ≤ 0.53 | ≤ 2/4 | 0/4 |

**Next-bar high/low price levels** (1h and 4h) also beat their baselines: HAR is 2–7% better than the 20-bar average
move, and calibrated 10/90% bands are breached 8.7–9.9% of the time (target 10%), ready for stops and take-profits.
A 5-day return forecast showed no edge.

Short-term BTC return signs mean-revert, so "bet against the last bar" was added as a baseline (2026-11-22).
Most of the apparent intraday skill is that effect; the model's own edge beyond it is about 1–1.5 macro-F1 points
at 1h, stable across all 17 quarterly refits. The probabilities are well calibrated (ECE ≤ 0.04). Triple-barrier
models predict volatility, not direction.

Details: [`docs/06_evaluation.md`](docs/06_evaluation.md).

## Modeling Iteration

Four iterations: midterm 1D models (failed against persistence) → calibrated walk-forward (1/4 windows) → 9 tuned
models on 1D (4/4 vs persistence, but no reversal check) → multi-horizon with per-window selection and the reversal
baseline. Final model saved in `src/modeling/selected_model.json`.

A fifth iteration added next-bar high/low price levels with conformal bands (`src/modeling/selected_price_model.json`).
The agent that trades on it is evaluated in [`docs/13`](docs/13_agent_evaluation.md).

Details: [`docs/07_modeling_iteration.md`](docs/07_modeling_iteration.md), [`docs/08_conclusions.md`](docs/08_conclusions.md).

## Trading Agents & Dashboard

```bash
python -m dashboard                 # admin dashboard at http://127.0.0.1:8000 (syncs Binance data every 5 min)
python -m src.data.update --watch   # or keep the data up to date from a terminal
```

* **The LLM reads, the rules decide.** The TradingAgents roles - seven analysts (trend, momentum, volatility,
  volume/order flow, sentiment, the team's LightGBM as the quant analyst, and news scored once a day by an
  open-weights LLM), a bull/bear debate, a research manager with the 5-tier
  Buy/Overweight/Hold/Underweight/Sell rating, a trader, an aggressive/conservative/neutral risk team, a portfolio
  manager with hard limits, and a memory that re-weights analysts by their hit rate - each implemented as explicit,
  testable rules.
* **Always up to date.** Each sync downloads only the candles that closed since the last one, plus futures funding
  rates and the Fear & Greed index. The frozen snapshot stays untouched for the notebooks.
* **Dashboard.** Replay any period (including the newly synced data) or trade live on 1m-4h candles, and inspect every
  decision's full pipeline. Paper trading only.

Details and design choices: [`docs/11_rule_based_agents.md`](docs/11_rule_based_agents.md); evaluation:
[`docs/13_agent_evaluation.md`](docs/13_agent_evaluation.md).

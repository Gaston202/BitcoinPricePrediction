# CRISP-DM Phase 1 — Business Understanding

## 1. Business objectives

**Background.** Cryptocurrencies trade 24/7, are highly volatile and are driven by a mix of market
microstructure, sentiment and news. The project aims to **predict cryptocurrency price movements** and use
those predictions in a simulated portfolio. We use **Bitcoin (BTCUSDT on Binance) as the case study**: it is
the largest and most liquid crypto asset, with the longest clean price history. The pipeline is written
for any Binance spot pair (`config.SYMBOL`), so other coins can be added later without changing the code.

Bitcoin itself is a highly volatile asset (annualized volatility ≈ 70%) that trades 24/7.
Discretionary trading is emotional and inconsistent. A systematic, data-driven approach can be tested
objectively on historical data before any real money is risked.

**Objective.** Build an AI-driven Bitcoin portfolio simulation system:

1. A **machine-learning model** trained on historical Binance OHLCV data predicts the direction of the next price move.
2. An **AI agent** combines the model's predictions with news sentiment and market indicators to make
   portfolio allocation and trading decisions.
3. A **historical simulation (backtest)** evaluates the strategy's performance.

**Scope of the midterm:** CRISP-DM from Business Understanding through Feature Engineering (data
collection, data understanding, preprocessing, EDA with hypothesis testing, feature engineering).
Modeling, the agent and the backtest are future work (see `docs/08_conclusions.md`).

## 2. Research questions

| # | Question | Answered in |
|---|---|---|
| RQ1 | Is the direction of Bitcoin's next price move predictable from its own past returns? | `03_eda.ipynb` items 7, 10 |
| RQ2 | What are the statistical properties of crypto returns (distribution, stationarity, outliers), and what do they imply for modeling? | `02_data_quality.ipynb`, `03_eda.ipynb` items 3, 4, 6 |
| RQ3 | Does trading activity (volume, range) carry information about the size of the next move? | `03_eda.ipynb` items 10, 13 (Mann–Whitney test) |
| RQ4 | Which prediction horizon and "flat" threshold make a predicted move worth its trading fees? | `03_eda.ipynb` item 11 |

**Midterm success criteria** (measurable, for the data-preparation work itself):

1. The cleaned dataset is ≥ 99% complete on its time grid, with 0 duplicate timestamps and 0 invalid prices.
2. Every research question is answered with a statistical test or a quantified measure, not a visual impression.
3. The model-ready dataset has 0 NaN/inf values, a chronological split with no overlap, and features that use
   only data available at time *t* (no leakage).
4. Every final feature is justified (EDA finding or train-only relevance test), and noisy or redundant features (|ρ| > 0.85) are removed.

## 3. Business success criterion

> The agent's strategy achieves a **higher Sharpe ratio than buy-and-hold BTC** over the out-of-sample
> test period, **net of 0.1% trading fees per side**.

The Sharpe ratio is used because it measures return per unit of risk. Raw return alone is a weak
criterion: BTC rose about 19× over the dataset, so almost any long strategy "makes money".

The comparison must use **exactly the same test window** for both strategies. Buy-and-hold Sharpe
varies from −1.3 (2022) to +3.4 (2017) depending on the year (see `notebooks/03_eda.ipynb`, item 12).

## 4. Data-mining goal

Predict the **direction of the next price move** as a 3-class classification:

| Class | Condition (forward log return r over the horizon) |
|---|---|
| UP | r > +θ |
| FLAT | −θ ≤ r ≤ +θ |
| DOWN | r < −θ |

The model outputs **class probabilities**, which the agent uses as a confidence signal.
The FLAT class stops the agent from paying fees on moves too small to be worth trading.

Horizon and θ are chosen from the EDA evidence. **Recommended: 1-day horizon, θ = 1%** (to be confirmed by the team).

## 5. Data-mining success criterion

> On a **chronologically held-out** test set, the classifier beats both the **majority-class** baseline and the
> **persistence** baseline ("the next move repeats the last realized move") on **macro-F1** and **ROC-AUC**.

Reference values (full dataset, 1D / θ = 1%): majority macro-F1 ≈ 0.18, persistence macro-F1 ≈ 0.36.

Realistic expectation: crypto direction is close to unpredictable from price alone (EDA item 7).
A few points above the baselines is a meaningful result. An accuracy far above that almost
certainly indicates **data leakage** and must be investigated, not celebrated.

## 6. Resources, constraints, risks

| Item | Detail |
|---|---|
| Data | Binance public REST API: free, no API key, about 80k hourly candles |
| Tools | Python 3.11+, pandas, statsmodels, scikit-learn, python-binance |
| Team | 5 students |
| Risk: look-ahead leakage | Labels use future prices. Features must only use data up to `close[t]`, with a chronological split. |
| Risk: regime change | Bull/bear regimes differ strongly. The test period must be reported with its regime. |
| Risk: overfitting the backtest | Tune only on validation data. The test set is touched once. |
| Risk: geo-blocking | `api.binance.com` returns HTTP 451 in some countries → `BINANCE_USE_DATA_API=1` |

## 7. Decisions log

| # | Decision | Choice |
|---|---|---|
| 1 | Prediction target | 3-class direction (up/flat/down) with probabilities |
| 2 | Base interval | `1h`, resampled to `4h` and `1D` |
| 3 | Market & range | Binance spot BTCUSDT, 2017-08-17 → 2026-09-30 23:00 UTC (frozen) |
| 4 | Collection | REST API only, via `python-binance`, all 12 kline fields |
| 5 | Storage | Immutable Parquet snapshot + metadata JSON in `data/raw/`, committed to Git |
| 6 | Business success | Beat buy-and-hold on Sharpe, net of fees |
| 7 | Gap policy | Reindex to an hourly grid, flag gaps, forward-fill gaps ≤ 3h, leave longer gaps NaN |
| 8 | EDA scope | Core items 1, 3, 4, 6, 7, 10, 11, 12 |
| 9 | Code layout | Notebooks for the narrative, `src/` for tested logic, venv + `requirements.txt` |
| 10 | Outliers | Flag daily returns with robust z-score > 5; keep them (real market events) |
| 11 | Feature selection | Train split only: drop scale-dependent `log_volume`, drop noise (ANOVA + MI vs. shuffled labels), drop one of each pair with \|ρ\| > 0.85 |

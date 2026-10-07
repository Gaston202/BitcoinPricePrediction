# CRISP-DM Phase 1 — Business Understanding

## 1. Business objectives

**Background.** Bitcoin is a highly volatile asset (annualized volatility ≈ 70%) that trades 24/7.
Discretionary trading is emotional and inconsistent. A systematic, data-driven approach can be tested
objectively on historical data before any real money is risked.

**Objective.** Build an AI-driven Bitcoin portfolio simulation system:

1. A **machine-learning model** trained on historical Binance OHLCV data predicts the direction of the next price move.
2. An **AI agent** combines the model's predictions with news sentiment and market indicators to make
   portfolio allocation and trading decisions.
3. A **historical simulation (backtest)** evaluates the strategy's performance.

**Scope of the current iteration:** data collection, data understanding and EDA for the ML model (CRISP-DM phases 1–2).

## 2. Business success criterion

> The agent's strategy achieves a **higher Sharpe ratio than buy-and-hold BTC** over the out-of-sample
> test period, **net of 0.1% trading fees per side**.

The Sharpe ratio is used because it measures return per unit of risk. Raw return alone is a weak
criterion: BTC rose about 19× over the dataset, so almost any long strategy "makes money".

The comparison must use **exactly the same test window** for both strategies. Buy-and-hold Sharpe
varies from −1.3 (2022) to +3.4 (2017) depending on the year (see `notebooks/03_eda.ipynb`, item 12).

## 3. Data-mining goal

Predict the **direction of the next price move** as a 3-class classification:

| Class | Condition (forward log return r over the horizon) |
|---|---|
| UP | r > +θ |
| FLAT | −θ ≤ r ≤ +θ |
| DOWN | r < −θ |

The model outputs **class probabilities**, which the agent uses as a confidence signal.
The FLAT class stops the agent from paying fees on moves too small to be worth trading.

Horizon and θ are chosen from the EDA evidence. **Recommended: 1-day horizon, θ = 1%** (to be confirmed by the team).

## 4. Data-mining success criterion

> On a **chronologically held-out** test set, the classifier beats both the **majority-class** baseline and the
> **persistence** baseline ("the next move repeats the last realized move") on **macro-F1** and **ROC-AUC**.

Reference values (full dataset, 1D / θ = 1%): majority macro-F1 ≈ 0.18, persistence macro-F1 ≈ 0.36.

Realistic expectation: crypto direction is close to unpredictable from price alone (EDA item 7).
A few points above the baselines is a meaningful result. An accuracy far above that almost
certainly indicates **data leakage** and must be investigated, not celebrated.

## 5. Resources, constraints, risks

| Item | Detail |
|---|---|
| Data | Binance public REST API: free, no API key, about 80k hourly candles |
| Tools | Python 3.11+, pandas, statsmodels, scikit-learn, python-binance |
| Team | 5 students |
| Risk: look-ahead leakage | Labels use future prices. Features must only use data up to `close[t]`, with a chronological split. |
| Risk: regime change | Bull/bear regimes differ strongly. The test period must be reported with its regime. |
| Risk: overfitting the backtest | Tune only on validation data. The test set is touched once. |
| Risk: geo-blocking | `api.binance.com` returns HTTP 451 in some countries → `BINANCE_USE_DATA_API=1` |

## 6. Decisions log

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

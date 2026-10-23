# Conclusions and Next Steps (midterm scope)

Scope: CRISP-DM Business Understanding → Feature Engineering, on Bitcoin (BTCUSDT) as the case study for
cryptocurrency price prediction. Evidence: `notebooks/01`–`04`, `docs/01`, `02`, `04`.

## 1. Answers to the research questions

| # | Question | Answer | Evidence |
|---|---|---|---|
| RQ1 | Is the next move's direction predictable from past returns? | **Barely.** There is a tiny reversal effect, but it explains ≈ 0.25% of the variance. | Lag-1 autocorrelation ≈ −0.05 (Ljung-Box significant only because n is large); every \|ρ\| with the next return's sign < 0.07 |
| RQ2 | What are the statistical properties of crypto returns? | **Fat-tailed, negatively skewed, stationary**; prices are non-stationary; 49 extreme days are real events. | Excess kurtosis 16–37, Jarque-Bera p ≈ 0; ADF/KPSS agree (price p = 0.56 / ≤ 0.01, returns p ≈ 0 / ≥ 0.1); robust z-score > 5 on 1.5% of days |
| RQ3 | Does activity carry information about the size of the next move? | **Yes, for magnitude, not direction.** | Spearman ≈ 0.35 (range) and 0.17 (volume) with \|next move\|; Mann–Whitney U p ≈ 2×10⁻⁷, median next move 1.67% vs. 1.33%, small effect (r = 0.12); strong volatility clustering |
| RQ4 | Which horizon and threshold make a move worth its fees? | **1 day, θ = 1%.** | Median daily move 1.47% ≈ 7× the 0.2% round-trip fee (4h: only 2.5×); classes ≈ 30 / 38 / 33% |

## 2. Midterm success criteria

| Criterion (`docs/01` §2) | Result |
|---|---|
| Clean data ≥ 99% complete, 0 duplicates, 0 invalid prices | **Met:** 99.84%, 0, 0 |
| Every research question answered with a test or a measure | **Met:** see the table above |
| Model-ready data: no NaN/inf, chronological non-overlapping split, no leakage | **Met:** asserted in notebook 04; 1-row embargo at each boundary; scaler fit on train only |
| Every final feature justified; noisy and redundant features removed | **Met:** 12 of 22 candidates kept (7 noise, 3 correlated at \|ρ\| > 0.85, `log_volume` removed on domain grounds); PCA variant also built |

## 3. Limitations

* **One asset, one exchange.** Results describe BTCUSDT on Binance. Other coins are more volatile and less
  liquid; the findings must be re-checked before generalizing to "crypto".
* **Price and volume only.** No news, sentiment or on-chain data yet, and RQ1 shows price alone carries
  very little directional signal.
* **Small daily sample.** About 2,800 model-ready days, of which 1,954 are for training. Mutual-information
  estimates at this size are noisy.
* **Regime dependence.** The test window (2025-08 → 2026-09) is a bear market (buy-and-hold Sharpe −0.33);
  any evaluation on it must be read with that regime in mind.
* **Test assumptions.** Mann–Whitney and Ljung-Box assume independent observations; volatility clustering
  weakens this, so p-values are optimistic. Effect sizes are reported for that reason.
* **Survivorship.** Bitcoin is the crypto that succeeded; a strategy tuned on it may look better than it
  would on a randomly chosen coin.

## 4. Future work (project roadmap)

1. **Modeling.** Train classifiers (logistic regression, random forest, gradient boosting) on the 12 selected
   features (and the PCA variant), select on `val`, and test once on `test` against the majority and persistence baselines
   (macro-F1 and ROC-AUC). Calibrate the decision threshold, and use walk-forward evaluation across bull,
   bear and range-bound windows. A first exploratory pass (notebooks 05–07) already suggests that
   price/volume features alone are not regime-robust, which motivates steps 2–3.
2. **Richer inputs.** Daily **news sentiment** (aligned with the 1D horizon), **market indicators** (RSI, MACD,
   moving-average trends, funding rates, Fear & Greed index), and hour/weekday seasonality.
3. **AI agent.** Combine the model's class probabilities with sentiment and indicators to decide portfolio
   allocation (long / cash, position size). Use the volatility signal from RQ3 for position sizing.
4. **Backtesting.** Simulate the agent's portfolio on historical data, net of 0.1% fees per side, and compare
   its Sharpe ratio and drawdown with buy-and-hold on the same windows (the business success criterion).
5. **More cryptocurrencies.** The pipeline is symbol-agnostic (`config.SYMBOL`); extend it to ETH, SOL and others,
   and test whether the findings hold across coins.

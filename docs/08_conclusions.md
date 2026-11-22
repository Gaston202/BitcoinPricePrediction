# Conclusions and Next Steps

Scope: CRISP-DM Business Understanding → Evaluation, on Bitcoin (BTCUSDT) as the case study for cryptocurrency
price prediction. Sections 1–2 are the midterm answers (`notebooks/01`–`03`). Section 2b adds the modeling results
(updated 2026-11-22; `notebooks/04`–`07`, `docs/04`–`07`).

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

## 2b. Modeling results (updated 2026-11-22)

| Finding | Evidence |
|---|---|
| **1h direction is predictable, a little.** The 1h binary model beats majority, persistence **and reversal** in 4/4 walk-forward windows. Its lift over reversal is +0.6 to +1.4 macro-F1 points, significant in 3 of 4 windows | `docs/06` §2, §4 |
| **Most short-term "skill" is mean reversion.** Return signs are negatively autocorrelated at 1h (−0.06) and 4h (−0.08), so "bet against the last bar" alone has a sign-AUC of 0.53–0.54. This answers RQ1 more precisely: the tiny reversal effect is real, and it is strongest intraday | `docs/06` §1 |
| **No daily direction edge from price.** 1D models never significantly beat reversal | `docs/06` §2 |
| **Triple-barrier models predict volatility, not direction** (class ROC-AUC up to 0.69, sign-AUC about 0.50 at 4h/1D), consistent with RQ3 | `docs/06` §3 |
| **Confidence is informative at 1h:** 54% accuracy on all bars, 61% on the 5% most confident | `docs/06` §7 |
| **New short-term features earn their place:** 6 of 21 survive the 1D selection (16 features in total), and they are 40–50% of the per-window multi-horizon selections. `close_location` is the most-selected feature overall (27 of 36 runs) | `docs/04` §7, `notebooks/05` §2b |
| **Final model:** 1h, binary, LightGBM, Platt-calibrated (`src/modeling/selected_model.json`) | `docs/07` |

## 3. Limitations

* **One asset, one exchange.** Results describe BTCUSDT on Binance. Other coins are more volatile and less
  liquid; the findings must be re-checked before generalizing to "crypto".
* **Price and volume only.** No news, sentiment or on-chain data yet. Price alone carries a small 1h edge and no
  daily edge (§2b).
* **ML metrics are not trading results.** About 54% hourly accuracy is small next to 0.1% fees per side, and naive
  hourly sign trading loses money after costs in the literature (`docs/09` #21). The business criterion is only
  tested once the agent trades selectively.
* **Small daily sample.** About 2,800 model-ready days, of which 1,954 are for training. Mutual-information
  estimates at this size are noisy.
* **Regime dependence.** Addressed with four walk-forward windows (bear, bull, range, latest) and 17 quarterly refits.
  The 1h edge is weakest in the latest window (2025-07 → 2026-09: +0.6 points, p = 0.12).
* **Test assumptions.** Mann–Whitney and Ljung-Box assume independent observations; volatility clustering
  weakens this, so p-values are optimistic. Effect sizes are reported for that reason.
* **Survivorship.** Bitcoin is the crypto that succeeded; a strategy tuned on it may look better than it
  would on a randomly chosen coin.

## 4. Future work (project roadmap)

1. **Modeling (done, 2026-11-22).** 9 models × 3 horizons × 3 targets under walk-forward with per-window selection,
   calibration and a reversal baseline (§2b). **Next modeling step (agreed):** next-bar high/low price levels with
   10/90% bands, and the 5-day close (return quantiles + direction) on 1h and 4h bars.
2. **Richer inputs.** Daily **news sentiment** (aligned with the 1D horizon), **market indicators** (RSI, MACD,
   moving-average trends, funding rates, Fear & Greed index), and hour/weekday seasonality.
3. **AI agent.** Combine the model's class probabilities with sentiment and indicators to decide portfolio
   allocation (long / cash, position size). Use the volatility signal from RQ3 for position sizing.
4. **Backtesting.** Simulate the agent's portfolio on historical data, net of 0.1% fees per side, and compare
   its Sharpe ratio and drawdown with buy-and-hold on the same windows (the business success criterion).
5. **More cryptocurrencies.** The pipeline is symbol-agnostic (`config.SYMBOL`); extend it to ETH, SOL and others,
   and test whether the findings hold across coins.

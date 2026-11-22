# CRISP-DM Phase 5 — Evaluation

Rewritten 2026-11-22. Code and figures: `notebooks/06_evaluation.ipynb`, `reports/figures/06_*.png`.

## 1. Baselines and the reversal effect

From 2022 on, the **signs** of consecutive returns are negatively correlated: −0.061 at 1h and −0.080 at 4h
(noise bands ±0.010 and ±0.019), and −0.040 at 1D, which is inside its ±0.047 band. Persistence is therefore worse
than a coin flip intraday (sign-AUC 0.47 at 1h, 0.46 at 4h). **Reversal** ("bet against the last bar") reaches 0.53
at 1h and 0.54 at 4h with no model at all.

Mean binary macro-F1 of the baselines:

| Horizon | Majority | Persistence | Reversal |
|---|---|---|---|
| 1h | 0.336 | 0.471 | 0.528 |
| 4h | 0.338 | 0.463 | 0.536 |
| 1D | 0.340 | 0.484 | 0.516 |

The criterion was revised accordingly (`docs/01` §5, decision 14): a model must beat majority, persistence **and**
reversal in every window.

## 2. Revised data-mining criterion

| Horizon | Target | Best model | Windows clearing all baselines | Significant vs reversal | Mean sign-AUC | Models clearing 4/4 (of 9) |
|---|---|---|---|---|---|---|
| 1h | binary | stacking | 4/4 | 3/4 | 0.558 | 8 |
| 1h | vol3 | soft voting | 4/4 | 4/4 | 0.556 | 8 |
| 1h | triple barrier | LightGBM | 4/4 | 4/4 | 0.543 | 8 |
| 4h | binary | soft voting | 4/4 | 2/4 | 0.555 | 5 |
| 4h | vol3 | soft voting | 4/4 | 3/4 | 0.551 | 4 |
| 4h | triple barrier | soft voting | 4/4 | 4/4 | 0.499 | 3 |
| 1D | binary | gradient boosting | 2/4 | 0/4 | 0.512 | 0 |
| 1D | vol3 | gradient boosting | 2/4 | 1/4 | 0.507 | 0 |
| 1D | triple barrier | soft voting | 3/4 | 2/4 | 0.505 | 1 |

## 3. Direction vs volatility

The **shared sign-AUC** scores every model on one question: does P(up) − P(down) rank up-bars above down-bars? It is
about 0.55 at 1h and 4h for every model and target, except triple barrier at 4h (0.50), and about 0.50–0.53 at 1D.
Triple barrier has the highest class ROC-AUC (0.62–0.69), but that comes from predicting *whether* a barrier gets
hit, i.e. volatility. Binary is the clean direction target. The triple-barrier probabilities are kept as a
volatility signal.

## 4. 1h binary in detail

LightGBM, stacking and XGBoost are within about 0.001 of each other. Per window (2022 / 2023 / 2025 / `original`),
LightGBM beats reversal by +1.4, +1.4, +1.3 and +0.6 macro-F1 points (p = 0.005, 0.004, 0.002, 0.117). Its sign-AUC
is 0.573, 0.578, 0.540 and 0.542.

## 5. Calibration

After Platt scaling, ECE is at most 0.039. Uncalibrated, it was 0.07–0.11 at 1D. The reliability curves sit on the
diagonal. The predictions are narrow: 90% of the 1h P(up) values lie between 0.41 and 0.60.

## 6. Stability (17 quarterly refits per intraday horizon)

- **1h:** sign-AUC between 0.533 and 0.586 (mean 0.557), and it beats persistence in all 17 quarters.
- **4h:** mean sign-AUC 0.552, but it dips below 0.5 in 2 quarters. It beats persistence in 16 of 17.

## 7. Selective prediction

At 1h, accuracy rises from 54.1% on all bars to 59.4% on the 20% most confident and 60.8% on the top 5%. At 4h it
stays at about 55–56%. 1D shows no gain.

## 8. Verdict

- The **data-mining criterion is met at 1h**, and more thinly at 4h. It is **not met at 1D**: price alone gives no
  daily direction edge.
- The **business criterion is not evaluated in this phase** (agreed 2026-11-21). Trading is the agent's job, using a
  confidence band.
- Hand-off to the agent: calibrated 1h P(up) from LightGBM (binary target), plus the triple-barrier probabilities as
  a volatility signal.

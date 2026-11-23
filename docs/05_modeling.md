# CRISP-DM Phase 4 — Modeling

Rewritten 2026-11-22. Code and figures: `notebooks/05_modeling.ipynb`. Implementation:
`src/modeling/multi_horizon.py`, `src/modeling/train.py`, `src/modeling/tuning.py`, `src/features/targets.py`.
Runner: `scripts/run_multi_horizon_experiment.py` (results in `reports/results/multi_horizon/`).

## 1. Walk-forward protocol

There are four test windows (from `03_eda.ipynb` item 12): `2022_bear`, `2023_bull`, `2025_range` and `original`
(2025-07-09 → 2026-09-29). For each window, all earlier rows are split 85/15 into inner train and inner validation,
with a one-bar embargo at each boundary. The window itself is scored once. Example, 1h `original`: inner train
2017-08 → 2024-05 (57k bars), inner validation 2024-05 → 2025-07 (10k bars), test 10.7k bars.

## 2. Targets (all one bar ahead)

| Target | Classes | Rule |
|---|---|---|
| `binary` | down / up | Sign of the next bar's log return |
| `vol3` | down / flat / up | Flat if the return is within ±*k*σₜ. σₜ = rolling std of past bar returns (30 days for 1D, 1 week for 4h/1h) |
| `triple_barrier` | down / flat / up | On the next bar's 1h path: up if `close·exp(kσₜ)` is touched first, down if `close·exp(−kσₜ)` is, flat if neither is touched before the bar ends |

*k* is set on inner train so that about a third of the rows are flat. It is never tuned for profit. In practice it
came out the same in almost every window: 0.30 for `vol3` and 0.70–0.80 for `triple_barrier`.

## 3. Models

Logistic regression, random forest, sklearn gradient boosting, XGBoost (binary objective for 2 classes), LightGBM,
RBF-SVM, kNN, soft voting, and stacking (LR + RF + LightGBM + SVM). For 1h and 4h, the single-threaded models train
on their most recent rows only: SVM 10k, kNN and GB 20k. On 20k rows, one SVM fit took 78 s and one GB fit 37 s. A
GPU did not help: XGBoost took 0.46 s for 60k rows on both CPU and CUDA, and the installed LightGBM has no GPU build.

## 4. Tuning and calibration

Optuna TPE (seed 0): 50 trials for 1D and 25 for 4h/1h, maximising inner-validation macro-F1. The tuned model is fit
on inner train, then **Platt-calibrated** on inner validation. Class weights are tuned on the same rows, using the
calibrated probabilities. Both variants are scored: raw (uncalibrated argmax) and cal (Platt + weights).

## 5. Reproducibility

Notebook 05 reruns one cell of the experiment from scratch (1h, binary, LightGBM, `original`) in about 95 s. It
selects the same 10 features and reproduces every saved metric to 4 decimals (macro-F1 0.5256, sign-AUC 0.5421,
ECE 0.0118).

## 6. Next-bar high/low price levels (1h, 4h)

Code: `src/modeling/price_targets.py`, runner `scripts/run_price_targets_experiment.py`, results in
`reports/results/price_targets/`.

| Target | Definition | Output |
|---|---|---|
| `high_next` | log(high[t+1] / close[t]) | predicted high = close[t] · exp(prediction) |
| `low_next` | log(low[t+1] / close[t]) | predicted low = close[t] · exp(prediction) |

- Models train on the distance from the close, because tree models cannot predict prices outside their training
  range. The prediction converts back to an exact price level.
- **Features:** the candidate pool plus bar-scale range features: the last bar's up/down move and range, and their
  averages over the last day and the last week (1/24/168 bars on 1h, 1/6/42 on 4h). These are the HAR inputs.
  Features are re-selected inside every window (F-test + MI noise filter, |ρ| > 0.85, count chosen on validation MAE).
- **Models:** HAR, Ridge, random forest, XGBoost, LightGBM (25 Optuna trials, validation MAE).
- **Bands:** LightGBM quantile regression at 10/50/90%, plus a split-conformal shift on inner validation, so that
  each level's breach rate matches its promise.
- The same machinery was run for the 5-day return (120 bars on 1h, 30 on 4h, 5-day embargo). It has no edge (`docs/06` §8).

## 7. Runtime

Per target: 8–19 min for 1D, 1.6–2.9 h for 4h, 2.7–5.7 h for 1h (triple barrier is the slowest). The 9 horizon ×
target runs ran in parallel with 3 threads each, about 5.7 h in total. High/low: 16 min (4h) and 52 min (1h).

# Modeling Iteration — Calibration and Regime Robustness

**Still CRISP-DM Phase 4/5** (iterating on Modeling, re-running Evaluation). Not Deployment.

Full analysis: `notebooks/07_modeling_iteration.ipynb`. Figures: `reports/figures/07_*.png`.
Implementation: `src/modeling/train.py` (`tune_class_weights`, `weighted_predict`).

`docs/06_evaluation.md` left two open items before Deployment could be considered. This notebook
addresses both — and the honest result is that neither is actually fixed.

## Fix 1: calibrated decision threshold

`model.predict()` uses `argmax(proba)`, which collapses to "flat" whenever the model is uncertain, since
flat sits between the other two classes. `tune_class_weights()` instead picks per-class weights that
maximize macro-F1 **on validation only**, then `weighted_predict()` applies them — no retraining, no
peeking at test.

**Result: not a reliable fix.** Walk-forward tested across four windows, the calibrated model clears the
data-mining criterion (beats both baselines on **both** macro-F1 and ROC-AUC) in only 2 of 4, both by a
margin of about 0.01 macro-F1:

| Window | Model macro-F1 | Persistence macro-F1 | Data-mining criterion |
|---|---|---|---|
| original (2025-07 → 2026-09) | 0.359 | 0.356 | Met (barely) |
| 2022 (bear) | 0.330 | 0.361 | Not met |
| 2023 (bull) | 0.358 | 0.347 | Met (barely) |
| 2025 (range) | 0.359 | 0.363 | Not met |

> **Revision note.** An earlier run of this notebook kept the last row before each window in the training
> pool. That row's label is computed from the window's first close, so it leaked one outcome across the
> boundary; it is now purged (`.iloc[:-LABEL_HORIZON]`, the same embargo `chronological_split` applies).
> The strategy backtest also now uses exact short-position returns. Removing a single training row moved
> the 2023 macro-F1 from 0.301 to 0.358 and its strategy Sharpe from −1.47 to +1.19 — the random forest's
> bootstrap reshuffles completely. That sensitivity is itself a result: **window-level numbers this
> unstable are noise-dominated**, which supports the verdict below rather than weakening it.

Upweighting the under-predicted classes trades the "always flat" bias for a differently biased rule — it
doesn't add real signal, it redistributes errors.

## Fix 2: walk-forward across regimes

Windows chosen from the full-year buy-and-hold table (`03_eda.ipynb` item 12): **2022** (bear, BH Sharpe
−1.34), **2023** (bull, BH Sharpe 2.34), **2025** (range, BH Sharpe ≈0.02), plus the **original** test
window, all retrained walk-forward (train on everything strictly before the window, inner 85/15
train/val split for calibration) so every window uses the identical recipe.

| Window | Strategy Sharpe | Buy-and-hold Sharpe | Strategy return | BH return | Business criterion |
|---|---|---|---|---|---|
| original (now recalibrated) | **−0.44** | −0.33 | −20.5% | −24.9% | **Not met** |
| 2022 (bear) | 0.28 | −1.34 | +2.9% | −65.4% | Met |
| 2023 (bull) | 1.19 | 2.34 | **+28.5%** | +154.0% | **Not met** |
| 2025 (range) | 0.37 | 0.02 | +6.7% | −7.5% | Met |

**This is the more important result, and it's bad news.** Forcing the model to take more directional
positions (instead of sitting mostly in cash, which the calibration fix partially corrects) removes the
earlier apparent win entirely: on the *original* window the recalibrated strategy now **underperforms**
buy-and-hold. The Phase 5 "technical pass" was an artifact of low market exposure, confirmed by removing
it. Across regimes, the strategy wins in bear/range markets but **badly lags buy-and-hold in a bull
market** (+28.5% vs. +154.0%, sitting flat 72% of the days) — a
strategy that only wins when the market is falling or flat is not a robust edge; it's a defensive-by-
accident model wearing a trading strategy's clothes.

## Verdict

| | Before this iteration | After |
|---|---|---|
| Data-mining criterion | Not met | Still not reliably met (2 of 4 windows, by ~0.01 macro-F1) |
| Business criterion | Met on 1 (bear) window | Not robust — fails in the bull and the original window |

**Still not ready for Deployment.** The honest conclusion is that price/volume features and these three
model families, even with better-calibrated decision thresholds, do not produce a regime-robust
directional edge on 1D Bitcoin returns — a stronger confirmation of the EDA's original warning
(`03_eda.ipynb` item 10: no raw variable linearly predicts direction) than the first-pass Phase 4/5 result
suggested.

## Next, still without touching Deployment

1. The project's own design (`README.md`) treats the ML model as one input to an agent that also uses
   **news sentiment and market indicators**. This iteration suggests those are not optional extras — they
   may be necessary for any real edge to exist at all.
2. Calibration is confirmed not to be the bottleneck — the next lever for the ML leg alone is features
   (or a different horizon/θ, e.g. the 4h/θ=0.5% fallback noted in `docs/02`), not more threshold tuning.
3. Any future result should be re-run through this same multi-regime check before trusting a
   "beats buy-and-hold" claim again — a single-window pass is not sufficient evidence, as this notebook
   demonstrated directly.

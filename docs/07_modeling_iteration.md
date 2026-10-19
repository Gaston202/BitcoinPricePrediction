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
data-mining criterion (beats both baselines on **both** macro-F1 and ROC-AUC) in only 1 of 4:

| Window | Model macro-F1 | Persistence macro-F1 | Data-mining criterion |
|---|---|---|---|
| original (2025-07 → 2026-09) | 0.359 | 0.356 | Met (barely) |
| 2022 (bear) | 0.358 | 0.361 | Not met |
| 2023 (bull) | 0.301 | 0.347 | Not met |
| 2025 (range) | 0.352 | 0.363 | Not met |

Upweighting the under-predicted classes trades the "always flat" bias for a differently biased rule — it
doesn't add real signal, it redistributes errors.

## Fix 2: walk-forward across regimes

Windows chosen from the full-year buy-and-hold table (`03_eda.ipynb` item 12): **2022** (bear, BH Sharpe
−1.34), **2023** (bull, BH Sharpe 2.34), **2025** (range, BH Sharpe ≈0.02), plus the **original** test
window, all retrained walk-forward (train on everything strictly before the window, inner 85/15
train/val split for calibration) so every window uses the identical recipe.

| Window | Strategy Sharpe | Buy-and-hold Sharpe | Strategy return | BH return | Business criterion |
|---|---|---|---|---|---|
| original (now recalibrated) | **−0.59** | −0.33 | −24.8% | −24.9% | **Not met** |
| 2022 (bear) | 1.24 | −1.34 | +59.9% | −65.4% | Met |
| 2023 (bull) | **−1.47** | 2.34 | **−27.7%** | +154.0% | **Not met** |
| 2025 (range) | 0.25 | 0.02 | +3.3% | −7.5% | Met |

**This is the more important result, and it's bad news.** Forcing the model to take more directional
positions (instead of sitting mostly in cash, which the calibration fix partially corrects) removes the
earlier apparent win entirely: on the *original* window the recalibrated strategy now **underperforms**
buy-and-hold. The Phase 5 "technical pass" was an artifact of low market exposure, confirmed by removing
it. Across regimes, the strategy wins in bear/range markets but **loses badly in a bull market** — a
strategy that only wins when the market is falling or flat is not a robust edge; it's a defensive-by-
accident model wearing a trading strategy's clothes.

## Verdict

| | Before this iteration | After |
|---|---|---|
| Data-mining criterion | Not met | Still not reliably met (1 of 4 windows, barely) |
| Business criterion | Met on 1 (bear) window | Not robust — fails hard in a bull regime |

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

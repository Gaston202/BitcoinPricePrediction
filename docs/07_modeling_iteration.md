# Modeling Iteration (CRISP-DM Phase 4 ↔ 5)

Rewritten 2026-11-22. Code and figures: `notebooks/07_modeling_iteration.ipynb`.

| Iteration | Setup | Outcome |
|---|---|---|
| 1 · midterm `05` | 1D, 3-class θ = 1%, LR/RF/GB, single chronological split | No model beat persistence; argmax predicted flat on 89% of days |
| 2 · midterm `07` | + calibrated decision rule, 4-window walk-forward | Random forest beat persistence in 1 of 4 windows |
| 3 · 2026-11-21 | 1D, 9 models, Optuna, fixed 12 features (+ a 13-indicator ablation) | LightGBM beat persistence 4/4, +3.9 macro-F1. The paper indicators made 8 of 9 models worse. Accuracy (~44%) did not beat "always flat" |
| 4 · 2026-11-21/22 | 1h/4h/1D × 3 targets, 23 short-term features, per-window selection, Platt calibration, reversal baseline | 1h binary clears the revised criterion 4/4; 1D still has no direction edge |

## Why iteration 4 changed what it changed

| Change | Evidence |
|---|---|
| Intraday horizons | More data and documented short-term structure (`docs/09` #21, #34) |
| Binary / vol-scaled / triple-barrier targets | Fixed next-bar labels rank last in head-to-head comparisons (#22, #26) |
| Short-term features | Requested by the team; intraday realized volatility is a standard predictor |
| Per-window selection | Removes the overlap of the fixed 12 with the 2022/2023 windows |
| Calibration + log-loss/ECE | The agent consumes probabilities (#38) |
| Shared sign-AUC | Compares targets with different classes on direction skill |
| Bootstrap + quarterly folds | Small edges need p-values and more data points (#21) |
| Reversal baseline | Found in the results: return signs mean-revert intraday |

## Lesson

Most of the intraday lift over persistence (about 7 macro-F1 points) is the reversal effect. The model's own edge
beyond reversal is about **1–1.5 points at 1h**. It is small, but it is significant, stable across regimes and
quarters, and concentrated in the most confident predictions.

## Final model

**1h · binary · LightGBM · Platt-calibrated**, saved in `src/modeling/selected_model.json` with the hyperparameters
and the 10 features of the latest window. `model_from_choice` rebuilds it. Features: `rsi_14`, `rsi_3`,
`mom7_x_volratio`, `stoch_d`, `body_ratio`, `obv_slope`, `taker_buy_ratio`, `close_location`, `vol_ratio`, `streak`.
It was chosen over the nominal #1 (stacking) for simplicity: the same sign-AUC within 0.001, and one model instead
of five.

## Next

Next-bar high/low price levels with 10/90% bands, and the 5-day close (return quantiles + direction) on 1h and 4h
bars (agreed 2026-11-22). Then the agent: confidence-band trading, judged on Sharpe net of fees.

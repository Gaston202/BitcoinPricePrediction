# CRISP-DM Phase 3 — Data Preparation

Rewritten 2026-11-22 for the multi-horizon iteration. Code and figures: `notebooks/04_feature_engineering.ipynb`,
`reports/figures/04_*.png`. Implementation: `src/features/engineering.py`, `src/features/indicators.py`,
`src/features/short_term.py` (new families + `candidate_pool`), `src/features/targets.py` (labels),
`src/features/selection.py` (filter), `src/modeling/multi_horizon.py` (`select_window_features`).

## 1. Bar sizes and labels

There are three bar sizes, each with its own model: **1h** (77.8k model-ready rows), **4h** (19.4k) and **1D**
(2.8k), all built from the frozen 1h snapshot. Every label looks **one bar ahead**: binary up/down,
volatility-scaled 3-class, or triple barrier (definitions in `docs/05` §2). Up bars are 51% at every horizon.

## 2. Candidate pool (56 features for 1D, 58 for 1h/4h)

| Family | n | Columns |
|---|---|---|
| Original 1D set (midterm) | 22 | `ret_lag1..3`, `mom_7d/14d/30d`, `vol_7d/14d/30d`, `atr_14`, `vol_ratio`, `dist_ma20/50`, `rsi_14`, `volume_z`, `volume_ratio`, `taker_buy_ratio`, `dow_sin/cos`, `is_weekend`, `ret1_x_volz`, `mom7_x_volratio` |
| Paper indicators | 13 | MACD (3), stochastic %K/%D, Bollinger %B/width, ROC-10, Williams %R, CCI-20, OBV slope, MFI-14, BOP |
| **Fast indicators** (new) | 6 | `rsi_3`, `rsi_5`, `mom_3`, `bb5_pct_b`, `stoch5_k`, `ema3_8_cross` |
| **Candle / price action** (new) | 6 | `body_ratio`, `upper_wick`, `lower_wick`, `close_location`, `range_pct`, `streak` |
| **Intraday from 1h candles** (new) | 9 | `rv_4h`, `rv_24h`, `rv_168h`, `rv_ratio_day_week`, `ret_last_1h`, `ret_last_4h`, `hour_volume_spike`, `up_hours_share`, `max_hour_share` |
| **Hour of day** (new, 1h/4h) | 2 | `hour_sin`, `hour_cos` |

Window lengths are in bars: on the 1h model, `mom_7d` means 7 hours. Intraday features are always in hours. For a
bar that opens at *t* and lasts *B* hours, they are read from the 1h candle that opens at *t + B − 1h* and closes
exactly when the bar closes.

## 3. Leakage and scale checks

These run as unit tests (`tests/test_short_term.py`, `tests/test_targets.py`) and again on the real data in notebook 04:

- **Look-ahead:** multiplying every price after a cut date by 3 changes **0 features** before the cut, at all three
  horizons. The 4h/1D bars are rebuilt from the shocked 1h data, so the resampling step is covered too.
- **Scale:** multiplying all prices by 10 changes nothing at 4h or 1D. At 1h it changes 15 of the 77.8k rows, all
  in `mfi_14`. MFI uses the *sign* of the hourly change in typical price, and ×10 rounding can flip that sign when
  the price barely moves. This is floating-point tie-breaking, not a dependence on the price level.
- Triple-barrier labels read only the next bar's 1h path. A hand-built path checks that hits are read in order.

## 4. Redundancy

On pre-2022 1h rows, 63 feature pairs exceed |Spearman ρ| = 0.85. Examples: `ret_lag1` ≡ `ret_last_1h` (on 1h),
`bop` ≡ `body_ratio`, `stoch_k` ≡ `williams_r`, `vol_30d` ≈ `rv_24h` (0.97). About half the principal components
carry 90% of the variance. PCA is **not** used for modeling: the correlation filter removes the duplicates and
keeps the features interpretable, the same decision as in the midterm version.

## 5. Relevance (ANOVA + mutual information, inner train of the `original` window)

Features failing both tests (noise): 15 of 58 at 1h, 22 of 58 at 4h, 39 of 56 at 1D. New short-term features in the
MI top 15: 4 at 1h, 7 at 4h, **11 at 1D**.

## 6. Selection inside every walk-forward window

The midterm selected 12 features once, on 2017-10 → 2024-06, which overlaps the 2022/2023 test windows. Now
`select_window_features` runs separately for every horizon × target × window, on that window's inner training rows:

1. **Filter** (`select_features`): drop near-constant and zero-variance features, noise, and the lower-MI member of
   each |ρ| > 0.85 pair.
2. **Rank** the survivors by MI with that window's target (using the last 20k rows for 1h).
3. **Count:** keep the top 10 / 15 / 20 / 30, whichever gives a default LightGBM the lowest
   **inner-validation log-loss**.

Over the 36 runs, about 13 features are kept on average for 1D and 24 for 1h/4h. The new short-term features are
40% of the pool and 40–50% of the selected sets. `close_location` and `streak` survive almost everywhere, and the
intraday family matters most at 4h and 1D. Example (1D binary, `original` window): `rv_ratio_day_week`,
`ema3_8_cross`, `rv_168h`, `bb5_pct_b`, `close_location`, `dow_sin`, `up_hours_share`, `ret_lag1`, `ret1_x_volz`,
`streak`.

## 7. Outputs

- No processed files are written any more. Features are built on the fly by `candidate_pool`.
- `src/features/selected_features.json` (the midterm 12) is kept, because iteration 3 (`docs/07`) and
  `scripts/run_direction_experiment.py` still use it.
- The per-window selections are in `reports/results/multi_horizon/<horizon>_<target>_meta.json`.

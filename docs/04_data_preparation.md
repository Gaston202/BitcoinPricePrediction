# CRISP-DM Phase 3 — Data Preparation

Full analysis with code and figures: `notebooks/04_feature_engineering.ipynb`, structured as
creation → curse of dimensionality → selection → extraction → selected indicators.
Figures for the report: `reports/figures/04_*.png`. Implementation: `src/features/engineering.py` (original
candidates), `src/features/short_term.py` (new short-term indicators) and `src/features/selection.py`
(selection + PCA).

Updated 2026-11-22: 21 new short-term indicators were added to the candidates, and the selection was re-run.
Everything else is unchanged from the midterm version.

## 1. Label (open question #1 from `docs/02`, resolved)

**1D horizon, θ = 1%**, the option recommended in `03_eda.ipynb` item 11. It is kept unchanged on purpose, so any
change in results can be attributed to the features alone.

| | |
|---|---|
| Rule | UP if forward log return > +1%, DOWN if < −1%, FLAT otherwise |
| Class balance | ≈ 30% down / 38% flat / 33% up on the model-ready rows |
| Why not 4h/θ=0.5% | 1D's typical move is ≈ 7× the round-trip fee, vs ≈ 2.5× for 4h: fewer samples but a much cleaner signal |

## 2. Candidate features (43)

Every candidate is built from data up to and including `close[t]`. A unit test shocks future prices and asserts
that no past feature moves, and notebook 04 repeats that check for the new indicators on the real data.

| Family | Columns |
|---|---|
| Mathematical transformations | `ret_lag1..3` (true lags of the 1-day log return), `mom_7d/14d/30d` (cumulative log return), `vol_7d/14d/30d`, `atr_14` / close |
| Aggregations | rolling windows above, `volume_z` (14d z-score of log volume), `volume_ratio` (volume / 30d mean) |
| Domain knowledge | `dist_ma20`, `dist_ma50` (close / MA − 1), `vol_ratio` (vol_7d / vol_30d), `rsi_14`, `taker_buy_ratio` |
| Date / time | `dow_sin`, `dow_cos` (cyclical day of week), `is_weekend` |
| Interactions | `ret1_x_volz` (ret_lag1 × volume_z), `mom7_x_volratio` (mom_7d × vol_ratio) |
| **New · fast indicators** | `rsi_3`, `rsi_5`, `mom_3`, `bb5_pct_b`, `stoch5_k`, `ema3_8_cross` |
| **New · candle / price action** | `body_ratio`, `upper_wick`, `lower_wick`, `close_location`, `range_pct`, `streak` |
| **New · intraday (from 1h candles)** | `rv_4h`, `rv_24h`, `rv_168h`, `rv_ratio_day_week`, `ret_last_1h`, `ret_last_4h`, `hour_volume_spike`, `up_hours_share`, `max_hour_share` |

The new indicators are all scale-free. The intraday ones are read from the last 1h candle of each day, which closes
exactly when the day closes.

**Changed in the first version:**
- The old `ret_lag5/7/14` were overlapping multi-day returns, not lags. True lags stop at 3 days; multi-day returns
  now live in the honestly named `mom_*` columns.
- `log_volume` is removed on domain grounds: the level trends upward over the years, so the values learned on train
  never reappear later. A statistical test on train cannot catch this. Only relative volume is kept.

The 50-day moving average sets the warm-up, and every unfilled data gap costs 50 rows after it. The new indicators
look back at most 7 days, so they cost no extra rows: **2,793 model-ready rows**.

## 3. Chronological split (open question #2, resolved)

A 70/15/15 split by position, never shuffled, with the boundary rows purged (`embargo=1`). **The split happens
before selection**, and every selection or extraction step is fit on `train` only.

| Split | Rows | Window |
|---|---|---|
| Train | 1,954 | 2017-10-26 → 2024-06-12 |
| Validation | 417 | 2024-06-14 → 2025-08-04 |
| Test | 420 | 2025-08-06 → 2026-09-29 |

**Test window regime:** a bear/declining market (buy-and-hold over the original test window: −25%, Sharpe −0.33,
max drawdown −53%). Per `03_eda.ipynb` item 12, a strategy that mostly sits in cash can beat buy-and-hold here
without being "smart".

## 4. Curse of dimensionality

1,954 train rows / 43 candidates ≈ **45 rows per feature** (≈ 15 per feature per class), half the midterm ratio.
That is thin for a signal as weak as daily BTC direction, which makes selection and extraction more important.

## 5. Feature selection — train only

| Step | Rule | Removed |
|---|---|---|
| Correlation heatmap | Spearman (robust to fat tails) | — (diagnostic, 25 pairs above \|ρ\| = 0.85) |
| Feature vs target | Boxplots per label class | — (diagnostic) |
| Near-constant | One value in > 95% of rows | none |
| Low variance | std ≈ 0 | none |
| Clear noise | ANOVA p > 0.05 **and** MI ≤ 95th pct of shuffled-label MI | 19: `ret_lag2`, `ret_lag3`, `mom_14d`, `rsi_14`, `taker_buy_ratio`, `is_weekend`, `mom7_x_volratio`, and 12 new: `rsi_3`, `rsi_5`, `stoch5_k`, `bb5_pct_b`, `streak`, `upper_wick`, `lower_wick`, `up_hours_share`, `max_hour_share`, `hour_volume_spike`, `ret_last_1h`, `rv_ratio_day_week` |
| Correlated pairs | \|ρ\| > 0.85, keep higher MI | 8: `body_ratio` (↔ `ret_lag1`, 0.96), `dist_ma20` (↔ `ema3_8_cross`, 0.95), `mom_30d` (↔ `dist_ma50`, 0.93), `rv_24h` (↔ `range_pct`, 0.93), `ema3_8_cross` (↔ `mom_7d`, 0.91), `vol_14d` (↔ `atr_14`, 0.91), `volume_ratio` (↔ `volume_z`, 0.89), `atr_14` (↔ `rv_168h`, 0.88) |

**Selected (16):** 10 original (`ret_lag1`, `mom_7d`, `vol_7d`, `vol_30d`, `vol_ratio`, `dist_ma50`, `volume_z`,
`dow_sin`, `dow_cos`, `ret1_x_volz`) and 6 new (`mom_3`, `close_location`, `range_pct`, `rv_4h`, `rv_168h`,
`ret_last_4h`). They are stored with every drop reason in the tracked manifest `src/features/selected_features.json`,
which `src/modeling/train.py` loads as `FEATURE_COLUMNS`.

## 6. Feature extraction: PCA — train only

Standardized PCA (equivalent to PCA on the correlation matrix) on all 43 candidates. **17 components explain
91.2%** of the variance (10 of 22 before). The first three: PC1 ≈ 27% trend and oscillators (`rsi_5`, `rsi_3`,
`ema3_8_cross`, `dist_ma20`, `mom_7d`); PC2 ≈ 16% volatility (`rv_24h`, `range_pct`, `rv_168h`, `vol_*`, `atr_14`);
PC3 ≈ 9% where the day closed (`close_location`, `body_ratio`, `ret_lag1`, `up_hours_share`, `streak`), a block the
original candidates did not have. Selection remains the main input because it stays interpretable; the PCA variant
is persisted for a head-to-head comparison.

## 7. Which indicators were selected, and why

The label is 3-class, so a feature can help by telling *move* from *flat* (volatility) or *up* from *down*
(direction). Almost all the information is of the first kind.

| Kept (new) | Why |
|---|---|
| `rv_168h` | Highest MI of all 43 candidates (0.048 vs a 0.015 chance level); replaces `atr_14` as the cleaner volatility measure |
| `range_pct` | MI 0.024, ANOVA p < 0.001: a wide day predicts a big next day; replaces `rv_24h` |
| `rv_4h` | MI 0.019, ANOVA p < 0.001: whether the day ended calm or agitated |
| `ret_last_4h` | MI 0.017 above its 0.014 baseline: a non-linear signal from how the day finished |
| `close_location` | ANOVA p = 0.035: one of the few direction signals (closing near the day's high or low) |
| `mom_3` | Just above its baseline (MI 0.0129 vs 0.0125): the weakest survivor |

12 of the new indicators are noise on daily bars. The fast oscillators, the wicks, the streak and the intraday
shape measures carry short-memory signals that average out before the next day. Three more duplicate existing
features: `body_ratio` ≈ `ret_lag1`, `rv_24h` ≈ `range_pct`, `ema3_8_cross` ≈ `mom_7d`. Two original features lost
their place to new ones: `atr_14` to `rv_168h`, and `dist_ma20` to `ema3_8_cross`, which was then itself dropped.

## 8. Start date (open question #3, resolved)

The full 2017-08-17 start is kept. The feature warm-up already trims the thinnest early rows, and the
chronological split pushes 2017–2018 entirely into `train`.

## 9. Output

- `data/processed/btcusdt_1D_{train,val,test}.parquet`: all 43 candidates + `label` (gitignored).
- `data/processed/btcusdt_1D_pca_{train,val,test}.parquet`: `pc1..pc17` + `label` (gitignored).
- `src/features/selected_features.json`: tracked; the single source of truth for the 1D model's columns.

All are regenerated by `notebooks/04_feature_engineering.ipynb`. The model-comparison run of 2026-11-21
(`docs/07`, iteration 3) used the earlier 12-feature manifest, which is preserved in git history.
The multi-horizon models (`docs/05`) do not use this manifest. They re-select features inside every walk-forward
window from a wider pool, as shown in `notebooks/05_modeling.ipynb`.

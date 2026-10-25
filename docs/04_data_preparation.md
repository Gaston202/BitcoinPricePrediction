# CRISP-DM Phase 3 — Data Preparation

Full analysis with code and figures: `notebooks/04_feature_engineering.ipynb`, structured as
creation → curse of dimensionality → selection → extraction.
Figures for the report: `reports/figures/04_*.png`. Implementation: `src/features/engineering.py`
(creation) and `src/features/selection.py` (selection + PCA).

## 1. Label (open question #1 from `docs/02`, resolved)

**1D horizon, θ = 1%** — the option recommended in `03_eda.ipynb` item 11. Deliberately unchanged in this
version, so any change in results can be attributed to the features alone.

| | |
|---|---|
| Rule | UP if forward log return > +1%, DOWN if < −1%, FLAT otherwise |
| Class balance | ≈ 30% down / 38% flat / 33% up on the model-ready rows |
| Why not 4h/θ=0.5% | 1D's typical move is ≈7× the round-trip fee vs. ≈2.5× for 4h; fewer samples but a much cleaner signal |

## 2. Candidate features

22 candidates, every one built from data up to and including `close[t]` (a unit test shocks future prices
and asserts no past feature moves).

| Family | Columns |
|---|---|
| Mathematical transformations | `ret_lag1..3` (true lags of the 1-day log return), `mom_7d/14d/30d` (cumulative log return), `vol_7d/14d/30d`, `atr_14` / close |
| Aggregations | rolling windows above, `volume_z` (14d z-score of log volume), `volume_ratio` (volume / 30d mean) |
| Domain knowledge | `dist_ma20`, `dist_ma50` (close / MA − 1), `vol_ratio` (vol_7d / vol_30d), `rsi_14`, `taker_buy_ratio` |
| Date / time | `dow_sin`, `dow_cos` (cyclical day of week), `is_weekend` |
| Interactions | `ret1_x_volz` (ret_lag1 × volume_z), `mom7_x_volratio` (mom_7d × vol_ratio) |

**Changed from the first version:**
- The old `ret_lag5/7/14` were overlapping multi-day returns, not lags. True lags stop at 3 days; multi-day
  returns now live in the honestly named `mom_*` columns.
- `log_volume` is removed on domain grounds: the level trends upward over the years, so the values learned
  on train never reappear later. A statistical test on train cannot catch this. Only relative volume is kept.

The 50-day moving average sets the warm-up, and every unfilled data gap costs 50 rows after it:
2,793 model-ready rows (previously 2,983 with a 30-day longest window).

## 3. Chronological split (open question #2, resolved)

70/15/15 split by position, never shuffled, with the boundary rows purged (`embargo=1`). **The split now
happens before selection**, and every selection/extraction step is fit on `train` only.

| Split | Rows | Window |
|---|---|---|
| Train | 1,954 | 2017-10-26 → 2024-06-12 |
| Validation | 417 | 2024-06-14 → 2025-08-04 |
| Test | 420 | 2025-08-06 → 2026-09-29 |

**Test window regime:** a bear/declining market (buy-and-hold over the original test window: −25%, Sharpe
−0.33, max drawdown −53%). Per `03_eda.ipynb` item 12, a strategy that mostly sits in cash can beat
buy-and-hold here without being "smart".

## 4. Curse of dimensionality

1,954 train rows / 22 candidates ≈ 89 rows per feature (≈ 30 per feature per class). Not extreme, but thin
for a signal as weak as daily BTC direction, which motivates both selection and extraction.

## 5. Feature selection — train only

| Step | Rule | Removed |
|---|---|---|
| Correlation heatmap | Spearman (robust to fat tails) | — (diagnostic) |
| Feature vs target | Boxplots per label class | — (diagnostic) |
| Near-constant | one value in > 95% of rows | none |
| Low variance | std ≈ 0 | none |
| Clear noise | ANOVA p > 0.05 **and** MI ≤ 95th pct of shuffled-label MI | `ret_lag2`, `ret_lag3`, `mom_14d`, `rsi_14`, `taker_buy_ratio`, `is_weekend`, `mom7_x_volratio` |
| Correlated pairs | \|ρ\| > 0.85, keep higher MI | `mom_30d` (↔ `dist_ma50`, 0.93), `vol_14d` (↔ `atr_14`, 0.91), `volume_ratio` (↔ `volume_z`, 0.89) |

**Selected (12):** `ret_lag1`, `mom_7d`, `vol_7d`, `vol_30d`, `atr_14`, `vol_ratio`, `dist_ma20`, `dist_ma50`,
`volume_z`, `dow_sin`, `dow_cos`, `ret1_x_volz`. Stored with every drop reason in the tracked manifest
`src/features/selected_features.json`, which `src/modeling/train.py` loads as `FEATURE_COLUMNS`.

**What the evidence says:** the volatility features carry by far the most information, but they separate
*flat* from *move*, not *up* from *down*. No feature clearly shifts the up distribution against the down
one: the direction signal in price/volume data is weak. With 22 tests at α = 0.05 about one false positive
is expected, so a lone significant result (e.g. the day-of-week terms) is a hypothesis, not a finding.

## 6. Feature extraction: PCA — train only

Standardized PCA (equivalent to PCA on the correlation matrix) on all 22 candidates.
**10 components explain 91.7%** of the variance. The first three recover the heatmap's blocks:
PC1 ≈ 29% trend (`dist_ma*`, `rsi_14`, `mom_*`), PC2 ≈ 17% volatility (`vol_*`, `atr_14`), PC3 ≈ 11%
activity/calendar (volume vs. weekend). Selection remains the main input (interpretable); the PCA
variant is persisted for a head-to-head comparison (compactness vs. interpretability).

## 7. Start date (open question #3, resolved)

Kept the full 2017-08-17 start. The feature warm-up already trims the thinnest early rows, and the
chronological split pushes 2017–2018 entirely into `train`.

## 8. Output

- `data/processed/btcusdt_1D_{train,val,test}.parquet` — all 22 candidates + `label` (gitignored).
- `data/processed/btcusdt_1D_pca_{train,val,test}.parquet` — `pc1..pc10` + `label` (gitignored).
- `src/features/selected_features.json` — tracked; the single source of truth for the model's columns.

All regenerated by `notebooks/04_feature_engineering.ipynb`.

## 9. Open questions for Modeling

1. Do the selected 12 features or the PCA variant beat the original 13-feature set on `val`?
   (Compare on identical rows; the new warm-up shortens every split.)
2. The label itself is the next lever: a volatility-scaled θ instead of a fixed 1%, as a separate iteration.

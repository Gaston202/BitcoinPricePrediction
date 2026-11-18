# Modeling Handoff — Findings So Far and Next Steps

**Date:** 2026-11-18 · **Branch:** `feat/model-comparison` · **Status:** code done and tested (107 tests), full
experiments **not run yet**. They will run on a second laptop.

This file records what was decided, what was found, how to run the experiments, and what's left.

---

## 1. Decisions (agreed 2026-11-18)

| Topic | Decision |
|---|---|
| Literature search | `paper-search-mcp` (user-scope MCP, `uvx --with "mcp<2" paper-search-mcp`), direction-classification papers on OHLCV/technical features, ~20 papers → `docs/09_literature_review.md` |
| "Best model" judged on | **Classification metrics only**: macro-F1 and ROC-AUC vs majority + persistence baselines |
| Direction target | Unchanged: 3-class next-day direction, θ = 1 % |
| Models | LR, RF, GB (existing) + **XGBoost, LightGBM, SVM, kNN, soft voting, stacking**. No deep learning for now (future work) |
| Features | The fixed 12 from `selected_features.json` (main run) + 13 paper indicators (comparison run) |
| Tuning | Optuna, 50 trials per model per window, objective = macro-F1 on inner validation |
| Calibration | Every model reported **raw** (argmax) and **calibrated** (`tune_class_weights` on inner validation) |
| Winner rule | 4-window walk-forward. Rank by: windows beating persistence → mean macro-F1 → mean ROC-AUC |
| Feature selection per window | **Keep the fixed 12** (decided). Caveat: they were selected on 2017-10 → 2024-06, which overlaps the 2022 and 2023 windows, so those two windows are optimistic |
| Range model (added later) | Also explore next-day **high/low as % from close** as the risk leg for the portfolio |
| Notebooks | Rewrite 05–07 (05 = candidates + tuning + feature comparison, 06 = classification evaluation + short business section, 07 = walk-forward + final model) |

## 2. Findings so far

### 2.1 Literature (`docs/09_literature_review.md`)
- Honest next-day direction accuracy from price features is **52–56 %** (Akyıldırım 2021, Jaquart 2021/2022, Ortu 2022).
- **No consistent winner**: SVM is best in 2 clean daily studies, boosting in 3, LR in 2 (Chen 2020, Dimitriadou 2023).
- **Match complexity to sample size** (Chen 2020): on ~730 daily rows LR/LDA beat the complex ML models. We have ~2.8k rows.
- **Leakage confirmed in the full text** of three highly cited "high accuracy" papers:
  - Öktem 2025 (99.2 %) and Jung 2023 (90.6 %): the feature/label comes from the **same candle**.
  - Radmand 2025 (91–92 %): **shuffled 80/20 split** with labels that span several days.
- Worth reusing: volatility-scaled 3-class label (Fernandes 2025), training-median thresholds (Jaquart 2021),
  accuracy on the most confident predictions only (Jaquart 2022: +5 points on the top 10 %).

### 2.2 Methodology issues found in our own pipeline
- The fixed 12 features were chosen on data that overlaps the 2022/2023 walk-forward windows (see §1).
  `docs/07` has the same flaw. Only the **original** and **2025_range** windows are fully clean.
- `SVC(probability=True)` is deprecated in scikit-learn 1.9, so it was replaced by
  `CalibratedClassifierCV(SVC(), ensemble=False)` with unshuffled folds.

### 2.3 Preliminary smoke-run results (NOT final)
These come from a **2-trial** run on **3 models** (LR, XGBoost, soft voting), fixed-12 features. Treat them as
a sanity check only. The full run (50 trials, 9 models, 2 feature sets) replaces them.

**Direction (calibrated variant):**

| Model | Windows beating persistence | Windows meeting full criterion | Mean macro-F1 | Mean ROC-AUC |
|---|---|---|---|---|
| soft_voting | **4 / 4** | **4 / 4** | 0.394 | 0.588 |
| logistic_regression | 3 / 4 | 3 / 4 | 0.393 | 0.595 |
| xgboost | 3 / 4 | 3 / 4 | 0.386 | 0.576 |

Persistence macro-F1 per window is 0.347–0.363, so the margins are **+2 to +7 points**. That's clearly better
than `docs/07` (random forest: 1 of 4 windows), and in line with the "a few points above baseline" ceiling from
the literature. Calibration helps: raw argmax still predicts "flat" on 50–77 % of days.

**Range (R² of next-day upside/downside, per window):**

| Target | HAR | Ridge | Persistence | 20-day mean |
|---|---|---|---|---|
| up_next | 0.003 – 0.060 | 0.000 – 0.082 | −0.96 – −0.52 | −0.06 – 0.00 |
| down_next | −0.004 – 0.079 | 0.005 – 0.119 | −0.92 – −0.40 | −0.03 – 0.00 |

- Models beat both baselines on MAE (≈ 0.0128–0.0131 vs 0.0139 for the 20-day mean), but R² is small (≈ 0.05).
  Daily high/low are noisier than expected. The signal is real but modest.
- **Breach rate ≈ 33–39 %**: the models predict the *average* move, so the actual high/low exceeds the
  prediction on about 1 day in 3. For stop-loss/take-profit levels we need **quantile regression** (e.g. the
  90th percentile of the upside), not the mean. See §4.

## 3. How to run on the other laptop

```bash
git clone https://github.com/Gaston202/BitcoinPricePrediction.git
cd BitcoinPricePrediction
git checkout feat/model-comparison          # the branch must be pushed first
python -m venv .venv
source .venv/Scripts/activate               # Windows Git Bash; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt             # includes xgboost, lightgbm, optuna
pytest -q                                   # expect 107 passed

# data/interim is gitignored: regenerate it from the committed raw snapshot
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_quality.ipynb

# Experiments (results go to reports/results/)
python scripts/run_range_experiment.py                   # ~15-30 min
python scripts/run_direction_experiment.py               # ~1-1.5 h per feature set at 50 trials
# faster options:
python scripts/run_direction_experiment.py --trials 25
python scripts/run_direction_experiment.py --feature-sets selected12
```

Each script prints `window <name>: start` with elapsed time, so you can see progress. Bring back the whole
`reports/results/` folder (CSV + JSON files).

| Output file | Content |
|---|---|
| `direction_wf_<set>.csv` | One row per (window, model): raw/cal macro-F1, ROC-AUC, baselines, flat share, criterion flags |
| `direction_params_<set>.json` | Tuned hyperparameters per window and model |
| `direction_ranking_<set>.csv` | Ranking by the agreed rule, calibrated and raw |
| `range_wf.csv` | One row per (window, target, model/baseline): MAE, RMSE, R², breach rate |
| `range_predictions.csv` | Every out-of-sample range prediction (input for the portfolio simulation) |
| `range_summary.csv` | Mean metrics per (target, model) across windows |

## 4. What's left (in order)

1. [ ] **Push** `feat/model-comparison` so the other laptop can check it out.
2. [ ] **Run both experiments** (§3) and commit `reports/results/`.
3. [ ] **Analyze direction results:** winner per the agreed rule; fixed-12 vs +paper indicators; raw vs
   calibrated; flag the 2022/2023 windows as optimistic.
4. [ ] **Rewrite notebooks 05–07 + docs 05–07** around the results (structure in §1). Notebook 05 saves the
   val-selected model with `save_model_choice()` → `src/modeling/selected_model.json`; notebook 06 rebuilds it
   with `model_from_choice()`.
5. [ ] **Range model follow-ups:**
   - quantile regression (LightGBM `objective="quantile"`, α = 0.1 / 0.9) for stop/take-profit levels;
   - check breach rates match the target quantile.
6. [ ] **Portfolio simulation:** `position = clip((P(up) − P(down)) / predicted_range × target_vol, −1, 1)`, stops
   from the quantile range model. Use **1h candles** to decide whether stop or take-profit hit first inside a day.
   Compare with buy-and-hold on Sharpe, net of 0.1 % fees, over the same 4 windows.
7. [ ] Update `docs/08_conclusions.md` and the README.

### Future work (not in current scope)
- Volatility-scaled or training-quantile direction labels (Fernandes 2025, Jaquart 2021).
- Re-select features inside each walk-forward window (removes the 2022/2023 caveat).
- LSTM/GRU (only competitive with much more data in the literature).
- External data: on-chain, funding rates, sentiment. The literature suggests this is where most of the real gain is.
- Permutation test for significance of the edge over persistence (Mesfin 2026 template).
- Verify Alonso-Monsalve 2020 and Máté 2024 from full text (PDFs not retrieved).

## 5. Code map (new on this branch)

| File | What |
|---|---|
| `src/modeling/train.py` | All candidates, `LabelEncodedClassifier` (XGBoost → −1/0/1), ensembles, model-choice manifest |
| `src/modeling/tuning.py` | Optuna search spaces + `tune_model` |
| `src/modeling/walk_forward.py` | `WINDOWS`, `window_splits`, `evaluate_window`, `run_walk_forward`, `rank_models` |
| `src/modeling/range_model.py` | High/low targets, range features, HAR + ML regressors, baselines, range walk-forward |
| `src/features/indicators.py` | 13 paper indicators, look-ahead- and scale-tested |
| `scripts/run_direction_experiment.py`, `scripts/run_range_experiment.py` | Runnable experiments |
| `docs/09_literature_review.md` | 20-paper review with leakage flags |

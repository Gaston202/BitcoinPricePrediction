# CRISP-DM Phase 4 — Modeling

Full analysis with code and figures: `notebooks/05_modeling.ipynb`.
Figures for the report: `reports/figures/05_*.png`. Implementation: `src/modeling/train.py`, `src/evaluation/strategy.py`.

## 1. Candidates and selection

Three classifiers trained on `train`, selected on `val` only (`test` untouched until §3):

| Model | Val accuracy | Val macro-F1 |
|---|---|---|
| majority baseline | 0.413 | 0.195 |
| persistence baseline | 0.337 | **0.327** |
| logistic regression | 0.413 | 0.306 |
| random forest | 0.410 | 0.318 |
| gradient boosting | 0.406 | 0.293 |

**Best candidate: random forest** — but it does **not** beat the persistence baseline on macro-F1. None of
the three candidates do. This is the honest, expected outcome given EDA finding 10 (`docs/02`): no raw
variable linearly predicts direction, and nothing in this feature set changes that conclusion.

## 2. Final test evaluation (first and only look at `test`)

| | Test accuracy | Test macro-F1 |
|---|---|---|
| majority baseline | 0.451 | 0.207 |
| persistence baseline | 0.380 | **0.356** |
| random forest (selected) | 0.471 | 0.302 |

Random forest beats majority on raw accuracy, but like on `val`, it does **not** clear the persistence
baseline on macro-F1, which is the metric that actually matters here (accuracy is misleading on an
imbalanced 3-class target — see `docs/02` finding 12).

## 3. Business read: strategy vs. buy-and-hold on the test window

| | Total return | Sharpe | Max drawdown |
|---|---|---|---|
| random forest strategy | **+9.5%** | **0.44** | **−17.1%** |
| buy-and-hold | −24.9% | −0.33 | −53.0% |

At face value this looks like a clear win. **It is not evidence of predictive skill.** The model predicts
"flat" on 89% of test days, versus an actual flat rate of 45%:

| | down | flat | up |
|---|---|---|---|
| predicted | 4.0% | 89.3% | 6.7% |
| actual | 29.0% | 45.1% | 25.9% |

The strategy wins because it is in the market only ~11% of the time during a window that happened to be a
bear market (buy-and-hold Sharpe −0.33, flagged in `docs/04_data_preparation.md` as an open risk). Sitting
mostly in cash dodges the drawdown; it does not demonstrate the model learned direction. This is exactly
the caveat `03_eda.ipynb` item 12 raised in advance: *"A strategy that is mostly in cash can beat
buy-and-hold in a bear-market test year without being 'smart'."*

## 4. Conclusion

| Criterion | Result |
|---|---|
| Data-mining success criterion (beat persistence macro-F1) | **Not met** |
| Business success criterion (beat buy-and-hold on this window) | Met, but attributable to low market exposure, not direction skill |

This feature set and these three model families do not extract a usable directional signal beyond what
`03_eda.ipynb` already found: 1D direction is close to unpredictable with raw/engineered price-volume
features alone. The result is still useful — it is the honest ML-leg baseline the rest of the project
(news sentiment, market indicators, the full agent) needs to beat, and it rules out "just add a classifier"
as sufficient on its own.

## 5. Options for the next iteration

1. Treat this as the baseline and proceed to the full agent (news sentiment + market indicators per the
   README) — the business case may still work if the agent's other inputs add real signal.
2. Revisit the label: re-run EDA item 11 with a different horizon/threshold (e.g. 4h/θ=0.5%, the fallback
   noted in `docs/02`) to see if a different target is more learnable.
3. Address the "always flat" shortcut directly — e.g. a cost-sensitive loss or class-balanced threshold
   tuning that penalizes collapsing to the majority-adjacent class, rather than relying on `class_weight`
   alone.

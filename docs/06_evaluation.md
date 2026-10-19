# CRISP-DM Phase 5 — Evaluation

Full analysis with code: `notebooks/06_evaluation.ipynb`. Implementation: `src/evaluation/criteria.py`.

This phase does not train anything new. It checks the Phase 4 result against the two criteria fixed
*before any data was seen*, in `docs/01_business_understanding.md` — not against whichever metric looks
best in hindsight.

## 1. Data-mining success criterion

> `docs/01` Sec. 4: beat **both** majority and persistence baselines on **both** macro-F1 **and** ROC-AUC.

| | macro-F1 (test) | ROC-AUC (test) |
|---|---|---|
| majority baseline | 0.207 | 0.500 |
| persistence baseline | 0.356 | 0.518 |
| random forest (selected model) | 0.302 | **0.579** |

**Not met.** The model's ROC-AUC does beat both baselines — its predicted probabilities rank the true
class better than chance, a genuine if modest signal. But the criterion requires winning on both metrics
against both baselines, and macro-F1 loses to persistence. A model can have a better-than-chance
probability ranking and still make worse hard decisions than a trivial rule if its decision threshold is
miscalibrated — consistent with `docs/05_modeling.md`'s finding that it defaults to "flat" on 89% of test
days versus an actual 45% flat rate. This is a diagnosable calibration problem, not evidence of zero
signal.

## 2. Business success criterion

> `docs/01` Sec. 2: beat buy-and-hold BTC on Sharpe ratio, net of 0.1%-per-side fees, over the same test
> window.

| | Total return | Sharpe | Max drawdown |
|---|---|---|---|
| strategy (random forest positions) | +9.5% | **0.44** | −17.1% |
| buy-and-hold | −24.9% | −0.33 | −53.0% |

**Technically met.** But per the risk `docs/01` Sec. 5 flagged in advance — *"regime change: the test
period must be reported with its regime"* — this window is a bear market. The strategy wins by being in
the market only ~11% of the time, mechanically avoiding most of the drawdown. This has not been shown to
generalize to a bull or range-bound regime; it has only been shown to avoid losing in this specific
downturn.

## 3. Overall verdict

| Criterion | Status |
|---|---|
| Data-mining success | **Not met** |
| Business success | **Technically met, on a single bear-market window** |

**Recommendation: do not proceed to Deployment (Phase 6) on this model as-is.** A business-criterion pass
driven by one favorable test window is exactly the false positive the project's own risk log anticipated.

## 4. Before trusting this result

1. **Re-test across regimes.** Rerun the same pipeline with a bull-market and a range-bound test window
   (the yearly Sharpe table in `03_eda.ipynb` item 12 identifies candidate years) to check whether the
   business criterion still holds outside a downturn.
2. **Address the calibration gap.** The AUC-without-F1 pattern points to a threshold/calibration problem,
   not an absence of signal — worth trying class-balanced decision thresholds or a cost-sensitive
   objective before concluding this feature set is exhausted.
3. Only after (1) and (2) should this feed into the full agent (news sentiment + market indicators)
   with confidence that the ML leg contributes real signal rather than a favorable test window.

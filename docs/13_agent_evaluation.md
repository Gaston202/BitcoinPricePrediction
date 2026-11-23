# Agent Evaluation: the Business Criterion (CRISP-DM Phase 5)

Notebook: `notebooks/08_agent_evaluation.ipynb`. Code: `src/evaluation/portfolio.py` (paper wallet, baselines,
metrics, bootstrap test), `src/evaluation/agent_backtest.py` (data and windows), `src/evaluation/agent_experiments.py`
(tuning, selection, evaluation, ablation, sensitivity). Reproduce everything: `make agent`
(= `python scripts/run_agent_evaluation.py all`, ~6 min; the preset grid `make tune` takes ~1.5 h).
Results: `reports/results/agent/*.csv`.

## 1. Question

> **Business success criterion (docs/01 §3):** the strategy achieves a higher Sharpe ratio than buy-and-hold BTC
> over the out-of-sample test period, net of 0.1% trading fees per side.

Beating buy-and-hold in a falling market only requires holding less BTC, so the agent is also compared with five
simpler strategies. If a one-line rule does as well, the agent's complexity is not justified.

| Baseline | Rule |
|---|---|
| `cash` | Never invest |
| `buy_and_hold` | 100% BTC from the first candle |
| `static_50` | 50% BTC, rebalanced when it drifts |
| `sma200_filter` | 100% BTC when the close is above its 200-candle average, else cash |
| `ema20_50_cross` | 100% BTC while EMA20 > EMA50 |
| `vol_target` | Exposure = 40% target volatility ÷ recent volatility (max 100%) |
| `model_only` (1h) | Follow the team's LightGBM alone: 100% when P(up) > 52% |

All of them, and the agent, trade through the same `Wallet` with the same fee and rebalance band. A unit test checks
that replaying the agent's targets through `simulate_exposure` reproduces its equity exactly.

## 2. Protocol

| Period | Dates | Used for |
|---|---|---|
| **Development** | 2023-01-01 → 2025-06-30 | Choosing presets and the analyst team, by Sharpe. Nothing else |
| 2022_bear | 2022 | Out-of-sample |
| 2023_bull | 2023 | In-sample (inside development) |
| 2025_range | 2025 | Half in development |
| **test** | 2025-07-09 → 2026-09-30 | Out-of-sample: the team's test window |

* **Presets** (conviction smoothing, cooldown, minimum rebalance): a 48-point grid per candle size on the development
  period. The `marouen` presets ranked 3rd (1h), 12th (4h) and 17th (1D) of 48; the winners now sit in
  `INTERVAL_PRESETS`, and a unit test keeps them equal to `preset_tuning.csv`.
* **Team** (development Sharpe): `core` (trend, momentum, volatility, volume), `core+sentiment` (+ funding, long/short,
  Fear & Greed), `hybrid_lexicon` (+ news scored by the lexicon), `hybrid_llm` (+ news scored by the LLM, once its
  scores exist). Selected: **1h `core+sentiment`, 4h `hybrid_lexicon`, 1D `core`**.
* **The quant analyst's model** was itself chosen in phase 4 with the test window visible. Re-running that choice
  without the test window (`scripts/check_model_selection_without_test.py`) still picks the 1h binary target, with
  LightGBM 2nd of 9 models (3rd with all windows): the choice does not depend on the agent's test window.
* **Inputs are frozen** (`data/external/`): funding, Fear & Greed and news are snapshots, so the numbers below do not
  change when the APIs do.

## 3. Results

Sharpe ratio, net of 0.1% fees (★ = out-of-sample):

| Strategy | 1h 2022★ | 1h test★ | 4h 2022★ | 4h test★ | 1D 2022★ | 1D test★ |
|---|---|---|---|---|---|---|
| **Agent (selected team)** | −1.54 | −1.17 | −1.11 | **−0.11** | −1.87 | **+0.18** |
| Buy-and-hold | −1.31 | −0.31 | −1.39 | −0.32 | −1.34 | −0.34 |
| SMA200 filter | −2.52 | −1.01 | −0.84 | +0.22 | 0.00 | +0.41 |
| EMA20/50 cross | −1.46 | −0.42 | −1.55 | +0.26 | −1.91 | −0.45 |
| Static 50% | −1.45 | −0.20 | −1.51 | −0.20 | −1.43 | −0.25 |
| Vol target | −1.60 | −0.43 | −1.39 | −0.54 | −1.26 | −0.51 |

Test window (2025-07-09 → 2026-09-30; BTC −23 to −25%):

| | Return | Sharpe | Max drawdown | Trades |
|---|---|---|---|---|
| Agent 1h | −21.4% | −1.17 | −31.8% | 262 |
| Agent 4h | −2.6% | −0.11 | −19.4% | 56 |
| Agent 1D | +1.9% | +0.18 | −16.0% | 17 |
| Buy-and-hold | −23 to −25% | −0.31 to −0.34 | −53% | 1 |
| SMA200 filter 4h / 1D | +3.1% / +8.4% | +0.22 / +0.41 | −30.0% / −19.8% | |

**Significance** (paired circular block bootstrap of the Sharpe difference, one-week blocks, 2,000 resamples): no
out-of-sample difference between the agent and buy-and-hold is significant. Test window: 4h +0.21 [−1.54, +1.98],
p = 0.41; 1D +0.52 [−1.07, +2.36], p = 0.26.

## 4. Verdict on the business criterion

| Candles | Test window | Significant? | Beats SMA200? |
|---|---|---|---|
| 1h | **FAIL** (−1.17 vs −0.31) | — | no |
| 4h | **PASS** (−0.11 vs −0.32) | no (p = 0.41) | no |
| 1D | **PASS** (+0.18 vs −0.34) | no (p = 0.26) | no |

**The criterion as written is met on 4h and 1D candles, but the evidence is weak**:
* In the six out-of-sample cases (2 windows × 3 candle sizes) the agent beats buy-and-hold in 3, never significantly.
* The SMA200 filter has the higher Sharpe in 5 of the 6. The agent's edge over buy-and-hold is mostly being out of
  the market in a downtrend, which a one-line trend filter also achieves.
* The agent's clear, measurable benefit is **risk**: a test-window max drawdown of −16% (1D) and −19% (4h) against
  −53% for buy-and-hold, with a written reason for every trade.

## 5. Ablation (remove one part, rerun; Δ Sharpe, negative = the part helps)

| Part | 1h dev / test | 4h dev / test | 1D dev / test | Reading |
|---|---|---|---|---|
| Conviction smoothing | −1.40 / −0.62 | −0.50 / +0.30 | −0.67 / −0.38 | Helps nearly everywhere: it is what keeps fees down |
| Risk team | −0.51 / −0.41 | 0.00 / +0.07 | −0.09 / +0.22 | Helps on 1h |
| Stop-loss / take-profit | +0.62 / +0.86 | −0.17 / +0.28 | −0.05 / −0.09 | Hurts on 1h (stopped out by noise); helps in 2022 everywhere (−0.8 on 4h and 1D) |
| Trend analyst | +0.07 / +1.11 | −1.33 / +0.27 | −0.25 / −0.14 | Essential on 4h in development; mixed elsewhere |
| Sentiment (funding, F&G) | −0.18 / −0.35 | −0.21 / +0.36 | — | Helps on 1h; sign flips on 4h |
| News (lexicon) | — | −0.18 / +0.10 | — | Within noise |
| Memory re-weighting | −0.06 / −0.57 | +0.05 / +0.07 | −0.23 / −0.13 | Small, mostly helps |
| Kill switch | 0.00 / +0.01 | 0.00 / 0.00 | 0.00 / 0.00 | Fired only on 1h in 2022 (removing it: +0.26 there); elsewhere drawdowns stayed under 25% |

Most analyst contributions are under ±0.4 Sharpe and change sign between the development and test periods, which is
the size of the bootstrap noise. Only conviction smoothing helps consistently.

## 6. Robustness

* **Fees.** 4h test Sharpe goes from −0.01 (0.05% fee) to −0.49 (0.2% + 10 bp slippage); 1D from +0.21 to +0.07.
  1h is fee-bound: development Sharpe 1.04 at 0.1%, 0.51 at 0.2%.
* **Execution delay.** Filling one candle after the decision instead of at its close: 4h −0.11 → −0.13 (robust),
  **1D +0.18 → −1.04 (fragile)**. The 1D pass depends on same-close execution, which a daily strategy can only
  approximate by trading just before midnight UTC.
* **Rule parameters** (stop 1.5–3 ATR, take-profit 3–6 ATR, thresholds ×0.75–1.25, smoothing ×0.5–2, three risk
  profiles): 4h test Sharpe ranges from −0.44 to +0.24 and 1D from −0.17 to +0.43. The chosen settings are mid-range,
  not cherry-picked, and the sign of the test result is not stable under reasonable changes.

## 7. Limitations

* One test window of 15 months, in a falling market. A strategy that holds less BTC wins by construction; a rising
  test market would be a harder exam.
* The team selection (4h `hybrid_lexicon` over `core`) did not carry over to the test window (−0.11 vs +0.33):
  with these sample sizes, development Sharpe ranks are noisy. Fewer choices would be more honest than more tuning.
* News is from one tech-leaning source, with no predictive value on its own ([docs/12](12_news_sentiment.md)).
* No shorting, no leverage, no funding cost; fills at the candle close with fixed fees.

## 8. What would make the agent better (future work)

1. Make the trend filter the backbone and let the other analysts only scale the position *inside* an uptrend.
2. Evaluate on more out-of-sample history with walk-forward re-tuning (e.g. yearly), not one split.
3. Execute 1D decisions at a fixed time before the close, and model that delay in the backtest.

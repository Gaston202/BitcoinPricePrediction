# Hybrid Multi-Agent Trader: LLM-read news, rule-based decisions (CRISP-DM Phase 5–6)

**Paper trading only.** A simulated USD/BTC wallet; no order is ever sent to an exchange.

Code: `src/agents/` (engine), `src/data/update.py` + `src/data/binance.py` (always up-to-date data),
`src/data/news.py` + `src/agents/news_sentiment.py` (news and its LLM score), `dashboard/` (admin dashboard).
Tests: `tests/test_agents.py`, `tests/test_news.py`, `tests/test_portfolio.py`, `tests/test_update.py`,
`tests/test_dashboard.py`. **Evaluation of this agent: [`docs/13_agent_evaluation.md`](13_agent_evaluation.md).**

> **The hybrid principle: the LLM reads, the rules decide.** One open-weights LLM turns each day's Bitcoin
> headlines into a sentiment score (cached, so backtests never call it). Every buy and sell decision and every
> risk limit stays an explicit, testable rule. See [`docs/12_news_sentiment.md`](12_news_sentiment.md).

## 1. Why this design

The project brief asks for an **AI agent** that decides when to buy and sell with a $1,000 wallet. Our reference
was [TradingAgents](https://github.com/TauricResearch/TradingAgents) (Tauric Research), a multi-agent trading
framework in which LLM "employees" (analysts, researchers, a trader, a risk team and a portfolio manager) debate
each decision. We kept its **organisation** and replaced every LLM call with **explicit rules**:

| | TradingAgents | This project |
|---|---|---|
| Agents | LLM prompts | Deterministic Python rules |
| Reproducible? | No (sampling, model versions) | Yes - same data, same decision |
| Explainable? | Free text | Every argument is a rule with a threshold and a weight |
| Cost / latency | API calls per decision | ~0.6 ms per candle, offline; the LLM runs once per day of news, cached |
| Data | Stocks: fundamentals, news, social | BTC: Binance candles, futures funding, long/short ratio, Fear & Greed, news headlines |

## 2. The pipeline (one candle)

```
Binance candles ──► indicators (src/agents/market_data.py, built on src/features/*)
                      │
   ┌──────────────────┼──────────────────────────────────────────────────────────────┐
   ▼                  ▼               ▼            ▼           ▼        ▼              ▼
 Trend          Momentum        Volatility     Volume/flow  Sentiment  Quant        News ◄── daily headlines
                                                                   (LightGBM)          scored once by an LLM
   └──────────── reports: score −1…+1, confidence, evidence items ─────────────────────┘
                      │
          Bull researcher ◄── debate, N rounds ──► Bear researcher
                      │   (rebuttal rules discount the other side's arguments)
               Research manager ── 5-tier rating: Buy / Overweight / Hold / Underweight / Sell
                      │
                   Trader ── target BTC share + ATR stop-loss / take-profit
                      │
     Aggressive ◄── risk debate, M rounds ──► Conservative, Neutral
                      │
             Portfolio manager ── final decision + hard limits ──► wallet
                      │
                   Memory ── settles decisions 6 candles later; analysts' hit rates re-weight them
```

### Analysts (`src/agents/analysts.py`)

| Analyst | Reads | Example rules |
|---|---|---|
| Trend | EMA20/50, SMA200, MACD | EMA20 > EMA50 → bullish (stronger the wider the gap in ATR units) |
| Momentum | RSI14, stochastic, ROC10 | RSI < 30 oversold → bullish; %K crosses %D above 80 → bearish |
| Volatility | Bollinger %B and width, 20-candle range | breakout above the 20-candle high after a squeeze → bullish |
| Volume / order flow | taker-buy ratio, OBV slope, MFI, volume z-score | > 53% of volume from aggressive buyers → bullish |
| Sentiment | futures funding rate, long/short ratio, Fear & Greed | funding > 0.03 %/8h (crowded longs) → bearish (contrarian) |
| Quant | team's LightGBM `P(next 1h up)` | P > 52% → bullish; abstains on other candle sizes |
| News | yesterday's news score (3-day average), from the LLM or the lexicon | score > +0.15 → bullish, < −0.15 → bearish; confidence = the scorer's |

An analyst with no data (e.g. sentiment before syncing) **abstains** (confidence 0) instead of guessing.

### Debate (`src/agents/researchers.py`)

Round 1: each researcher presents its side's evidence (strength × analyst weight × confidence).
Rounds 2+: each attacks the other side's strongest unrebutted argument. A rebuttal only lands when a rule matches
the market, for example *"oversold in a downtrend is a falling knife"* (×0.5) or *"overbought in a strong uptrend
just means strength"* (×0.5). The research manager turns `(bull − bear) / (bull + bear)` into the 5-tier rating,
using an EMA of the conviction so one noisy candle cannot flip the position.

### Trader, risk team, portfolio manager (`src/agents/execution.py`)

* **Trader:** Buy → 100% BTC, Overweight → 60%, Hold → unchanged, Underweight → at most 25%, Sell → 0%.
  Stop-loss 2 × ATR below entry, take-profit 4 × ATR above.
* **Risk team:** the aggressive analyst sizes up when trend and conviction agree; the conservative analyst caps the
  BTC share at 50% in turbulent markets, during drawdowns, on low conviction or below the 200-candle average; the
  neutral analyst takes the midpoint.
* **Portfolio manager:** adopts the chosen risk profile's view, then applies the hard limits: stop-loss / take-profit
  exits (also checked intrabar in live mode), a 25% drawdown kill switch with a pause, a cooldown after each trade and
  a minimum rebalance size so fees do not eat the wallet.
  * *Fixed on `hybrid-version`:* after the kill-switch pause the drawdown is measured from the equity at restart
    (before, the old peak made the first re-entry trigger the kill switch again: 4 kills instead of 1 in the
    regression test). After a take-profit, adding exposure needs a fresh signal (before, the half just banked was
    bought straight back on the next candle, paying the fee twice).

### Memory (`src/agents/memory.py`)

Like TradingAgents' decision log and settlement: every decision is scored 6 candles later. Each analyst's hit rate
on its directional calls (last 200, Laplace-smoothed) multiplies its weight between ×0.5 and ×1.5. The dashboard
shows the hit rates and the "lessons" the memory derives from them.

## 3. Always up-to-date data (`src/data/update.py`)

* The frozen snapshot in `data/raw/` (to 2026-09-30) is never modified, so the notebooks stay reproducible.
* Each sync downloads only the hourly candles that **closed** after the last stored one (Binance `/api/v3/klines`,
  falling back to `data-api.binance.vision`), appends them to `data/live/`, and rebuilds
  `data/interim/btcusdt_{1h,4h,1D}_latest.parquet` with the project's own gap policy (`src.data.clean.fill_gaps`).
* It also syncs the futures **funding-rate history** (since 2019) and the **Fear & Greed** index.
* The dashboard syncs on start-up and every 5 minutes; `python -m src.data.update --watch` does the same from a terminal.

## 4. Settings were chosen without looking at the test window

Rule thresholds are textbook values. The only tuned settings, the conviction smoothing, cooldown and minimum rebalance
for each candle size (`INTERVAL_PRESETS` in `src/agents/config.py`), are picked by Sharpe on the **development period
2023-01 → 2025-06**: `python scripts/run_agent_evaluation.py tune` reruns the grid search and
`reports/results/agent/preset_tuning.csv` holds every grid point. A unit test checks that `INTERVAL_PRESETS` equals the
winners in that file. The analyst team is chosen the same way (`select` step). The test window
(2025-07-09 → 2026-09-30) and the 2022 bear market are only used to report results.

## 5. Results

The full evaluation (every window, seven baselines, bootstrap tests, ablation, sensitivity) is in
[`docs/13_agent_evaluation.md`](13_agent_evaluation.md) and `notebooks/08_agent_evaluation.ipynb`.

For the record, the table first reported on the `marouen` branch (before the two bug fixes, compared with
buy-and-hold only) was: 4h test window −5.1% vs −22.9% buy-and-hold, 1D −3.8% vs −24.9%, 1h −28.9% vs −23.3%.
The audit of that branch found that a one-line 200-candle moving-average filter beat it in every period, which is
why the agent is now always compared with simple strategies, not only with buy-and-hold.

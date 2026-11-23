# Rule-Based Multi-Agent Trader (CRISP-DM Phase 5–6: the agent and the business criterion)

**Paper trading only.** A simulated USD/BTC wallet; no order is ever sent to an exchange.

Code: `src/agents/` (engine), `src/data/update.py` + `src/data/binance.py` (always up-to-date data),
`dashboard/` (admin dashboard). Tests: `tests/test_agents.py`, `tests/test_update.py`, `tests/test_dashboard.py`.

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
| Cost / latency | API calls per decision | ~0.6 ms per candle, offline |
| Data | Stocks: fundamentals, news, social | BTC: Binance candles, futures funding, long/short ratio, Fear & Greed |

## 2. The pipeline (one candle)

```
Binance candles ──► indicators (src/agents/market_data.py, built on src/features/*)
                      │
   ┌──────────────────┼────────────────────────────────────────────────┐
   ▼                  ▼               ▼            ▼           ▼        ▼
 Trend          Momentum        Volatility     Volume/flow  Sentiment  Quant (team's LightGBM)
   └──────────── reports: score −1…+1, confidence, evidence items ─────┘
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
for each candle size (`INTERVAL_PRESETS` in `src/agents/config.py`), were picked by Sharpe on **2023-01 → 2025-06**.
The team's unseen test window (2025-07-09 → 2026-09-30) was only used to report the results below.

## 5. Results ($1,000 start, 0.1% fee per trade; trend, momentum, volatility and volume analysts; quant added on 1h)

| Candles | Period | Agents | Buy-and-hold | Agent max DD | Trades | Fees |
|---|---|---|---|---|---|---|
| 4h | Test window (unseen, bear) | **−5.1%** | −22.9% | −23.5% | 103 | $54 |
| 1D | Test window (unseen, bear) | **−3.8%** | −24.9% | −18.7% | 19 | $10 |
| 1h | Test window (unseen, bear) | −28.9% | −23.3% | −36.1% | 420 | $222 |
| 4h | Tuning period 2023-01 → 2025-06 (bull) | +165.5% | +547.4% | - | 251 | $243 |

What this means:

* **The edge is defensive.** In the falling test market the team cut exposure early and lost 4-5× less than holding,
  with a much smaller drawdown. In the strong bull market it made money but far less than simply holding.
* **Fast candles lose to fees.** On 1h candles the signals flip too often; every setting we tried lost money on the
  tuning period, and it lost on the test window too. 4h is the recommended default.
* **The team's ML model helps a little as one voice among six.** As the quant analyst on 1h candles, its directional
  calls were right 61% of the time over the 6-candle settlement horizon, but that was not enough to overcome the fees.
* Compared with the earlier single-model agent (−20.2% on the same window), the rule-based team cuts the loss by
  three quarters on 4h and 1D candles.

Nothing here is evidence of a money-making strategy. It is evidence that a transparent, rule-based risk process
protects capital better than buying and holding in a downturn, at the cost of upside in rallies.

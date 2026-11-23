# Trading Agent & Admin Dashboard

**Still a research tool, not Deployment.** Everything below is *paper trading*: a simulated
USD/BTC wallet. No order is ever sent to an exchange.

Code: `src/agent/` (model service, policy, market data, sessions), `dashboard/` (FastAPI + Chart.js).
Tests: `tests/test_agent.py`, `tests/test_dashboard.py`.

## How the agent decides

1. **Model** - the project's random forest (1D horizon, θ = 1%), trained on `train`, decision weights
   calibrated on `val` (`tune_class_weights`). An hourly twin (θ = 0.25%) exists for live mode, because a
   daily model would only decide once a day.
2. **Forecast** - expected next-candle log return = Σ P(class) × mean return of that class in `train`.
   The dashboard plots `price × e^expected` one candle ahead, next to the real price.
3. **Policy** (long-only spot wallet, all-in / all-out via `PortfolioSimulator`):
   * BUY when in cash and the model predicts UP (optionally only if the expected move covers the 0.2%
     round-trip fee - "fee-aware", and only above a minimum confidence);
   * SELL when holding and the model predicts DOWN, or the stop-loss / take-profit is hit;
   * HOLD otherwise. Every decision is logged with its reason.

## Default risk settings were chosen on validation, not test

A small grid (min. confidence 0/45/55/65%, stop-loss off/5/10%, fee-aware on/off) was scored by Sharpe on
the **validation** window. Best: 1D → no confidence filter, 5% stop-loss, fee-aware off; 1h → 45% confidence,
5% stop-loss, fee-aware off. These are the dashboard defaults.

## Results on the unseen test window (2025-07 → 2026-09, bear market)

| | AI agent (defaults) | Buy-and-hold |
|---|---|---|
| Final wallet from $1,000 | $798 (−20.2%) | $751 (−24.9%) |
| Trades / win rate | 74 / 43% | 1 / - |
| Max drawdown | −49.3% | −53.0% |
| Fees paid | $57 | ≈ $2 (one entry, one exit) |

The agent **lost less than buy-and-hold, it did not make money**, and its Sharpe is essentially the
same as buy-and-hold's (−0.35 vs −0.33).

## Findings worth knowing

* **Validation settings don't transfer.** The validation window was a bull market (buy-and-hold +76%),
  the test window a bear market. The policy that scored best on validation is just "the least-bad"
  on test. Same lesson as `docs/07`: single-regime tuning is not evidence of an edge.
* **Fees dominate at hourly frequency.** The hourly model's expected moves are ~±0.8% at most and
  usually far smaller. With *fee-aware* on, it makes **0 trades** over the whole test window. With it off
  and no confidence filter, it makes 1,008 trades and loses 62% - $603 of that is fees.
* **The forecast hugs the price.** Expected moves are a fraction of a percent, so the forecast line is
  almost the price shifted one candle. That is an honest picture of how little directional signal the
  price/volume features carry (`03_eda` item 10, `docs/07`).

## Live mode

Polls Binance (`api.binance.com`, falling back to `data-api.binance.vision`) every 3 s for the price
and re-runs the hourly model on the still-forming candle every 30 s - 1 h (volume of the open candle is
extrapolated to a full candle so it is not read as "abnormally quiet"). It needs internet access; after
5 consecutive failures the session stops with an error shown in the dashboard.

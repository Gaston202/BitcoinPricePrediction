# How to Run the Project

Bitcoin price prediction plus a **hybrid multi-agent trader** (modelled on TradingAgents: an open-weights LLM reads
the daily news, explicit rules make every trade) that manages a $1,000 paper wallet on always up-to-date Binance data.
The LLM is optional: without an API key the news analyst uses a word-list scorer, and everything else works offline.
All commands are run from the project root folder `BitcoinPricePrediction`.

> Commands are shown for **Windows PowerShell**. Git Bash / macOS / Linux alternatives are noted under each step.

---

## 1. Requirements

- Python **3.11 or newer** (tested with 3.12 and 3.14)
- Git
- Internet access for installing packages, syncing new Binance data and the **Live** mode
  (everything else works offline on the stored data)

## 2. Get the code

```powershell
git clone https://github.com/Gaston202/BitcoinPricePrediction.git
cd BitcoinPricePrediction
git checkout hybrid-version
```

Already have the folder? Open it in a terminal and update it:

```powershell
git fetch origin
git checkout hybrid-version
git pull
```

## 3. Create and activate a virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- If PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
- Git Bash: `source .venv/Scripts/activate` · macOS/Linux: `source .venv/bin/activate`
- **Tip:** OneDrive can corrupt a `.venv` while syncing. If pip ever breaks, delete `.venv` and recreate it,
  or keep the environment outside OneDrive (`py -3 -m venv C:\venvs\btc`).

## 4. Install the dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import pandas, sklearn, lightgbm, fastapi, uvicorn; print('Ready')"
```

## 5. Prepare the data (first time only)

The raw Binance snapshot (2017-08 → 2026-09-30) is already in `data/raw/` - **no download needed** for it.
The notebooks build the cleaned and model-ready files:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_quality.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_feature_engineering.ipynb
```

(Or open them with `jupyter notebook notebooks` and use **Run → Run All Cells**.)

## 6. Bring the data up to date with Binance

```powershell
python -m src.data.update                  # download every candle that closed since the last sync
python -m src.data.update --watch          # keep syncing every 5 minutes (Ctrl+C to stop)
python -m src.data.update --watch --every 60
python -m src.data.update --no-sentiment   # candles only (skip funding rate / Fear & Greed)
```

Each sync only downloads what is missing: new hourly candles go to `data/live/` and the combined, cleaned history to
`data/interim/btcusdt_{1h,4h,1D}_latest.parquet`. The frozen snapshot is never modified, so the notebooks stay
reproducible. **The dashboard does this automatically**, so this step is optional.

## 7. Run the admin dashboard and the agent team

```powershell
python -m dashboard
```

Your browser opens at **http://127.0.0.1:8000** (stop the server with `Ctrl+C`).

| Option | Command |
|---|---|
| Different port | `python -m dashboard --port 8080` |
| Don't open the browser | `python -m dashboard --no-browser` |
| Sync Binance data more often | `python -m dashboard --sync-every 60` |
| Offline (no background sync) | `python -m dashboard --no-sync` |
| Reachable from your network | `python -m dashboard --host 0.0.0.0` |

In the dashboard:

1. **Header** - shows how fresh the data is; **Sync now** downloads the newest candles immediately.
2. **Overview** - price history (new data highlighted), agent backtests vs. buy-and-hold, what the agent team thinks
   right now, Fear & Greed, volatility, and the team's ML model results.
3. **Agent Trading** - choose a mode and press **▶ Start agents**:
   - **Replay** - the agents trade stored candles (1h / 4h / 1D) over a period: the new data since the snapshot,
     the last 30-365 days, or the team's test window.
   - **Live** - the agents trade the real Bitcoin price right now on 1m / 5m / 15m / 1h / 4h candles.

   You can change the risk profile, the debate and risk rounds, and which analysts take part. You see the price with
   every buy/sell and the active stop-loss, the debate conviction, the wallet vs. buy-and-hold, the **full pipeline**
   of each decision (analysts → bull/bear debate → trader → risk team → portfolio manager), the memory's hit rates,
   and a decision log (click any row to open that decision).
4. **Report figures** - every chart exported by the notebooks.

Models for the quant analyst are trained on first use (a few seconds) and cached in `models/`.
This is **paper trading only** - no real money, no real orders.

## 8. Use the agents from Python

```python
from src.agents.config import config_for
from src.agents.graph import TradingAgentsGraph, run_backtest
from src.agents.market_data import compute_indicators
from src.data.update import load_latest

ind = compute_indicators(load_latest("4h").dropna(subset=["close"]))
graph, steps = run_backtest(ind.loc["2026-01-01":], config_for("4h", risk_profile="conservative"), "4h")
print(graph.summary()["pnl_pct"], graph.last_state.plan.rationale)
```

## 9. Run the tests

```powershell
python -m pytest            # whole suite
python -m pytest -q tests/test_agents.py tests/test_update.py tests/test_dashboard.py   # agents, data sync, dashboard
python -m pytest -q tests/test_news.py tests/test_portfolio.py                          # news, LLM scorer, evaluation
```

The tests never call Binance, Hacker News or OpenRouter (the network is faked), so they pass offline. GitHub runs
them on every push (`.github/workflows/tests.yml`). On macOS/Linux, `make test` does the same.

## 10. Run all the notebooks (full analysis)

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_collection.ipynb` | Data source, metadata, data dictionary |
| 02 | `02_data_quality.ipynb` | Gaps, cleaning policy, outliers, resampling |
| 03 | `03_eda.ipynb` | Returns, stationarity, autocorrelation, correlations, baselines |
| 04 | `04_feature_engineering.ipynb` | Features, labels, chronological split, selection, PCA |
| 05 | `05_modeling.ipynb` | Walk-forward protocol, targets, 9 models, calibration |
| 06 | `06_evaluation.ipynb` | Baselines, success criteria, calibration, stability |
| 07 | `07_modeling_iteration.ipynb` | Four iterations and the final model |
| 08 | `08_agent_evaluation.ipynb` | The agent vs. seven baselines: business criterion, bootstrap, ablation, robustness |
| 09 | `09_news_sentiment.ipynb` | News coverage, LLM vs lexicon scores, predictive tests, look-ahead check |

Notebook 08 reads `reports/results/agent/`, produced by step 10b.

```powershell
Get-ChildItem notebooks\0*.ipynb | ForEach-Object { jupyter nbconvert --to notebook --execute --inplace $_.FullName }
```

Git Bash: `for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done`

## 10b. News sentiment and the agent evaluation

The frozen inputs are committed (`data/external/`), so this step only reruns the experiments:

```powershell
python scripts/run_agent_evaluation.py all    # team selection, evaluation, ablation, sensitivity (~6 min)
python scripts/run_agent_evaluation.py tune   # optional: the 48-point preset grid (~1.5 h)
```

macOS/Linux: `make agent` / `make tune`. Results: `reports/results/agent/*.csv`.

**Scoring the news with the LLM (optional).** The LLM is `google/gemma-4-31b-it:free` on OpenRouter (free, open
weights).

1. Create a free account at openrouter.ai and an API key.
2. Copy `.env.example` to `.env` and paste the key: `OPENROUTER_API_KEY=sk-or-...` (`.env` is never committed).
3. Score every day once (the result is cached, so backtests never call the API):

```powershell
python -m src.agents.news_sentiment --scorer llm            # resumable: rerun after a daily-limit stop
python -m src.agents.news_sentiment --scorer llm --model <other-openrouter-model>   # optional
python scripts/run_agent_evaluation.py all                 # now also evaluates the hybrid_llm team
```

Re-downloading the headlines (optional, ~1 h): `python -m src.data.news --source hn` then
`python -m src.data.news --source hn --snapshot`, and `python -m src.agents.news_sentiment --scorer lexicon`.

## 11. Optional: rebuild the frozen snapshot

```powershell
python -m src.data.collect               # rebuild the snapshot (skips if it exists)
python -m src.data.collect --force       # overwrite it
$env:BINANCE_USE_DATA_API = "1"; python -m src.data.collect   # if Binance answers HTTP 451
```

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Install Python with **Add to PATH**, or use `py -3` |
| pip crashes with `ImportError ... pip._internal` | The `.venv` is damaged (often by OneDrive): delete it and redo steps 3-4 |
| `No module named fastapi` / `lightgbm` | Activate `.venv`, then `python -m pip install -r requirements.txt` |
| Header says "Sync failed" | No internet / firewall / VPN blocks Binance; the agents keep working on stored data |
| Sentiment analyst always abstains | Its data comes from the sync - press **Sync now** with internet access |
| `Address already in use` | Another program uses port 8000: `python -m dashboard --port 8080` |
| A notebook can't find `src` | Start Jupyter from the project root, not inside `notebooks/` |
| macOS: `Library not loaded: libomp.dylib` (LightGBM) | `brew install libomp`, or `export DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.11/site-packages/sklearn/.dylibs` |
| `OPENROUTER_API_KEY is not set` | Only needed for `--scorer llm`; see step 10b |
| LLM scoring stops with "daily limit" | The free tier's daily quota: rerun the next day, it resumes where it stopped |

Agent design: [`docs/11_rule_based_agents.md`](docs/11_rule_based_agents.md) · news sentiment: [`docs/12_news_sentiment.md`](docs/12_news_sentiment.md) · agent evaluation: [`docs/13_agent_evaluation.md`](docs/13_agent_evaluation.md).

---

# Tools Used

## Language & environment

| Tool | Used for |
|---|---|
| **Python 3.11+** | The whole project: data pipeline, models, agents, web server |
| **venv** + **pip** + `requirements.txt` | Isolated environment with pinned versions |

## Data sources

| Tool | Used for |
|---|---|
| **Binance Spot REST API** (`/api/v3/klines`, `/api/v3/ticker/price`) | Hourly BTC/USDT candles 2017 → now, live candles (1m-4h) and the live price |
| **data-api.binance.vision** | Binance market-data mirror, fallback when the main API is blocked (HTTP 451) |
| **Binance USDT-M Futures API** (`/fapi/v1/fundingRate`, `globalLongShortAccountRatio`) | Funding-rate history and crowd long/short ratio for the sentiment analyst |
| **Alternative.me Fear & Greed API** | Daily crypto Fear & Greed index (sentiment analyst) |
| **Hacker News search API** (Algolia) | Daily Bitcoin headlines 2022 → now for the news analyst (`src/data/news.py`) |
| **GDELT DOC 2.0 API** | Alternative headline source (broader, but heavily rate-limited) |
| **OpenRouter** (`google/gemma-4-31b-it:free`) | Open-weights LLM that scores each day's headlines (optional, cached) |
| **python-binance** | Client used to build the original frozen snapshot (`src/data/collect.py`) |
| **requests** | HTTP calls for the incremental updater and live trading (`src/data/binance.py`) |

## Data handling & storage

| Tool | Used for |
|---|---|
| **pandas**, **NumPy** | Cleaning, gap filling, resampling 1h → 4h → 1D, indicators, features |
| **Apache Parquet** via **PyArrow** | Typed storage of raw / live / interim / processed data |
| **JSON** | Snapshot metadata, sync status, model choice, decision memory log |

## Statistics & machine learning

| Tool | Used for |
|---|---|
| **SciPy**, **statsmodels** | Distribution and stationarity tests (ADF/KPSS), autocorrelation (ACF/PACF, Ljung-Box) |
| **scikit-learn** | Baseline models, feature selection, PCA, Platt calibration, metrics |
| **LightGBM**, **XGBoost** | Gradient-boosted models; the selected 1h LightGBM is the quant analyst |
| **Optuna** | Hyperparameter search in the walk-forward experiments |
| **Block bootstrap** (own code, `src/evaluation/portfolio.py`) | Is the agent's Sharpe really above a baseline's? |
| **Granger causality, Spearman, event study** (statsmodels, SciPy) | Does news predict the next day's return? |
| **joblib** | Caching trained models (`models/`) |

## Agents

| Tool | Used for |
|---|---|
| **TradingAgents** (Tauric Research, reference design) | Agent roles, debate rounds, 5-tier rating, risk team, memory - re-implemented as rules |
| **LLM for text only** | Headlines → daily sentiment score; a word-list (lexicon) scorer is the offline fallback and the control |
| **Technical analysis rules** (EMA, SMA, MACD, RSI, stochastic, Bollinger, ATR, OBV, MFI) | The analysts' logic (`src/agents/analysts.py`) |
| **Python dataclasses / threading** | Agent messages (`src/agents/state.py`), background replay / live sessions and data sync |

## Visualization, notebooks & dashboard

| Tool | Used for |
|---|---|
| **Matplotlib** | Report figures in `reports/figures/` |
| **Jupyter**, **ipykernel**, **nbconvert**, **nbformat**, **Jinja2** | Notebooks, headless runs, styled tables |
| **FastAPI**, **Uvicorn**, **Pydantic** | Dashboard API, web server, request validation |
| **HTML / CSS / JavaScript** + **Chart.js** 4.4 (+ date-fns adapter) | Dashboard front end and real-time charts (vendored, works offline) |

## Testing & collaboration

| Tool | Used for |
|---|---|
| **pytest**, **httpx** | Automated tests, including the dashboard API with a faked exchange |
| **Git** + **GitHub** | Version control, team branches (`marouen`, `master`, ...) and pull requests |
| **CRISP-DM** | Project methodology |
| **Claude (AI assistant)** | Code review and bug fixes, the rule-based agent team, the data updater and the dashboard |

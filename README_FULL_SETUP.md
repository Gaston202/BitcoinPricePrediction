# Bitcoin Price Prediction and Portfolio Simulation

This guide explains how to install, run, test, and reproduce the complete
project from scratch, including the offline portfolio simulation.

The project uses historical BTCUSDT data already stored in the repository. The
portfolio simulation does **not** fetch market data from Binance or any other
external API.

## 1. Requirements

- Windows 10 or later
- Python 3.11 or newer
- Git, optional
- At least 2 GB of free disk space
- Internet access only for installing Python packages

The repository currently includes a `.venv` directory, but the commands below
also show how to create a fresh environment.

## 2. Open the project folder

Open PowerShell and run:

```powershell
cd "C:\Users\Mega Pc\OneDrive - South Mediterranean University\Desktop\DataScience_Lab\BitcoinPricePrediction"
```

Confirm that the project contains these directories:

```text
data\
docs\
notebooks\
reports\
src\
tests\
```

## 3. Create and activate a virtual environment

To create a new environment:

```powershell
py -3 -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, allow locally-created
scripts for your user account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of the terminal prompt.

For Git Bash, use:

```bash
source .venv/Scripts/activate
```

## 4. Install the project dependencies

With the virtual environment activated:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python --version
python -c "import numpy, pandas, sklearn, pyarrow, matplotlib; print('Dependencies are ready')"
```

## 5. Use the included local data

The raw BTCUSDT 1-hour snapshot is already included:

```text
data\raw\btcusdt_1h_20170817_20260930.parquet
data\raw\btcusdt_1h_20170817_20260930_metadata.json
```

No download is required.

The notebooks generate these derived files:

```text
data\interim\
  btcusdt_1h_clean.parquet
  btcusdt_4h_clean.parquet
  btcusdt_1D_clean.parquet

data\processed\
  btcusdt_1D_train.parquet
  btcusdt_1D_val.parquet
  btcusdt_1D_test.parquet
```

The `data\interim\` and `data\processed\` directories are generated outputs and
may be ignored by Git. They can be regenerated at any time by rerunning the
notebooks.

## 6. Run the automated tests

Run all tests from the project root:

```powershell
python -m pytest
```

The suite covers:

- Data cleaning and gap handling
- Resampling and return calculations
- Feature engineering
- Model helpers and baselines
- Trading strategy logic
- Portfolio simulation and transaction fees

## 7. Run the notebooks in the correct order

Start Jupyter:

```powershell
jupyter notebook notebooks
```

Open each notebook and choose **Run → Run All Cells** in this order:

### 7.1 Data collection and metadata

Open:

```text
notebooks\01_data_collection.ipynb
```

This notebook documents the local raw snapshot and its schema. It does not
need to download data for the standard workflow.

### 7.2 Data quality and cleaning

Open:

```text
notebooks\02_data_quality.ipynb
```

This creates the cleaned hourly, four-hour, and daily files under
`data\interim\`.

### 7.3 Exploratory data analysis

Open:

```text
notebooks\03_eda.ipynb
```

This analyzes returns, stationarity, autocorrelation, distributions,
correlations, class balance, and baseline performance.

It also generates figures under:

```text
reports\figures\
```

### 7.4 Feature engineering and chronological splitting

Open:

```text
notebooks\04_feature_engineering.ipynb
```

This creates model features and chronological train, validation, and test
datasets under `data\processed\`.

The split is chronological rather than random to reduce look-ahead leakage.

### 7.5 Model training and first evaluation

Open:

```text
notebooks\05_modeling.ipynb
```

This trains the candidate classifiers, selects a model using validation data,
evaluates the test window, and compares the strategy with buy-and-hold.

### 7.6 Formal evaluation

Open:

```text
notebooks\06_evaluation.ipynb
```

This checks the model against the project’s predefined data-mining and
business success criteria.

### 7.7 Modeling iteration across market regimes

Open:

```text
notebooks\07_modeling_iteration.ipynb
```

This evaluates calibrated predictions across multiple bear, bull, and
range-bound market windows.

The model should not be treated as deployment-ready solely because it performs
well in one historical window.

## 8. Run the offline portfolio simulation

After notebooks 02 and 04 have generated the local files, open:

```text
notebooks\simulation_demo.ipynb
```

Run all cells. The demo:

1. Loads `data\interim\btcusdt_1D_clean.parquet`.
2. Loads local train and test features from `data\processed\`.
3. Trains the existing random-forest model using local training data.
4. Generates predictions for the local test split.
5. Converts model labels into portfolio signals.
6. Joins the signals with local daily close prices.
7. Runs `PortfolioSimulator`.
8. Calculates performance metrics.
9. Plots the strategy equity curve against fee-adjusted buy-and-hold.

The simulator implementation is:

```text
src\simulation.py
```

It starts with:

```text
Cash:        $1,000
BTC holding: 0.0 BTC
Trading fee: 0.001 per side
```

### Signal meanings

The simulator accepts:

| Signal | Meaning |
|---|---|
| `1` or `"BUY"` | Invest all available cash into BTC |
| `-1` or `"SELL"` | Sell all BTC holdings for cash |
| `0` | Sell all BTC holdings for cash |
| `"HOLD"` | Keep the current position unchanged |

The model’s `0` class means **flat**, so the demo deliberately converts model
prediction `0` to `"HOLD"` rather than passing it directly as a sell signal.
This preserves the model’s intended behavior.

## 9. Run the simulator from a Python script

The simulator can also be used directly:

```python
import pandas as pd

from src.simulation import PortfolioSimulator

prices = pd.read_parquet("data/interim/btcusdt_1D_clean.parquet")
signals = pd.Series("HOLD", index=prices.index)

simulator = PortfolioSimulator(initial_cash=1000.0, fee=0.001)
ledger = simulator.run(prices[["close"]], signals)
metrics = simulator.performance_metrics(prices["close"])

print(ledger.tail())
print(metrics)
```

The ledger contains:

```text
timestamp
price
signal
cash
btc_holding
total_portfolio_value
```

## 10. Expected simulation output

Using the repository’s current local test window, the simulation produced
approximately:

```text
Strategy return:        -15.31%
Buy-and-hold return:    -24.86%
Strategy advantage:       9.55 percentage points
Strategy max drawdown:   -43.05%
Benchmark max drawdown:  -52.97%
```

These are historical backtest results, not a guarantee of future performance.
The model’s advantage in this window is influenced by the market regime and
should not be interpreted as proof of a robust trading edge.

## 11. Optional: recollect the raw data

The normal workflow should use the committed local snapshot. If you explicitly
want to rebuild it, run:

```powershell
python -m src.data.collect
```

To fetch up to the latest closed candle:

```powershell
python -m src.data.collect --end now
```

This optional command uses an external Binance endpoint and is not needed for
the offline portfolio simulation.

If Binance returns HTTP 451 because of regional restrictions, use:

```powershell
$env:BINANCE_USE_DATA_API = "1"
python -m src.data.collect
```

Do not run this step if the goal is to reproduce results strictly from the
repository’s frozen data.

## 12. Troubleshooting

### `python` is not recognized

Install Python and enable **Add Python to PATH**, or use the Python launcher:

```powershell
py -3 --version
```

### `pytest` is not recognized

Make sure the virtual environment is activated, then run:

```powershell
python -m pytest
```

### PowerShell activation is blocked

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

### A notebook cannot find `src`

Start Jupyter from the project root, not from inside `notebooks`:

```powershell
cd "C:\Users\Mega Pc\OneDrive - South Mediterranean University\Desktop\DataScience_Lab\BitcoinPricePrediction"
jupyter notebook notebooks
```

### A processed file is missing

Run the notebooks in order, especially:

```text
02_data_quality.ipynb
04_feature_engineering.ipynb
```

### Jupyter shows running kernels after closing notebooks

Open the **Running** tab and select **Shut Down All**, or stop Jupyter with
`Ctrl+C` in the terminal where it was started.

## 13. Stop using the virtual environment

When finished:

```powershell
deactivate
```


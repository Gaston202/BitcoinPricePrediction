# Beginner's Guide — How the Data Was Prepared, and Why

This guide explains the first part of the project (business understanding, notebooks `01` to `04`) for
someone new to data science. For every step it answers three questions:

1. **What did we do?**
2. **Why did we do it that way?** (and what would go wrong otherwise)
3. **What did we find?**

The technical write-ups with all the numbers are in `docs/01` to `docs/04`. This file is the
"explain it to me like I'm new" companion. Words in **bold** the first time they appear are defined in the
[glossary](#glossary) at the end.

---

## 0 · The big picture

### What is the project trying to do?

We want to know whether a computer can predict **whether Bitcoin's price will go up, down, or stay roughly
flat tomorrow**, using only its past prices and trading volumes. If it can, a trading strategy built on
those predictions might do better than simply buying Bitcoin and holding it.

### How is the work organised? CRISP-DM

We follow **CRISP-DM**, the most widely used process for data-mining projects. Think of it as a recipe
with six steps, where you often loop back:

```
 1. Business Understanding   → What problem are we solving? How will we know we succeeded?
 2. Data Understanding       → What data do we have? Is it clean? What does it look like?
 3. Data Preparation         → Turn raw data into something a model can learn from
 4. Modeling                 → Train models
 5. Evaluation               → Did we meet the success criteria from step 1?
 6. Deployment               → Use it for real (not reached in this project)
```

This guide covers steps 1 to 3:

| Step | Where it lives |
|---|---|
| 1. Business Understanding | `docs/01_business_understanding.md` |
| 2. Data Understanding | `notebooks/01_data_collection.ipynb`, `02_data_quality.ipynb`, `03_eda.ipynb` |
| 3. Data Preparation | `notebooks/04_feature_engineering.ipynb` |

**Why use a process at all?** Without one it's tempting to jump straight to "train a model and look at the
accuracy". That is how people fool themselves: they never define in advance what "good" means, so any
number can be spun as a success. CRISP-DM forces the success criteria to be written down **before**
looking at any results.

---

## 1 · Business Understanding (`docs/01`)

### 1.1 The goal

The full system has three parts:

1. A **machine-learning model** predicts the direction of Bitcoin's next price move.
2. An **AI agent** combines those predictions with news and market indicators to decide what to trade.
3. A **backtest** replays history to check how that strategy would have performed.

### 1.2 Decision: success = beating buy-and-hold on the **Sharpe ratio**, after fees

**What:** the strategy must earn a higher Sharpe ratio than "buy Bitcoin and never sell" (**buy-and-hold**),
over the same period, after paying 0.1% in fees on every trade.

**Why the Sharpe ratio and not just profit?** Bitcoin rose about 19× over our data. Almost *any* strategy
that owns Bitcoin "makes money", so profit alone proves nothing. The Sharpe ratio measures **return per
unit of risk**.

> 🧠 **Analogy:** two drivers both arrive in 1 hour. One drove calmly; the other sped and nearly crashed
> five times. Same result, very different risk. Sharpe rewards the calm driver.

**Why "after fees"?** A strategy that trades every day pays fees every day. Ignoring them can make a
money-losing strategy look profitable.

**Why the *same* period?** Buy-and-hold's Sharpe ranges from −1.3 (2022, a crash year) to +3.4 (2017, a
boom year). Comparing a strategy tested in 2022 with buy-and-hold in 2017 would be meaningless.

### 1.3 Decision: predict 3 classes (UP / FLAT / DOWN), not the exact price

**What:** instead of predicting tomorrow's price ("$64,312"), the model predicts a category:

| Class | Meaning |
|---|---|
| UP | price rises by more than θ (theta, a threshold) |
| FLAT | price moves less than θ either way |
| DOWN | price falls by more than θ |

**Why not predict the price?** Tomorrow's price is almost always very close to today's. A model that
just says "same as today" looks extremely accurate on price, yet it is useless for trading. What
matters for trading is the *direction*.

**Why a FLAT class?** Every trade costs a fee. If the expected move is tiny, trading on it loses money
even when the prediction is right. FLAT means "not worth trading".

### 1.4 Decision: the model must beat simple **baselines**

**What:** the model counts as successful only if it beats two "dumb" strategies:

- **Majority baseline:** always predict the most common class.
- **Persistence baseline:** predict that tomorrow repeats the last known move.

It must beat both on two metrics, **macro-F1** and **ROC-AUC** (see glossary).

**Why compare against baselines?** "60% accuracy" sounds good, but if always guessing "FLAT" already
gets 40%, the model is much less impressive. A baseline gives the number context.

**Why expect only a small improvement?** Financial markets are close to unpredictable. If any simple
pattern worked reliably, traders would exploit it until it disappeared. So **a model that is far better
than the baselines is a warning sign** of a bug (usually **data leakage**), not a reason to celebrate.

### 1.5 Risks identified up front

| Risk | What it means | How we guard against it |
|---|---|---|
| **Look-ahead leakage** | The model accidentally "sees the future" during training | Features use only data up to today's close. Data is split by time, never shuffled. |
| **Regime change** | Bull and bear markets behave differently | Always report which kind of market the test period was |
| **Overfitting the backtest** | Tweaking until results look good on the test period | Tune only on validation data. Touch the test set once. |

---

## 2 · Data Collection (`notebooks/01_data_collection.ipynb`)

### 2.1 Decision: Binance, BTCUSDT, hourly candles

**What:** we downloaded every hourly **candle** (open, high, low, close prices plus volume) of the
Bitcoin/Tether pair from Binance's public API, from August 2017 to September 2026. That's 79,836 rows.

**Why Binance?** It is the largest crypto exchange, its API is free, and it needs no account.

**Why BTCUSDT?** It's the most traded Bitcoin pair, so prices reflect real activity, not a few
scattered trades.

**Why hourly when we predict daily?** You can always combine small candles into bigger ones (24 hourly
candles → 1 daily candle), but you can never split a daily candle back into hours. Hourly keeps every
option open (we also built 4-hour candles to compare).

### 2.2 Decision: a frozen, immutable snapshot

**What:** the data was downloaded **once**, saved as a file in `data/raw/`, and never changed.
A metadata file records when and how it was downloaded.

**Why?** If each team member downloaded the data on a different day, everyone would have slightly
different data, and results couldn't be compared or reproduced. A frozen file means everyone, including
whoever grades this, gets exactly the same results. This is called **reproducibility**.

### 2.3 What the data contains

Besides prices and volume, Binance gives the **taker-buy volume**: the share of volume where the buyer
was impatient and bought at the asking price. Above 50% means buyers are more aggressive than sellers
("buy pressure"). Most price datasets don't have this, so it's a bonus.

**First observation:** the price went from a few thousand to over a hundred thousand dollars. Analysis
on raw prices would be dominated by recent years, which is why the EDA uses **returns** (percentage
changes) instead (see section 4.1).

---

## 3 · Data Quality (`notebooks/02_data_quality.ipynb`)

**Why check quality before analysing anything?** "Garbage in, garbage out." If the data has errors, every
conclusion built on it is wrong, however clever the model.

### 3.1 Four checks

| Dimension | Question | Result |
|---|---|---|
| **Completeness** | Is every hour present? | 99.84%: 128 hours missing, in 28 gaps |
| **Uniqueness** | Any duplicate timestamps? | None |
| **Validity** | Are prices positive, and is low ≤ open/close ≤ high? | Always |
| **Consistency** | Anything odd, like hours with no trades? | 4 zero-volume hours (kept; see below) |

### 3.2 Finding: the gaps are exchange maintenance

**How we found them:** Binance simply skips missing hours, so they're invisible in the raw file. We
compared the data against a complete hourly calendar and looked for holes.

**What they are:** most gaps are in 2018–2021, and 13 of the 28 start at **exactly 02:00 UTC**, a
quiet time of night. That pattern points to scheduled maintenance, not random errors.

**Why does this matter?** The exchange going offline has nothing to do with whether the price was about
to rise or fall. Missing data of this kind is called **MCAR** (Missing Completely At Random), and it's
the safe kind: removing or filling it doesn't bias the results.

### 3.3 Decision: the gap policy (fill short gaps, leave long ones empty)

**What:**
- Gaps of **3 hours or less** → filled with the last known price (**forward-fill**).
- Longer gaps (up to 33 hours) → left empty (NaN).

**Why fill short gaps?** A 1–3 hour hole barely changes a daily picture, and filling it keeps the day usable.

**Why *not* fill long gaps?** Filling 33 hours with the same price would invent a perfectly calm day and
night that never happened. The model would learn from fake data.

### 3.4 Decision: keep zero-volume hours

Hours where nothing traded are real ("the market was quiet"), not errors, so they stay. The taker-buy
share is undefined for them (0 ÷ 0), so it's marked as missing for those hours.

### 3.5 Decision: a daily candle with any missing hour is discarded

When 24 hourly candles are combined into a daily one, any day with an unfilled missing hour is set to
NaN entirely. **Why?** A "daily" candle built from 9 hours of data would claim a high and low for the
whole day while missing most of it.

---

## 4 · Exploratory Data Analysis (`notebooks/03_eda.ipynb`)

**What is EDA?** Before building models, you study the data to understand its behaviour. The findings
here drive almost every decision in step 3.

### 4.1 Decision: work with **log returns**, not prices

**What:** the log return is `ln(price today / price yesterday)`, which is roughly the percentage change.

**Why?**
- **Scale-free:** a 1% move means the same in 2017 at $4,000 and in 2026 at $100,000.
- **Additive:** a week's return is the sum of the daily returns, which makes the maths simpler.
- **Stationary** (see 4.4), which is the most important reason.

### 4.2 Finding: the noise is about 40× bigger than the trend

On an average day Bitcoin rises by about **0.09%**, but a typical daily swing (the standard deviation)
is about **3.5%**. The day-to-day randomness completely swamps the upward drift.

> 🧠 **Analogy:** walking slowly uphill while being shoved around in a crowd. Over a year you end up
> higher, but whether your next step goes up or down is almost a coin flip.

This is why predicting direction is hard **by nature**, not because of a bad model.

### 4.3 Finding: extreme moves are far more common than a "normal" bell curve predicts (**fat tails**)

- Moves bigger than 3 standard deviations happen **6–7× more often** than a bell curve says.
- Crashes are sharper than rallies (**negative skew**).

**Decision on the test:** to check normality we used **Jarque-Bera** and **D'Agostino** rather than the
better-known **Shapiro-Wilk**. **Why?** Shapiro-Wilk becomes unreliable above 5,000 data points, and we
have 3,300 to 80,000. The two tests we used are designed for large samples.

**Consequences:**
- Prefer **robust** statistics (median, Spearman correlation) that aren't thrown off by a few extreme days.
- Prefer models that don't assume bell-shaped data (tree-based models).
- Predict a category (up/flat/down) rather than an exact number: one crash day can wreck a model that
  tries to predict exact returns.

### 4.4 Finding: prices are non-stationary, returns are stationary

**What is stationarity?** A series is **stationary** if its basic behaviour (average, spread) stays
the same over time.

- **Price:** not stationary. It wanders and trends: $4,000 in 2017, $100,000 later.
- **Returns:** stationary. Daily percentage moves behave similarly across the years.

**Why does it matter?** A model learns patterns from the past and applies them to the future. If the
data's behaviour keeps changing (prices), patterns learned on 2019 values don't apply in 2026. With
stationary data (returns) they can.

**Decision on the tests:** we used **two** tests, **ADF** and **KPSS**, which have **opposite starting
assumptions**. ADF assumes "not stationary" and tries to disprove it; KPSS assumes "stationary" and tries
to disprove that. When both point the same way, the conclusion is much stronger than either alone. They
agreed: prices are non-stationary, returns are stationary.

**➜ Decision for later: every feature is built from returns and ratios, never raw prices.**

### 4.5 Finding: direction is barely predictable, volatility clearly is

We used **autocorrelation**, which asks whether today's value is related to yesterday's.

- **Direction:** yesterday's return predicts today's with a correlation of only about **−0.05**: a tiny
  tendency to partially reverse. It is *statistically* significant (with tens of thousands of data points,
  even tiny effects are), but it explains only **0.25%** of the variation.
  **Statistically significant ≠ practically useful.**
- **Volatility:** big moves are clearly followed by big moves, and calm by calm (**volatility
  clustering**). This effect is strong.

> 🧠 **Analogy:** weather. You can't predict whether a coin lands heads, but if it's been stormy all
> week, tomorrow is probably stormy too. You can't predict the *direction* of the wind, but you can
> predict *how strong* it will be.

### 4.6 Finding: no single variable predicts direction; several predict move size

We correlated each variable with tomorrow's return:
- With the **direction**, every correlation is within ±0.06, so essentially nothing.
- With the **size** of tomorrow's move, the day's high-low range (~0.35) and volume (~0.17) are clearly
  related.

**Decision on the method:** we computed both **Pearson** (measures straight-line relationships, sensitive
to extreme values) and **Spearman** (works on rankings, robust to extremes). Given the fat tails from 4.3,
Spearman is the more trustworthy of the two here.

### 4.7 Decision: the label is **1-day horizon, θ = 1%**

We compared predicting 4 hours ahead vs. 1 day ahead:

| | 4-hour horizon | 1-day horizon |
|---|---|---|
| Training examples | ~20,000 | ~3,300 |
| Typical move | ~0.50% | ~1.47% |
| Share of a typical move eaten by fees (0.2% round trip) | **~40%** | **~14%** |

**Why 1 day?** 4 hours gives more data, but fees eat 40% of a typical move, so even correct predictions
would lose money. A daily move is about 7× the fee.

**Why θ = 1%?** It splits the days into three roughly equal groups (≈ 30% down / 38% flat / 33% up),
and a 1% move is big enough to be worth trading. **Balanced classes** matter because a model trained on
"90% flat" data learns to just say "flat".

### 4.8 The baselines (the bars to clear)

| Baseline | Score |
|---|---|
| Always predict the most common class | macro-F1 ≈ 0.18 |
| Persistence (repeat the last move) | macro-F1 ≈ 0.36, the real bar |
| Buy-and-hold (for the strategy) | Sharpe ≈ 0.82, but with a worst drop of **−83%** |

**Important caveat:** if the test period happens to be a crash, a strategy that mostly sits in cash will
beat buy-and-hold without being smart at all. That's why the test period's market type must always be
reported.

---

## 5 · Feature Engineering (`notebooks/04_feature_engineering.ipynb`)

**What is a feature?** One piece of information the model gets about each day, for example "how much
did the price move yesterday?" or "how volatile has the last week been?". **Feature engineering** is
designing those inputs. It often matters more than which model you pick.

The notebook has four stages: **create** many candidate features → **split** the data → **select** the
useful ones → **compress** them (PCA) as a comparison.

### 5.1 The golden rule: no peeking at the future

Every feature for day *t* uses only information available at the close of day *t*. The label for day *t*
is about day *t+1*.

**Why is this so important?** If a feature accidentally contains tomorrow's information, the model looks
brilliant in testing and fails completely in real life, because in real life tomorrow hasn't happened
yet. This is **data leakage**, the most common reason for "too good to be true" results.

**How we checked it:** a unit test changes all prices *after* a given day and verifies that no feature
*before* that day changes. If one did, it would be secretly using the future.

### 5.2 Creating 22 candidate features

| Family | Examples | Idea in plain words |
|---|---|---|
| Mathematical transformations | `ret_lag1..3`, `mom_7d/14d/30d`, `vol_7d/14d/30d`, `atr_14` | Yesterday's move, the trend over a week or a month, how jumpy the price has been |
| Aggregations (summaries) | `volume_z`, `volume_ratio` | Is today's volume unusual compared with recent days? |
| Domain knowledge | `dist_ma20/50`, `vol_ratio`, `rsi_14`, `taker_buy_ratio` | Indicators traders use: is the price far above its average? Is volatility rising? Are buyers aggressive? |
| Date / time | `dow_sin`, `dow_cos`, `is_weekend` | Day of the week; weekends are quieter |
| Interactions | `ret1_x_volz`, `mom7_x_volratio` | Combinations: "a big move **on** huge volume" can mean something different from the same move on quiet volume |

**Why so many?** At this stage we don't know which ones are useful, so we cast a wide net and let the
selection step (5.5) decide, using evidence rather than gut feeling.

**Why the strange "sin/cos" for the day of the week?** If days are numbered 0 (Monday) to 6 (Sunday), the
model thinks Sunday and Monday are far apart (6 vs 0), when they're actually neighbours. Placing the
days on a circle, like a clock face, using sine and cosine keeps neighbours next to each other.

**Why divide things by the price** (e.g. `atr_14 / close`)? A $500 daily range is huge at $4,000 and tiny
at $100,000. Dividing by the price makes the feature mean the same thing in every year (the
stationarity lesson from 4.4).

### 5.3 Two features deliberately removed or fixed

- **`log_volume` removed.** Total trading volume grew steadily over the years. A model would learn "high
  volume means X" from 2017–2024 levels that never appear again. We kept only *relative* volume
  ("unusual compared with the last few weeks"), which stays meaningful over time. This was a judgement
  call based on understanding the data, which no statistical test on the training data alone could catch.
- **Lagged returns fixed.** The first version's "lag 5/7/14" features were really overlapping 5/7/14-day
  totals that contained each other (the 14-day total includes the 7-day one). They're now honest 1-day
  lags (`ret_lag1..3`) plus clearly named trend features (`mom_*`).

### 5.4 Decision: split the data by time, **before** choosing features

**What:** the data is split chronologically into three parts:

```
 2017 ─────────────── 2024-06 │ 2024-06 ── 2025-08 │ 2025-08 ── 2026-09
        TRAIN (70%)           │   VALIDATION (15%) │     TEST (15%)
   the model learns here      │  we tune here      │  final exam, used once
```

**Why by time and not randomly?** With a random shuffle the model would train on, say, March 2025 and be
tested on February 2025, so it would effectively know the future. In real life you only ever have the
past to predict the future. The split must imitate that.

**Why three parts?** Train = studying. Validation = practice exams, used for making choices. Test = the
final exam. If you keep peeking at the final exam while studying, your grade no longer measures what you
learned.

**What is the "embargo"?** The last day of training has a label that depends on the *first* day of
validation (tomorrow's price). That one boundary day is removed so no information leaks across.

**Why split *before* selecting features?** If we chose features by looking at all the data, test data
included, the test would have influenced the model's design, a subtle form of leakage. So **every
selection decision below uses the training data only.**

### 5.5 The curse of dimensionality: why not keep all 22?

With 1,954 training days and 22 features, there are about 89 days per feature. More features make the
data "sparser": the model has more ways to find coincidental patterns that don't repeat. This is called
**overfitting**.

> 🧠 **Analogy:** with 10 students and 50 facts about each one (shoe size, birthday, favourite colour…),
> you will *always* find some fact that "predicts" exam grades perfectly, purely by coincidence.

Since our signal is weak (section 4), this risk is real. There are two remedies: **selection** (keep the
best features) and **extraction** (compress them, see 5.7).

### 5.6 Feature selection, step by step (on training data only)

**Step 1, correlation heatmap.** Which features are near-copies of each other? We used **Spearman**
correlation because of the fat tails (4.3). Three clusters appeared: trend features (momentum, distance
from moving average, RSI), volatility features, and volume features.

**Step 2, feature vs. target plots.** For each feature, one box plot per class (up/flat/down). If the
boxes sit at different heights, the feature helps tell the classes apart.
*Finding:* volatility features separate "flat" from "a move happened", but almost nothing separates "up"
from "down". This matches the EDA.

**Step 3, two relevance tests.**
- **ANOVA F-test:** are the feature's averages different between the three classes? It only detects
  simple, straight-line differences.
- **Mutual information (MI):** how much does knowing the feature reduce uncertainty about the class?
  It also catches complex, non-linear relationships.

**The shuffled-label baseline (an important trick):** MI is never exactly zero, even for pure noise. So
we shuffled the labels randomly 20 times and measured how much MI each feature gets **by pure chance**. A
feature only counts as informative if it beats its own chance level.

**Decision:** a feature is dropped as noise only if it fails **both** tests. **Why both?** Our signals are
weak, and a feature with a non-linear signal would fail ANOVA unfairly. Requiring both failures avoids
throwing away useful features.

**Watch out for multiple testing:** testing 22 features at a 5% significance level means about **one
feature will look significant by pure luck**. So a single surprising result (like a day-of-week effect)
is treated as a hypothesis to check later, not a discovery.

**Step 4, the filter pass**, in this order:
1. Remove **near-constant** features (one value in more than 95% of rows). None found.
2. Remove **zero-variance** features. None found.
3. Remove **noise** (fails both tests). 7 removed: `ret_lag2`, `ret_lag3`, `mom_14d`, `rsi_14`,
   `taker_buy_ratio`, `is_weekend`, `mom7_x_volratio`.
4. For each pair of features correlated above 0.85, keep the one with more MI. 3 removed: `mom_30d`,
   `vol_14d`, `volume_ratio`.

**Result: 12 of 22 features kept.** The list, with the reason for every removal, is saved in
`src/features/selected_features.json`, so the decision is transparent and the modeling code reads it from
one single place.

### 5.7 Feature extraction: PCA (as a comparison)

**What is PCA?** Instead of *choosing* features, **PCA** (Principal Component Analysis) *combines* all of
them into a smaller number of new "summary" features (components), ordered by how much of the data's
variation each one captures.

> 🧠 **Analogy:** describing a person with 22 body measurements. Most of them move together, so
> "overall size" plus "body shape" capture most of the information in 2 numbers instead of 22.

**Decision: standardize first.** Each feature is rescaled to the same spread before PCA. **Why?** PCA
chases variation, so without rescaling RSI (values around 50, spread ~10) would dominate daily returns
(spread ~0.03) purely because of its units, not its importance.

**Findings:**
- 10 components capture **92%** of the information in the 22 features.
- The first three components rediscover, on their own, the clusters seen in the heatmap: **trend**
  (29%), **volatility** (17%), and **activity/calendar** (11%: busy weekdays vs quiet weekends). Two
  different methods agreeing on where the redundancy is makes the conclusion more trustworthy.

**Decision: the selected features stay the main input; PCA is only a comparison.** **Why?** A component
like "PC3" can't be explained in trading terms, while "7-day volatility" can. When performance is
similar, interpretability wins.

### 5.8 What notebook 04 produces

| Output | Content |
|---|---|
| `data/processed/btcusdt_1D_{train,val,test}.parquet` | All 22 features + label, per split |
| `data/processed/btcusdt_1D_pca_{train,val,test}.parquet` | The PCA version |
| `src/features/selected_features.json` | The 12 chosen features + why each other one was dropped |

---

## 6 · Questions you might be asked (and short answers)

**"Why not just use deep learning / an LSTM?"**
With about 2,000 training days and a very weak signal, complex models mostly memorise noise. Simple
models with well-designed features are the honest starting point. Published studies with strict methods
reach only about 52–60% direction accuracy, even with deep learning.

**"Why is your accuracy so low compared with papers claiming 80–90%?"**
Such numbers usually come from leakage: shuffled (non-chronological) splits, indicators computed with
future data, or one lucky test period. We deliberately prevented all three.

**"Why didn't you use all the data for training?"**
Because then nothing would be left to test on honestly. The validation and test sets simulate "the future".

**"Why remove features? Doesn't more information help?"**
Not when the data is limited: extra weak features add more chances to learn coincidences (5.5).

**"What is the single most important finding so far?"**
Bitcoin's daily *direction* is nearly unpredictable from its own price and volume, but its *volatility*
(how big the moves will be) is predictable. That shapes everything that follows.

---

## Glossary

| Term | Plain-language meaning |
|---|---|
| **ADF / KPSS** | Two stationarity tests with opposite starting assumptions; agreement between them makes a strong conclusion |
| **ANOVA F-test** | Checks whether a feature's average differs between classes |
| **Autocorrelation** | How much a series is related to its own past values |
| **Backtest** | Replaying a trading strategy on historical data to see how it would have performed |
| **Baseline** | A deliberately simple strategy that a model must beat to be worth anything |
| **Buy-and-hold** | Buy once at the start, sell at the end, do nothing in between |
| **Candle (OHLCV)** | One time period's Open, High, Low, Close prices and Volume |
| **Chronological split** | Splitting data by time: past for training, future for testing |
| **CRISP-DM** | A standard six-step process for data-mining projects |
| **Data leakage** | When a model accidentally gets information it wouldn't have in real life, usually from the future |
| **Embargo** | Removing the boundary rows between splits so labels can't leak across |
| **EDA** | Exploratory Data Analysis: studying the data before modeling |
| **Fat tails** | Extreme events happening much more often than a bell curve predicts |
| **Feature** | One input variable given to the model |
| **Forward-fill** | Filling a missing value with the last known value |
| **Label / target** | What the model tries to predict (here: UP / FLAT / DOWN) |
| **Log return** | `ln(price today / price yesterday)`, roughly the percentage change |
| **Macro-F1** | A score from 0 to 1 that averages performance across all classes equally, so ignoring a class is penalised |
| **MCAR** | Missing Completely At Random: missing data unrelated to the values themselves, so it's safe to handle |
| **Mutual information (MI)** | How much knowing a feature reduces uncertainty about the label; catches non-linear links |
| **Overfitting** | Learning coincidences in the training data that don't repeat on new data |
| **PCA** | Combining many correlated features into a few uncorrelated summary components |
| **Pearson / Spearman** | Correlation measures; Pearson for straight-line links, Spearman uses ranks and resists extreme values |
| **Persistence baseline** | "Tomorrow repeats the last known move" |
| **ROC-AUC** | How well a model's confidence ranks the right answer above wrong ones; 0.5 = random guessing |
| **Sharpe ratio** | Return per unit of risk; higher is better |
| **Standardize** | Rescale a feature to mean 0 and spread 1 so features with different units are comparable |
| **Stationary** | A series whose average and spread stay stable over time |
| **θ (theta)** | The threshold (1%) separating UP/DOWN from FLAT |
| **Validation set** | Data held aside to make choices (tuning); the "practice exam" |
| **Volatility clustering** | Big moves tend to follow big moves, calm follows calm |

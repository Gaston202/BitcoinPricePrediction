# News Sentiment: the LLM half of the hybrid agent

Notebook: `notebooks/09_news_sentiment.ipynb`. Code: `src/data/news.py` (collection), `src/agents/news_sentiment.py`
(scoring), `src/evaluation/news_analysis.py` (statistics), `src/agents/analysts.py::news_analyst` (use in the agent).
Tests: `tests/test_news.py` (no network: every API is faked).

## 1. Why news, and why only here

The agent's other analysts all read prices, volume, funding and Fear & Greed. Those are numbers. Headlines are
text, and turning text into a number is the one job a language model does better than a hand-written rule. So the
hybrid design gives the LLM **only** that job:

```
headlines of day D ──► LLM (once per day, cached) ──► score −1…+1, confidence, one-line reason
                                                           │  usable from D+1 00:00 UTC
                                                           ▼
                        News analyst (a rule: vote if |3-day score| ≥ 0.15) ──► debate ──► rules decide the trade
```

Every trade, every position size and every risk limit is still an explicit rule. That keeps backtests fast,
reproducible and free, and it means the LLM can never place an order.

## 2. Data

| | |
|---|---|
| Source | Hacker News search (Algolia API): stories whose title mentions bitcoin / btc / crypto, ranked by points |
| Period | 2022-01-01 → 2026-09-30 (1,734 days) |
| Volume | 3,642 headlines; 1,395 days (80.4%) have at least one; median 2 per day, 10% of days have 5 or more |
| Frozen copy | `data/external/news_headlines.parquet` (rebuild: `make news`) |

GDELT (thousands of news sites, up to 75 articles a day) is also implemented, but it blocked this client for hours
after a burst of requests, so it could not deliver a full history. A snapshot uses **one** source for the whole
period: mixing sources would change the outlet mix halfway through and look like a regime change.

**Limitations of the data.** Hacker News is tech-leaning (regulation, security, mining, protocol news) rather than
market news, and coverage falls from 92% of days in 2022 to 66% in 2026. On days without headlines the news analyst
abstains, and a score older than 3 days is dropped.

**Leak-free join.** A day's headlines are only complete at midnight, so day D's score is joined to candles from
D+1 00:00 UTC on (`attach_news`; unit test `test_attach_news_uses_a_day_only_after_it_has_ended`).

## 3. Scoring

| Scorer | What it is | Role |
|---|---|---|
| **LLM** | `nvidia/nemotron-3-ultra-550b-a55b:free` (NVIDIA Nemotron 3 Ultra) on OpenRouter. Open weights (anyone can rerun it locally), free, reliable JSON. Chosen over Gemma 4 31B, whose free pool was rate-limited upstream. 14 days per request (28 for the last 919 days, to fit the free daily quota), top 20 headlines per day, temperature 0, fixed seed | The hybrid agent's news reader |
| **Lexicon** | ~110 weighted words ("approval", "rally" / "hack", "ban") with a negation rule | The control: it cannot know the future |

**Look-ahead guard in the prompt.** The prompt never contains a date (days are "Day 1…7") and tells the model to
judge the wording only, not what it remembers about prices. This reduces, but cannot remove, the risk that a model
trained on later web text "knows" what happened next: see §5.

Running the LLM: put `OPENROUTER_API_KEY=...` in `.env` (template: `.env.example`, never committed), then
`make sentiment-llm`. The free tier has a daily request limit; scoring saves after every batch and resumes where it
stopped, and a different model can be chosen with `--model` or `OPENROUTER_MODEL`.

## 4. Does the news predict the next day?

The LLM rated all 1,395 news days (none unreadable). It agrees with the lexicon on the direction on 77% of days
where both have an opinion (Spearman ρ = 0.39), so it reads the same news, with more nuance (the lexicon finds a
sentiment word on only 505 days).

| Test | LLM (1,394 days) | Lexicon (505 days) | Reading |
|---|---|---|---|
| Spearman, score vs **same-day** return | ρ = +0.069, **p = 0.01** | ρ = +0.074, p = 0.10 | Headlines *describe* the day's move |
| Spearman, score vs **next-day** return | ρ = −0.014, p = 0.62 | ρ = −0.049, p = 0.27 | No forecast |
| Spearman, score vs next-day absolute return | ρ = −0.006, p = 0.82 | ρ = −0.019, p = 0.66 | No volatility forecast |
| Granger, score → return (lags 1–3) | p = 0.67 / 0.68 / 0.28 | p = 0.10 / 0.29 / 0.43 | No |
| Granger, return → score (lags 1–3) | **p < 0.0001** (F = 32 / 25 / 18) | p = 0.06 / 0.08 / 0.11 | News **follows** price |
| Event study, 5 days after the most bullish 10% | −0.0% [−1.0%, +0.8%] | −1.2% [−3.2%, +0.5%] | Not different from an average day |
| Event study, 5 days after the most bearish 10% | −0.2% [−1.2%, +0.7%] | +0.4% [−1.1%, +1.7%] | Not different from an average day |

The LLM's reading is the more precise of the two, and it shows the mechanism clearly: yesterday's price move
strongly predicts today's news tone, but the news tone does not predict tomorrow's price.

## 5. Look-ahead check

If the LLM were using memory of later prices, its scores would predict returns much better before its training
cutoff (~2025) than after, and much better than the lexicon. The notebook splits at 2025-01-01. For the lexicon the
next-day correlation is −0.099 (p = 0.06) before 2025 and +0.080 (p = 0.36) after: no stable sign, as expected
from a scorer that cannot know the future. For the LLM it is −0.012 (p = 0.72) before 2025 and −0.018 (p = 0.71)
after, and the same-day correlation is the same in both periods (+0.068 / +0.069). **No sign of look-ahead**: the
model does not forecast better on the years it may have read about.

## 6. Verdict

* Free daily headline sentiment from this source, read by an LLM or a word list, has **no measurable predictive value** for Bitcoin's next-day
  return or volatility (all p-values above 0.05). This is consistent with an efficient, heavily watched market:
  public headlines are priced in within the day.
* The news analyst is therefore kept **weak by design**: weight 0.8 out of ~6, it votes only on clearly non-neutral
  news, and a rebuttal discounts good news into an overbought market (and bad news after a deep sell-off). Whether
  it helps the agent at all is tested by the ablation in [`docs/13_agent_evaluation.md`](13_agent_evaluation.md):
  the `hybrid_llm` team ranked last of four on the development period (1D, 4h) and was not selected.
* The real value of the LLM in this project is **explanation**: the dashboard shows, in words, which news the agent
  saw on a given day. That is useful to a user even when the score adds no edge.

"""Turn daily news headlines into one sentiment score per day: the LLM half of the hybrid agent.

Two scorers share the same output (one row per day: score in [-1, 1], confidence in [0, 1],
number of headlines, a short reason):

LLMScorer      an open-weights LLM served for free by OpenRouter (default
               nvidia/nemotron-3-ultra-550b-a55b:free). It reads the headlines of several days per request
               and rates each day. Needs OPENROUTER_API_KEY (environment or .env file).
LexiconScorer  a transparent word list (rally, approval, inflow ... / crash, hack, lawsuit ...).
               It cannot know the future, so it is the control for the LLM's look-ahead risk.

The LLM only *reads*; it never trades. Its scores are cached per day in
data/external/news_sentiment_<scorer>.parquet, so a backtest never calls the API, is
deterministic, and costs nothing to re-run. Scoring is resumable: days already in the cache are
skipped, so a free-tier daily limit just means running the command again the next day.

Look-ahead guard: the LLM was trained on data that overlaps the backtest years. The prompt
therefore hides every date (days are labelled "Day 1..N"), asks it to judge the wording only,
and the notebook compares it with the lexicon scorer. See docs/12_news_sentiment.md.

    python -m src.agents.news_sentiment --scorer lexicon
    python -m src.agents.news_sentiment --scorer llm          # needs OPENROUTER_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import config
from src.data.news import load_headlines

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Open weights (NVIDIA, 550B mixture-of-experts), free on OpenRouter, reliable JSON. Chosen over
# google/gemma-4-31b-it:free, whose free pool was rate-limited upstream on every attempt (2026-11-24).
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# 14 days per request keeps a full history (~1,400 days) near 100 requests, i.e. within two days
# of the free tier's daily request limit; a median day has only 2 headlines, so prompts stay short.
DAYS_PER_REQUEST = 14
HEADLINES_PER_DAY = 20        # headlines come ranked by relevance, so these are the most relevant ones
MAX_RETRIES = 5
TIMEOUT = 120
SENTIMENT_COLUMNS = ["day", "score", "confidence", "n_headlines", "reason", "scorer"]


def cache_path(scorer_name: str) -> Path:
    return config.DATA_DIR / "external" / f"news_sentiment_{scorer_name}.parquet"


# --------------------------------------------------------------------------- lexicon
POSITIVE = {
    "surge": 1.0, "surges": 1.0, "soar": 1.0, "soars": 1.0, "rally": 1.0, "rallies": 1.0,
    "jump": 0.8, "jumps": 0.8, "gain": 0.6, "gains": 0.6, "rise": 0.6, "rises": 0.6, "climb": 0.6,
    "climbs": 0.6, "record": 0.8, "high": 0.4, "highs": 0.6, "bull": 0.8, "bullish": 1.0,
    "approve": 1.0, "approves": 1.0, "approval": 1.0, "approved": 1.0, "adoption": 0.8,
    "adopt": 0.6, "adopts": 0.6, "inflow": 0.8, "inflows": 0.8, "buy": 0.4, "buys": 0.6,
    "accumulate": 0.6, "breakout": 0.8, "rebound": 0.8, "rebounds": 0.8, "recover": 0.6,
    "recovers": 0.6, "upgrade": 0.6, "partnership": 0.4, "launch": 0.3, "launches": 0.3,
    "optimism": 0.8, "boost": 0.6, "boosts": 0.6, "milestone": 0.6, "legal": 0.3,
}
NEGATIVE = {
    "crash": 1.0, "crashes": 1.0, "plunge": 1.0, "plunges": 1.0, "tumble": 0.9, "tumbles": 0.9,
    "slump": 0.9, "slumps": 0.9, "drop": 0.6, "drops": 0.6, "fall": 0.6, "falls": 0.6, "sink": 0.7,
    "sinks": 0.7, "slide": 0.6, "slides": 0.6, "low": 0.3, "lows": 0.5, "bear": 0.8, "bearish": 1.0,
    "hack": 1.0, "hacked": 1.0, "exploit": 0.9, "stolen": 0.9, "theft": 0.9, "scam": 0.9,
    "fraud": 1.0, "ban": 1.0, "bans": 1.0, "banned": 1.0, "crackdown": 1.0, "lawsuit": 0.8,
    "sue": 0.7, "sues": 0.7, "probe": 0.6, "investigation": 0.6, "reject": 0.9, "rejects": 0.9,
    "rejected": 0.9, "delay": 0.5, "delays": 0.5, "outflow": 0.8, "outflows": 0.8, "sell": 0.4,
    "selloff": 1.0, "liquidation": 0.8, "liquidations": 0.8, "bankruptcy": 1.0, "collapse": 1.0,
    "fear": 0.7, "fears": 0.7, "warning": 0.6, "warns": 0.6, "risk": 0.3, "volatile": 0.3,
}
NEGATORS = {"not", "no", "never", "fails", "fail", "without"}
_WORD = re.compile(r"[a-z]+")


def lexicon_headline_score(title: str) -> float | None:
    """Score one headline in [-1, 1]; None when it has no sentiment word."""
    words = _WORD.findall(title.lower().replace("sell-off", "selloff"))
    total, hits = 0.0, 0
    for i, w in enumerate(words):
        weight = POSITIVE.get(w, 0.0) - NEGATIVE.get(w, 0.0)
        if weight:
            if i > 0 and words[i - 1] in NEGATORS:
                weight = -weight
            total += weight
            hits += 1
    return float(np.tanh(total)) if hits else None


class LexiconScorer:
    name = "lexicon"

    def score_days(self, days: dict[pd.Timestamp, list[str]]) -> list[dict]:
        out = []
        for day, titles in days.items():
            scores = [s for s in map(lexicon_headline_score, titles) if s is not None]
            if scores:
                score = float(np.mean(scores))
                # More scored headlines that agree with each other -> more confidence.
                agreement = abs(np.mean(np.sign(scores)))
                confidence = float(min(1.0, len(scores) / 10) * (0.5 + 0.5 * agreement))
                reason = f"{len(scores)} of {len(titles)} headlines carry sentiment words"
            else:
                score, confidence, reason = 0.0, 0.0, "no sentiment words"
            out.append({"day": day, "score": score, "confidence": confidence,
                        "n_headlines": len(titles), "reason": reason})
        return out


# --------------------------------------------------------------------------- LLM
SYSTEM_PROMPT = """You are a financial news analyst. For each day you receive the most relevant English \
headlines about Bitcoin. Rate how the news of that day would most likely move the Bitcoin price over the \
next 24 hours.

Rules:
- Judge only from the wording of the headlines. Do not use anything you may know about what happened later.
- score: -1 (clearly bearish news) ... 0 (neutral, mixed or irrelevant) ... +1 (clearly bullish news).
- confidence: 0 ... 1, how clear and important the news is (routine price commentary is low confidence).
- reason: one short sentence naming the main driver.
Answer with JSON only, exactly in this shape:
{"days": [{"day": 1, "score": 0.0, "confidence": 0.0, "reason": "..."}]}"""


def load_api_key(env_file: Path = config.PROJECT_ROOT / ".env") -> str | None:
    """OPENROUTER_API_KEY from the environment, else from a local .env file (gitignored)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key or not env_file.exists():
        return key
    for line in env_file.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None


class QuotaExhausted(RuntimeError):
    """The free tier's daily request limit is used up: run the command again later."""


def parse_llm_json(text: str, n_days: int) -> list[dict]:
    """Extract and validate the {"days": [...]} answer; missing days come back neutral."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    by_day = {}
    for item in data.get("days", []):
        try:
            k = int(item["day"])
            by_day[k] = {"score": float(np.clip(float(item["score"]), -1, 1)),
                         "confidence": float(np.clip(float(item.get("confidence", 0.5)), 0, 1)),
                         "reason": str(item.get("reason", ""))[:200]}
        except (KeyError, TypeError, ValueError):
            continue
    return [by_day.get(k, {"score": 0.0, "confidence": 0.0, "reason": "not rated by the model"})
            for k in range(1, n_days + 1)]


class LLMScorer:
    name = "llm"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 session: requests.Session | None = None, pause: float = 3.5) -> None:
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or load_api_key()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Create a free key at openrouter.ai/keys and "
                               "put OPENROUTER_API_KEY=... in the project's .env file.")
        self.session = session or requests.Session()
        self.pause = pause  # free models allow about 20 requests per minute

    def _prompt(self, days: dict[pd.Timestamp, list[str]]) -> str:
        blocks = []
        for k, titles in enumerate(days.values(), 1):
            lines = "\n".join(f"- {t}" for t in titles) or "- (no headlines)"
            blocks.append(f"Day {k}:\n{lines}")
        return "\n\n".join(blocks)

    def _post(self, payload: dict) -> dict:
        wait = 20.0
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.post(OPENROUTER_URL, json=payload, timeout=TIMEOUT,
                                         headers={"Authorization": f"Bearer {self.api_key}",
                                                  "X-Title": "BTC news sentiment (student project)"})
                data = resp.json() if resp.status_code == 200 else None
                if isinstance(data, dict) and data.get("choices"):
                    return data
            except (requests.RequestException, ValueError) as exc:  # dropped connection, cut-off body
                logger.info("Network error (%s), retry %d in %.0fs", type(exc).__name__, attempt, wait)
                time.sleep(wait)
                wait *= 2
                continue
            body = resp.text[:300]
            if resp.status_code == 429 and "per-day" in body.lower():
                raise QuotaExhausted(body)
            # A 200 without "choices" carries an upstream provider error in its body: retry it.
            if resp.status_code in (200, 429, 500, 502, 503, 504):
                logger.info("OpenRouter %s, retry %d in %.0fs", resp.status_code, attempt, wait)
                time.sleep(wait)
                wait *= 2
                continue
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {body}")
        raise RuntimeError("OpenRouter kept failing; try again later or set OPENROUTER_MODEL")

    def score_days(self, days: dict[pd.Timestamp, list[str]]) -> list[dict]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "seed": 0,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": self._prompt(days)}],
        }
        answer = self._post(payload)
        text = answer["choices"][0]["message"]["content"] or ""
        time.sleep(self.pause)
        try:
            rated = parse_llm_json(text, len(days))
        except ValueError:  # malformed JSON (json.JSONDecodeError is a ValueError)
            return self._rescore_in_halves(days)
        return [{"day": day, "n_headlines": len(titles), **r}
                for (day, titles), r in zip(days.items(), rated)]

    def _rescore_in_halves(self, days: dict[pd.Timestamp, list[str]]) -> list[dict]:
        """A long answer occasionally comes back as broken JSON: ask again for each half.
        A single day that still fails is kept as 'not rated' (confidence 0: the analyst abstains)."""
        if len(days) == 1:
            (day, titles), = days.items()
            logger.warning("Unreadable LLM answer for %s; day left unrated", day.date())
            return [{"day": day, "n_headlines": len(titles), "score": 0.0, "confidence": 0.0,
                     "reason": "unreadable model answer"}]
        items = list(days.items())
        mid = len(items) // 2
        logger.info("Unreadable LLM answer for %d days; retrying in two halves", len(items))
        return self.score_days(dict(items[:mid])) + self.score_days(dict(items[mid:]))


# --------------------------------------------------------------------------- batch scoring
def daily_headlines(headlines: pd.DataFrame, per_day: int = HEADLINES_PER_DAY) -> dict[pd.Timestamp, list[str]]:
    """Day -> its `per_day` most relevant headlines (GDELT's relevance rank)."""
    ranked = headlines.sort_values(["day", "rank"])
    return {day: grp["title"].head(per_day).tolist() for day, grp in ranked.groupby("day", sort=True)}


def load_sentiment(scorer_name: str) -> pd.DataFrame:
    path = cache_path(scorer_name)
    if not path.exists():
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)
    return pd.read_parquet(path)


def best_available() -> tuple[str, pd.DataFrame]:
    """The LLM scores when they exist, else the lexicon scores (name, table)."""
    for name in ("llm", "lexicon"):
        table = load_sentiment(name)
        if len(table):
            return name, table
    return "none", load_sentiment("lexicon")


def score_all(headlines: pd.DataFrame, scorer, batch_days: int = DAYS_PER_REQUEST,
              path: Path | None = None) -> pd.DataFrame:
    """Score every day not yet cached, saving after each batch so the run can be resumed."""
    path = path or cache_path(scorer.name)
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=SENTIMENT_COLUMNS)
    seen = set(pd.to_datetime(done["day"], utc=True)) if len(done) else set()
    todo = {d: t for d, t in daily_headlines(headlines).items() if d not in seen}
    items = list(todo.items())
    logger.info("%s: %d days cached, %d to score", scorer.name, len(seen), len(items))
    for i in range(0, len(items), batch_days):
        batch = dict(items[i:i + batch_days])
        try:
            rows = scorer.score_days(batch)
        except QuotaExhausted:
            logger.warning("Free-tier daily limit reached after %d days; run again later to continue.", i)
            break
        new = pd.DataFrame(rows).assign(scorer=getattr(scorer, "model", scorer.name))
        done = pd.concat([done, new[SENTIMENT_COLUMNS]], ignore_index=True) if len(done) else new[SENTIMENT_COLUMNS]
        path.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(path, index=False)
        if (i // batch_days) % 10 == 0:
            logger.info("%s: %d/%d days scored", scorer.name, min(i + batch_days, len(items)), len(items))
    return done.sort_values("day").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorer", choices=["lexicon", "llm"], default="lexicon")
    parser.add_argument("--model", default=None, help=f"OpenRouter model id (default {DEFAULT_MODEL})")
    parser.add_argument("--days-per-request", type=int, default=DAYS_PER_REQUEST)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    scorer = LexiconScorer() if args.scorer == "lexicon" else LLMScorer(model=args.model)
    batch = 365 if args.scorer == "lexicon" else args.days_per_request
    out = score_all(load_headlines(), scorer, batch_days=batch)
    print(f"{len(out):,} days scored -> {cache_path(scorer.name)}")


if __name__ == "__main__":
    main()

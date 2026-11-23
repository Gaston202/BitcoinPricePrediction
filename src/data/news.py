"""Daily Bitcoin news headlines from two free, key-less sources with full history.

gdelt  GDELT DOC 2.0 API: up to 75 English articles per day from thousands of news sites,
       ranked by relevance. The richest source, but it throttles hard (one request per
       5 seconds, and long blocks after a burst).
hn     Hacker News search API (Algolia): stories whose title mentions Bitcoin, ranked by
       points. Fewer headlines per day (typically 3-20) and tech-leaning, but reliable.

One request per UTC day. Each day is cached as its own JSON file under
data/external/news/<source>/, so a long download can be stopped and resumed and a cached day
is never requested again. `build_snapshot()` merges one source's cache into the frozen file
data/external/news_headlines.parquet that the analysis reads. A snapshot always comes from a
single source, so the headline mix does not change halfway through the history.

Timing (no look-ahead): a headline belongs to the UTC day it was first published or seen, and
the headlines of day D are treated as known only at D+1 00:00 UTC
(`src.agents.market_data.attach_news`).

    python -m src.data.news --source hn --start 2022-01-01 --end 2026-09-30   # resumable
    python -m src.data.news --source hn --snapshot                            # merge the cache
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

NEWS_DIR = config.DATA_DIR / "external" / "news"
HEADLINES_FILE = config.DATA_DIR / "external" / "news_headlines.parquet"
COLUMNS = ["day", "source", "rank", "seen_at", "title", "domain", "url"]
TIMEOUT = 60
MAX_RETRIES = 4
BLOCKED_PAUSE = 600.0  # after MAX_RETRIES failures in a row: pause, then try the same day again

# A headline must name Bitcoin or crypto to count (search APIs also match body text).
RELEVANT = re.compile(r"\b(?:bitcoins?|btc|crypto|cryptocurrenc\w*)\b", re.IGNORECASE)


class NewsError(RuntimeError):
    pass


# --------------------------------------------------------------------------- sources
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = "(bitcoin OR btc) sourcelang:english"
HN_URL = "https://hn.algolia.com/api/v1/search_by_date"


def _day_bounds(day: pd.Timestamp) -> tuple[int, int]:
    start = int(day.timestamp())
    return start, start + 86_400


def fetch_gdelt(day: pd.Timestamp, session: requests.Session) -> list[dict]:
    params = {"query": GDELT_QUERY, "mode": "artlist", "format": "json", "sort": "hybridrel",
              "maxrecords": 75, "startdatetime": f"{day:%Y%m%d}000000", "enddatetime": f"{day:%Y%m%d}235959"}
    resp = session.get(GDELT_URL, params=params, timeout=TIMEOUT)
    text = resp.text.strip()
    if text.startswith("{"):
        return json.loads(text).get("articles", [])
    if not text:  # GDELT answers an empty body when nothing matched
        return []
    raise NewsError(f"GDELT busy: {text[:60]}")


def fetch_hn(day: pd.Timestamp, session: requests.Session) -> list[dict]:
    start, end = _day_bounds(day)
    params = {"query": "bitcoin", "tags": "story", "hitsPerPage": 200,
              "numericFilters": f"created_at_i>={start},created_at_i<{end}"}
    resp = session.get(HN_URL, params=params, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise NewsError(f"HN search answered {resp.status_code}")
    return [{k: h.get(k) for k in ("title", "url", "created_at", "points", "num_comments")}
            for h in resp.json().get("hits", [])]


def tidy_gdelt(articles: list[dict]) -> pd.DataFrame:
    if not articles:
        return pd.DataFrame(columns=["rank", "seen_at", "title", "domain", "url"])
    df = pd.DataFrame(articles)
    return pd.DataFrame({
        "rank": range(len(df)),  # GDELT's relevance order, most relevant first
        "seen_at": pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True),
        "title": df["title"], "domain": df["domain"], "url": df["url"],
    })


def tidy_hn(hits: list[dict]) -> pd.DataFrame:
    if not hits:
        return pd.DataFrame(columns=["rank", "seen_at", "title", "domain", "url"])
    df = pd.DataFrame(hits)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df = df.sort_values("points", ascending=False)  # community attention as the relevance rank
    url = df["url"].fillna("")
    return pd.DataFrame({
        "rank": range(len(df)),
        "seen_at": pd.to_datetime(df["created_at"], utc=True),
        "title": df["title"].fillna(""),
        "domain": url.str.extract(r"https?://(?:www\.)?([^/]+)", expand=False).fillna("news.ycombinator.com"),
        "url": url,
    })


SOURCES: dict[str, tuple[Callable, Callable, float]] = {
    # name -> (fetch one day, tidy raw rows, seconds to wait between requests)
    "gdelt": (fetch_gdelt, tidy_gdelt, 6.0),
    "hn": (fetch_hn, tidy_hn, 0.5),
}


# --------------------------------------------------------------------------- cleaning
def clean_title(title: str) -> str:
    """GDELT tokenises titles ("Bitcoin rises , analysts say ."); put the punctuation back."""
    t = re.sub(r"\s+([,.;:!?%')\]])", r"\1", str(title))
    t = re.sub(r"([(\[$])\s+", r"\1", t)
    t = re.sub(r"\s+'\s*s\b", "'s", t)
    return re.sub(r"\s+", " ", t).strip()


def tidy_day(raw: list[dict], day: pd.Timestamp, source: str = "gdelt") -> pd.DataFrame:
    """Raw rows of one day -> relevant, de-duplicated headlines in relevance order."""
    df = SOURCES[source][1](raw)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["title"] = df["title"].map(clean_title)
    df = df[df["title"].str.contains(RELEVANT)]
    # The same story is syndicated many times: keep its first sighting only.
    key = df["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df.assign(_key=key).sort_values("seen_at").drop_duplicates("_key").sort_values("rank")
    df = df.assign(day=day, source=source, rank=range(len(df)))
    return df[COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- download
def cache_dir(source: str) -> Path:
    return NEWS_DIR / source


def _fetch_with_retries(fetch: Callable, day: pd.Timestamp, session: requests.Session) -> list[dict]:
    wait = 30.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch(day, session)
        except (NewsError, requests.RequestException, ValueError) as exc:
            logger.info("%s on %s, retry %d in %.0fs", exc, f"{day:%Y-%m-%d}", attempt, wait)
            time.sleep(wait)
            wait *= 2
    raise NewsError(f"no answer for {day:%Y-%m-%d} after {MAX_RETRIES} tries")


def download(start: str, end: str, source: str = "hn", fetch: Callable | None = None,
             gap: float | None = None) -> int:
    """Fill the per-day cache of one source for [start, end]. Returns the days downloaded."""
    default_fetch, _, default_gap = SOURCES[source]
    fetch, gap = fetch or default_fetch, default_gap if gap is None else gap
    folder = cache_dir(source)
    folder.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1D")
    todo = [d for d in days if not (folder / f"{d:%Y-%m-%d}.json").exists()]
    logger.info("%s: %d of %d days to download", source, len(todo), len(days))
    session = requests.Session()
    for i, day in enumerate(todo, 1):
        while True:
            try:
                raw = _fetch_with_retries(fetch, day, session)
                break
            except NewsError as exc:  # long throttling block: wait it out, then resume
                logger.warning("%s; pausing %.0f minutes", exc, BLOCKED_PAUSE / 60)
                time.sleep(BLOCKED_PAUSE)
        (folder / f"{day:%Y-%m-%d}.json").write_text(json.dumps(raw))
        if i % 50 == 0 or i == len(todo):
            logger.info("%s: %d/%d days done (last %s, %d rows)", source, i, len(todo), f"{day:%Y-%m-%d}", len(raw))
        if gap and i < len(todo):
            time.sleep(gap)
    return len(todo)


# --------------------------------------------------------------------------- snapshot
def build_snapshot(source: str = "hn", out: Path = HEADLINES_FILE) -> pd.DataFrame:
    """Merge one source's cached days into the frozen headline table."""
    frames = [tidy_day(json.loads(p.read_text()), pd.Timestamp(p.stem, tz="UTC"), source)
              for p in sorted(cache_dir(source).glob("*.json"))]
    if not frames:
        raise NewsError(f"no cached days in {cache_dir(source)}")
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load_headlines(path: Path = HEADLINES_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.news --snapshot")
    return pd.read_parquet(path)


def coverage(headlines: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Headlines per day over [start, end], including days with none (data-quality check)."""
    days = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1D")
    return headlines.groupby("day").size().reindex(days, fill_value=0).rename("headlines").to_frame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), default="hn")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=f"{config.CUTOFF_DATE:%Y-%m-%d}")
    parser.add_argument("--snapshot", action="store_true", help="only merge the cache into the frozen file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.snapshot:
        download(args.start, args.end, args.source)
    df = build_snapshot(args.source)
    print(f"{len(df):,} {args.source} headlines over {df['day'].nunique():,} days -> {HEADLINES_FILE}")


if __name__ == "__main__":
    main()

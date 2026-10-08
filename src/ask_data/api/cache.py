"""Redis answer cache (FR-12): the same first question on the same dataset version and model is
answered from Redis for ANSWER_CACHE_TTL_S instead of running the graph again.

Only a thread's first question is cached or served: a follow-up ("now only finals") means
something different in every conversation. The key holds the dataset *version*, so reloading a
dataset makes its old answers unreachable (they expire on their own). Like the rate limiter, the
cache fails open: a Redis problem means a normal graph run, never an error.
"""

import hashlib
import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal

log = logging.getLogger(__name__)

PUNCT_END_RE = re.compile(r"[\s?.!]+$")
FIELDS = ("question", "sql", "attempts", "tables", "table_selection", "columns", "rows",
          "row_count", "truncated", "answer", "chart", "assumption")  # fmt: skip


def _json_default(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, date | datetime):
        return v.isoformat()
    return str(v)


def normalize_question(question: str) -> str:
    """'  Who won  the 2016 final? ' and 'who won the 2016 final' share a cache entry."""
    return PUNCT_END_RE.sub("", " ".join(question.split()).lower())


def cache_key(dataset: str, version: str, question: str, model: str) -> str:
    raw = json.dumps([dataset, version, normalize_question(question), model])
    return "answer:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


class AnswerCache:
    def __init__(self, redis, ttl_s: int):
        self.redis, self.ttl_s = redis, ttl_s

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0

    async def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        try:
            raw = await self.redis.get(key)
        except Exception as e:  # noqa: BLE001
            log.warning("answer cache unavailable, running the graph: %s", e)
            return None
        return json.loads(raw) if raw else None

    async def put(self, key: str, state: dict) -> None:
        if not self.enabled:
            return
        value = {f: state.get(f) for f in FIELDS}
        try:
            await self.redis.set(key, json.dumps(value, default=_json_default), ex=self.ttl_s)
        except Exception as e:  # noqa: BLE001
            log.warning("could not write the answer cache: %s", e)

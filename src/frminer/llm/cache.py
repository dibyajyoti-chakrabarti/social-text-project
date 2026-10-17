"""SQLite-backed store for LLM responses and per-provider daily request usage.

Every request is keyed by a hash of everything that can influence the output
(provider, model, messages, parameters), so re-running a pipeline stage never
re-spends quota and results stay reproducible.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_s REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    provider TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, provider)
);
"""


def request_key(request: dict[str, Any]) -> str:
    blob = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response, prompt_tokens, completion_tokens, latency_s "
                "FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "text": row[0],
            "prompt_tokens": row[1],
            "completion_tokens": row[2],
            "latency_s": row[3],
        }

    def put(
        self,
        key: str,
        request: dict[str, Any],
        text: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_s: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    request["provider"],
                    request["model"],
                    json.dumps(request, ensure_ascii=False),
                    text,
                    prompt_tokens,
                    completion_tokens,
                    latency_s,
                    time.time(),
                ),
            )
            self._conn.commit()

    @staticmethod
    def quota_day(tz: str = "UTC") -> str:
        """Today's date in the timezone where the provider's daily quota resets."""
        return datetime.now(ZoneInfo(tz)).date().isoformat()

    def requests_today(self, provider: str, tz: str = "UTC") -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT requests FROM usage WHERE day = ? AND provider = ?",
                (self.quota_day(tz), provider),
            ).fetchone()
        return row[0] if row else 0

    def record_request(self, provider: str, tz: str = "UTC") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage (day, provider, requests) VALUES (?, ?, 1) "
                "ON CONFLICT(day, provider) DO UPDATE SET requests = requests + 1",
                (self.quota_day(tz), provider),
            )
            self._conn.commit()

    def exhaust(self, provider: str, limit: int, tz: str = "UTC") -> None:
        """Record that the provider refused more requests today (counter ≥ `limit`)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage (day, provider, requests) VALUES (?, ?, ?) "
                "ON CONFLICT(day, provider) DO UPDATE SET requests = MAX(requests, ?)",
                (self.quota_day(tz), provider, limit, limit),
            )
            self._conn.commit()

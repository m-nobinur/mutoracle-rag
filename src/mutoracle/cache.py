"""SQLite cache and cost ledger for provider-backed RAG runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any


@dataclass(frozen=True)
class CachedCompletion:
    """A cached chat completion."""

    answer: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CachedOracleScore:
    """A cached normalized oracle score."""

    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class UsageSummary:
    """Aggregate usage recorded in the local ledger."""

    requests: int
    live_requests: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    total_latency_seconds: float


class SQLiteCacheLedger:
    """Small SQLite-backed response cache and cost ledger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def lookup_completion(self, cache_key: str) -> CachedCompletion | None:
        """Return a cached completion for a key, if present."""

        with self._connect() as connection:
            row = connection.execute(
                "select answer, metadata_json from completions where cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return CachedCompletion(
            answer=str(row["answer"]),
            metadata=json.loads(str(row["metadata_json"])),
        )

    def store_completion(
        self,
        *,
        cache_key: str,
        answer: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store or replace a provider completion."""

        with self._connect() as connection:
            connection.execute(
                """
                insert into completions(cache_key, answer, metadata_json, created_at)
                values (?, ?, ?, ?)
                on conflict(cache_key) do update set
                    answer = excluded.answer,
                    metadata_json = excluded.metadata_json
                """,
                (cache_key, answer, json.dumps(metadata, sort_keys=True), time()),
            )

    def lookup_oracle_score(self, cache_key: str) -> CachedOracleScore | None:
        """Return a cached oracle score for a key, if present."""

        with self._connect() as connection:
            row = connection.execute(
                "select score, metadata_json from oracle_scores where cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return CachedOracleScore(
            score=float(row["score"]),
            metadata=json.loads(str(row["metadata_json"])),
        )

    def store_oracle_score(
        self,
        *,
        cache_key: str,
        oracle_name: str,
        input_hash: str,
        score: float,
        metadata: dict[str, Any],
    ) -> None:
        """Store or replace a normalized oracle score."""

        with self._connect() as connection:
            connection.execute(
                """
                insert into oracle_scores(
                    cache_key,
                    oracle_name,
                    input_hash,
                    score,
                    metadata_json,
                    created_at
                )
                values (?, ?, ?, ?, ?, ?)
                on conflict(cache_key) do update set
                    oracle_name = excluded.oracle_name,
                    input_hash = excluded.input_hash,
                    score = excluded.score,
                    metadata_json = excluded.metadata_json
                """,
                (
                    cache_key,
                    oracle_name,
                    input_hash,
                    score,
                    json.dumps(metadata, sort_keys=True),
                    time(),
                ),
            )

    def record_usage(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_seconds: float,
        cache_hit: bool,
    ) -> None:
        """Record one model request or cache hit in the usage ledger."""

        with self._connect() as connection:
            connection.execute(
                """
                insert into usage_ledger(
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    latency_seconds,
                    cache_hit,
                    created_at
                )
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    latency_seconds,
                    int(cache_hit),
                    time(),
                ),
            )

    def usage_summary(self) -> UsageSummary:
        """Return aggregate ledger totals."""

        with self._connect() as connection:
            row = connection.execute(
                """
                select
                    count(*) as requests,
                    coalesce(sum(case when cache_hit = 0 then 1 else 0 end), 0)
                        as live_requests,
                    coalesce(sum(case when cache_hit = 1 then 1 else 0 end), 0)
                        as cache_hits,
                    coalesce(sum(prompt_tokens), 0) as prompt_tokens,
                    coalesce(sum(completion_tokens), 0) as completion_tokens,
                    coalesce(sum(cost_usd), 0.0) as total_cost_usd,
                    coalesce(sum(latency_seconds), 0.0) as total_latency_seconds
                from usage_ledger
                """
            ).fetchone()
        return UsageSummary(
            requests=int(row["requests"]),
            live_requests=int(row["live_requests"]),
            cache_hits=int(row["cache_hits"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            total_cost_usd=float(row["total_cost_usd"]),
            total_latency_seconds=float(row["total_latency_seconds"]),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists completions (
                    cache_key text primary key,
                    answer text not null,
                    metadata_json text not null,
                    created_at real not null
                )
                """
            )
            connection.execute(
                """
                create table if not exists usage_ledger (
                    id integer primary key autoincrement,
                    model text not null,
                    prompt_tokens integer not null,
                    completion_tokens integer not null,
                    cost_usd real not null,
                    latency_seconds real not null default 0.0,
                    cache_hit integer not null,
                    created_at real not null
                )
                """
            )
            connection.execute(
                """
                create table if not exists oracle_scores (
                    cache_key text primary key,
                    oracle_name text not null,
                    input_hash text not null,
                    score real not null,
                    metadata_json text not null,
                    created_at real not null
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("pragma table_info(usage_ledger)")
            }
            if "latency_seconds" not in columns:
                connection.execute(
                    "alter table usage_ledger add column latency_seconds real not null "
                    "default 0.0"
                )


def prompt_hash(prompt: str) -> str:
    """Return the stable hash used in manifests and cache keys."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def completion_cache_key(
    *,
    model: str,
    prompt: str,
    temperature: float,
    provider_route: str = "openrouter",
    seed: int = 0,
) -> str:
    """Return a stable cache key for provider inputs."""

    payload = json.dumps(
        {
            "model": model,
            "prompt_hash": prompt_hash(prompt),
            "provider_route": provider_route,
            "seed": seed,
            "temperature": temperature,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def oracle_cache_key(
    *,
    oracle_name: str,
    model: str,
    payload: dict[str, Any],
) -> str:
    """Return a stable cache key for oracle inputs."""

    encoded = json.dumps(
        {
            "model": model,
            "oracle_name": oracle_name,
            "payload": payload,
        },
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

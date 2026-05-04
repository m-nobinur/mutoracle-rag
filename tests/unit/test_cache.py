from __future__ import annotations

from pathlib import Path

from mutoracle.cache import SQLiteCacheLedger, completion_cache_key


def test_sqlite_cache_and_usage_ledger(tmp_path: Path) -> None:
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    key = completion_cache_key(model="test", prompt="hello", temperature=0.0)

    assert ledger.lookup_completion(key) is None

    ledger.store_completion(
        cache_key=key,
        answer="world",
        metadata={"model": "test"},
    )
    cached = ledger.lookup_completion(key)

    assert cached is not None
    assert cached.answer == "world"
    assert cached.metadata == {"model": "test"}

    ledger.record_usage(
        model="test",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.25,
        latency_seconds=0.5,
        cache_hit=False,
    )
    summary = ledger.usage_summary()

    assert summary.requests == 1
    assert summary.live_requests == 1
    assert summary.cache_hits == 0
    assert summary.prompt_tokens == 10
    assert summary.completion_tokens == 5
    assert summary.total_cost_usd == 0.25
    assert summary.total_latency_seconds == 0.5

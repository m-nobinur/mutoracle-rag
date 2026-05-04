from __future__ import annotations

from pathlib import Path

import pytest

from mutoracle.cache import SQLiteCacheLedger, completion_cache_key
from mutoracle.config import (
    CostConfig,
    MutOracleConfig,
    OpenRouterConfig,
    RuntimeConfig,
)
from mutoracle.provider import OpenRouterProvider


def test_provider_serves_cached_completion_without_api_key(tmp_path: Path) -> None:
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    key = completion_cache_key(
        model=config.models.generator,
        prompt="cached prompt",
        temperature=float(config.models.temperature),
        provider_route="openrouter",
        seed=config.runtime.seed,
    )
    ledger.store_completion(
        cache_key=key,
        answer="cached answer",
        metadata={"model": config.models.generator},
    )

    completion = OpenRouterProvider(config, ledger).complete("cached prompt")

    assert completion.answer == "cached answer"
    assert completion.metadata["cache_hit"] is True
    assert ledger.usage_summary().cache_hits == 1


def test_provider_enforces_live_query_budget(tmp_path: Path) -> None:
    config = MutOracleConfig(
        openrouter=OpenRouterConfig(api_key="test-key"),
        cost=CostConfig(max_queries=1),
        runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"),
    )
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    ledger.record_usage(
        model=config.models.generator,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.0,
        latency_seconds=0.1,
        cache_hit=False,
    )

    with pytest.raises(RuntimeError, match="Remote query budget exhausted"):
        OpenRouterProvider(config, ledger).complete("new prompt")

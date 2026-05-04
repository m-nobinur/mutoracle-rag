from __future__ import annotations

from pathlib import Path

from mutoracle.config import MutOracleConfig, RuntimeConfig
from mutoracle.rag import FixtureRAGPipeline


def test_fixture_rag_pipeline_runs_without_credentials(tmp_path: Path) -> None:
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    pipeline = FixtureRAGPipeline(config=config)

    run = pipeline.run("How does MutOracle-RAG localize faults?")

    assert run.query == "How does MutOracle-RAG localize faults?"
    assert len(run.passages) == config.rag.top_k
    assert "MutOracle-RAG localizes hallucination faults" in run.answer
    assert run.metadata["generation"]["provider"] == "fixture"
    assert run.metadata["generation"]["usage"]["total_tokens"] > 0
    assert run.metadata["generation"]["prompt_hash"]
    assert run.metadata["latency"]["retrieval_seconds"] == 0.0
    assert run.metadata["latency"]["generation_seconds"] == 0.0
    assert run.metadata["retrieval"][0]["id"] == "mutoracle-purpose"


def test_fixture_rag_pipeline_is_reproducible(tmp_path: Path) -> None:
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    pipeline = FixtureRAGPipeline(config=config)

    first = pipeline.run("What is MutOracle-RAG?")
    second = pipeline.run("What is MutOracle-RAG?")

    assert first == second

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from mutoracle.cache import SQLiteCacheLedger
from mutoracle.config import MutOracleConfig, RuntimeConfig
from mutoracle.contracts import RAGRun
from mutoracle.oracles import (
    LLMJudgeOracle,
    NLIOracle,
    SemanticSimilarityOracle,
    clamp_score,
    cosine_to_unit_interval,
    parse_judge_response,
)
from mutoracle.oracles.nli import _split_pipeline_output
from mutoracle.provider import ProviderCompletion


class CountingEmbeddingBackend:
    """Mock embedding backend for testing that tracks call count."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode texts into embeddings, returning a vector per text."""
        self.calls += 1
        vectors = {
            "MutOracle localizes RAG failures.": [1.0, 0.0],
        }
        return [vectors.get(text, [0.0, 1.0]) for text in texts]


class CountingNLIBackend:
    def __init__(self) -> None:
        self.calls = 0

    def probabilities(self, *, premise: str, hypothesis: str) -> dict[str, float]:
        self.calls += 1
        return {
            "entailment": 0.82,
            "neutral": 0.1,
            "contradiction": 0.08,
        }


class BatchCountingNLIBackend:
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls = 0

    def probabilities(self, *, premise: str, hypothesis: str) -> dict[str, float]:
        del premise, hypothesis
        self.single_calls += 1
        return {"entailment": 0.1}

    def probabilities_many(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[dict[str, float]]:
        self.batch_calls += 1
        return [{"entailment": 0.2 + index * 0.1} for index, _ in enumerate(pairs)]


class SequenceJudgeProvider:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0
        self.temperatures: list[float | None] = []

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        request_kind: str = "generation",
    ) -> ProviderCompletion:
        del prompt, model, request_kind
        self.temperatures.append(temperature)
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return ProviderCompletion(answer=answer, metadata={"call": self.calls})


class FailingJudgeProvider:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        request_kind: str = "generation",
    ) -> ProviderCompletion:
        del prompt, model, temperature, request_kind
        self.calls += 1
        raise RuntimeError(self.message)


def test_score_normalization_helpers_return_unit_interval() -> None:
    assert clamp_score(-0.5) == 0.0
    assert clamp_score(1.5) == 1.0
    assert clamp_score(0.4) == 0.4
    assert cosine_to_unit_interval(-1.0) == 0.0
    assert cosine_to_unit_interval(1.0) == 1.0
    assert cosine_to_unit_interval(0.0) == 0.5


def test_semantic_oracle_scores_tiny_fixture_and_uses_cache(tmp_path: Path) -> None:
    run = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    backend = CountingEmbeddingBackend()
    oracle = SemanticSimilarityOracle(
        ledger=ledger,
        backend=backend,
        model_name="fake-embedding",
    )

    first = oracle.score_result(run)
    second = oracle.score_result(run)

    assert first.value == pytest.approx(1.0)
    assert second.value == pytest.approx(1.0)
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert backend.calls == 1


def test_semantic_oracle_batches_uncached_runs(tmp_path: Path) -> None:
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    backend = CountingEmbeddingBackend()
    oracle = SemanticSimilarityOracle(
        ledger=ledger,
        backend=backend,
        model_name="fake-embedding",
    )
    runs = [
        RAGRun(
            query=f"What does MutOracle do {index}?",
            passages=["MutOracle localizes RAG failures."],
            answer="MutOracle localizes RAG failures.",
        )
        for index in range(3)
    ]

    first = oracle.score_results(runs)
    second = oracle.score_results(runs)

    assert [result.value for result in first] == pytest.approx([1.0, 1.0, 1.0])
    assert all(result.metadata["cache_hit"] is False for result in first)
    assert all(result.metadata["cache_hit"] is True for result in second)
    assert backend.calls == 1


def test_nli_oracle_scores_tiny_fixture_and_uses_cache(tmp_path: Path) -> None:
    run = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    backend = CountingNLIBackend()
    oracle = NLIOracle(ledger=ledger, backend=backend, model_name="fake-nli")

    first = oracle.score_result(run)
    second = oracle.score_result(run)

    assert first.value == pytest.approx(0.82)
    assert second.value == pytest.approx(0.82)
    assert second.metadata["cache_hit"] is True
    assert backend.calls == 1


def test_nli_oracle_uses_backend_batch_api_for_uncached_runs(tmp_path: Path) -> None:
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    backend = BatchCountingNLIBackend()
    oracle = NLIOracle(ledger=ledger, backend=backend, model_name="fake-nli")
    runs = [
        RAGRun(
            query=f"What does MutOracle do {index}?",
            passages=[f"Premise {index}"],
            answer=f"Hypothesis {index}",
        )
        for index in range(3)
    ]

    results = oracle.score_results(runs)
    cached = oracle.score_results(runs)

    assert [result.value for result in results] == pytest.approx([0.2, 0.3, 0.4])
    assert all(result.metadata["cache_hit"] is False for result in results)
    assert all(result.metadata["cache_hit"] is True for result in cached)
    assert backend.batch_calls == 1
    assert backend.single_calls == 0


def test_nli_batch_handles_empty_inputs_and_pipeline_output_shapes(
    tmp_path: Path,
) -> None:
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    backend = BatchCountingNLIBackend()
    oracle = NLIOracle(ledger=ledger, backend=backend, model_name="fake-nli")

    empty_result = oracle.score_results([RAGRun(query="q", passages=[], answer="")])[0]

    assert empty_result.value == 0.0
    assert empty_result.metadata["reason"] == "empty_premise_or_hypothesis"
    assert backend.batch_calls == 0
    assert _split_pipeline_output([], expected=0) == []
    assert _split_pipeline_output(
        {"label": "ENTAILMENT", "score": 0.9}, expected=1
    ) == [[{"label": "ENTAILMENT", "score": 0.9}]]


def test_nli_cache_reuses_scores_for_query_only_changes(tmp_path: Path) -> None:
    baseline = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )
    query_mutation = RAGRun(
        query="How does MutOracle work?",
        passages=baseline.passages,
        answer=baseline.answer,
    )
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    backend = CountingNLIBackend()
    oracle = NLIOracle(ledger=ledger, backend=backend, model_name="fake-nli")

    first = oracle.score_result(baseline)
    second = oracle.score_result(query_mutation)

    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert backend.calls == 1


def test_semantic_cache_reuses_scores_for_query_only_changes(tmp_path: Path) -> None:
    baseline = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )
    query_mutation = RAGRun(
        query="How does MutOracle work?",
        passages=baseline.passages,
        answer=baseline.answer,
    )
    ledger = SQLiteCacheLedger(tmp_path / "cache.sqlite3")
    backend = CountingEmbeddingBackend()
    oracle = SemanticSimilarityOracle(
        ledger=ledger,
        backend=backend,
        model_name="fake-embedding",
    )

    first = oracle.score_result(baseline)
    second = oracle.score_result(query_mutation)

    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert backend.calls == 1


def test_judge_cache_reuses_scores_for_query_only_changes(tmp_path: Path) -> None:
    baseline = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )
    query_mutation = RAGRun(
        query="How does MutOracle work?",
        passages=baseline.passages,
        answer=baseline.answer,
    )
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    provider = SequenceJudgeProvider(
        ['{"verdict": "faithful", "confidence": 0.7, "reason": "Supported."}']
    )
    oracle = LLMJudgeOracle(config=config, ledger=ledger, provider=provider)

    first = oracle.score_result(baseline)
    second = oracle.score_result(query_mutation)

    assert first.value == pytest.approx(0.7)
    assert second.value == pytest.approx(0.7)
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert provider.calls == 1


def test_judge_json_parsing_is_strict() -> None:
    parsed = parse_judge_response(
        '{"verdict": "faithful", "confidence": 0.91, "reason": "Supported."}'
    )

    assert parsed.verdict == "faithful"
    assert parsed.confidence == 0.91


def test_invalid_judge_json_retries_once_then_records_failure(tmp_path: Path) -> None:
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    provider = SequenceJudgeProvider(["not json", '{"unexpected": true}'])
    oracle = LLMJudgeOracle(config=config, ledger=ledger, provider=provider)
    run = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )

    result = oracle.score_result(run)
    cached = oracle.score_result(run)

    assert result.value == 0.0
    assert result.metadata["cache_hit"] is False
    assert result.metadata["failure"]["kind"] == "invalid_judge_response"
    assert result.metadata["attempts"] == 2
    assert result.metadata["temperature"] == 0.0
    assert cached.metadata["cache_hit"] is True
    assert provider.calls == 2


def test_judge_uses_zero_temperature_even_if_generation_is_nonzero(
    tmp_path: Path,
) -> None:
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    config = config.model_copy(
        update={"models": config.models.model_copy(update={"temperature": 0.8})}
    )
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    provider = SequenceJudgeProvider(
        ['{"verdict": "faithful", "confidence": 0.7, "reason": "Supported."}']
    )
    oracle = LLMJudgeOracle(config=config, ledger=ledger, provider=provider)
    run = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )

    result = oracle.score_result(run)

    assert result.value == pytest.approx(0.7)
    assert provider.temperatures == [0.0]


def test_non_retryable_provider_error_propagates(tmp_path: Path) -> None:
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    provider = FailingJudgeProvider(
        "OPENROUTER_API_KEY is required for remote RAG generation."
    )
    oracle = LLMJudgeOracle(config=config, ledger=ledger, provider=provider)
    run = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="MutOracle localizes RAG failures.",
    )

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        oracle.score_result(run)
    assert provider.calls == 1


def test_judge_hallucinated_verdict_maps_to_inverse_confidence(tmp_path: Path) -> None:
    config = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"))
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    provider = SequenceJudgeProvider(
        [('{"verdict": "hallucinated", "confidence": 0.8, "reason": "Unsupported."}')]
    )
    oracle = LLMJudgeOracle(config=config, ledger=ledger, provider=provider)
    run = RAGRun(
        query="What does MutOracle do?",
        passages=["MutOracle localizes RAG failures."],
        answer="It makes coffee.",
    )

    result = oracle.score_result(run)

    assert result.value == pytest.approx(0.2)
    assert result.metadata["verdict"]["verdict"] == "hallucinated"

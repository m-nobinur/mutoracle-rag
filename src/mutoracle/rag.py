"""Reproducible RAG system under test for Phase 2."""

from __future__ import annotations

from pathlib import Path

from mutoracle.cache import SQLiteCacheLedger, prompt_hash
from mutoracle.config import MutOracleConfig
from mutoracle.contracts import RAGRun
from mutoracle.prompts import build_rag_prompt
from mutoracle.provider import OpenRouterProvider
from mutoracle.retrieval import LexicalRetriever, Passage, load_corpus


class FixtureRAGPipeline:
    """Black-box RAG pipeline backed by a deterministic fixture corpus."""

    def __init__(
        self,
        *,
        config: MutOracleConfig,
        corpus_path: Path | None = None,
        use_remote: bool = False,
    ) -> None:
        self._config = config
        self._corpus = load_corpus(corpus_path)
        self._retriever = LexicalRetriever(self._corpus)
        self._ledger = SQLiteCacheLedger(config.runtime.cache_path)
        self._provider = (
            OpenRouterProvider(config, self._ledger) if use_remote else None
        )

    def run(self, query: str) -> RAGRun:
        """Run retrieval, prompt construction, and generation for one query."""

        hits = self._retriever.search(query, top_k=self._config.rag.top_k)
        passages = [hit.passage for hit in hits]
        prompt = build_rag_prompt(query, passages)

        if self._provider is None:
            answer = _deterministic_answer(query, passages)
            usage = _fixture_usage(prompt, answer)
            generation_metadata = {
                "provider": "fixture",
                "cache_hit": False,
                "model": "fixture-deterministic",
                "provider_route": "fixture",
                "prompt_hash": prompt_hash(prompt),
                "seed": self._config.runtime.seed,
                "temperature": 0.0,
                "usage": usage,
                "latency_seconds": 0.0,
                "estimated_cost_usd": 0.0,
            }
        else:
            completion = self._provider.complete(prompt)
            answer = completion.answer
            generation_metadata = completion.metadata

        return RAGRun(
            query=query,
            passages=[passage.text for passage in passages],
            answer=answer,
            metadata={
                "prompt": prompt,
                "retrieval": [
                    {
                        "id": hit.passage.id,
                        "title": hit.passage.title,
                        "score": hit.score,
                    }
                    for hit in hits
                ],
                "generation": generation_metadata,
                "latency": {
                    "retrieval_seconds": 0.0,
                    "generation_seconds": generation_metadata["latency_seconds"],
                },
                "cache_path": str(self._config.runtime.cache_path),
            },
        )

    def usage_summary(self) -> dict[str, int | float]:
        """Return provider usage totals as JSON-friendly data."""

        summary = self._ledger.usage_summary()
        return {
            "requests": summary.requests,
            "live_requests": summary.live_requests,
            "cache_hits": summary.cache_hits,
            "prompt_tokens": summary.prompt_tokens,
            "completion_tokens": summary.completion_tokens,
            "total_cost_usd": summary.total_cost_usd,
            "total_latency_seconds": summary.total_latency_seconds,
        }


def _deterministic_answer(query: str, passages: list[Passage]) -> str:
    if not passages:
        return "The provided context is insufficient to answer the question."
    first = passages[0]
    return (
        f"Based on {first.title}, {first.text} "
        f"This addresses the question: {query.strip()}"
    )


def _fixture_usage(prompt: str, answer: str) -> dict[str, int]:
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(answer)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

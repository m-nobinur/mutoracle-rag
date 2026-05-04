from __future__ import annotations

from mutoracle.pipeline.prompt import build_rag_prompt
from mutoracle.pipeline.rag import FixtureRAGPipeline
from mutoracle.pipeline.retriever import LexicalRetriever
from mutoracle.providers.openrouter_provider import OpenRouterProvider
from mutoracle.storage.sqlite_cache import SQLiteCacheLedger


def test_phase_two_module_layout_exports_expected_symbols() -> None:
    assert build_rag_prompt
    assert FixtureRAGPipeline
    assert LexicalRetriever
    assert OpenRouterProvider
    assert SQLiteCacheLedger

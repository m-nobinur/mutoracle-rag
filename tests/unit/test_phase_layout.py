from __future__ import annotations

from mutoracle.mutations import get_operator, mutation_registry
from mutoracle.oracles import LLMJudgeOracle, NLIOracle, SemanticSimilarityOracle
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


def test_phase_three_module_layout_exports_expected_symbols() -> None:
    assert get_operator("CI")
    assert set(mutation_registry()) == {"CI", "CR", "CS", "QP", "QN", "FS", "FA"}


def test_phase_four_module_layout_exports_expected_symbols() -> None:
    assert SemanticSimilarityOracle
    assert NLIOracle
    assert LLMJudgeOracle

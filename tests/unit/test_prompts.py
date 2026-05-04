from __future__ import annotations

from mutoracle.prompts import build_rag_prompt
from mutoracle.retrieval import Passage


def test_build_rag_prompt_is_stable() -> None:
    prompt = build_rag_prompt(
        "What is RAG?",
        [Passage(id="p1", title="RAG", text="RAG grounds answers in context.")],
    )

    assert "Question:\nWhat is RAG?" in prompt
    assert "[1] RAG\nRAG grounds answers in context." in prompt
    assert prompt.endswith("Answer:")

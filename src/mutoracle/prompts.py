"""Deterministic prompt construction for the RAG system under test."""

from __future__ import annotations

from mutoracle.retrieval import Passage


def build_rag_prompt(query: str, passages: list[Passage]) -> str:
    """Build a stable grounded-answer prompt from retrieved passages."""

    context = "\n\n".join(
        f"[{index}] {passage.title}\n{passage.text}"
        for index, passage in enumerate(passages, start=1)
    )
    return (
        "You are the generator inside MutOracle-RAG's system under test.\n"
        "Answer the question using only the provided context. "
        "If the context is insufficient, say so.\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

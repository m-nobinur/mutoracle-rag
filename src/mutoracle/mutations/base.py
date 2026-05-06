"""Shared helpers for deterministic RAG mutation operators."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from mutoracle.contracts import RAGRun, Stage

CANONICAL_OPERATOR_IDS = (
    "CI",
    "CR",
    "CS",
    "QP",
    "QN",
    "QD",
    "QI",
    "FS",
    "FA",
    "FE",
    "GN",
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")

STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "can",
    "could",
    "describe",
    "does",
    "do",
    "explain",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "please",
    "question",
    "the",
    "this",
    "to",
    "what",
    "why",
    "you",
}


def clone_run(
    run: RAGRun,
    *,
    query: str | None = None,
    passages: list[str] | None = None,
    answer: str | None = None,
    operator_id: str,
    operator_name: str,
    stage: Stage,
    rejected: bool = False,
    rejection_reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> RAGRun:
    """Return an immutable-style copy of a run with mutation metadata attached."""

    metadata = deepcopy(run.metadata)
    record: dict[str, Any] = {
        "operator_id": operator_id,
        "operator_name": operator_name,
        "stage": stage,
        "rejected": rejected,
    }
    if rejection_reason is not None:
        record["rejection_reason"] = rejection_reason
    if details:
        record["details"] = deepcopy(details)

    history = metadata.get("mutations", [])
    if not isinstance(history, list):
        history = []
    history = [*history, record]
    metadata["mutation"] = record
    metadata["mutations"] = history

    return RAGRun(
        query=run.query if query is None else query,
        passages=list(run.passages) if passages is None else list(passages),
        answer=run.answer if answer is None else answer,
        metadata=metadata,
    )


def content_similarity(left: str, right: str) -> float:
    """Return Jaccard similarity over non-stopword content tokens."""

    left_tokens = set(content_tokens(left))
    right_tokens = set(content_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def content_tokens(text: str) -> list[str]:
    """Tokenize text for mutation quality checks."""

    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    ]


def preserve_capitalization(source: str, replacement: str) -> str:
    """Apply simple capitalization from a matched span to its replacement."""

    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return f"{replacement[:1].upper()}{replacement[1:]}"
    return replacement

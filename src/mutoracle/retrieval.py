"""Local retrieval primitives for the Phase 2 RAG system under test."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Passage:
    """A fixture corpus passage."""

    id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    """A retrieved passage with its deterministic lexical score."""

    passage: Passage
    score: float


def load_corpus(path: Path | None = None) -> list[Passage]:
    """Load a corpus from JSON or the packaged Phase 2 fixture corpus."""

    if path is None:
        raw_text = (
            resources.files("mutoracle.fixtures")
            .joinpath("corpus.json")
            .read_text(encoding="utf-8")
        )
    else:
        raw_text = path.read_text(encoding="utf-8")

    raw = json.loads(raw_text)
    if not isinstance(raw, list):
        msg = "Corpus JSON must contain a list of passage objects."
        raise ValueError(msg)

    return [_parse_passage(item) for item in raw]


class LexicalRetriever:
    """Small deterministic TF-IDF retriever for smoke and fixture runs."""

    def __init__(self, corpus: list[Passage]) -> None:
        if not corpus:
            msg = "Retriever corpus must contain at least one passage."
            raise ValueError(msg)

        self._corpus = corpus
        self._doc_tokens = [
            _token_counts(f"{passage.title} {passage.text}") for passage in corpus
        ]
        self._doc_freq = _document_frequencies(self._doc_tokens)

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        """Return the top-k passages for a query."""

        if top_k < 1:
            msg = "top_k must be at least 1."
            raise ValueError(msg)

        query_tokens = _token_counts(query)
        scored = [
            RetrievalHit(passage=passage, score=self._score(query_tokens, doc_tokens))
            for passage, doc_tokens in zip(self._corpus, self._doc_tokens, strict=True)
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.passage.id))
        return scored[:top_k]

    def _score(self, query_tokens: dict[str, int], doc_tokens: dict[str, int]) -> float:
        score = 0.0
        total_docs = len(self._corpus)
        for token, query_count in query_tokens.items():
            if token not in doc_tokens:
                continue
            idf = math.log((1 + total_docs) / (1 + self._doc_freq[token])) + 1
            score += query_count * doc_tokens[token] * idf
        return round(score, 6)


def _parse_passage(item: Any) -> Passage:
    if not isinstance(item, dict):
        msg = "Corpus entries must be mappings."
        raise ValueError(msg)
    try:
        return Passage(
            id=str(item["id"]),
            title=str(item["title"]),
            text=str(item["text"]),
        )
    except KeyError as error:
        msg = f"Corpus entry missing required field: {error.args[0]}"
        raise ValueError(msg) from error


def _token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in TOKEN_PATTERN.findall(text.lower()):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _document_frequencies(doc_tokens: list[dict[str, int]]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for tokens in doc_tokens:
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies

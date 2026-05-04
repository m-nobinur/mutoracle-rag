"""Prompt-stage mutation operators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

from mutoracle.contracts import RAGRun, Stage
from mutoracle.mutations.base import clone_run, content_similarity

QUESTION_PUNCTUATION = re.compile(r"\?+\s*$")
TOKEN_OR_PUNCTUATION = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?|[^\w\s]")
QUESTION_WORDS = {"how", "what", "when", "where", "why", "which", "who"}
DO_AUXILIARIES = {"do", "does", "did", "can", "could", "should", "would", "will"}
BE_AUXILIARIES = {"is", "are", "was", "were", "has", "have", "had"}
AUXILIARIES = DO_AUXILIARIES | BE_AUXILIARIES


@dataclass(frozen=True)
class QueryParaphraseMutation:
    """Replace a query with a deterministic high-overlap paraphrase."""

    name: str = "Query Paraphrase"
    stage: Stage = "prompt"
    operator_id: str = "QP"
    minimum_similarity: float = 0.7

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        del rng
        candidate = _paraphrase(run.query)
        if candidate is None:
            return clone_run(
                run,
                operator_id=self.operator_id,
                operator_name=self.name,
                stage=self.stage,
                rejected=True,
                rejection_reason="no deterministic paraphrase pattern matched",
            )

        similarity = content_similarity(run.query, candidate)
        if similarity < self.minimum_similarity:
            return clone_run(
                run,
                operator_id=self.operator_id,
                operator_name=self.name,
                stage=self.stage,
                rejected=True,
                rejection_reason="paraphrase failed content-similarity gate",
                details={"candidate": candidate, "similarity": similarity},
            )

        return clone_run(
            run,
            query=candidate,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            details={"original_query": run.query, "similarity": similarity},
        )


@dataclass(frozen=True)
class QueryNegationMutation:
    """Negate the query while rejecting unsupported grammar shapes."""

    name: str = "Query Negation"
    stage: Stage = "prompt"
    operator_id: str = "QN"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        del rng
        candidate = _negate(run.query)
        if candidate is None or not _looks_grammatical_negation(candidate):
            return clone_run(
                run,
                operator_id=self.operator_id,
                operator_name=self.name,
                stage=self.stage,
                rejected=True,
                rejection_reason="query shape cannot be grammatically negated",
            )

        return clone_run(
            run,
            query=candidate,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            details={"original_query": run.query},
        )


def _paraphrase(query: str) -> str | None:
    stripped = QUESTION_PUNCTUATION.sub("", query.strip())
    lowered = stripped.lower()
    patterns = (
        ("what is ", "Explain "),
        ("what are ", "Explain "),
        ("how does ", "Describe how "),
        ("how do ", "Describe how "),
        ("why does ", "Explain why "),
        ("why do ", "Explain why "),
    )
    for prefix, replacement in patterns:
        if lowered.startswith(prefix):
            return f"{replacement}{stripped[len(prefix) :]}?"
    return None


def _negate(query: str) -> str | None:
    tokens = TOKEN_OR_PUNCTUATION.findall(query.strip())
    if len(tokens) < 3:
        return None

    lower_tokens = [token.lower() for token in tokens]
    auxiliary_index = next(
        (index for index, token in enumerate(lower_tokens) if token in AUXILIARIES),
        None,
    )
    if auxiliary_index is None:
        return None

    auxiliary = lower_tokens[auxiliary_index]
    if auxiliary in DO_AUXILIARIES:
        insert_index = _next_word_index(tokens, auxiliary_index + 1)
        if insert_index is None:
            return None
        insert_index += 1
    else:
        insert_index = auxiliary_index + 1

    if insert_index < len(tokens) and lower_tokens[insert_index] == "not":
        return None

    negated = [*tokens[:insert_index], "not", *tokens[insert_index:]]
    return _join_tokens(negated)


def _next_word_index(tokens: list[str], start: int) -> int | None:
    for index in range(start, len(tokens)):
        if re.search(r"[A-Za-z0-9]", tokens[index]):
            return index
    return None


def _looks_grammatical_negation(query: str) -> bool:
    tokens = TOKEN_OR_PUNCTUATION.findall(query)
    lower_tokens = [token.lower() for token in tokens]
    if "not" not in lower_tokens:
        return False
    not_index = lower_tokens.index("not")
    if not_index == len(tokens) - 1:
        return False
    if tokens[not_index + 1] in {"?", ".", ",", ";", ":"}:
        return False
    return not (
        lower_tokens[0] in QUESTION_WORDS
        and not any(token in AUXILIARIES for token in lower_tokens[:3])
    )


def _join_tokens(tokens: list[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif re.fullmatch(r"[^\w\s]", token) or text.endswith("-"):
            text += token
        else:
            text += f" {token}"
    return text

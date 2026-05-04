"""Generation-stage mutation operators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

from mutoracle.contracts import RAGRun, Stage
from mutoracle.mutations.base import clone_run, preserve_capitalization

SYNONYM_REPLACEMENTS = {
    "answer": "response",
    "confidence": "certainty",
    "context": "evidence",
    "dataset": "corpus",
    "fault": "defect",
    "faults": "defects",
    "generation": "answering",
    "hallucination": "unsupported claim",
    "oracle": "checker",
    "oracles": "checkers",
    "pipeline": "workflow",
    "query": "question",
    "retrieval": "search",
}

ANTONYM_REPLACEMENTS = {
    "accurate": "inaccurate",
    "complete": "incomplete",
    "detect": "miss",
    "detected": "missed",
    "faithful": "unfaithful",
    "localize": "obscure",
    "localizes": "obscures",
    "positive": "negative",
    "same": "different",
    "stable": "unstable",
    "sufficient": "insufficient",
    "supported": "unsupported",
}


@dataclass(frozen=True)
class FactoidSynonymSubstitutionMutation:
    """Substitute one supported answer span with a near-synonym."""

    name: str = "Factoid Synonym Substitution"
    stage: Stage = "generation"
    operator_id: str = "FS"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        return _replace_supported_span(
            run,
            rng=rng,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            replacements=SYNONYM_REPLACEMENTS,
            rejection_reason="no supported synonym span found in answer",
        )


@dataclass(frozen=True)
class FactoidAntonymSubstitutionMutation:
    """Substitute one supported answer span with an antonym."""

    name: str = "Factoid Antonym Substitution"
    stage: Stage = "generation"
    operator_id: str = "FA"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        return _replace_supported_span(
            run,
            rng=rng,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            replacements=ANTONYM_REPLACEMENTS,
            rejection_reason="no supported antonym span found in answer",
        )


def _replace_supported_span(
    run: RAGRun,
    *,
    rng: Random,
    operator_id: str,
    operator_name: str,
    stage: Stage,
    replacements: dict[str, str],
    rejection_reason: str,
) -> RAGRun:
    matches = _find_supported_matches(run.answer, replacements)
    if not matches:
        return clone_run(
            run,
            operator_id=operator_id,
            operator_name=operator_name,
            stage=stage,
            rejected=True,
            rejection_reason=rejection_reason,
        )

    match = matches[rng.randrange(len(matches))]
    replacement = preserve_capitalization(
        match.group(0),
        replacements[match.group(0).lower()],
    )
    answer = f"{run.answer[: match.start()]}{replacement}{run.answer[match.end() :]}"
    return clone_run(
        run,
        answer=answer,
        operator_id=operator_id,
        operator_name=operator_name,
        stage=stage,
        details={
            "original_span": match.group(0),
            "replacement": replacement,
            "span": [match.start(), match.end()],
        },
    )


def _find_supported_matches(
    text: str,
    replacements: dict[str, str],
) -> list[re.Match[str]]:
    alternatives = sorted(
        (re.escape(key) for key in replacements), key=len, reverse=True
    )
    pattern = re.compile(rf"\b({'|'.join(alternatives)})\b", flags=re.IGNORECASE)
    return list(pattern.finditer(text))

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

ENTITY_REPLACEMENTS = {
    "Apollo": "Gemini",
    "Elon Musk": "Jeff Bezos",
    "Guido van Rossum": "Dennis Ritchie",
    "Marie Curie": "Rosalind Franklin",
    "Python": "Java",
    "Transformer architecture": "recurrent neural network",
    "computer program": "database schema",
    "efficient similarity search": "relational indexing",
    "embedded": "serialized",
    "relational database": "document store",
    "retrieved context": "training data",
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


@dataclass(frozen=True)
class FactoidEntitySwapMutation:
    """Swap a supported named entity or domain phrase for a nearby distractor."""

    name: str = "Factoid Entity Swap"
    stage: Stage = "generation"
    operator_id: str = "FE"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        return _replace_supported_span(
            run,
            rng=rng,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            replacements=ENTITY_REPLACEMENTS,
            rejection_reason="no supported entity or domain phrase found in answer",
        )


@dataclass(frozen=True)
class AnswerNegationMutation:
    """Add a local negation to short factoid answers."""

    name: str = "Answer Negation"
    stage: Stage = "generation"
    operator_id: str = "GN"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        del rng
        answer = run.answer.strip()
        if not answer or answer.lower().startswith(("not ", "no ")):
            return clone_run(
                run,
                operator_id=self.operator_id,
                operator_name=self.name,
                stage=self.stage,
                rejected=True,
                rejection_reason="answer is empty or already negated",
            )
        auxiliary_pattern = (
            r"\b(is|are|was|were|has|have|had|can|should|would|will)\b"
        )
        if re.search(auxiliary_pattern, answer):
            candidate = re.sub(
                auxiliary_pattern,
                lambda match: f"{match.group(1)} not",
                answer,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            candidate = f"Not {answer}"
        return clone_run(
            run,
            answer=candidate,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            details={"original_answer": run.answer},
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
    replacement_map = {key.lower(): value for key, value in replacements.items()}
    replacement = preserve_capitalization(
        match.group(0),
        replacement_map[match.group(0).lower()],
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

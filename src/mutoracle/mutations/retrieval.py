"""Retrieval-stage mutation operators."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from mutoracle.contracts import RAGRun, Stage
from mutoracle.mutations.base import clone_run

DISTRACTOR_PASSAGES = (
    "Control distractor: The Pacific Ocean is the largest ocean on Earth.",
    "Control distractor: SQLite stores relational data in a local file.",
    "Control distractor: A leap year usually has 366 calendar days.",
)


@dataclass(frozen=True)
class ContextInjectionMutation:
    """Inject an unrelated passage into retrieved context."""

    name: str = "Context Injection"
    stage: Stage = "retrieval"
    operator_id: str = "CI"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        passages = list(run.passages)
        distractor = DISTRACTOR_PASSAGES[rng.randrange(len(DISTRACTOR_PASSAGES))]
        insert_at = rng.randrange(len(passages) + 1)
        passages.insert(insert_at, distractor)
        return clone_run(
            run,
            passages=passages,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            details={"inserted_index": insert_at, "inserted_passage": distractor},
        )


@dataclass(frozen=True)
class ContextRemovalMutation:
    """Remove one retrieved passage."""

    name: str = "Context Removal"
    stage: Stage = "retrieval"
    operator_id: str = "CR"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        if not run.passages:
            return clone_run(
                run,
                operator_id=self.operator_id,
                operator_name=self.name,
                stage=self.stage,
                rejected=True,
                rejection_reason="no passages available to remove",
            )

        passages = list(run.passages)
        removed_index = rng.randrange(len(passages))
        removed_passage = passages.pop(removed_index)
        return clone_run(
            run,
            passages=passages,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            details={
                "removed_index": removed_index,
                "removed_passage": removed_passage,
            },
        )


@dataclass(frozen=True)
class ContextShuffleMutation:
    """Shuffle retrieved passages while preserving passage text."""

    name: str = "Context Shuffle"
    stage: Stage = "retrieval"
    operator_id: str = "CS"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        if len(run.passages) < 2:
            return clone_run(
                run,
                operator_id=self.operator_id,
                operator_name=self.name,
                stage=self.stage,
                rejected=True,
                rejection_reason="at least two passages are required to shuffle",
            )

        passages = list(run.passages)
        before = list(passages)
        rng.shuffle(passages)
        if passages == before:
            passages = [*passages[1:], passages[0]]

        return clone_run(
            run,
            passages=passages,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
            details={"original_order": before, "shuffled_order": passages},
        )

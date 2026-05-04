"""Registry for canonical mutation operators."""

from __future__ import annotations

from typing import cast

from mutoracle.contracts import MutationOperator, Stage
from mutoracle.mutations.generation import (
    FactoidAntonymSubstitutionMutation,
    FactoidSynonymSubstitutionMutation,
)
from mutoracle.mutations.prompt import QueryNegationMutation, QueryParaphraseMutation
from mutoracle.mutations.retrieval import (
    ContextInjectionMutation,
    ContextRemovalMutation,
    ContextShuffleMutation,
)


def mutation_registry() -> dict[str, MutationOperator]:
    """Return the canonical Phase 3 operator registry keyed by operator ID."""

    operators: list[tuple[str, MutationOperator]] = [
        ("CI", cast("MutationOperator", ContextInjectionMutation())),
        ("CR", cast("MutationOperator", ContextRemovalMutation())),
        ("CS", cast("MutationOperator", ContextShuffleMutation())),
        ("QP", cast("MutationOperator", QueryParaphraseMutation())),
        ("QN", cast("MutationOperator", QueryNegationMutation())),
        ("FS", cast("MutationOperator", FactoidSynonymSubstitutionMutation())),
        ("FA", cast("MutationOperator", FactoidAntonymSubstitutionMutation())),
    ]
    return dict(operators)


def get_operator(operator_id: str) -> MutationOperator:
    """Return one operator by canonical ID."""

    registry = mutation_registry()
    normalized = operator_id.upper()
    try:
        return registry[normalized]
    except KeyError as error:
        valid = ", ".join(registry)
        msg = f"Unknown mutation operator '{operator_id}'. Valid IDs: {valid}."
        raise ValueError(msg) from error


def operators_by_stage(stage: Stage) -> dict[str, MutationOperator]:
    """Return canonical operators targeting one pipeline stage."""

    return {
        operator_id: operator
        for operator_id, operator in mutation_registry().items()
        if operator.stage == stage
    }


def list_operator_ids() -> list[str]:
    """Return canonical mutation IDs in registry order."""

    return list(mutation_registry())

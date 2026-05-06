from __future__ import annotations

from random import Random

from mutoracle.contracts import RAGRun
from mutoracle.mutations import (
    get_operator,
    list_operator_ids,
    mutation_registry,
    operators_by_stage,
)
from mutoracle.mutations.prompt import QueryParaphraseMutation


def _fixture_run(
    *,
    query: str = "How does MutOracle-RAG localize faults?",
    answer: str = (
        "MutOracle-RAG localizes hallucination faults by comparing oracle "
        "confidence across the retrieval pipeline."
    ),
    passages: list[str] | None = None,
) -> RAGRun:
    return RAGRun(
        query=query,
        passages=["retrieval context", "prompt context", "generation context"]
        if passages is None
        else passages,
        answer=answer,
        metadata={"generation": {"provider": "fixture"}},
    )


def test_registry_exposes_canonical_operator_ids_by_stage() -> None:
    assert list_operator_ids() == [
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
    ]
    assert set(operators_by_stage("retrieval")) == {"CI", "CR", "CS"}
    assert set(operators_by_stage("prompt")) == {"QP", "QN", "QD", "QI"}
    assert set(operators_by_stage("generation")) == {"FS", "FA", "FE", "GN"}


def test_all_operators_preserve_rag_run_schema_and_are_deterministic() -> None:
    run = _fixture_run()

    for operator_id, operator in mutation_registry().items():
        first = operator.apply(run, rng=Random(7))
        second = operator.apply(run, rng=Random(7))

        assert first == second
        assert isinstance(first.query, str), operator_id
        assert isinstance(first.passages, list), operator_id
        assert all(isinstance(passage, str) for passage in first.passages), operator_id
        assert isinstance(first.answer, str), operator_id
        assert first.metadata["mutation"]["operator_id"] == operator_id
        assert first.metadata["mutation"]["stage"] == operator.stage


def test_context_removal_handles_single_and_empty_passages() -> None:
    operator = get_operator("CR")

    one_passage = operator.apply(_fixture_run(passages=["only context"]), rng=Random(1))
    assert one_passage.passages == []
    assert one_passage.metadata["mutation"]["rejected"] is False

    empty = operator.apply(_fixture_run(passages=[]), rng=Random(1))
    assert empty.passages == []
    assert empty.metadata["mutation"]["rejected"] is True
    assert "no passages" in empty.metadata["mutation"]["rejection_reason"]


def test_query_paraphrase_rejects_low_similarity_candidate() -> None:
    operator = QueryParaphraseMutation(minimum_similarity=1.1)
    mutated = operator.apply(
        _fixture_run(query="What is MutOracle-RAG?"), rng=Random(1)
    )

    assert mutated.query == "What is MutOracle-RAG?"
    assert mutated.metadata["mutation"]["rejected"] is True
    assert "similarity" in mutated.metadata["mutation"]["rejection_reason"]


def test_query_negation_labels_invalid_outputs_for_rejection() -> None:
    mutated = get_operator("QN").apply(
        _fixture_run(query="Explain MutOracle-RAG fault localization."),
        rng=Random(1),
    )

    assert mutated.query == "Explain MutOracle-RAG fault localization."
    assert mutated.metadata["mutation"]["rejected"] is True
    assert "grammatically" in mutated.metadata["mutation"]["rejection_reason"]


def test_generation_operators_only_mutate_supported_factoid_spans() -> None:
    unsupported = _fixture_run(answer="Banana telescope river.")

    synonym_rejected = get_operator("FS").apply(unsupported, rng=Random(1))
    antonym_rejected = get_operator("FA").apply(unsupported, rng=Random(1))

    assert synonym_rejected.answer == unsupported.answer
    assert synonym_rejected.metadata["mutation"]["rejected"] is True
    assert antonym_rejected.answer == unsupported.answer
    assert antonym_rejected.metadata["mutation"]["rejected"] is True

    supported = _fixture_run()
    synonym = get_operator("FS").apply(supported, rng=Random(1))
    antonym = get_operator("FA").apply(supported, rng=Random(1))

    assert synonym.answer != supported.answer
    assert synonym.metadata["mutation"]["rejected"] is False
    assert antonym.answer != supported.answer
    assert antonym.metadata["mutation"]["rejected"] is False


def test_richer_prompt_and_generation_mutations_cover_factoid_queries() -> None:
    run = _fixture_run(
        query="Who founded SpaceX in 2002?",
        answer="Elon Musk founded SpaceX.",
    )

    detail_drop = get_operator("QD").apply(run, rng=Random(1))
    instruction = get_operator("QI").apply(run, rng=Random(1))
    entity_swap = get_operator("FE").apply(run, rng=Random(1))
    negation = get_operator("GN").apply(run, rng=Random(1))

    assert detail_drop.query == "Who founded SpaceX?"
    assert instruction.query.endswith("explicitly supported by context.")
    assert "Jeff Bezos" in entity_swap.answer
    assert "not" in negation.answer.lower() or negation.answer.startswith("Not ")

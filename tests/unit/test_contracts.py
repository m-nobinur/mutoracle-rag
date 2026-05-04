from __future__ import annotations

from mutoracle.contracts import FaultReport, RAGRun


def test_rag_run_has_empty_metadata_by_default() -> None:
    run = RAGRun(
        query="What is RAG?",
        passages=["context"],
        answer="retrieval augmented generation",
    )

    assert run.metadata == {}


def test_fault_report_defaults_to_empty_evidence() -> None:
    report = FaultReport(
        stage="no_fault_detected",
        confidence=0.0,
        deltas={},
        stage_deltas={"retrieval": 0.0, "prompt": 0.0, "generation": 0.0},
    )

    assert report.evidence == []

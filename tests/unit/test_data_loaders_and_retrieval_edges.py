from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutoracle.data import loaders
from mutoracle.data.loaders import (
    build_noise_pool,
    dataset_manifests,
    load_rgb_subset,
    load_triviaqa_subset,
)
from mutoracle.retrieval import LexicalRetriever, Passage, load_corpus


def test_source_loaders_read_jsonl_paths_and_limit(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "\n".join(
            [
                "",
                json.dumps(
                    {
                        "source_qid": "q1",
                        "source": "custom",
                        "query": "Who?",
                        "gt_answer": "Ada",
                        "supporting_doc_id": "doc1",
                        "supporting_passage": "Ada support.",
                        "distractor_doc_id": "doc2",
                        "distractor_passage": "Wrong.",
                    }
                ),
                json.dumps(
                    {
                        "source_qid": "q2",
                        "query": "What?",
                        "gt_answer": "Babbage",
                        "supporting_doc_id": "doc3",
                        "supporting_passage": "Babbage support.",
                        "distractor_doc_id": "doc4",
                        "distractor_passage": "Wrong.",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rgb = load_rgb_subset(source_path, limit=1)
    trivia = load_triviaqa_subset(source_path, limit=2)

    assert len(rgb) == 1
    assert rgb[0].source == "custom"
    assert [item.source_qid for item in trivia] == ["q1", "q2"]


def test_noise_pool_reads_json_and_validates_shape(tmp_path: Path) -> None:
    noise_path = tmp_path / "noise.json"
    noise_path.write_text(
        json.dumps(
            [
                {"doc_id": "n1", "text": "noise one"},
                {"doc_id": "n2", "text": "noise two"},
            ]
        ),
        encoding="utf-8",
    )
    invalid_path = tmp_path / "invalid-noise.json"
    invalid_path.write_text("{}", encoding="utf-8")

    assert build_noise_pool(noise_path, limit=1) == [
        {"doc_id": "n1", "text": "noise one"}
    ]
    with pytest.raises(ValueError, match="Noise pool"):
        build_noise_pool(invalid_path)


def test_dataset_manifest_uses_staged_rgb_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_revision": "rgb-revision",
                "records": {"source_en_jsonl": {"checksum": "a" * 64}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(loaders, "DEFAULT_RGB_MANIFEST_PATH", manifest_path)

    manifests = dataset_manifests(build_date="2026-05-05")

    assert manifests[0].revision == "rgb-revision"
    assert manifests[0].checksum == f"sha256:{'a' * 64}"

    manifest_path.write_text("{bad json", encoding="utf-8")
    assert dataset_manifests(build_date="2026-05-05")[0].revision


def test_retrieval_load_and_validation_edges(tmp_path: Path) -> None:
    invalid_shape = tmp_path / "invalid.json"
    invalid_shape.write_text("{}", encoding="utf-8")
    missing_field = tmp_path / "missing.json"
    missing_field.write_text(json.dumps([{"id": "x", "title": "X"}]), encoding="utf-8")
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps([{"id": "a", "title": "Ada", "text": "Ada wrote notes."}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="list"):
        load_corpus(invalid_shape)
    with pytest.raises(ValueError, match="missing required field"):
        load_corpus(missing_field)
    with pytest.raises(ValueError, match="at least one passage"):
        LexicalRetriever([])

    retriever = LexicalRetriever(load_corpus(valid))
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("Ada", top_k=0)
    assert retriever.search("missing token", top_k=1)[0].score == 0.0

    corpus = [
        Passage(id="b", title="Same", text="token"),
        Passage(id="a", title="Same", text="token"),
    ]
    assert LexicalRetriever(corpus).search("token", top_k=2)[0].passage.id == "a"

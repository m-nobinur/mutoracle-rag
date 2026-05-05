"""Deterministic Phase 6 source loaders.

The public functions are intentionally small and local-first. They provide
stable fixture-backed RGB, TriviaQA, and noise pools for CI and offline work,
while keeping the same shape expected from future Hugging Face `datasets`
downloads.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from mutoracle.data.manifest import DatasetManifest, sha256_text

DEFAULT_RGB_SOURCE_PATH = Path("data/rgb/rgb_v1.0.0/source_en.jsonl")
DEFAULT_RGB_MANIFEST_PATH = Path("data/rgb/manifest.json")


@dataclass(frozen=True)
class SourceExample:
    """One source question with support and distractor passages."""

    source_qid: str
    source: str
    query: str
    gt_answer: str
    supporting_doc_id: str
    supporting_passage: str
    distractor_doc_id: str
    distractor_passage: str


TOPIC_FACTS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "SpaceX",
        "Who founded SpaceX in 2002?",
        "Elon Musk",
        "spacex_founder",
        "Gwynne Shotwell",
    ),
    (
        "Python",
        "Who created the Python programming language?",
        "Guido van Rossum",
        "python_creator",
        "Dennis Ritchie",
    ),
    (
        "Ada Lovelace",
        "What is Ada Lovelace often credited with writing?",
        "the first computer program",
        "ada_program",
        "the first telephone patent",
    ),
    (
        "Apollo 11",
        "Which mission first landed humans on the Moon?",
        "Apollo 11",
        "apollo_landing",
        "Apollo 13",
    ),
    (
        "Marie Curie",
        "Which scientist won Nobel Prizes in both physics and chemistry?",
        "Marie Curie",
        "curie_nobel",
        "Rosalind Franklin",
    ),
    (
        "Django",
        "Which programming language is the Django web framework written in?",
        "Python",
        "django_language",
        "Ruby",
    ),
    (
        "FAISS",
        "What does FAISS provide for vector search?",
        "efficient similarity search",
        "faiss_vector_search",
        "relational transaction logging",
    ),
    (
        "RAG",
        "What does retrieval-augmented generation combine with text generation?",
        "retrieved context",
        "rag_context",
        "manual database migrations",
    ),
    (
        "SQLite",
        "Which type of database is SQLite?",
        "an embedded relational database",
        "sqlite_embedded",
        "a hosted vector database",
    ),
    (
        "Transformers",
        "Which architecture popularized attention-based language models?",
        "the Transformer architecture",
        "transformer_attention",
        "the MapReduce architecture",
    ),
)


def load_rgb_subset(
    path: Path | None = None,
    *,
    limit: int = 150,
) -> list[SourceExample]:
    """Load a local RGB subset or return deterministic fixture examples."""

    resolved_path = path
    if resolved_path is None and DEFAULT_RGB_SOURCE_PATH.exists():
        resolved_path = DEFAULT_RGB_SOURCE_PATH
    if resolved_path is not None and resolved_path.exists():
        return _read_source_examples(resolved_path, source="rgb", limit=limit)
    return _fixture_examples(source="rgb", limit=limit)


def load_triviaqa_subset(
    path: Path | None = None,
    *,
    limit: int = 150,
) -> list[SourceExample]:
    """Load a local TriviaQA subset or return deterministic fixture examples."""

    if path is not None and path.exists():
        return _read_source_examples(path, source="triviaqa", limit=limit)
    return _fixture_examples(source="triviaqa", limit=limit)


def build_noise_pool(
    path: Path | None = None,
    *,
    limit: int = 300,
) -> list[dict[str, str]]:
    """Load or build a deterministic noise/distractor passage pool."""

    if path is not None and path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            msg = f"Noise pool must be a JSON list: {path}"
            raise ValueError(msg)
        return [
            {"doc_id": str(item["doc_id"]), "text": str(item["text"])}
            for item in raw[:limit]
        ]

    passages: list[dict[str, str]] = []
    for index in range(limit):
        topic, _, answer, fact_id, distractor = TOPIC_FACTS[index % len(TOPIC_FACTS)]
        passages.append(
            {
                "doc_id": f"noise_{index:05d}_{fact_id}",
                "text": (
                    f"Noise passage {index} mentions {topic} but emphasizes "
                    f"{distractor}, not the ground-truth answer {answer}."
                ),
            }
        )
    return passages


def dataset_manifests(*, build_date: str | None = None) -> list[DatasetManifest]:
    """Return Phase 6 source manifests with required provenance fields."""

    manifest_date = build_date or date.today().isoformat()
    rgb_staging = _read_staged_rgb_manifest()
    rgb_preview = "\n".join(
        json.dumps(asdict(example), sort_keys=True)
        for example in load_rgb_subset(limit=10)
    )
    trivia_preview = "\n".join(
        json.dumps(asdict(example), sort_keys=True)
        for example in load_triviaqa_subset(limit=10)
    )
    noise_preview = "\n".join(
        json.dumps(item, sort_keys=True) for item in build_noise_pool(limit=10)
    )
    rgb_checksum = sha256_text(rgb_preview)
    rgb_revision = "phase6-fixture-preview"
    rgb_notes = (
        "Primary RAG evaluation source. The offline build uses a "
        "schema-compatible fixture preview until raw downloads are staged."
    )
    if rgb_staging is not None:
        checksum = rgb_staging.get("checksum")
        if isinstance(checksum, str) and checksum.strip():
            checksum_value = checksum.strip()
            rgb_checksum = (
                checksum_value
                if checksum_value.startswith("sha256:")
                else f"sha256:{checksum_value}"
            )
        revision = rgb_staging.get("source_revision")
        if isinstance(revision, str) and revision.strip():
            rgb_revision = revision.strip()
        rgb_notes = (
            "Primary RAG evaluation source staged from the RGB benchmark "
            "repository with local source and Phase 8 split artifacts."
        )

    return [
        DatasetManifest(
            dataset_id="rgb",
            name="RGB Benchmark",
            url="https://github.com/chen700564/RGB",
            license="source license; verify upstream before redistribution",
            revision=rgb_revision,
            checksum=rgb_checksum,
            date=manifest_date,
            notes=rgb_notes,
        ),
        DatasetManifest(
            dataset_id="triviaqa",
            name="TriviaQA filtered subset",
            url="https://huggingface.co/datasets/mandarjoshi/trivia_qa",
            license="Apache-2.0",
            revision="phase6-fixture-preview",
            checksum=sha256_text(trivia_preview),
            date=manifest_date,
            notes=(
                "Generalization and FITS source. The Phase 6 loader keeps a "
                "local deterministic subset for offline reproducibility."
            ),
        ),
        DatasetManifest(
            dataset_id="wikipedia_noise",
            name="Wikipedia/noise passage pool",
            url="https://huggingface.co/datasets/wikimedia/wikipedia",
            license="CC BY-SA 3.0",
            revision="phase6-fixture-preview",
            checksum=sha256_text(noise_preview),
            date=manifest_date,
            notes="Distractor pool used for CI and FITS generation faults.",
        ),
    ]


def _fixture_examples(*, source: str, limit: int) -> list[SourceExample]:
    examples: list[SourceExample] = []
    for index in range(limit):
        topic, query, answer, fact_id, distractor = TOPIC_FACTS[
            index % len(TOPIC_FACTS)
        ]
        variant = index // len(TOPIC_FACTS)
        source_qid = f"{source}_{index:05d}"
        supporting_doc_id = f"{source}_support_{index:05d}_{fact_id}"
        distractor_doc_id = f"{source}_distractor_{index:05d}_{fact_id}"
        examples.append(
            SourceExample(
                source_qid=source_qid,
                source=source,
                query=_variant_query(query, variant),
                gt_answer=answer,
                supporting_doc_id=supporting_doc_id,
                supporting_passage=(
                    f"{topic} reference {variant}: {query} The supported "
                    f"answer is {answer}."
                ),
                distractor_doc_id=distractor_doc_id,
                distractor_passage=(
                    f"{topic} distractor {variant}: a plausible but wrong "
                    f"answer is {distractor}."
                ),
            )
        )
    return examples


def _variant_query(query: str, variant: int) -> str:
    if variant == 0:
        return query
    return f"{query} (source variant {variant})"


def _read_source_examples(
    path: Path,
    *,
    source: str,
    limit: int,
) -> list[SourceExample]:
    examples: list[SourceExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        examples.append(
            SourceExample(
                source_qid=str(item["source_qid"]),
                source=str(item.get("source", source)),
                query=str(item["query"]),
                gt_answer=str(item["gt_answer"]),
                supporting_doc_id=str(item["supporting_doc_id"]),
                supporting_passage=str(item["supporting_passage"]),
                distractor_doc_id=str(item["distractor_doc_id"]),
                distractor_passage=str(item["distractor_passage"]),
            )
        )
        if len(examples) >= limit:
            break
    return examples


def _read_staged_rgb_manifest() -> dict[str, Any] | None:
    if not DEFAULT_RGB_MANIFEST_PATH.exists():
        return None
    try:
        raw = json.loads(DEFAULT_RGB_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None

    records = raw.get("records")
    if not isinstance(records, dict):
        return None
    source_info = records.get("source_en_jsonl")
    if not isinstance(source_info, dict):
        return None
    return {
        "source_revision": raw.get("source_revision"),
        "checksum": source_info.get("checksum"),
    }

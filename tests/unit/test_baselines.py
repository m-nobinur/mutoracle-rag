from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mutoracle.baselines import (
    BaselineExample,
    LabeledScore,
    MetaRAGBaseline,
    NoopVariantGenerator,
    RagasBaseline,
    SentenceClaimExtractor,
    SimpleMetamorphicVariantGenerator,
    run_baselines,
    tune_threshold_validation_only,
    write_baseline_outputs,
)
from mutoracle.baselines.metarag_baseline import (
    LexicalNLIBackend,
    NLIClaimVerifier,
    SpacyClaimExtractor,
    _entailment_probability,
    _is_variant_violation,
    _wordnet_replacement,
    _wordnet_variants,
    score_claims,
)
from mutoracle.contracts import RAGRun


class StubVerifier:
    model_id = "stub-nli"

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, str]] = []

    def score_claim(self, *, context: str, claim: str) -> float:
        self.calls.append((context, claim))
        return self._scores[len(self.calls) - 1]


class StubRagasScorer:
    model_id = "stub-ragas-judge"

    def score(self, run: RAGRun, *, reference: str | None = None) -> float:
        assert reference == "Ada Lovelace"
        assert run.query == "Who wrote the notes?"
        return 0.72


class StubRagasMetricScorer(StubRagasScorer):
    def score_metrics(
        self,
        run: RAGRun,
        *,
        reference: str | None = None,
    ) -> dict[str, float]:
        assert reference == "Ada Lovelace"
        assert run.query == "Who wrote the notes?"
        return {
            "faithfulness": 0.72,
            "answer_relevancy": 0.81,
            "context_precision": 0.64,
            "context_recall": 0.58,
        }


def test_ragas_consumes_same_ragrun_fixture_objects() -> None:
    run = _run()
    result = RagasBaseline(scorer=StubRagasMetricScorer()).run(
        run,
        threshold=0.6,
        reference="Ada Lovelace",
    )

    assert result.baseline_name == "ragas"
    assert result.query == run.query
    assert result.score == 0.72
    assert result.predicted_label == "faithful"
    assert result.model_ids == ["fixture", "stub-ragas-judge"]
    assert result.latency_seconds >= 0.0
    assert result.cost_usd == 0.0
    assert result.run_id
    assert result.metadata["cost_scope"] == "generation_only"
    assert result.metadata["latency_breakdown_seconds"]["generation"] == 0.0
    assert result.metadata["cost_breakdown_usd"]["generation"] == 0.0
    assert result.scores == {
        "faithfulness": 0.72,
        "answer_relevancy": 0.81,
        "context_precision": 0.64,
        "context_recall": 0.58,
    }


def test_metarag_approximation_scores_claims_with_nli_verifier() -> None:
    run = _run(answer="Ada wrote the notes. The notes described an algorithm.")
    verifier = StubVerifier([0.9, 0.2])
    baseline = MetaRAGBaseline(
        verifier=verifier,
        variant_generator=NoopVariantGenerator(),
    )

    result = baseline.run(run, threshold=0.75)

    assert result.baseline_name == "metarag"
    assert result.score == 0.5
    assert result.predicted_label == "hallucinated"
    assert result.metadata["claim_count"] == 2
    assert result.metadata["supported_claims"] == 1
    assert result.metadata["cost_scope"] == "generation_only"
    assert result.model_ids == ["fixture", "stub-nli"]
    assert result.cost_usd == 0.0
    assert len(verifier.calls) == 2


def test_metarag_generates_and_scores_metamorphic_variants() -> None:
    run = _run(answer="Ada wrote 2 notes.")
    verifier = StubVerifier([0.9, 0.7, 0.9])
    baseline = MetaRAGBaseline(
        extractor=SentenceClaimExtractor(min_words=3),
        verifier=verifier,
        variant_generator=SimpleMetamorphicVariantGenerator(max_variants_per_claim=2),
    )

    result = baseline.run(run, threshold=0.5)

    assert result.metadata["variant_count"] == 2
    assert result.metadata["variant_violations"] == 1
    assert result.score == 0.5
    assert [call[1] for call in verifier.calls] == [
        "Ada wrote 2 notes",
        "Ada authored 2 notes",
        "Ada wrote 3 notes",
    ]


def test_metarag_approximation_handles_empty_claim_sets() -> None:
    run = _run(answer="Yes.")
    verifier = StubVerifier([])
    baseline = MetaRAGBaseline(
        extractor=SentenceClaimExtractor(min_words=3),
        verifier=verifier,
        variant_generator=NoopVariantGenerator(),
    )

    result = baseline.run(run)

    assert result.score == 1.0
    assert result.predicted_label == "faithful"
    assert result.metadata["empty_claim_set"] is True
    assert verifier.calls == []


def test_thresholds_are_learned_only_from_validation_data() -> None:
    calibration = tune_threshold_validation_only(
        [
            LabeledScore(score=0.9, expected_label="faithful"),
            LabeledScore(score=0.8, expected_label="faithful"),
            LabeledScore(score=0.2, expected_label="hallucinated"),
            LabeledScore(score=0.1, expected_label="hallucinated"),
        ]
    )

    assert 0.2 < calibration.threshold < 0.8
    assert calibration.validation_f1 == 1.0
    with pytest.raises(ValueError, match="validation split"):
        tune_threshold_validation_only(
            [
                LabeledScore(
                    score=0.1,
                    expected_label="hallucinated",
                    split="test",
                )
            ]
        )


def test_shared_runner_writes_comparable_result_schema(tmp_path: Path) -> None:
    run = _run()
    baseline = MetaRAGBaseline(
        verifier=StubVerifier([1.0]),
        variant_generator=NoopVariantGenerator(),
    )
    results = run_baselines(
        examples=[BaselineExample(run=run)],
        baselines=[baseline],
        thresholds={"metarag": 0.5},
    )
    manifest = write_baseline_outputs(
        results=results,
        output_path=tmp_path / "baselines.jsonl",
        thresholds={"metarag": 0.5},
        metadata={"split": "smoke"},
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "baselines.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert manifest.result_schema == "mutoracle.baseline_result.v1"
    assert rows[0]["baseline_name"] == "metarag"
    assert {"latency_seconds", "cost_usd", "model_ids", "run_id"} <= set(rows[0])
    assert (tmp_path / "baselines.manifest.json").exists()


def test_metarag_records_generation_metadata_for_comparability() -> None:
    run = _run(
        generation_cost=0.07,
        generation_latency=0.2,
        generation_model="fixture-generation-model",
    )
    baseline = MetaRAGBaseline(
        verifier=StubVerifier([1.0]),
        variant_generator=NoopVariantGenerator(),
    )

    result = baseline.run(run, threshold=0.5)

    assert result.cost_usd == 0.07
    assert result.latency_seconds >= 0.2
    assert result.model_ids == ["fixture-generation-model", "stub-nli"]
    assert result.metadata["latency_breakdown_seconds"]["generation"] == 0.2
    assert result.metadata["cost_breakdown_usd"]["generation"] == 0.07


def test_metarag_validation_and_helper_branches() -> None:
    with pytest.raises(ValueError, match="entailment_threshold"):
        MetaRAGBaseline(entailment_threshold=1.2)

    assert SentenceClaimExtractor(min_words=3).extract("No. Ada wrote notes.") == [
        "Ada wrote notes"
    ]
    assert LexicalNLIBackend().probabilities(premise="anything", hypothesis="") == {
        "entailment": 1.0,
        "neutral": 0.0,
    }
    assert _entailment_probability({"neutral": 1.0}) == 0.0
    assert _is_variant_violation(
        SimpleNamespace(expected_supported=True),
        score=0.1,
        threshold=0.5,
    )
    assert _is_variant_violation(
        SimpleNamespace(expected_supported=False),
        score=0.9,
        threshold=0.5,
    )

    class ProbBackend:
        def probabilities(self, *, premise: str, hypothesis: str) -> dict[str, float]:
            assert (premise, hypothesis) == ("context", "claim")
            return {"entailment": 0.4}

    verifier = NLIClaimVerifier(backend=ProbBackend(), model_id="stub")
    assert score_claims(claims=["claim"], context="context", verifier=verifier) == [0.4]


def test_spacy_claim_extractor_falls_back_and_uses_sentence_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_spacy = ModuleType("spacy")

    def raise_os_error(model_name: str):
        del model_name
        raise OSError("missing model")

    missing_spacy.load = raise_os_error
    monkeypatch.setitem(sys.modules, "spacy", missing_spacy)
    assert SpacyClaimExtractor().extract("Ada wrote notes.") == ["Ada wrote notes"]

    class Sentence:
        def __init__(self, text: str) -> None:
            self.text = text

    class NLP:
        def __call__(self, answer: str):
            del answer
            return SimpleNamespace(
                sents=[Sentence("Ada wrote notes."), Sentence("No.")]
            )

    good_spacy = ModuleType("spacy")
    good_spacy.load = lambda model_name: NLP()
    monkeypatch.setitem(sys.modules, "spacy", good_spacy)
    assert SpacyClaimExtractor().extract("ignored") == ["Ada wrote notes"]


def test_wordnet_variant_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class Lemma:
        def __init__(self, name: str, antonyms: list[object] | None = None) -> None:
            self._name = name
            self._antonyms = antonyms or []

        def name(self) -> str:
            return self._name

        def antonyms(self) -> list[object]:
            return self._antonyms

    class Synset:
        def lemmas(self) -> list[object]:
            return [Lemma("authored", [Lemma("unwrote")])]

    class WordNet:
        def synsets(self, word: str) -> list[object]:
            if word == "lookup":
                raise LookupError("missing")
            return [Synset()]

    corpus = ModuleType("nltk.corpus")
    corpus.wordnet = WordNet()
    nltk = ModuleType("nltk")
    nltk.corpus = corpus
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setitem(sys.modules, "nltk.corpus", corpus)

    variants = _wordnet_variants("Ada wrote")
    assert {variant.kind for variant in variants} == {"synonym", "antonym"}
    assert _wordnet_replacement("lookup", wn=WordNet(), antonym=False) is None
    assert _wordnet_replacement("authored", wn=WordNet(), antonym=False) is None


def _run(
    *,
    answer: str = "Ada Lovelace wrote notes about the Analytical Engine.",
    generation_cost: float = 0.0,
    generation_latency: float = 0.0,
    generation_model: str = "fixture",
) -> RAGRun:
    return RAGRun(
        query="Who wrote the notes?",
        passages=["Ada Lovelace wrote notes about the Analytical Engine."],
        answer=answer,
        metadata={
            "generation": {
                "estimated_cost_usd": generation_cost,
                "latency_seconds": generation_latency,
                "model": generation_model,
            }
        },
    )

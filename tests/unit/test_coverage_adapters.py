from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mutoracle import provider as provider_module
from mutoracle.baselines import RagasBaseline
from mutoracle.baselines import ragas_baseline as ragas_module
from mutoracle.baselines.ragas_baseline import (
    OfficialRagasFaithfulnessScorer,
    _ragas_metric,
    _ragas_metric_scores,
    _score_metric,
)
from mutoracle.cache import SQLiteCacheLedger
from mutoracle.config import (
    CostConfig,
    MutOracleConfig,
    OpenRouterConfig,
    RuntimeConfig,
)
from mutoracle.contracts import RAGRun
from mutoracle.oracles.nli import (
    NLIOracle,
    TransformersNLIBackend,
    _entailment_probability,
    _flatten_pipeline_output,
)
from mutoracle.oracles.semantic import (
    SemanticSimilarityOracle,
    SentenceTransformerBackend,
)
from mutoracle.provider import (
    OpenRouterProvider,
    _estimate_cost_usd,
    _extract_answer,
    _extract_usage,
)
from mutoracle.retrieval import Passage
from mutoracle.storage import faiss_index as faiss_module
from mutoracle.storage.faiss_index import FaissIndex


class SequenceEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        start = self.calls
        self.calls += len(texts)
        return self._vectors[start : start + len(texts)]


class LiveProvider(OpenRouterProvider):
    def _request_completion(
        self,
        prompt: str,
        api_key: str,
        *,
        model: str,
        temperature: float,
    ) -> dict[str, object]:
        assert prompt == "live prompt"
        assert api_key == "test-key"
        assert model == "test-model"
        assert temperature == 0.2
        return {
            "choices": [{"message": {"content": [{"text": " live answer "}]}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


def test_provider_live_completion_records_usage_and_cache(tmp_path: Path) -> None:
    config = MutOracleConfig(
        openrouter=OpenRouterConfig(api_key="test-key"),
        cost=CostConfig(
            prompt_cost_per_1m_tokens=1.0,
            completion_cost_per_1m_tokens=2.0,
        ),
        runtime=RuntimeConfig(cache_path=tmp_path / "cache.db"),
    )
    ledger = SQLiteCacheLedger(config.runtime.cache_path)

    result = LiveProvider(config, ledger).complete(
        "live prompt",
        model="test-model",
        temperature=0.2,
        request_kind="unit",
    )

    assert result.answer == "live answer"
    assert result.metadata["request_kind"] == "unit"
    assert result.metadata["usage"]["total_tokens"] == 15
    assert result.metadata["estimated_cost_usd"] == pytest.approx(0.00002)
    assert ledger.usage_summary().live_requests == 1


def test_provider_requires_api_key_and_enforces_cost_budget(tmp_path: Path) -> None:
    missing_key = MutOracleConfig(runtime=RuntimeConfig(cache_path=tmp_path / "a.db"))
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(
            missing_key,
            SQLiteCacheLedger(missing_key.runtime.cache_path),
        ).complete("uncached")

    config = MutOracleConfig(
        openrouter=OpenRouterConfig(api_key="test-key"),
        cost=CostConfig(max_cost_usd=0.01),
        runtime=RuntimeConfig(cache_path=tmp_path / "b.db"),
    )
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    ledger.record_usage(
        model=config.models.generator,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.01,
        latency_seconds=0.1,
        cache_hit=False,
    )
    with pytest.raises(RuntimeError, match="Remote cost budget exhausted"):
        OpenRouterProvider(config, ledger).complete("uncached")


def test_provider_response_parsing_helpers_cover_edge_cases() -> None:
    assert (
        _extract_answer({"choices": [{"message": {"content": [" a ", {"text": "b"}]}}]})
        == "a\nb"
    )
    assert _extract_usage({}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert (
        _extract_usage({"usage": {"prompt_tokens": 2, "completion_tokens": 3}})[
            "total_tokens"
        ]
        == 5
    )
    assert (
        _estimate_cost_usd(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            prompt_cost_per_1m=1.0,
            completion_cost_per_1m=2.0,
        )
        == 2.0
    )
    for response, message in (
        ({}, "choices"),
        ({"choices": [None]}, "choice must be an object"),
        ({"choices": [{}]}, "message object"),
        ({"choices": [{"message": {"content": ""}}]}, "text content"),
    ):
        with pytest.raises(RuntimeError, match=message):
            _extract_answer(response)


def test_provider_request_completion_builds_client_and_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Completion:
        def model_dump(self, mode: str):
            assert mode == "json"
            return {"choices": [{"message": {"content": "ok"}}]}

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return Completion()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(provider_module, "OpenAI", FakeOpenAI)
    config = MutOracleConfig(
        openrouter=OpenRouterConfig(
            api_key="key",
            app_url="https://example.test",
            timeout_seconds=7,
        )
    )

    response = OpenRouterProvider(
        config,
        SQLiteCacheLedger(tmp_path / "cache.db"),
    )._request_completion("prompt", "key", model="model", temperature=0.0)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert calls[0]["client"]["timeout"] == 7
    assert calls[1]["extra_headers"]["HTTP-Referer"] == "https://example.test"


def test_provider_request_completion_rejects_non_mapping_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completion:
        def model_dump(self, mode: str):
            del mode
            return []

    class Completions:
        def create(self, **kwargs):
            del kwargs
            return Completion()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(provider_module, "OpenAI", FakeOpenAI)

    with pytest.raises(RuntimeError, match="JSON object"):
        OpenRouterProvider(
            MutOracleConfig(),
            SQLiteCacheLedger(tmp_path / "cache.db"),
        )._request_completion("prompt", "key", model="model", temperature=0.0)


def test_ragas_official_scorer_metric_paths_and_validation() -> None:
    class ValueMetric:
        def score(self, **payload):
            assert payload["reference"] == "ref"
            return SimpleNamespace(value=1.4)

    run = _run()
    scorer = OfficialRagasFaithfulnessScorer(
        config=MutOracleConfig(),
        metric={"faithfulness": ValueMetric()},
    )

    assert scorer.score(run, reference="ref") == 1.0
    assert scorer.score_metrics(run, reference="ref") == {"faithfulness": 1.0}
    with pytest.raises(RuntimeError, match="official ragas package"):
        OfficialRagasFaithfulnessScorer(config=MutOracleConfig())


def test_ragas_build_metrics_with_stubbed_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ragas = ModuleType("ragas")
    llms = ModuleType("ragas.llms")
    llms.llm_factory = lambda model, client: f"llm:{model}"
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms)
    monkeypatch.setattr(
        ragas_module,
        "_ragas_metric",
        lambda *names, llm: {"names": names, "llm": llm},
    )

    metrics = OfficialRagasFaithfulnessScorer(
        config=MutOracleConfig(openrouter=OpenRouterConfig(api_key="key"))
    )._metrics

    assert set(metrics) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }


def test_ragas_private_metric_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class ScoreOnly:
        model_id = "score-only"

        def score(self, run: RAGRun, *, reference: str | None = None) -> float:
            assert reference == "ref"
            return -0.5

    class MissingFaithfulness:
        model_id = "bad"

        def score_metrics(self, run: RAGRun, *, reference: str | None = None):
            return {"context_precision": 0.5}

    assert _ragas_metric_scores(ScoreOnly(), _run(), reference="ref") == {
        "faithfulness": 0.0
    }
    with pytest.raises(ValueError, match="faithfulness"):
        _ragas_metric_scores(MissingFaithfulness(), _run(), reference=None)

    class AsyncMetric:
        async def ascore(self, **payload):
            assert payload["user_input"] == "q"
            return 0.6

    assert (
        _score_metric(
            AsyncMetric(),
            user_input="q",
            response="a",
            retrieved_contexts=["c"],
            reference=None,
        )
        == 0.6
    )

    ragas = ModuleType("ragas")

    class SingleTurnSample:
        def __init__(self, **payload):
            self.payload = payload

    ragas.SingleTurnSample = SingleTurnSample
    monkeypatch.setitem(sys.modules, "ragas", ragas)

    class LegacyMetric:
        async def single_turn_ascore(self, sample):
            assert sample.payload["reference"] == "ref"
            return 0.7

    assert (
        _score_metric(
            LegacyMetric(),
            user_input="q",
            response="a",
            retrieved_contexts=["c"],
            reference="ref",
        )
        == 0.7
    )
    with pytest.raises(TypeError, match="Unsupported RAGAS"):
        _score_metric(
            object(),
            user_input="q",
            response="a",
            retrieved_contexts=["c"],
            reference=None,
        )

    class Faithfulness:
        def __init__(self, *, llm):
            self.llm = llm

    monkeypatch.setattr(
        ragas_module,
        "import_module",
        lambda name: SimpleNamespace(Faithfulness=Faithfulness),
    )
    assert isinstance(_ragas_metric("Faithfulness", llm="llm"), Faithfulness)
    with pytest.raises(RuntimeError, match="does not expose"):
        _ragas_metric("MissingMetric", llm="llm")


def test_ragas_baseline_fallback_scorer_metadata() -> None:
    class ScalarScorer:
        model_id = "scalar-ragas"

        def score(self, run: RAGRun, *, reference: str | None = None) -> float:
            assert reference == "ref"
            return 0.25

    result = RagasBaseline(scorer=ScalarScorer()).run(
        _run(), threshold=0.5, reference="ref"
    )

    assert result.predicted_label == "hallucinated"
    assert result.scores == {"faithfulness": 0.25}
    assert result.metadata["headline_metric"] == "faithfulness"


def test_faiss_index_validation_fallback_and_native_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="at least one passage"):
        FaissIndex([], SequenceEmbedder([]))
    with pytest.raises(ValueError, match="non-empty"):
        FaissIndex([_passage("a")], SequenceEmbedder([[]]))
    with pytest.raises(ValueError, match="consistent"):
        FaissIndex(
            [_passage("a"), _passage("b")], SequenceEmbedder([[1.0], [1.0, 2.0]])
        )

    index = FaissIndex(
        [_passage("a"), _passage("b")], SequenceEmbedder([[1, 0], [0, 1], [1, 0]])
    )
    with pytest.raises(ValueError, match="top_k"):
        index.search("query", top_k=0)
    assert index.search("query", top_k=2)[0].passage.id == "a"
    index._native_index = None
    with pytest.raises(RuntimeError, match="Native FAISS"):
        index._native_search([1.0, 0.0], top_k=1)

    class FakeArray(list):
        shape = (2, 2)

    class FakeNumpy:
        def asarray(self, values, dtype):
            del dtype
            return FakeArray(values)

    class FakeIndex:
        def __init__(self, dim):
            self.dim = dim

        def add(self, vectors):
            self.vectors = vectors

        def search(self, query_array, top_k):
            del query_array, top_k
            return [[0.9, 0.1]], [[1, 0]]

    class FakeFaiss:
        def __init__(self) -> None:
            self.IndexFlatIP = FakeIndex

        def normalize_L2(self, values):  # noqa: N802 - mirrors faiss API.
            self.normalized = values

    fake_faiss = FakeFaiss()

    def fake_import(name: str):
        if name == "faiss":
            return fake_faiss
        if name == "numpy":
            return FakeNumpy()
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(faiss_module, "import_module", fake_import)
    native = FaissIndex(
        [_passage("a"), _passage("b")], SequenceEmbedder([[1, 0], [0, 1], [0, 1]])
    )
    assert native.search("query", top_k=2)[0].passage.id == "b"


def test_nli_and_semantic_lazy_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = ModuleType("transformers")

    def pipeline(task, model, top_k):
        assert (task, model, top_k) == ("text-classification", "fake-nli", None)
        return lambda payload: [[{"label": "ENTAILMENT", "score": 0.8}]]

    transformers.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert TransformersNLIBackend("fake-nli").probabilities(
        premise="premise",
        hypothesis="hypothesis",
    ) == {"entailment": 0.8}
    assert _flatten_pipeline_output({"label": "neutral", "score": 1.0}) == [
        {"label": "neutral", "score": 1.0}
    ]
    assert _flatten_pipeline_output("bad") == []
    assert _entailment_probability({"not_entailment": 2.0}) == 1.0
    assert _entailment_probability({"neutral": 1.0}) == 0.0

    sentence_transformers = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is False
            return [[float(index), 1.0] for index, _ in enumerate(texts)]

    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    assert SentenceTransformerBackend("fake-embed").encode(["a", "b"]) == [
        [0.0, 1.0],
        [1.0, 1.0],
    ]


def test_oracle_empty_inputs_return_structured_zero_scores(tmp_path: Path) -> None:
    empty = RAGRun(query="q", passages=[], answer="")
    ledger = SQLiteCacheLedger(tmp_path / "cache.db")

    nli = NLIOracle(ledger=ledger, backend=object(), model_name="fake")
    semantic = SemanticSimilarityOracle(
        ledger=ledger, backend=object(), model_name="fake"
    )

    assert nli.score_result(empty).metadata["reason"] == "empty_premise_or_hypothesis"
    assert semantic.score_result(empty).metadata["reason"] == "empty_context_or_answer"


def _run() -> RAGRun:
    return RAGRun(
        query="q",
        passages=["c"],
        answer="a",
        metadata={
            "generation": {
                "model": "fixture",
                "estimated_cost_usd": 0.1,
                "latency_seconds": 0.2,
            }
        },
    )


def _passage(identifier: str) -> Passage:
    return Passage(id=identifier, title=identifier.upper(), text=f"{identifier} text")

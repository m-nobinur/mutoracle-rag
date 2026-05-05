"""Official RAGAS baseline adapter."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from openai import AsyncOpenAI

from mutoracle.baselines.schema import (
    BaselineResult,
    classify_faithfulness,
    merge_model_ids,
    run_id_for,
    run_metadata_model_ids,
    run_metadata_value,
)
from mutoracle.config import MutOracleConfig
from mutoracle.contracts import RAGRun


class RagasScorer(Protocol):
    """Scoring interface used by the RAGAS wrapper."""

    model_id: str

    def score(self, run: RAGRun, *, reference: str | None = None) -> float:
        """Return the official RAGAS faithfulness score."""


class RagasMetricScorer(RagasScorer, Protocol):
    """Scorer that can return the full Phase 7 RAGAS metric set."""

    def score_metrics(
        self,
        run: RAGRun,
        *,
        reference: str | None = None,
    ) -> dict[str, float]:
        """Return RAGAS metrics keyed by canonical metric name."""


@dataclass(frozen=True)
class RagasBaseline:
    """Run RAGAS faithfulness on the shared RAGRun schema."""

    scorer: RagasScorer
    name: str = "ragas"

    def run(
        self,
        run: RAGRun,
        *,
        threshold: float = 0.5,
        reference: str | None = None,
    ) -> BaselineResult:
        """Score one RAG output with official RAGAS."""

        started = time.perf_counter()
        metric_scores = _ragas_metric_scores(
            self.scorer,
            run,
            reference=reference,
        )
        score = min(1.0, max(0.0, metric_scores["faithfulness"]))
        scorer_latency = time.perf_counter() - started
        generation_latency = run_metadata_value(
            run,
            ("generation", "latency_seconds"),
            run_metadata_value(run, ("latency", "generation_seconds"), 0.0),
        )
        total_latency = generation_latency + scorer_latency
        generation_cost = run_metadata_value(
            run,
            ("generation", "estimated_cost_usd"),
            0.0,
        )
        model_ids = merge_model_ids(
            run_metadata_model_ids(run),
            [self.scorer.model_id],
        )
        return BaselineResult(
            run_id=run_id_for(run),
            baseline_name=self.name,
            query=run.query,
            score=score,
            threshold=threshold,
            predicted_label=classify_faithfulness(score=score, threshold=threshold),
            latency_seconds=total_latency,
            cost_usd=generation_cost,
            model_ids=model_ids,
            scores=metric_scores,
            metadata={
                "official_package": "ragas",
                "ragas_metrics": sorted(metric_scores),
                "headline_metric": "faithfulness",
                "cost_scope": "generation_only",
                "latency_breakdown_seconds": {
                    "generation": generation_latency,
                    "baseline": scorer_latency,
                },
                "cost_breakdown_usd": {
                    "generation": generation_cost,
                },
            },
        )


class OfficialRagasFaithfulnessScorer:
    """RAGAS Faithfulness scorer using the currently documented collections API."""

    def __init__(
        self,
        *,
        config: MutOracleConfig,
        metric: Any | None = None,
    ) -> None:
        self.model_id = config.models.judge
        self._metrics = metric or self._build_metrics(config)

    def score(self, run: RAGRun, *, reference: str | None = None) -> float:
        """Return a RAGAS faithfulness score for one run."""

        return self.score_metrics(run, reference=reference)["faithfulness"]

    def score_metrics(
        self,
        run: RAGRun,
        *,
        reference: str | None = None,
    ) -> dict[str, float]:
        """Return the Phase 7 RAGAS metric set for one run."""

        scores: dict[str, float] = {}
        for metric_name, metric in self._metrics.items():
            value = _score_metric(
                metric,
                user_input=run.query,
                response=run.answer,
                retrieved_contexts=run.passages,
                reference=reference,
            )
            if hasattr(value, "value"):
                value = value.value
            scores[metric_name] = min(1.0, max(0.0, float(value)))
        return scores

    def _build_metrics(self, config: MutOracleConfig) -> dict[str, Any]:
        try:
            from ragas.llms import llm_factory  # type: ignore[import-not-found]
        except ImportError as error:
            msg = (
                "The official ragas package is required for RagasBaseline. "
                "Install ragas in the experiment environment before running this "
                "baseline."
            )
            raise RuntimeError(msg) from error

        client = AsyncOpenAI(
            api_key=config.openrouter.api_key,
            base_url=config.openrouter.base_url,
            timeout=config.openrouter.timeout_seconds,
        )
        llm = llm_factory(config.models.judge, client=client)
        return {
            "faithfulness": _ragas_metric("Faithfulness", llm=llm),
            "answer_relevancy": _ragas_metric(
                "AnswerRelevancy",
                "ResponseRelevancy",
                llm=llm,
            ),
            "context_precision": _ragas_metric("ContextPrecision", llm=llm),
            "context_recall": _ragas_metric("ContextRecall", llm=llm),
        }


def _ragas_metric_scores(
    scorer: RagasScorer,
    run: RAGRun,
    *,
    reference: str | None,
) -> dict[str, float]:
    score_metrics = getattr(scorer, "score_metrics", None)
    if callable(score_metrics):
        raw_scores = score_metrics(run, reference=reference)
    else:
        raw_scores = {"faithfulness": scorer.score(run, reference=reference)}
    if "faithfulness" not in raw_scores:
        msg = "RAGAS baseline scores must include faithfulness."
        raise ValueError(msg)
    return {
        name: min(1.0, max(0.0, float(value))) for name, value in raw_scores.items()
    }


def _ragas_metric(*class_names: str, llm: Any) -> Any:
    metrics_module = import_module("ragas.metrics.collections")
    for class_name in class_names:
        metric_class = getattr(metrics_module, class_name, None)
        if metric_class is not None:
            return metric_class(llm=llm)
    names = ", ".join(class_names)
    msg = f"Installed ragas package does not expose any of: {names}"
    raise RuntimeError(msg)


def _score_metric(
    metric: Any,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str | None,
) -> Any:
    """Score with either the new collection API or the legacy sample API."""

    payload: dict[str, Any] = {
        "user_input": user_input,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
    }
    if reference is not None:
        payload["reference"] = reference

    if hasattr(metric, "score"):
        return metric.score(**payload)
    if hasattr(metric, "ascore"):
        return asyncio.run(metric.ascore(**payload))
    if hasattr(metric, "single_turn_ascore"):
        try:
            from ragas import SingleTurnSample  # type: ignore[import-not-found]
        except ImportError:
            from ragas.dataset_schema import (  # type: ignore[import-not-found]
                SingleTurnSample,
            )

        sample = SingleTurnSample(**payload)
        return asyncio.run(metric.single_turn_ascore(sample))
    msg = "Unsupported RAGAS metric object; expected score/ascore API."
    raise TypeError(msg)

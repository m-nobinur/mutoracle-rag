"""OpenRouter-backed LLM-as-judge oracle with strict JSON validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mutoracle.cache import SQLiteCacheLedger, oracle_cache_key, prompt_hash
from mutoracle.config import MutOracleConfig
from mutoracle.contracts import RAGRun
from mutoracle.oracles.base import (
    OracleScore,
    clamp_score,
    context_text,
    oracle_payload,
    stable_hash,
)
from mutoracle.provider import OpenRouterProvider, ProviderCompletion

JUDGE_SYSTEM_PROMPT = (
    "You are a strict factual-faithfulness evaluator. Given retrieved context "
    "and a generated response, decide whether the response is fully supported "
    "by the context. Output only valid JSON."
)
JUDGE_TEMPERATURE = 0.0


class JudgeVerdict(BaseModel):
    """Strict schema for LLM judge output."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["faithful", "hallucinated"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)


class JudgeProvider(Protocol):
    """Provider interface required by the LLM judge."""

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        request_kind: str = "generation",
    ) -> ProviderCompletion:
        """Return a provider completion."""


class LLMJudgeOracle:
    """Scores faithfulness with a strict JSON OpenRouter judge."""

    name = "llm_judge"

    def __init__(
        self,
        *,
        config: MutOracleConfig,
        ledger: SQLiteCacheLedger,
        provider: JudgeProvider | None = None,
        model_name: str | None = None,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self.model_name = model_name or config.models.judge
        self._provider = provider or OpenRouterProvider(config, ledger)

    def score(self, run: RAGRun) -> float:
        """Return a normalized faithfulness score in [0, 1]."""

        return self.score_result(run).value

    def score_result(self, run: RAGRun) -> OracleScore:
        """Return judge faithfulness score and validation metadata."""

        return self.score_results([run])[0]

    def score_results(self, runs: Sequence[RAGRun]) -> list[OracleScore]:
        """Return judge faithfulness scores for a batch of runs."""

        return [self._score_result_one(run) for run in runs]

    def _score_result_one(self, run: RAGRun) -> OracleScore:
        """Return judge faithfulness score and validation metadata."""

        payload = oracle_payload(run)
        input_hash = stable_hash(payload)
        base_prompt = build_judge_prompt(run)
        base_prompt_hash = prompt_hash(base_prompt)
        cache_key = oracle_cache_key(
            oracle_name=self.name,
            model=self.model_name,
            payload={
                "input_hash": input_hash,
                "prompt_hash": base_prompt_hash,
                "schema": "JudgeVerdict/v1",
                "temperature": JUDGE_TEMPERATURE,
            },
        )
        cached = self._ledger.lookup_oracle_score(cache_key)
        if cached is not None:
            metadata = dict(cached.metadata)
            metadata["cache_hit"] = True
            return OracleScore(
                oracle_name=self.name,
                value=clamp_score(cached.score),
                metadata=metadata,
            )

        result = self._score_uncached(
            base_prompt=base_prompt,
            base_prompt_hash=base_prompt_hash,
            input_hash=input_hash,
        )
        result = OracleScore(
            oracle_name=result.oracle_name,
            value=result.value,
            metadata={**result.metadata, "cache_hit": False},
        )
        self._ledger.store_oracle_score(
            cache_key=cache_key,
            oracle_name=self.name,
            input_hash=input_hash,
            score=result.value,
            metadata=result.metadata,
        )
        return result

    def _score_uncached(
        self,
        *,
        base_prompt: str,
        base_prompt_hash: str,
        input_hash: str,
    ) -> OracleScore:
        errors: list[str] = []
        for attempt in range(2):
            prompt = base_prompt if attempt == 0 else _retry_prompt(base_prompt)
            try:
                completion = self._provider.complete(
                    prompt,
                    model=self.model_name,
                    temperature=JUDGE_TEMPERATURE,
                    request_kind="llm_judge",
                )
                verdict = parse_judge_response(completion.answer)
            except (ValidationError, json.JSONDecodeError) as error:
                errors.append(str(error))
                continue
            except RuntimeError as error:
                if _is_retryable_provider_error(error):
                    errors.append(str(error))
                    continue
                raise

            score = judge_score(verdict)
            return OracleScore(
                oracle_name=self.name,
                value=score,
                metadata={
                    "input_hash": input_hash,
                    "model": self.model_name,
                    "prompt_hash": base_prompt_hash,
                    "attempts": attempt + 1,
                    "temperature": JUDGE_TEMPERATURE,
                    "verdict": verdict.model_dump(mode="json"),
                    "provider": completion.metadata,
                },
            )

        return OracleScore(
            oracle_name=self.name,
            value=0.0,
            metadata={
                "input_hash": input_hash,
                "model": self.model_name,
                "prompt_hash": base_prompt_hash,
                "attempts": 2,
                "temperature": JUDGE_TEMPERATURE,
                "failure": {
                    "kind": "invalid_judge_response",
                    "errors": errors,
                },
            },
        )


def build_judge_prompt(run: RAGRun) -> str:
    """Return the locked judge prompt for a RAG run."""

    passages = context_text(run) or "(no retrieved context)"
    response = run.answer.strip() or "(empty response)"
    return (
        "System:\n"
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        "User:\n"
        "Context passages:\n"
        f"{passages}\n\n"
        "Generated response:\n"
        f"{response}\n\n"
        "Output JSON:\n"
        '{"verdict": "faithful" | "hallucinated", "confidence": 0.0-1.0, '
        '"reason": "one short sentence"}'
    )


def parse_judge_response(raw: str) -> JudgeVerdict:
    """Parse and validate a raw judge response."""

    return JudgeVerdict.model_validate_json(_extract_json_object(raw))


def judge_score(verdict: JudgeVerdict) -> float:
    """Map a judge verdict into a faithfulness score."""

    confidence = clamp_score(verdict.confidence)
    if verdict.verdict == "faithful":
        return confidence
    return 1.0 - confidence


def _retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Your previous response was invalid. Return only the JSON object with "
        "the exact keys verdict, confidence, and reason."
    )


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        msg = "Judge response did not contain a JSON object."
        raise json.JSONDecodeError(msg, raw, 0)
    return text[start : end + 1]


def _is_retryable_provider_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "openrouter request failed" in message

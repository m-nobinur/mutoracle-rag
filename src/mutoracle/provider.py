"""OpenRouter provider wrapper using the OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from openai import APIError, OpenAI

from mutoracle.cache import SQLiteCacheLedger, completion_cache_key, prompt_hash
from mutoracle.config import MutOracleConfig


@dataclass(frozen=True)
class ProviderCompletion:
    """Text and metadata returned by a provider call."""

    answer: str
    metadata: dict[str, Any]


class OpenRouterProvider:
    """Minimal OpenRouter chat-completions client."""

    def __init__(self, config: MutOracleConfig, ledger: SQLiteCacheLedger) -> None:
        self._config = config
        self._ledger = ledger

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        request_kind: str = "generation",
    ) -> ProviderCompletion:
        """Return a cached or live completion for a prompt."""

        resolved_model = model or self._config.models.generator
        resolved_temperature = (
            float(temperature)
            if temperature is not None
            else float(self._config.models.temperature)
        )
        cache_key = completion_cache_key(
            model=resolved_model,
            prompt=prompt,
            temperature=resolved_temperature,
            provider_route="openrouter",
            seed=self._config.runtime.seed,
        )
        cached = self._ledger.lookup_completion(cache_key)
        if cached is not None:
            self._ledger.record_usage(
                model=resolved_model,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                latency_seconds=0.0,
                cache_hit=True,
            )
            metadata = dict(cached.metadata)
            metadata["cache_hit"] = True
            metadata["request_kind"] = request_kind
            return ProviderCompletion(answer=cached.answer, metadata=metadata)

        api_key = self._config.openrouter.api_key
        if not api_key:
            msg = "OPENROUTER_API_KEY is required for remote RAG generation."
            raise RuntimeError(msg)

        self._enforce_budget()
        started_at = perf_counter()
        response = self._request_completion(
            prompt,
            api_key,
            model=resolved_model,
            temperature=resolved_temperature,
        )
        latency_seconds = round(perf_counter() - started_at, 6)
        usage = _extract_usage(response)
        answer = _extract_answer(response)
        cost_usd = _estimate_cost_usd(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            prompt_cost_per_1m=float(self._config.cost.prompt_cost_per_1m_tokens),
            completion_cost_per_1m=float(
                self._config.cost.completion_cost_per_1m_tokens
            ),
        )
        metadata = {
            "model": resolved_model,
            "cache_hit": False,
            "provider": "openrouter",
            "provider_route": "openrouter",
            "request_kind": request_kind,
            "prompt_hash": prompt_hash(prompt),
            "temperature": resolved_temperature,
            "seed": self._config.runtime.seed,
            "usage": usage,
            "estimated_cost_usd": cost_usd,
            "latency_seconds": latency_seconds,
        }
        self._ledger.store_completion(
            cache_key=cache_key,
            answer=answer,
            metadata=metadata,
        )
        self._ledger.record_usage(
            model=resolved_model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cost_usd=cost_usd,
            latency_seconds=latency_seconds,
            cache_hit=False,
        )
        return ProviderCompletion(answer=answer, metadata=metadata)

    def _enforce_budget(self) -> None:
        summary = self._ledger.usage_summary()
        if summary.live_requests >= self._config.cost.max_queries:
            msg = (
                "Remote query budget exhausted: "
                f"{summary.live_requests}/{self._config.cost.max_queries}"
            )
            raise RuntimeError(msg)
        if summary.total_cost_usd >= self._config.cost.max_cost_usd:
            msg = (
                "Remote cost budget exhausted: "
                f"${summary.total_cost_usd:.4f}/${self._config.cost.max_cost_usd:.4f}"
            )
            raise RuntimeError(msg)

    def _request_completion(
        self,
        prompt: str,
        api_key: str,
        *,
        model: str,
        temperature: float,
    ) -> dict[str, Any]:
        client = OpenAI(
            api_key=api_key,
            base_url=self._config.openrouter.base_url,
            timeout=self._config.openrouter.timeout_seconds,
        )
        extra_headers = {"X-Title": self._config.openrouter.app_title}
        if self._config.openrouter.app_url:
            extra_headers["HTTP-Referer"] = self._config.openrouter.app_url

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self._config.models.max_tokens,
                extra_headers=extra_headers,
            )
        except APIError as error:
            msg = f"OpenRouter request failed: {error}"
            raise RuntimeError(msg) from error

        parsed = completion.model_dump(mode="json")
        if not isinstance(parsed, dict):
            msg = "OpenRouter response must be a JSON object."
            raise RuntimeError(msg)
        return parsed


def _extract_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "OpenRouter response did not include choices."
        raise RuntimeError(msg)
    first = choices[0]
    if not isinstance(first, dict):
        msg = "OpenRouter choice must be an object."
        raise RuntimeError(msg)
    message = first.get("message")
    if not isinstance(message, dict):
        msg = "OpenRouter choice did not include a message object."
        raise RuntimeError(msg)
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts:
            return "\n".join(parts)
    finish_reason = first.get("finish_reason")
    has_reasoning = bool(message.get("reasoning"))
    if finish_reason == "length":
        msg = (
            "OpenRouter choice message did not include text content "
            f"(finish_reason={finish_reason}, has_reasoning={has_reasoning}). "
            "Increase models.max_tokens for judge/generation calls."
        )
        raise RuntimeError(msg)
    msg = (
        "OpenRouter choice message did not include text content "
        f"(finish_reason={finish_reason}, has_reasoning={has_reasoning})."
    )
    raise RuntimeError(msg)


def _extract_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(
            usage.get("total_tokens") or prompt_tokens + completion_tokens
        ),
    }


def _estimate_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_1m: float,
    completion_cost_per_1m: float,
) -> float:
    return round(
        (prompt_tokens / 1_000_000 * prompt_cost_per_1m)
        + (completion_tokens / 1_000_000 * completion_cost_per_1m),
        8,
    )

"""Configuration models and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt

DEFAULT_CONFIG_PATH = Path("experiments/configs/dev.yaml")
PROJECT_ENV_PATH = Path(".env")


class OpenRouterConfig(BaseModel):
    """OpenRouter connection settings."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    app_title: str = "MutOracle-RAG"
    app_url: str | None = None
    timeout_seconds: PositiveInt = 60


class ModelConfig(BaseModel):
    """Model IDs used by the generator and judge."""

    model_config = ConfigDict(extra="forbid")

    generator: str = "openai/gpt-oss-20b:free"
    judge: str = "openai/gpt-oss-120b:free"
    temperature: NonNegativeFloat = 0.0
    max_tokens: PositiveInt = 512


class CostConfig(BaseModel):
    """Local budget limits for guarded model calls."""

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: NonNegativeFloat = 5.0
    max_queries: PositiveInt = 20
    prompt_cost_per_1m_tokens: NonNegativeFloat = 0.0
    completion_cost_per_1m_tokens: NonNegativeFloat = 0.0


class RAGConfig(BaseModel):
    """RAG system-under-test settings."""

    model_config = ConfigDict(extra="forbid")

    top_k: PositiveInt = 3


class RuntimeConfig(BaseModel):
    """Filesystem and reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 2026
    cache_path: Path = Path(".mutoracle/cache.sqlite3")


class MutOracleConfig(BaseModel):
    """Top-level project configuration."""

    model_config = ConfigDict(extra="forbid")

    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


def load_config(path: Path | None = None) -> MutOracleConfig:
    """Load config from YAML plus secret/environment overrides."""

    _load_project_env()
    resolved_path = resolve_config_path(path)
    raw = _read_yaml(resolved_path) if resolved_path else {}
    config = MutOracleConfig.model_validate(raw)
    return _apply_environment(config, allow_runtime_overrides=resolved_path is None)


def resolve_config_path(path: Path | None = None) -> Path | None:
    """Return the explicit or conventional project config path, if available."""

    if path is not None:
        return path
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Config file does not exist: {path}"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        msg = f"Config file must contain a YAML mapping: {path}"
        raise ValueError(msg)

    return raw


def _load_project_env(path: Path = PROJECT_ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _apply_environment(
    config: MutOracleConfig,
    *,
    allow_runtime_overrides: bool,
) -> MutOracleConfig:
    update: dict[str, Any] = {}

    openrouter_update = {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": os.getenv("OPENROUTER_BASE_URL"),
        "app_title": os.getenv("OPENROUTER_APP_TITLE"),
        "app_url": os.getenv("OPENROUTER_APP_URL"),
    }
    openrouter_update = {
        key: value for key, value in openrouter_update.items() if value
    }
    if openrouter_update:
        update["openrouter"] = config.openrouter.model_copy(update=openrouter_update)

    if allow_runtime_overrides:
        cost_update: dict[str, Any] = {}
        if max_cost := os.getenv("MUTORACLE_MAX_COST_USD"):
            cost_update["max_cost_usd"] = float(max_cost)
        if max_queries := os.getenv("MUTORACLE_MAX_QUERIES"):
            cost_update["max_queries"] = int(max_queries)
        if prompt_cost := os.getenv("MUTORACLE_PROMPT_COST_PER_1M_TOKENS"):
            cost_update["prompt_cost_per_1m_tokens"] = float(prompt_cost)
        if completion_cost := os.getenv("MUTORACLE_COMPLETION_COST_PER_1M_TOKENS"):
            cost_update["completion_cost_per_1m_tokens"] = float(completion_cost)
        if cost_update:
            update["cost"] = config.cost.model_copy(update=cost_update)

        runtime_update: dict[str, Any] = {}
        if cache_path := os.getenv("MUTORACLE_CACHE_PATH"):
            runtime_update["cache_path"] = Path(cache_path)
        if runtime_update:
            update["runtime"] = config.runtime.model_copy(update=runtime_update)

    return config.model_copy(update=update)

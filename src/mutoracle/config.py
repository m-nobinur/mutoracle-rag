"""Configuration models and loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt


class OpenRouterConfig(BaseModel):
    """OpenRouter connection settings."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    app_title: str = "MutOracle-RAG"
    app_url: str | None = None


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


def load_config(path: Path | None = None) -> MutOracleConfig:
    """Load config from optional YAML plus environment overrides."""

    raw = _read_yaml(path) if path else {}
    config = MutOracleConfig.model_validate(raw)
    return _apply_environment(config)


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


def _apply_environment(config: MutOracleConfig) -> MutOracleConfig:
    update: dict[str, Any] = {}

    openrouter_update = {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "app_title": os.getenv("OPENROUTER_APP_TITLE"),
        "app_url": os.getenv("OPENROUTER_APP_URL"),
    }
    openrouter_update = {
        key: value for key, value in openrouter_update.items() if value
    }
    if openrouter_update:
        update["openrouter"] = config.openrouter.model_copy(update=openrouter_update)

    cost_update: dict[str, Any] = {}
    if max_cost := os.getenv("MUTORACLE_MAX_COST_USD"):
        cost_update["max_cost_usd"] = float(max_cost)
    if max_queries := os.getenv("MUTORACLE_MAX_QUERIES"):
        cost_update["max_queries"] = int(max_queries)
    if cost_update:
        update["cost"] = config.cost.model_copy(update=cost_update)

    runtime_update: dict[str, Any] = {}
    if cache_path := os.getenv("MUTORACLE_CACHE_PATH"):
        runtime_update["cache_path"] = Path(cache_path)
    if runtime_update:
        update["runtime"] = config.runtime.model_copy(update=runtime_update)

    return config.model_copy(update=update)

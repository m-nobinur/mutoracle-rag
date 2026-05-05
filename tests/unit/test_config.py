from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mutoracle.config import MutOracleConfig, load_config, resolve_config_path


def test_default_config_loads_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = load_config()

    assert isinstance(config, MutOracleConfig)
    assert config.models.generator == "openai/gpt-oss-20b:free"
    assert config.openrouter.api_key is None


def test_yaml_config_overrides_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  generator: openai/gpt-5-nano
  judge: minimax/minimax-m2.5
cost:
  max_cost_usd: 1.25
  max_queries: 5
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.models.generator == "openai/gpt-5-nano"
    assert config.models.judge == "minimax/minimax-m2.5"
    assert config.cost.max_cost_usd == 1.25
    assert config.cost.max_queries == 5


def test_environment_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MUTORACLE_MAX_QUERIES", "7")

    config = load_config()

    assert config.openrouter.api_key == "test-key"
    assert config.cost.max_queries == 7


def test_conventional_dev_config_is_discovered() -> None:
    config = load_config()

    assert resolve_config_path() == Path("experiments/configs/dev.yaml")
    assert config.models.generator == "openai/gpt-5-nano"
    assert config.models.judge == "minimax/minimax-m2.5"
    assert config.runtime.cache_path == Path(".mutoracle/dev-cache.sqlite3")
    assert config.cost.max_queries == 5


def test_rag_and_provider_defaults_are_present() -> None:
    config = load_config()

    assert config.rag.top_k == 3
    assert config.openrouter.base_url.endswith("/api/v1")
    assert config.cost.prompt_cost_per_1m_tokens == 0.0


def test_uniform_strategy_ignores_unused_invalid_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "uniform.yaml"
    config_path.write_text(
        """
aggregation:
  strategy: uniform
  weights:
    nli: 2.0
    semantic_similarity: 0.0
    llm_judge: 0.0
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.aggregation.strategy == "uniform"
    assert config.aggregation.weights["nli"] == 2.0


def test_delta_threshold_above_one_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-threshold.yaml"
    config_path.write_text(
        """
aggregation:
  strategy: weighted
  delta_threshold: 2.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=r"delta_threshold"):
        load_config(config_path)

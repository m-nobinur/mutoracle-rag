from __future__ import annotations

from pathlib import Path

import pytest

from mutoracle.config import MutOracleConfig, load_config


def test_default_config_loads_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_environment_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MUTORACLE_MAX_QUERIES", "7")

    config = load_config()

    assert config.openrouter.api_key == "test-key"
    assert config.cost.max_queries == 7

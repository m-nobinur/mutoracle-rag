from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mutoracle.cli as cli_module
from mutoracle.contracts import RAGRun

app = cli_module.app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Mutation-driven fault localization" in result.output


def test_cli_version_and_config_commands() -> None:
    version = CliRunner().invoke(app, ["--version"])
    show = CliRunner().invoke(app, ["config", "show"])
    validate = CliRunner().invoke(app, ["config", "validate"])

    assert version.exit_code == 0
    assert "mutoracle-rag" in version.output
    assert show.exit_code == 0
    assert '"openrouter"' in show.output
    assert validate.exit_code == 0
    assert "Config valid" in validate.output


def test_cli_smoke() -> None:
    result = CliRunner().invoke(app, ["smoke"])

    assert result.exit_code == 0
    assert "Bootstrap smoke passed" in result.output


def test_cli_rag_smoke() -> None:
    result = CliRunner().invoke(
        app,
        ["rag", "smoke", "--query", "What is MutOracle-RAG?"],
    )

    assert result.exit_code == 0
    assert "RAG smoke passed" in result.output
    assert "mutoracle-purpose" in result.output


def test_cli_ten_query_smoke() -> None:
    result = CliRunner().invoke(app, ["smoke", "--queries", "10"])

    assert result.exit_code == 0
    assert "RAG batch smoke passed" in result.output
    assert '"queries": 10' in result.output


def test_cli_mutate_ci() -> None:
    result = CliRunner().invoke(app, ["mutate", "--operator", "CI"])

    assert result.exit_code == 0
    assert "Context Injection" in result.output
    assert '"operator": "CI"' in result.output
    assert '"rejected": false' in result.output


def test_cli_mutate_invalid_operator_reports_error() -> None:
    result = CliRunner().invoke(app, ["mutate", "--operator", "BAD"])

    assert result.exit_code == 2
    assert "Mutation error" in result.output


def test_cli_diagnose_uses_fixture_oracles() -> None:
    result = CliRunner().invoke(app, ["diagnose"])

    assert result.exit_code == 0
    assert "Fault diagnosis" in result.output
    assert '"stage":' in result.output
    assert '"deltas":' in result.output


def test_cli_diagnose_real_oracles_path(monkeypatch) -> None:
    calls: list[str] = []

    class StubOracle:
        def __init__(self, name: str) -> None:
            self.name = name

        def score(self, run) -> float:
            del run
            return 0.8

    def fake_real_oracles(config):
        del config
        calls.append("real")
        return [
            StubOracle("nli"),
            StubOracle("semantic_similarity"),
            StubOracle("llm_judge"),
        ]

    monkeypatch.setattr(cli_module, "_real_oracles", fake_real_oracles)
    result = CliRunner().invoke(app, ["diagnose", "--real-oracles"])

    assert result.exit_code == 0
    assert calls == ["real"]
    assert '"stage":' in result.output


def test_cli_diagnose_reports_runtime_errors(monkeypatch) -> None:
    class FailingLocalizer:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def diagnose(self, query: str):
            del query
            raise RuntimeError("diagnosis failed")

    monkeypatch.setattr(cli_module, "FaultLocalizer", FailingLocalizer)

    result = CliRunner().invoke(app, ["diagnose"])

    assert result.exit_code == 2
    assert "Diagnose error" in result.output


def test_cli_baseline_smoke_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline-smoke.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "baseline",
            "smoke",
            "--baseline",
            "metarag",
            "--queries",
            "2",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Phase 7 baselines" in result.output
    assert output_path.exists()
    manifest_path = output_path.with_suffix(".manifest.json")
    assert manifest_path.exists()

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[0]["baseline_name"] == "metarag"


def test_cli_baseline_invalid_and_ragas_error_paths() -> None:
    invalid = CliRunner().invoke(app, ["baseline", "smoke", "--baseline", "bad"])
    ragas = CliRunner().invoke(app, ["baseline", "smoke", "--baseline", "ragas"])

    assert invalid.exit_code == 2
    assert "Baseline error" in invalid.output
    assert ragas.exit_code == 2
    assert "RAGAS error" in ragas.output


def test_cli_rag_remote_error_path(monkeypatch) -> None:
    class FailingPipeline:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def run(self, query: str):
            del query
            raise RuntimeError("remote unavailable")

    monkeypatch.setattr(cli_module, "FixtureRAGPipeline", FailingPipeline)

    result = CliRunner().invoke(app, ["rag", "smoke", "--remote"])

    assert result.exit_code == 2
    assert "RAG smoke error" in result.output


def test_cli_rag_batch_error_path(monkeypatch) -> None:
    class FailingPipeline:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def run(self, query: str):
            del query
            raise RuntimeError("batch unavailable")

    monkeypatch.setattr(cli_module, "FixtureRAGPipeline", FailingPipeline)

    result = CliRunner().invoke(app, ["rag", "smoke", "--queries", "1"])

    assert result.exit_code == 2
    assert "RAG smoke error" in result.output


def test_cli_data_and_fits_build_use_shared_builder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_fits_dataset(**kwargs):
        calls.append(kwargs)
        manifest = tmp_path / "manifest.json"
        return {"manifest": manifest, "fits": tmp_path / "fits.jsonl"}

    monkeypatch.setattr(cli_module, "build_fits_dataset", fake_build_fits_dataset)

    data = CliRunner().invoke(
        app,
        ["data", "build", "--output-root", str(tmp_path), "--force"],
    )
    fits = CliRunner().invoke(
        app,
        ["fits", "build", "--output-root", str(tmp_path), "--force"],
    )

    assert data.exit_code == 0
    assert fits.exit_code == 0
    assert len(calls) == 2
    assert calls[0]["force_rebuild"] is True


def test_cli_private_helpers_cover_config_and_oracle_branches(tmp_path: Path) -> None:
    config = cli_module.MutOracleConfig(
        openrouter=cli_module.MutOracleConfig().openrouter.model_copy(
            update={"api_key": "secret"}
        )
    )
    encoded = cli_module._config_to_jsonable(config)
    assert encoded["openrouter"]["api_key"] == "***"

    with pytest.raises(cli_module.typer.Exit):
        cli_module._load_or_exit(tmp_path / "missing.yaml")

    summary = cli_module._run_summary(RAGRun(query="q", passages=["p"], answer="a"))
    assert summary["passage_count"] == 1

    oracles = cli_module._fixture_oracles()
    assert [oracle.name for oracle in oracles] == [
        "nli",
        "semantic_similarity",
        "llm_judge",
    ]
    assert oracles[0].score(RAGRun(query="q", passages=["same"], answer="same")) > 0
    assert [oracle.name for oracle in cli_module._real_oracles(config)] == [
        "nli",
        "semantic_similarity",
        "llm_judge",
    ]

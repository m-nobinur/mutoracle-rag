from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import mutoracle.cli as cli_module

app = cli_module.app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Mutation-driven fault localization" in result.output


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

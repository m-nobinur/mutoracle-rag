from __future__ import annotations

from typer.testing import CliRunner

from mutoracle.cli import app


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

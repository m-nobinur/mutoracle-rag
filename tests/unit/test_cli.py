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

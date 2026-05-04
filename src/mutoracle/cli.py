"""Command line interface for MutOracle-RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from mutoracle import __version__
from mutoracle.config import MutOracleConfig, load_config

app = typer.Typer(
    help="Mutation-driven fault localization for RAG pipelines.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Inspect and validate MutOracle-RAG configuration.")
app.add_typer(config_app, name="config")

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mutoracle-rag {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, help="Show version."),
    ] = False,
) -> None:
    """MutOracle-RAG command group."""


@config_app.command("show")
def show_config(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Print the resolved configuration as JSON."""

    resolved = _load_or_exit(config)
    console.print_json(data=_config_to_jsonable(resolved))


@config_app.command("validate")
def validate_config(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Validate configuration and print a short success message."""

    resolved = _load_or_exit(config)
    console.print(
        Panel.fit(
            f"Config valid\nGenerator: {resolved.models.generator}\n"
            f"Judge: {resolved.models.judge}",
            title="MutOracle-RAG",
        )
    )


@app.command()
def smoke() -> None:
    """Run a credential-free bootstrap smoke check."""

    resolved = load_config()
    console.print(
        Panel.fit(
            "Bootstrap smoke passed\n"
            f"Seed: {resolved.runtime.seed}\n"
            f"Cache: {resolved.runtime.cache_path}",
            title="MutOracle-RAG",
        )
    )


def _load_or_exit(path: Path | None) -> MutOracleConfig:
    try:
        return load_config(path)
    except (FileNotFoundError, ValueError, ValidationError) as error:
        console.print(f"[red]Configuration error:[/red] {error}")
        raise typer.Exit(code=2) from error


def _config_to_jsonable(config: MutOracleConfig) -> dict[str, Any]:
    encoded: dict[str, Any] = config.model_dump(mode="json")
    if encoded["openrouter"].get("api_key"):
        encoded["openrouter"]["api_key"] = "***"
    return encoded

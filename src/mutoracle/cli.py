"""Command line interface for MutOracle-RAG."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from mutoracle import __version__
from mutoracle.config import MutOracleConfig, load_config, resolve_config_path
from mutoracle.rag import FixtureRAGPipeline

app = typer.Typer(
    help="Mutation-driven fault localization for RAG pipelines.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Inspect and validate MutOracle-RAG configuration.")
rag_app = typer.Typer(help="Run the reproducible Phase 2 RAG system under test.")
app.add_typer(config_app, name="config")
app.add_typer(rag_app, name="rag")

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
    source = resolve_config_path(config)
    console.print_json(data=_config_to_jsonable(resolved))
    console.print(f"Config source: {source or 'built-in defaults'}")


@config_app.command("validate")
def validate_config(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Validate configuration and print a short success message."""

    resolved = _load_or_exit(config)
    source = resolve_config_path(config)
    console.print(
        Panel.fit(
            f"Config valid\nGenerator: {resolved.models.generator}\n"
            f"Judge: {resolved.models.judge}\n"
            f"Source: {source or 'built-in defaults'}",
            title="MutOracle-RAG",
        )
    )


@app.command()
def smoke(
    queries: Annotated[
        int,
        typer.Option(
            "--queries",
            min=0,
            help="Run N fixture RAG queries after the bootstrap smoke check.",
        ),
    ] = 0,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Run a credential-free bootstrap or fixture RAG smoke check."""

    resolved = _load_or_exit(config)
    console.print(
        Panel.fit(
            "Bootstrap smoke passed\n"
            f"Seed: {resolved.runtime.seed}\n"
            f"Cache: {resolved.runtime.cache_path}",
            title="MutOracle-RAG",
        )
    )
    if queries:
        _print_rag_batch_smoke(
            config=resolved,
            corpus=corpus,
            queries=_default_smoke_queries(limit=queries),
            remote=False,
        )


@rag_app.command("smoke")
def rag_smoke(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Question to run through the RAG SUT."),
    ] = "What is MutOracle-RAG?",
    queries: Annotated[
        int,
        typer.Option(
            "--queries",
            min=0,
            help="Run N packaged fixture smoke queries instead of --query.",
        ),
    ] = 0,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", exists=True, dir_okay=False),
    ] = None,
    remote: Annotated[
        bool,
        typer.Option("--remote", help="Use OpenRouter instead of fixture generation."),
    ] = False,
) -> None:
    """Run a reproducible RAG smoke query."""

    resolved = _load_or_exit(config)
    if queries:
        _print_rag_batch_smoke(
            config=resolved,
            corpus=corpus,
            queries=_default_smoke_queries(limit=queries),
            remote=remote,
        )
        return

    pipeline = FixtureRAGPipeline(
        config=resolved,
        corpus_path=corpus,
        use_remote=remote,
    )
    try:
        run = pipeline.run(query)
    except RuntimeError as error:
        console.print(f"[red]RAG smoke error:[/red] {error}")
        raise typer.Exit(code=2) from error

    console.print(
        Panel.fit(
            f"Question: {run.query}\n"
            f"Answer: {run.answer}\n"
            f"Passages: {len(run.passages)}",
            title="RAG smoke passed",
        )
    )
    console.print_json(data={"retrieval": run.metadata["retrieval"]})


def _print_rag_batch_smoke(
    *,
    config: MutOracleConfig,
    corpus: Path | None,
    queries: list[str],
    remote: bool,
) -> None:
    pipeline = FixtureRAGPipeline(
        config=config,
        corpus_path=corpus,
        use_remote=remote,
    )
    runs = []
    try:
        for query in queries:
            runs.append(pipeline.run(query))
    except RuntimeError as error:
        console.print(f"[red]RAG smoke error:[/red] {error}")
        raise typer.Exit(code=2) from error

    console.print(
        Panel.fit(
            f"RAG smoke queries passed: {len(runs)}\n"
            f"Provider: {runs[0].metadata['generation']['provider']}\n"
            f"Seed: {config.runtime.seed}",
            title="RAG batch smoke passed",
        )
    )
    console.print_json(
        data={
            "queries": len(runs),
            "first_retrieval": runs[0].metadata["retrieval"],
            "usage": pipeline.usage_summary(),
        }
    )


def _default_smoke_queries(*, limit: int) -> list[str]:
    raw_text = (
        resources.files("mutoracle.fixtures")
        .joinpath("queries.json")
        .read_text(encoding="utf-8")
    )
    queries = json.loads(raw_text)
    if not isinstance(queries, list) or not all(
        isinstance(query, str) for query in queries
    ):
        msg = "Packaged smoke queries must be a JSON list of strings."
        raise ValueError(msg)
    selected: list[str] = []
    while len(selected) < limit:
        selected.extend(queries)
    return selected[:limit]


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

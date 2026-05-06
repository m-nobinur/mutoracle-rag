"""Command line interface for MutOracle-RAG."""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from random import Random
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from mutoracle import __version__
from mutoracle.aggregation import build_aggregator
from mutoracle.baselines import (
    BaselineExample,
    LexicalNLIBackend,
    MetaRAGBaseline,
    NLIClaimVerifier,
    OfficialRagasFaithfulnessScorer,
    RagasBaseline,
    run_baselines,
    write_baseline_outputs,
)
from mutoracle.cache import SQLiteCacheLedger
from mutoracle.config import MutOracleConfig, load_config, resolve_config_path
from mutoracle.contracts import RAGRun
from mutoracle.data import build_fits_dataset
from mutoracle.localizer import FaultLocalizer, fault_report_to_dict
from mutoracle.mutations import get_operator, list_operator_ids
from mutoracle.mutations.base import content_similarity
from mutoracle.oracles import LLMJudgeOracle, NLIOracle, SemanticSimilarityOracle
from mutoracle.oracles.base import context_text
from mutoracle.rag import FixtureRAGPipeline

app = typer.Typer(
    help="Mutation-driven fault localization for RAG pipelines.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Inspect and validate MutOracle-RAG configuration.")
rag_app = typer.Typer(help="Run the reproducible RAG system under test.")
data_app = typer.Typer(help="Build data manifests and FITS artifacts.")
fits_app = typer.Typer(help="Build and validate the FITS fault-injection split.")
baseline_app = typer.Typer(help="Run response-level baselines.")
app.add_typer(config_app, name="config")
app.add_typer(rag_app, name="rag")
app.add_typer(data_app, name="data")
app.add_typer(fits_app, name="fits")
app.add_typer(baseline_app, name="baseline")

console = Console()

RELEASE_REQUIRED_PATHS = (
    Path("LICENSE"),
    Path("README.md"),
    Path("Makefile"),
    Path("pyproject.toml"),
    Path("experiments/configs/e1_detection.yaml"),
    Path("experiments/configs/e2_localization.yaml"),
    Path("experiments/configs/e3_ablation.yaml"),
    Path("experiments/configs/e4_separability.yaml"),
    Path("experiments/configs/e5_latency.yaml"),
    Path("experiments/configs/e6_weighted.yaml"),
    Path("data/manifests/datasets.json"),
    Path("data/fits/manifest.json"),
)
FULL_RESULTS_REQUIRED_MANIFESTS = (
    Path("experiments/results/e1_detection/e1_detection_full_manifest.json"),
    Path("experiments/results/e2_localization/e2_localization_full_manifest.json"),
    Path("experiments/results/e3_ablation/e3_ablation_full_manifest.json"),
    Path("experiments/results/e4_separability/e4_separability_full_manifest.json"),
    Path("experiments/results/e5_latency/e5_latency_full_manifest.json"),
    Path("experiments/results/e6_weighted/e6_weighted_full_manifest.json"),
)
SECRET_PATTERNS = (
    re.compile(r"OPENROUTER_API_KEY\s*=\s*(?!your-|<|$)[^\s]+", re.IGNORECASE),
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)
SECRET_SCAN_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "htmlcov",
    "dist",
    "build",
}
SECRET_SCAN_EXCLUDED_SUFFIXES = {".duckdb", ".pyc", ".png", ".jpg", ".jpeg", ".pdf"}
SECRET_SCAN_EXCLUDED_NAMES = {".env"}


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


@app.command("release-check")
def release_check(
    strict_full_results: Annotated[
        bool,
        typer.Option(
            "--strict-full-results",
            help="Fail unless full E1-E6 result manifests exist.",
        ),
    ] = False,
) -> None:
    """Check that release materials are present and public-ready."""

    report = _release_check_report(strict_full_results=strict_full_results)
    console.print_json(data=report)
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command()
def mutate(
    operator: Annotated[
        str,
        typer.Option(
            "--operator",
            "-o",
            help="Canonical mutation operator ID: CI, CR, CS, QP, QN, FS, or FA.",
        ),
    ],
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Question to run before mutation."),
    ] = "What is MutOracle-RAG?",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic mutation seed."),
    ] = 2026,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Run one canonical mutation against a fixture RAG run."""

    try:
        mutation_operator = get_operator(operator)
    except ValueError as error:
        console.print(f"[red]Mutation error:[/red] {error}")
        raise typer.Exit(code=2) from error

    resolved = _load_or_exit(config)
    pipeline = FixtureRAGPipeline(config=resolved, corpus_path=corpus)
    run = pipeline.run(query)
    mutated = mutation_operator.apply(run, rng=Random(seed))
    mutation = mutated.metadata["mutation"]
    status = "rejected" if mutation["rejected"] else "applied"

    console.print(
        Panel.fit(
            f"Operator: {operator.upper()} ({mutation_operator.name})\n"
            f"Stage: {mutation_operator.stage}\n"
            f"Status: {status}",
            title="Mutation",
        )
    )
    console.print_json(
        data={
            "operator": operator.upper(),
            "valid_operators": list_operator_ids(),
            "baseline": _run_summary(run),
            "mutated": _run_summary(mutated),
            "mutation": mutation,
        }
    )


@app.command()
def diagnose(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Question to diagnose."),
    ] = "What is MutOracle-RAG?",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", exists=True, dir_okay=False),
    ] = None,
    real_oracles: Annotated[
        bool,
        typer.Option(
            "--real-oracles",
            help="Use configured model-backed oracles instead of fixture oracles.",
        ),
    ] = False,
) -> None:
    """Run mutation-delta fault localization for one fixture RAG query."""

    resolved = _load_or_exit(config)
    pipeline = FixtureRAGPipeline(config=resolved, corpus_path=corpus)
    aggregator = build_aggregator(resolved.aggregation)
    oracles = _real_oracles(resolved) if real_oracles else _fixture_oracles()
    localizer = FaultLocalizer(
        pipeline=pipeline,
        oracles=oracles,
        aggregator=aggregator,
        delta_threshold=float(resolved.aggregation.delta_threshold),
        seed=resolved.runtime.seed,
    )
    try:
        report = localizer.diagnose(query)
    except RuntimeError as error:
        console.print(f"[red]Diagnose error:[/red] {error}")
        raise typer.Exit(code=2) from error

    console.print(
        Panel.fit(
            f"Question: {query}\n"
            f"Stage: {report.stage}\n"
            f"Confidence: {report.confidence:.4f}",
            title="Fault diagnosis",
        )
    )
    console.print_json(data=fault_report_to_dict(report))


@data_app.command("build")
def data_build(
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Directory for data manifests/artifacts."),
    ] = Path("data"),
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic FITS build seed."),
    ] = 2026,
    version: Annotated[
        str,
        typer.Option("--version", help="FITS artifact version directory."),
    ] = "fits_v1.0.0",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Rebuild an existing FITS artifact directory.",
        ),
    ] = False,
) -> None:
    """Build manifests plus FITS validation/test JSONL files."""

    _print_data_build(
        output_root=output_root,
        seed=seed,
        version=version,
        force_rebuild=force,
    )


@fits_app.command("build")
def fits_build(
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Directory for data manifests/artifacts."),
    ] = Path("data"),
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic FITS build seed."),
    ] = 2026,
    version: Annotated[
        str,
        typer.Option("--version", help="FITS artifact version directory."),
    ] = "fits_v1.0.0",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Rebuild an existing FITS artifact directory.",
        ),
    ] = False,
) -> None:
    """Build and validate FITS v1.0.0."""

    _print_data_build(
        output_root=output_root,
        seed=seed,
        version=version,
        force_rebuild=force,
    )


@baseline_app.command("smoke")
def baseline_smoke(
    baseline: Annotated[
        str,
        typer.Option(
            "--baseline",
            help="Baseline to run: metarag, ragas, or all.",
        ),
    ] = "metarag",
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Number of fixture queries to score."),
    ] = 2,
    output: Annotated[
        Path,
        typer.Option("--output", help="JSONL output path for baseline results."),
    ] = Path("experiments/results/baselines_smoke.jsonl"),
    threshold: Annotated[
        float,
        typer.Option("--threshold", min=0.0, max=1.0, help="Faithfulness threshold."),
    ] = 0.5,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Run a tiny shared-output baseline comparison smoke."""

    resolved = _load_or_exit(config)
    selected = baseline.lower()
    if selected not in {"metarag", "ragas", "all"}:
        console.print("[red]Baseline error:[/red] choose metarag, ragas, or all")
        raise typer.Exit(code=2)

    pipeline = FixtureRAGPipeline(config=resolved, corpus_path=corpus)
    examples = [
        BaselineExample(run=pipeline.run(query))
        for query in _default_smoke_queries(limit=queries)
    ]
    baselines: list[Any] = []
    if selected in {"metarag", "all"}:
        baselines.append(
            MetaRAGBaseline(
                verifier=NLIClaimVerifier(
                    backend=LexicalNLIBackend(),
                    model_id="fixture-lexical-nli",
                )
            )
        )
    if selected in {"ragas", "all"}:
        try:
            baselines.append(
                RagasBaseline(
                    scorer=OfficialRagasFaithfulnessScorer(config=resolved),
                )
            )
        except RuntimeError as error:
            console.print(f"[red]RAGAS error:[/red] {error}")
            raise typer.Exit(code=2) from error

    thresholds = {item.name: threshold for item in baselines}
    results = run_baselines(
        examples=examples,
        baselines=baselines,
        thresholds=thresholds,
    )
    manifest = write_baseline_outputs(
        results=results,
        output_path=output,
        thresholds=thresholds,
        metadata={
            "command": "mutoracle baseline smoke",
            "queries": queries,
        },
    )
    console.print(
        Panel.fit(
            f"Baseline smoke passed\nRuns: {manifest.run_count}\n"
            f"Baselines: {', '.join(manifest.baseline_names)}\n"
            f"Output: {output}",
            title="Baseline comparison",
        )
    )
    console.print_json(data=manifest.model_dump(mode="json"))


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


def _print_data_build(
    *,
    output_root: Path,
    seed: int,
    version: str,
    force_rebuild: bool,
) -> None:
    paths = build_fits_dataset(
        output_root=output_root,
        seed=seed,
        version=version,
        force_rebuild=force_rebuild,
    )
    console.print(
        Panel.fit(
            f"FITS build passed\nSeed: {seed}\nManifest: {paths['manifest']}",
            title="Data build",
        )
    )
    console.print_json(data={name: str(path) for name, path in paths.items()})


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


def _run_summary(run: RAGRun) -> dict[str, Any]:
    return {
        "query": run.query,
        "passage_count": len(run.passages),
        "passages": run.passages,
        "answer": run.answer,
    }


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


class _FixtureOracle:
    """Credential-free oracle used for deterministic localizer smoke runs."""

    def __init__(self, name: str, *, query_weight: float = 0.0) -> None:
        self.name = name
        self._query_weight = query_weight

    def score(self, run: RAGRun) -> float:
        support = content_similarity(context_text(run), run.answer)
        if self._query_weight == 0.0:
            return support
        query_alignment = content_similarity(run.query, run.answer)
        return (
            1.0 - self._query_weight
        ) * support + self._query_weight * query_alignment


def _fixture_oracles() -> list[_FixtureOracle]:
    return [
        _FixtureOracle("nli"),
        _FixtureOracle("semantic_similarity"),
        _FixtureOracle("llm_judge", query_weight=0.25),
    ]


def _real_oracles(
    config: MutOracleConfig,
) -> list[SemanticSimilarityOracle | NLIOracle | LLMJudgeOracle]:
    ledger = SQLiteCacheLedger(config.runtime.cache_path)
    return [
        NLIOracle(config=config, ledger=ledger),
        SemanticSimilarityOracle(config=config, ledger=ledger),
        LLMJudgeOracle(config=config, ledger=ledger),
    ]


def _release_check_report(*, strict_full_results: bool) -> dict[str, Any]:
    missing = [str(path) for path in RELEASE_REQUIRED_PATHS if not path.exists()]
    missing_full_results = [
        str(path) for path in FULL_RESULTS_REQUIRED_MANIFESTS if not path.exists()
    ]
    warnings: list[str] = []
    failures: list[str] = []
    if missing:
        failures.append("Missing required release files.")

    gitignore = Path(".gitignore")
    gitignore_text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    for required_pattern in (".env",):
        if required_pattern not in gitignore_text:
            failures.append(f".gitignore does not protect {required_pattern}.")

    full_results_mode = "complete" if not missing_full_results else "missing"
    if missing_full_results:
        message = (
            "Full E1-E6 result manifests are incomplete; run `make experiment-full` "
            "to regenerate missing artifacts."
        )
        if strict_full_results:
            failures.append(message)
        else:
            warnings.append(message)

    secret_hits = _scan_for_obvious_secrets(Path("."))
    if secret_hits:
        failures.append("Potential secrets found in public files.")

    status = "fail" if failures else "pass"
    return {
        "status": status,
        "strict_full_results": strict_full_results,
        "traceability_mode": "local-only",
        "full_results_mode": full_results_mode,
        "missing_required_files": missing,
        "missing_full_result_manifests": missing_full_results,
        "warnings": warnings,
        "failures": failures,
        "secret_hits": secret_hits,
    }


def _scan_for_obvious_secrets(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_secret_scan_excluded(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    hits.append(
                        {
                            "file": str(path),
                            "line": str(line_number),
                            "pattern": pattern.pattern,
                        }
                    )
    return hits


def _is_secret_scan_excluded(path: Path) -> bool:
    if path.name in SECRET_SCAN_EXCLUDED_NAMES:
        return True
    if path.suffix.lower() in SECRET_SCAN_EXCLUDED_SUFFIXES:
        return True
    return bool(set(path.parts) & SECRET_SCAN_EXCLUDED_DIRS)

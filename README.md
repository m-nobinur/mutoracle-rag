# MutOracle-RAG

MutOracle-RAG is a Python research package for mutation-driven fault
localization in RAG pipelines. The repository includes the FITS dataset, local
and LLM-backed oracle layers, experiment runners, baseline adapters, and
release-readiness checks.

## Start Here

```bash
uv sync --dev
uv run mutoracle config validate
make smoke
uv run pytest
```

The default smoke path is credential-free. Live LLM judging uses OpenRouter
through the OpenAI-compatible API when `OPENROUTER_API_KEY` is set.

## What Is Included

- a packaged CLI under `src/mutoracle/` for smoke checks, diagnosis, data
  builds, and release validation;
- a deterministic FITS dataset builder and frozen versioned data artifacts;
- mutation, oracle, aggregation, and calibrated localization workflows;
- baseline and experiment scripts that write reproducible result bundles.

## Common Tasks

| Task | Command |
| --- | --- |
| Inspect resolved config | `uv run mutoracle config show` |
| Build FITS manifests and data | `uv run mutoracle data build` |
| Run RAG smoke path | `uv run mutoracle rag smoke` |
| Run baseline smoke path | `uv run mutoracle baseline smoke --baseline metarag --queries 2` |
| Run development experiments | `make experiment-dev` |
| Regenerate analysis assets | `make analysis` |
| Run release readiness checks | `uv run mutoracle release-check` |

## Configuration

Development runs use `experiments/configs/dev.yaml` when it exists.
Use `.env` for secrets such as `OPENROUTER_API_KEY`, `HF_TOKEN`, or
`HUGGING_FACE_HUB_TOKEN`.

Install local model dependencies only when you need model-backed NLI or
semantic-oracle inference:

```bash
uv sync --extra oracles --dev
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for model choices,
aggregation settings, and the calibrated localizer config.

## Experiments, Data, and Release

- Experiment commands and artifact layout: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- FITS build process and schema: [docs/FITS_DATASET.md](docs/FITS_DATASET.md)
- Reproduction workflow from clean clone to full freeze: [docs/REPRODUCING.md](docs/REPRODUCING.md)
- Localizer decision rule and report schema: [docs/FAULT_LOCALIZER.md](docs/FAULT_LOCALIZER.md)
- MetaRAG baseline notes: [docs/METARAG_REIMPLEMENTATION.md](docs/METARAG_REIMPLEMENTATION.md)
- Public API surface: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

## Development Gate

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/mutoracle
uv run pytest
```

Equivalent Make targets are available:

```bash
make install
make lint
make test
make smoke
make release-check
```

# MutOracle-RAG

MutOracle-RAG is a Python research package for mutation-driven, stage-aware fault
localization in RAG pipelines. The committed design lock is in
[`docs/final-plan.md`](docs/final-plan.md), the phase index is in
[`docs/PHASE_PLAN.md`](docs/PHASE_PLAN.md), and completion is tracked in
[`docs/PHASE_STATUS.md`](docs/PHASE_STATUS.md).

## Current Status

Phase 1 bootstrap is in place:

- package metadata in `pyproject.toml`
- typed public contracts in `src/mutoracle/contracts.py`
- validated configuration loading in `src/mutoracle/config.py`
- a smoke CLI exposed as `mutoracle`
- pytest, ruff, and mypy configuration

Phase 2 is complete with a reproducible RAG system under test:

- fixture corpus and deterministic lexical retriever
- planned module layout under `pipeline/`, `providers/`, and `storage/`
- FAISS-compatible embedding index adapter with lightweight fixture fallback
- stable prompt builder
- OpenRouter generator wrapper through the OpenAI SDK
- SQLite response cache, cost ledger, token metadata, and latency metadata
- `mutoracle smoke --queries 10` for credential-free 10-query smoke runs

## Quickstart

```bash
uv sync --all-extras --dev
uv run mutoracle --help
uv run mutoracle config show
uv run mutoracle smoke --queries 10
uv run mutoracle rag smoke
uv run pytest
```

Model calls are not required for the bootstrap smoke path. Later phases will use
OpenRouter through the OpenAI-compatible API.

## Configuration

The project source of truth for development settings is
`experiments/configs/dev.yaml` when that file exists. Use `.env` for secrets,
especially `OPENROUTER_API_KEY`; the app loads `.env` automatically and masks the
key in `config show`.

```bash
uv run mutoracle config show
uv run mutoracle config validate
```

The current development config uses `openai/gpt-5-nano` as the Phase 2 generator
and `minimax/minimax-m2.5` as the later judge model. Change model IDs in
`experiments/configs/dev.yaml`.

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
```

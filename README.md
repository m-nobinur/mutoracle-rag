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

## Quickstart

```bash
uv sync --all-extras --dev
uv run mutoracle --help
uv run mutoracle config show
uv run pytest
```

Model calls are not required for the bootstrap smoke path. Later phases will use
OpenRouter through the OpenAI-compatible API.

## Configuration

Copy `.env.example` into your local shell or secret manager and set
`OPENROUTER_API_KEY` before running any command that calls an external model.
The bootstrap CLI can run without credentials.

```bash
uv run mutoracle config validate --config experiments/configs/dev.yaml
```

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
```

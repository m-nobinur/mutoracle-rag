# MutOracle-RAG

MutOracle-RAG is a Python research package for mutation-driven, stage-aware fault
localization in RAG pipelines. The committed design lock is in
[`docs/final-plan.md`](docs/final-plan.md), the phase index is in
[`docs/PHASE_PLAN.md`](docs/PHASE_PLAN.md), and completion is tracked in
[`docs/PHASE_STATUS.md`](docs/PHASE_STATUS.md).

## Current Status

Phases 1 through 4 are in place:

- repository bootstrap, typed contracts, config loading, CLI, and quality gates
- reproducible fixture RAG system under test with OpenRouter generator support
- seven canonical mutation operators: CI, CR, CS, QP, QN, FS, and FA
- NLI, semantic-similarity, and LLM-as-judge oracle modules
- shared SQLite response cache, oracle-score cache, cost ledger, and metadata
- `mutoracle smoke --queries 10` for credential-free 10-query smoke runs

## Quickstart

```bash
uv sync --dev
uv run mutoracle --help
uv run mutoracle config show
uv run mutoracle smoke --queries 10
uv run mutoracle rag smoke
uv run mutoracle mutate --operator CI
uv run pytest
```

Model calls are not required for the default smoke path. The LLM judge uses
OpenRouter through the OpenAI-compatible API when configured with
`OPENROUTER_API_KEY`.

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
and `minimax/minimax-m2.5` as the judge model. Local oracle model names are also
configured in `experiments/configs/dev.yaml`.

Install local model dependencies only when running real NLI or semantic oracle
inference:

```bash
uv sync --extra oracles --dev
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
make smoke
```

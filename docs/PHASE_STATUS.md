# Phase Status

## Phase 0: Design Lock

Status: complete.

Completed:

- Confirmed mutation IDs: CI, CR, CS, QP, QN, FS, FA.
- Confirmed v1 non-goals and minimum viable research fallback.
- Confirmed OpenRouter model IDs belong in config and run manifests.
- Confirmed `uv` for Python workflows and `gh` for PR/release flow.
- Added committed design lock at `docs/final-plan.md`.
- Added initial reference notes at `docs/REFERENCE_NOTES.md`.

Exit plan:

- `docs/final-plan.md` is the committed execution source of truth.
- Local phase notes under `dev-plan-source/phase-by-phase-dev-plan/` were updated to reflect completed Phase 0 and Phase 1 work.

## Phase 1: Repository Bootstrap

Status: complete.

Completed:

- Initialized package layout under `src/mutoracle`.
- Added `pyproject.toml`, `.python-version`, and `uv.lock`.
- Configured ruff, mypy, pytest, pytest-cov, pre-commit, and GitHub Actions.
- Added `.env.example` with OpenRouter and cost/cache settings.
- Added `mutoracle` CLI with help, config inspection, validation, and smoke.
- Added README quickstart and Makefile wrappers.
- Added unit tests for config, contracts, and CLI.

Validation:

- `uv sync --all-extras --dev`
- `uv run ruff check .`
- `uv run mypy src/mutoracle`
- `uv run pytest`
- `uv run mutoracle --help`

Exit plan:

- A clean clone can install, lint, type-check, test, and print CLI help without datasets or model credentials.
- `make install`, `make lint`, and `make test` are available.

## Phase 2: RAG System Under Test

Status: complete.

Completed:

- Added a packaged fixture corpus for reproducible local RAG runs.
- Added planned Phase 2 module exports under `src/mutoracle/pipeline/`,
  `src/mutoracle/providers/`, and `src/mutoracle/storage/`.
- Added deterministic lexical retrieval, prompt construction, and a FAISS-ready
  embedding index adapter with a pure-Python fixture fallback.
- Added an OpenRouter generator wrapper backed by the OpenAI SDK.
- Added a SQLite completion cache and usage ledger keyed by model, provider
  route, prompt hash, temperature, and seed.
- Added remote query and cost budget enforcement before live provider calls.
- Added prompt hash, token usage, seed, provider route, cost, and latency fields
  to RAG generation metadata.
- Added automatic `.env` loading for secrets and conventional discovery of
  `experiments/configs/dev.yaml` as the development config source of truth.
- Added `mutoracle rag smoke`, defaulting to credential-free fixture generation
  and supporting `--remote` for OpenRouter-backed generation.
- Added `mutoracle smoke --queries 10` and `make smoke` for the 10-query
  credential-free Phase 2 smoke path.

Validation:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy src/mutoracle`
- `uv run pytest`
- `uv run mutoracle smoke --queries 10`
- `uv run mutoracle rag smoke --query 'What is MutOracle-RAG?'`

Exit plan:

- The same seed and same cache state reproduce identical fixture `RAGRun`
  objects, including deterministic latency and token metadata fields.
- Live OpenRouter generation goes through the SQLite cache/cost ledger and is
  blocked when configured query or cost budgets are exhausted.

## Next Phase

Phase 3 should implement the mutation engine:

- seven operators: CI, CR, CS, QP, QN, FS, FA;
- registry by operator ID and stage;
- deterministic before/after examples for fixture `RAGRun` objects;
- unit tests proving each operator preserves the public schema.

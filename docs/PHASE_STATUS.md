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
- Local phase notes under `sources/phase-by-phase-dev-plan/` were updated to
  reflect completed Phase 0 and Phase 1 work.

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

- A clean clone can install, lint, type-check, test, and print CLI help without
  datasets or model credentials.
- `make install`, `make lint`, and `make test` are available.

## Next Phase

Phase 2 should implement the RAG system under test:

- local retriever and fixture corpus;
- deterministic prompt builder;
- OpenRouter provider wrapper;
- SQLite cache and cost ledger;
- reproducible RAG smoke command.

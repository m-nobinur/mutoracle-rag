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

## Phase 3: Mutation Engine

Status: complete.

Completed:

- Added `src/mutoracle/mutations/` with base copy helpers, registry lookup, and
  stage filtering.
- Implemented all seven canonical operators: CI, CR, CS, QP, QN, FS, and FA.
- Added deterministic rejection metadata for unsupported mutation cases.
- Added `mutoracle mutate --operator CI` for fixture-run mutation smoke checks.
- Added `make mutate` as the default mutation CLI wrapper.
- Documented operator behavior and before/after examples in
  `docs/MUTATION_TAXONOMY.md`.
- Added unit tests for schema preservation, determinism, CR edge cases,
  QP similarity rejection, QN grammar rejection, and FS/FA supported spans.

Validation:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy src/mutoracle`
- `uv run pytest`
- `uv run mutoracle smoke --queries 10`
- `uv run mutoracle mutate --operator CI`
- `uv run mutoracle mutate --operator CR`
- `uv run mutoracle mutate --operator CS`
- `uv run mutoracle mutate --operator QP`
- `uv run mutoracle mutate --operator QN`
- `uv run mutoracle mutate --operator FS`
- `uv run mutoracle mutate --operator FA`

Exit plan:

- Each canonical operator can run independently against fixture `RAGRun`
  objects and emit stable mutation metadata for downstream oracle scoring.

## Phase 4: Oracle Layer

Status: complete.

Completed:

- Added `src/mutoracle/oracles/` with base score helpers, detailed score
  metadata, and cache-backed oracle behavior.
- Implemented semantic similarity scoring with injectable or lazy-loaded
  `sentence-transformers` embeddings.
- Implemented NLI scoring with injectable or lazy-loaded Hugging Face
  `transformers` entailment probabilities.
- Implemented OpenRouter-backed LLM judge scoring with the configured judge
  model, locked prompt hash, strict Pydantic JSON validation, one retry, and
  structured invalid-response failure metadata.
- Extended the shared SQLite ledger with an `oracle_scores` table while keeping
  provider completions and usage/cost ledger behavior intact.
- Added optional `oracles` dependencies and config fields for local oracle model
  names.
- Added default pytest marker filtering so live provider tests are skipped unless
  explicitly selected.
- Documented score meaning, cache semantics, schema, and limitations in
  `docs/ORACLE_LAYER.md`.

Validation:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy src/mutoracle`
- `uv run pytest tests/unit/test_oracles.py tests/unit/test_cache.py tests/unit/test_provider.py tests/unit/test_phase_layout.py`
- `uv run pytest`

Exit plan:

- NLI, semantic similarity, and LLM judge oracles can score fixture `RAGRun`
  objects through normalized `[0, 1]` interfaces.
- Cached reruns avoid repeated injected model/provider calls.
- Invalid judge JSON retries once and then records a structured failure without
  blocking downstream aggregation.

## Phase 5: Aggregation and Localizer

Status: complete.

Completed:

- Added `src/mutoracle/aggregation/` with uniform, weighted, and
  confidence-gated aggregation strategies.
- Added config-backed aggregation weights, confidence gates, and
  `delta_threshold` validation.
- Added `src/mutoracle/localizer/` with transparent per-operator delta
  computation, per-stage max deltas, thresholded stage attribution, confidence,
  and evidence records.
- Added `mutoracle diagnose`, defaulting to credential-free fixture oracles and
  supporting `--real-oracles` for configured model-backed scoring.
- Added deterministic calibration script at `experiments/run_weight_search.py`.
- Added generated calibrated config at `experiments/configs/calibrated.yaml`.
- Documented the decision rule, config fields, CLI, and report schema in
  `docs/FAULT_LOCALIZER.md`.

Validation:

- `uv run python experiments/run_weight_search.py --seed 2026 --output experiments/configs/calibrated.yaml`
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy src/mutoracle`
- `uv run pytest tests/unit/test_aggregation.py tests/unit/test_localizer.py tests/unit/test_calibration.py tests/unit/test_cli.py tests/unit/test_config.py tests/unit/test_phase_layout.py`
- `uv run mutoracle diagnose`

Exit plan:

- Fixture examples produce stable `FaultReport` records with stage, confidence,
  per-operator deltas, stage deltas, and evidence.
- Aggregation weights and localizer thresholds are loaded from YAML config.
- The package is ready for Phase 6 FITS construction and validation splits.

## Phase 6: Data and FITS

Status: complete.

Completed:

- Added `src/mutoracle/data/` with source manifests, deterministic RGB and
  TriviaQA subset loaders, a Wikipedia/noise pool builder, FITS schema models,
  JSONL writing, checksums, and validation.
- Added `uv run mutoracle data build` and `uv run mutoracle fits build` for the
  Phase 6 data path.
- Added `data/README.md` manifest documentation and
  `data/fits/build_fits.py` as the script entry point requested by the phase
  plan.
- Added FITS v1.0.0 output generation with 75 examples each for retrieval,
  prompt, generation, and no-fault labels.
- Implemented prompt-stage FITS injection with canonical QN when supported and
  deterministic controlled-prompt fallback when QN rejects unsupported grammar.
- Added validation/test splits with no `qid` overlap and deterministic artifact
  checksums for fixed seeds.
- Added a 50-example audit sample and quality-gate report with label
  correctness threshold tracking.
- Added query-length distribution gate tracking source vs. FITS mean query
  lengths.
- Enforced frozen artifact behavior: existing `fits_v1.0.0` artifacts are reused
  by default and can only be rebuilt intentionally with `--force`.
- Documented the FITS schema, files, and quality gates in
  `docs/FITS_DATASET.md`.

Validation:

- `uv run ruff check src/mutoracle/data/fits.py src/mutoracle/cli.py tests/unit/test_data_fits.py data/fits/build_fits.py`
- `uv run mypy src/mutoracle`
- `uv run pytest tests/unit/test_data_fits.py`
- `uv run pytest tests/unit/test_cli.py tests/unit/test_phase_layout.py`
- `uv run pytest`
- `uv run mutoracle fits build --force`

Exit plan:

- FITS v1.0.0 is frozen locally with manifest, dataset card, quality-gate
  report, checksums, source provenance, and deterministic rebuild controls.

## Phase 7: Baselines

Status: complete.

Completed:

- Added `src/mutoracle/baselines/` with shared baseline result schemas,
  stable `RAGRun`-based run IDs, JSONL result writing, and sidecar manifests.
- Added an official RAGAS adapter for the `Faithfulness` metric on shared
  `RAGRun` objects plus answer relevancy, context precision, and context recall,
  with RAGAS imported lazily so non-baseline workflows remain credential-free.
- Added a documented MetaRAG approximation using spaCy-or-fallback claim
  extraction, synonym/antonym/factoid variants, and NLI-style claim
  verification.
- Added validation-only threshold tuning that rejects non-validation split
  records.
- Added shared generation/evaluator model ID merging and latency/cost breakdown
  metadata for directly comparable baseline records.
- Added `mutoracle baseline smoke`, `experiments/run_baselines.py`, and
  `make baseline` for tiny shared-output baseline result files.
- Documented MetaRAG implementation deviations in
  `docs/METARAG_REIMPLEMENTATION.md`.
- Added unit tests for RAGAS input compatibility, MetaRAG empty-claim handling,
  validation-only calibration, result schema fields, and baseline smoke output
  artifacts.

Validation:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy src/mutoracle`
- `uv run pytest tests/unit/test_baselines.py tests/unit/test_phase_layout.py tests/unit/test_cli.py`
- `uv run python experiments/run_baselines.py --baseline metarag --queries 2 --output /private/tmp/mutoracle-baselines-script.jsonl`
- `make baseline`
- `uv run pytest`

Exit plan:

- RAGAS and MetaRAG share comparable response-level result records over the same
  `RAGRun` outputs.
- Baseline results include latency, cost, model IDs, run IDs, thresholds, and
  score metadata for Phase 8 experiment scripts.
- The planned `experiments/run_baselines.py` entry point exists for Phase 8.
- Thresholds are selected from validation records only.

## Next Phase

Phase 8 implementation is complete, and phase-exit execution is in progress.

Completed:

- Added the E1-E6 experiment config suite under `experiments/configs/`.
- Added a shared Phase 8 experiment helper layer for config snapshots, raw
  JSONL output, summary CSVs, DuckDB import SQL, enriched manifests, failure
  logs, cost estimates, cost confirmation gates, `OPENROUTER_DAILY_USD_CAP`
  handling, and smoke-before-full checks.
- Added `experiments/run_mutoracle.py` for E2 FITS fault-attribution records.
- Upgraded `experiments/run_baselines.py` with a Phase 8 config-driven path for
  E1 detection comparison across RAGAS, MetaRAG, and a MutOracle detection
  variant while preserving the Phase 7 fixture CLI.
- Added `experiments/run_ablation.py` for E3 oracle ablations, E4 mutation
  operator ablations, and E6 aggregation comparisons.
- Added `experiments/run_latency.py` for E5 cost, latency, model ID, and
  overhead records.
- Added `docs/EXPERIMENT_PROTOCOL.md`, `docs/EXPERIMENTS.md`,
  `make experiment-smoke`, and `make experiment-full`.
- Added tests covering config presence, cost-gate blocking, smoke artifact
  creation, config snapshots, and seed recording.

Validation:

- `uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode smoke`
- `uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode smoke`
- `uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode smoke`
- `uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode smoke`
- `uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode smoke`
- `uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode smoke`
- `uv run mutoracle smoke --queries 20`
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy src/mutoracle`
- `uv run pytest`

Exit plan:

- E1-E6 can produce smoke outputs and full result records from config files.
- Every script writes raw JSONL, summary CSV, config snapshot, manifest, and
  failure log artifacts, plus DuckDB import SQL.
- Seeds `13`, `42`, and `91` are recorded in configs and manifests.
- Cost gates block above-cap runs unless confirmed.
- Full mode is protected by smoke-before-full gating.

Execution readiness notes:

- Smoke outputs for E1-E6 are present and reproducible.
- Full E1-E6 records across seeds are still required before declaring full
  Phase 8 exit.
- The master-plan dataset matrix still requires RGB-backed runs for E1, E3,
  E5, and E6 in addition to FITS-localization outputs.

## Next Phase

Phase 9 should consume the finalized Phase 8 result artifacts and produce
analysis and paper assets:

- generated tables for detection F1, FITS attribution accuracy, ablations,
  latency, and cost;
- figures for oracle-delta distributions and confusion matrices;
- reproducibility notes tying manifests and config snapshots to paper results.

# MutOracle-RAG

MutOracle-RAG is a Python research package for mutation-driven, stage-aware fault
localization in RAG pipelines. The canonical master plan is in
[`dev-plan-source/final-plan.md`](dev-plan-source/final-plan.md), and the
maintained documentation index is [`docs/README.md`](docs/README.md).

## Current Status

The end-to-end research artifact is complete and validated locally:

- repository bootstrap, typed contracts, config loading, CLI, and quality gates
- reproducible fixture RAG system under test with OpenRouter generator support
- eleven canonical mutation operators: CI, CR, CS, QP, QN, QD, QI, FS, FA,
  FE, and GN
- NLI, semantic-similarity, and LLM-as-judge oracle modules
- uniform, weighted, and confidence-gated aggregation strategies
- mutation-delta fault localization with calibrated `FaultReport` output
- deterministic data manifests and FITS v1.0.0 build/validation path
- RAGAS and MetaRAG-style baseline harnesses with shared result schemas
- E1-E6 experiment scripts, configs, smoke/full artifacts, and manifests
- DuckDB-backed analysis scripts with generated local analysis assets and
  run-ID traceability
- reproduction docs, API notes, and `mutoracle release-check`
- shared SQLite response cache, oracle-score cache, cost ledger, and metadata
- credential-free `smoke`, `mutate`, `diagnose`, and `data build` CLI paths

Full E1-E6 artifacts have been generated. Remaining release operations are
repository/tag governance tasks: final tag, GitHub release publication, and
optional FITS archival packaging.

## Quickstart

```bash
uv sync --dev
uv run mutoracle --help
uv run mutoracle config show
uv run mutoracle smoke --queries 10
uv run mutoracle rag smoke
uv run mutoracle mutate --operator CI
uv run mutoracle diagnose
uv run mutoracle data build
uv run mutoracle baseline smoke --help
uv run mutoracle release-check
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
Read-only Hugging Face access tokens can also live in `.env` as `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN`; the local model stack reads those from the environment
when it needs hub access.

```bash
uv run mutoracle config show
uv run mutoracle config validate
```

The current development config uses `openai/gpt-5-nano` as the generator and
`minimax/minimax-m2.5` as the judge model. Full-result runs use
`experiments/configs/phase8_real.yaml`, currently `morph/morph-v3-fast` for the
generator and `google/gemini-3.1-flash-lite-preview` for the judge. See
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the full model and cost
notes.

Install local model dependencies only when running real NLI or semantic oracle
inference:

```bash
uv sync --extra oracles --dev
```

Aggregation weights and the localizer delta threshold are configured under the
`aggregation` section. The calibrated config is generated with:

```bash
uv run python experiments/run_weight_search.py --seed 2026
```

See [`docs/FAULT_LOCALIZER.md`](docs/FAULT_LOCALIZER.md) for the localizer
decision rule and report schema.

## Baselines

The project includes response-level baseline comparison outputs for RAGAS and the local
MetaRAG approximation. The RAGAS adapter records faithfulness, answer relevancy,
context precision, and context recall when the official package is installed.
The shared runner writes JSONL records with stable `run_id`, normalized score,
threshold, predicted label, latency, model IDs, and per-metric details.
Rows also include metadata breakdowns that separate generation overhead from
baseline evaluator overhead for analysis.

```bash
uv run mutoracle baseline smoke --baseline metarag --queries 2
uv run python experiments/run_baselines.py --baseline metarag --queries 2
make baseline
```

RAGAS is loaded through the official package adapter when it is installed and an
evaluator model is configured. See
[`docs/METARAG_REIMPLEMENTATION.md`](docs/METARAG_REIMPLEMENTATION.md) for the
MetaRAG deviations and threshold-calibration rule.

## Experiments and Analysis

Experiment commands are documented in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Analysis assets regenerate from
saved result artifacts:

```bash
make experiment-smoke
make experiment-dev
make analysis
make analysis-dev
uv run python experiments/analyze_results.py --mode full
```

The analysis script imports raw JSONL into DuckDB, computes deterministic
bootstrap confidence intervals, writes local-only tables/figures/traceability,
and records run provenance. Use `dev` mode for small development runs with
progress logging. The localizer batches baseline and mutation oracle inputs,
and repeated runs still reuse the SQLite oracle cache.

## Release

Release artifacts and guides:

- reproduction guide: [`docs/REPRODUCING.md`](docs/REPRODUCING.md)
- API reference: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

Run the public-readiness check:

```bash
uv run mutoracle release-check
```

## Data and FITS

The data command builds source manifests and a deterministic FITS
fault-injection split:

```bash
uv run mutoracle data build
```

If `data/fits/fits_v1.0.0` already exists, the command reuses the frozen
artifact paths without rewriting files. To intentionally rebuild that version,
run:

```bash
uv run mutoracle fits build --force
```

The command writes dataset provenance to `data/manifests/datasets.json`, the
FITS build manifest to `data/fits/manifest.json`, and versioned validation/test
JSONL files under `data/fits/fits_v1.0.0/`. See
[`docs/FITS_DATASET.md`](docs/FITS_DATASET.md) for the schema and quality gates.

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

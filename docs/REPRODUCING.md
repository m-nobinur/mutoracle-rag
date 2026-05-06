# Reproducing MutOracle-RAG

This guide covers the shortest path from a clean clone to smoke validation,
development runs, and a full artifact freeze.

## Requirements

Required:

- Python 3.11 or 3.12
- `uv`
- Git

Optional:

- `OPENROUTER_API_KEY` for live generation and LLM judge calls
- local oracle dependencies for sentence-transformer and NLI inference

## 1. Clean Clone Setup

```bash
git clone <repo-url> mutoracle-rag
cd mutoracle-rag
uv sync --all-extras --dev
uv run mutoracle --help
uv run mutoracle config validate
```

The default smoke path is credential-free. Keep secrets in `.env`; do not
commit that file.

## 2. Smoke Reproduction

Run the core quality gate:

```bash
make check
make smoke
uv run mutoracle rag smoke
uv run mutoracle mutate --operator CI
uv run mutoracle diagnose
```

Then regenerate smoke experiment and analysis outputs:

```bash
make experiment-smoke
make analysis-smoke
uv run pytest tests/unit/test_analysis_assets.py --no-cov
```

Expected result: tests pass, E1-E6 smoke outputs are refreshed under
`experiments/results/`, and smoke analysis assets are refreshed under
`.local/analysis-assets/`.

## 3. Development-Scale Runs

```bash
make experiment-dev
make analysis-dev
```

Dev artifacts are intentionally separate from final full artifacts.

## 4. Full Artifact Freeze

```bash
make experiment-full
uv run python experiments/run_calibrated_localization.py
make analysis
uv run mutoracle release-check --strict-full-results
```

The strict release check passes when the full E1-E6 manifests exist.

## 5. Optional Live Model Setup

```bash
OPENROUTER_API_KEY=your-key-here
```

Install local oracle dependencies only when needed:

```bash
uv sync --extra oracles --dev
```

The configured SQLite cache reuses completions and oracle scores when the
prompt, model, provider route, temperature, and seed match.

## Notes

- Full runs may require local model downloads and longer runtime.
- FITS v1.0.0 JSONL files are frozen build artifacts and are ignored by default
  to avoid large generated data churn.

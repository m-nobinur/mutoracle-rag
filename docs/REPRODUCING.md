# Reproducing MutOracle-RAG

This guide provides a clean-clone workflow for:

- environment setup;
- credential-free smoke validation;
- development runs;
- full artifact regeneration.

## Prerequisites

Required:

- Python 3.11 or 3.12
- `uv`
- Git

Optional:

- OpenRouter API key for live generation and LLM judge calls
- Local oracle dependencies for sentence-transformer and NLI model inference

## 1) Clean Clone Setup

```bash
git clone <repo-url> mutoracle-rag
cd mutoracle-rag
uv sync --all-extras --dev
uv run mutoracle --help
uv run mutoracle config validate
```

The default smoke path is credential-free. Do not commit `.env`; it is ignored
by the repository.

## 2) Credential-Free Smoke Reproduction

Run the quality gate and core smoke commands:

```bash
make check
make smoke
uv run mutoracle rag smoke
uv run mutoracle mutate --operator CI
uv run mutoracle diagnose
```

Run smoke experiment scripts and regenerate smoke analysis assets:

```bash
make experiment-smoke
make analysis-smoke
uv run pytest tests/unit/test_analysis_assets.py --no-cov
```

Expected result: tests pass, E1-E6 smoke outputs are regenerated under
`experiments/results/`, and local analysis assets are regenerated under
`.local/analysis-assets/` with smoke provenance.

## 3) Development-Scale Reproduction

```bash
make experiment-dev
make analysis-dev
```

Development artifacts are intentionally separate from final full artifacts.
They are useful for inspecting runtime behavior before final release checks.

## 4) Full Artifact Freeze

Regenerate full artifacts and enforce strict full-result checks:

```bash
make experiment-full
uv run python experiments/run_calibrated_localization.py
make analysis
uv run mutoracle release-check --strict-full-results
```

The strict release check passes when full E1-E6 manifests exist.

## 5) Optional Live Model Setup

Create a local `.env` file:

```bash
OPENROUTER_API_KEY=your-key-here
```

Install local oracle dependencies only when needed:

```bash
uv sync --extra oracles --dev
```

The SQLite cache and usage ledger are stored under the configured runtime cache
path. Repeated calls reuse cached completions and oracle scores when the prompt,
model, provider route, temperature, and seed match.

## Reproduction Notes

- Full runs may require local model downloads or longer runtime.
- FITS v1.0.0 JSONL files are frozen build artifacts and are ignored by default
  to avoid large generated data churn.

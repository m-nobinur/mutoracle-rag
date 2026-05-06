# API Reference

This document summarizes the public Python and CLI surfaces used by the
experiments and release workflows. The source package is `src/mutoracle`.

## Core Contracts

`mutoracle.contracts`

- `RAGRun`: immutable-style record carrying `query`, retrieved `passages`,
  generated `answer`, and free-form `metadata`.
- `RAGPipeline`: protocol with `run(query: str) -> RAGRun`.
- `MutationOperator`: protocol for stage-tagged mutation operators.
- `Oracle`: protocol exposing `score(run: RAGRun) -> float`.
- `Aggregator`: protocol for combining oracle scores.
- `FaultReport`: localizer output with predicted stage, confidence,
  per-operator deltas, per-stage deltas, and evidence.

## RAG System Under Test

`mutoracle.rag` and `mutoracle.pipeline`

- `FixtureRAGPipeline`: deterministic fixture-backed RAG pipeline.
- `LexicalRetriever`: dependency-light lexical retriever for fixture and smoke
  runs.
- `build_rag_prompt`: prompt construction helper.
- `OpenRouterProvider`: OpenAI-compatible remote provider wrapper with cache
  and usage tracking.

## Mutation Operators

`mutoracle.mutations`

- `get_operator(operator_id)`: returns one canonical operator by ID.
- `list_operator_ids()`: returns `CI`, `CR`, `CS`, `QP`, `QN`, `FS`, and `FA`.
- `mutation_registry()`: registry used by the localizer and tests.

Canonical stages:

- retrieval/context: `CI`, `CR`, `CS`
- prompt/query: `QP`, `QN`
- generation/answer: `FS`, `FA`

## Oracles

`mutoracle.oracles`

- `NLIOracle`: entailment-style support score.
- `SemanticSimilarityOracle`: embedding similarity score.
- `LLMJudgeOracle`: strict JSON LLM judge through OpenRouter.

All oracle scores are normalized to `[0, 1]`, where larger values indicate
greater support or faithfulness. Cache-backed implementations store score
metadata in the shared SQLite ledger.

## Aggregation

`mutoracle.aggregation`

- `UniformAggregator`: equal-weight oracle combination.
- `WeightedAggregator`: configured or calibrated weighted combination.
- `ConfidenceGatedAggregator`: suppresses low-confidence oracle signals before
  combining.
- `build_aggregator(config)`: constructs the configured strategy.

Composite score:

```text
Omega = w_nli * Omega_nli + w_sim * Omega_sim + w_llm * Omega_llm
```

## Fault Localization

`mutoracle.localizer`

- `FaultLocalizer`: runs baseline and mutated variants, computes
  `Delta_i = Omega_0 - Omega_i`, aggregates stage deltas, and returns a
  `FaultReport`.
- `fault_report_to_dict(report)`: JSON-friendly serializer used by the CLI and
  experiments.

Decision rule:

```text
predicted_stage = argmax_stage Delta_stage
if max(Delta_stage) <= delta_threshold:
    predicted_stage = no_fault_detected
```

## Data and FITS

`mutoracle.data`

- `build_fits_dataset(...)`: builds or reuses the frozen FITS v1.0.0 artifacts.
- `validate_fits_records(...)`: validates schema and split constraints.
- dataset loaders and manifest helpers under `mutoracle.data.loaders` and
  `mutoracle.data.manifest`.

## Baselines

`mutoracle.baselines`

- `RagasBaseline`: adapter around official RAGAS faithfulness when installed.
- `MetaRAGBaseline`: documented local approximation of MetaRAG-style claim
  perturbation and verification.
- `run_baselines(...)`: runs shared-output baseline evaluation.
- `write_baseline_outputs(...)`: writes JSONL result rows and sidecar manifest.

## CLI

Installed entry point:

```bash
uv run mutoracle --help
```

Important commands:

- `mutoracle config show`
- `mutoracle config validate`
- `mutoracle smoke --queries 20`
- `mutoracle rag smoke`
- `mutoracle mutate --operator CI`
- `mutoracle diagnose`
- `mutoracle data build`
- `mutoracle fits build`
- `mutoracle baseline smoke`
- `mutoracle release-check`

## Experiment Entry Points

Scripts under `experiments/`:

- `run_baselines.py`: E1 detection and baseline smoke.
- `run_mutoracle.py`: E2 FITS localization.
- `run_ablation.py`: E3 oracle ablation, E4 mutation separability, and E6
  aggregation comparisons.
- `run_latency.py`: E5 runtime latency audit artifact.
- `run_weight_search.py`: calibration search.
- `analyze_results.py`: analysis table, figure, DuckDB, and traceability generation.

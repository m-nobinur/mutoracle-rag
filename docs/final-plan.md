# MutOracle-RAG Design Lock

This document is the committed Phase 0 design lock for MutOracle-RAG. The larger
local research notes under `sources/` are intentionally not part of the initial
repository history.

## Project Identity

Recommended title:

`Pipeline-Aware Fault Localization in Enterprise RAG Systems: A Mutation-Driven Oracle Aggregation Framework`

Short title:

`MutOracle-RAG`

Primary research question:

Can a mutation-driven oracle aggregation framework localize the source of
hallucination failures within a multi-stage RAG pipeline more effectively than
response-level detection baselines such as RAGAS and MetaRAG?

Core claim:

Existing RAG evaluation tools can detect that a response may be unfaithful, but
they usually do not identify whether the likely failure came from retrieval,
prompt/context construction, or generation. MutOracle-RAG probes the pipeline
with controlled mutations and uses multi-oracle score deltas to attribute the
fault to a pipeline stage.

## Locked Scope

Version 1 goals:

- Build a reproducible black-box RAG system under test.
- Implement seven canonical mutation operators grouped by pipeline stage.
- Implement NLI, semantic-similarity, and LLM-as-judge oracle signals.
- Aggregate oracle scores with uniform, weighted, and confidence-gated methods.
- Return fault stage, confidence, deltas, and evidence.
- Build FITS, a single-stage fault-injection split for localization evaluation.
- Compare against official RAGAS and a documented MetaRAG reimplementation.
- Produce a reproducible result package and a 6-8 page paper.

Version 1 non-goals:

- No white-box or mechanistic model inspection.
- No compound multi-stage failure attribution as a headline claim.
- No automated repair or remediation loop.
- No model fine-tuning.
- No multilingual evaluation.
- No production web app before paper-critical work is complete.

Minimum viable research fallback:

- Keep the mutation taxonomy, FITS construction, and localization analysis even
  if headline accuracy is mixed.
- Report negative results honestly with calibrated thresholds and shared splits.
- Preserve the paper claim as pipeline-stage fault localization, not generic
  hallucination detection.

## Mutation Taxonomy

Use these IDs everywhere in code, configs, docs, plots, and result records:

| ID | Operator | Stage |
| --- | --- | --- |
| CI | Context Injection | retrieval |
| CR | Context Removal | retrieval |
| CS | Context Shuffle | retrieval |
| QP | Query Paraphrase | prompt |
| QN | Query Negation | prompt |
| FS | Factoid Synonym Substitution | generation |
| FA | Factoid Antonym Substitution | generation |

CI, CR, CS, QP, and QN mutate pipeline inputs or intermediate artifacts. FS and
FA mutate generated answers to stress-test oracle behavior.

## Public Contracts

The initial package exposes the stable contracts in
`src/mutoracle/contracts.py`:

- `RAGRun`
- `RAGPipeline`
- `MutationOperator`
- `Oracle`
- `Aggregator`
- `FaultReport`

## Scoring Rule

Composite oracle score:

```text
Omega = w_nli * Omega_nli + w_sim * Omega_sim + w_llm * Omega_llm
```

Per-operator delta:

```text
Delta_i = Omega_0 - Omega_i
```

Stage decision:

```text
Delta_stage = max(Delta_i for operators targeting that stage)
predicted_stage = argmax_stage Delta_stage
```

If the maximum stage delta is below the calibrated threshold, the result is
`no_fault_detected`.

## Tooling Decisions

- Main runtime: Python 3.11 or 3.12, pinned locally to 3.12.
- Environment and locking: `uv`.
- CLI: Typer plus Rich.
- Config: Pydantic plus YAML.
- Provider: OpenRouter through the OpenAI-compatible API.
- Retrieval: sentence-transformers embeddings plus FAISS.
- Local storage: SQLite for model cache/cost ledger, DuckDB for result analysis.
- Quality gate: ruff, mypy, pytest, pytest-cov, pre-commit, GitHub Actions.

Model IDs are config values, not method requirements. The bootstrap defaults are
cheap smoke values and can be overridden by YAML or environment.

## Phase 0 Acceptance

- Canonical mutation IDs are locked: CI, CR, CS, QP, QN, FS, FA.
- Version 1 non-goals and fallback scope are locked.
- OpenRouter-through-OpenAI-SDK provider decision is locked for v1.
- `uv` and GitHub CLI workflow are accepted as the development path.
- The paper claim is pipeline-stage fault localization.

# Documentation Index

This directory contains the maintained documentation for the current
MutOracle-RAG artifact. Historical phase notes and planning drafts live under
`dev-plan-source/`; the files below are the ones to use when reproducing,
reviewing, or extending the project.

## Essential Guides

- [REPRODUCING.md](REPRODUCING.md): clean-clone setup, smoke checks, and full
  artifact regeneration.
- [EXPERIMENTS.md](EXPERIMENTS.md): E1-E6 commands, artifact layout, and
  analysis workflow.
- [CONFIGURATION.md](CONFIGURATION.md): model choices, aggregation settings,
  and current configuration assessment.
- [API_REFERENCE.md](API_REFERENCE.md): public package surfaces used by the
  experiment scripts.

## Method References

- [MUTATION_TAXONOMY.md](MUTATION_TAXONOMY.md): CI, CR, CS, QP, QN, FS, and FA.
- [ORACLE_LAYER.md](ORACLE_LAYER.md): NLI, semantic-similarity, and LLM-judge
  scoring.
- [FAULT_LOCALIZER.md](FAULT_LOCALIZER.md): aggregation, delta computation, and
  report schema.
- [FITS_DATASET.md](FITS_DATASET.md): FITS build, schema, and quality gates.
- [METARAG_REIMPLEMENTATION.md](METARAG_REIMPLEMENTATION.md): local MetaRAG
  approximation and deviations.

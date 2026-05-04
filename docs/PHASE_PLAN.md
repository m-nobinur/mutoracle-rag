# Phase-by-Phase Development Plan

| Phase | Name | Main output |
| --- | --- | --- |
| 0 | Design Lock | accepted scope and source decisions |
| 1 | Repository Bootstrap | package skeleton, `uv`, CI, smoke CLI |
| 2 | RAG System Under Test | reproducible RAG pipeline, provider, and cache |
| 3 | Mutation Engine | seven operators and registry |
| 4 | Oracle Layer | NLI, similarity, LLM judge, cache |
| 5 | Aggregation and Localizer | calibrated fault reports |
| 6 | Data and FITS | frozen FITS v1.0.0 |
| 7 | Baselines | RAGAS and MetaRAG harnesses |
| 8 | Experiments | E1-E6 result records |
| 9 | Analysis and Paper Assets | generated tables and figures |
| 10 | Paper and Release | final paper, docs, tag, optional DOI |

## Workflow

Each phase should become one branch or one small pull request:

```bash
uv sync --all-extras --dev
uv run ruff format .
uv run ruff check .
uv run mypy src/mutoracle
uv run pytest
gh pr create --fill
```

If a phase grows large, split by module ownership rather than by tooling layer.

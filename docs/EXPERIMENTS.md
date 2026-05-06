# Experiments

Experiments are configured from YAML files under `experiments/configs/`. Every
runner writes the same artifact bundle beside its raw result file:

- raw JSONL records;
- summary CSV;
- config snapshot YAML;
- run manifest JSON;
- failure JSONL with per-example reasons.

Use saved artifacts as the source of truth. Do not hand-edit metrics and do not
run release experiments from notebooks.

## Workflow

| Mode | When to use it | Commands |
| --- | --- | --- |
| Smoke | First check after a clean clone or CLI change | `make experiment-smoke` |
| Dev | Iterating on code or prompts | `make experiment-dev` and `make analysis-dev` |
| Full | Freezing release-facing artifacts | `make experiment-full`, `uv run python experiments/run_calibrated_localization.py`, `make analysis` |

Smoke mode is credential-free where possible. Dev mode writes separate
`*_dev_*` artifacts, uses 20 queries and seed `13`, and keeps output small.
Full mode records seeds `13`, `42`, and `91` unless the config overrides them.

## Raw Smoke Commands

```bash
uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode smoke
uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode smoke
uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode smoke
```

## Experiment Mapping

| Experiment | Config | Script | Purpose |
| --- | --- | --- | --- |
| E1 | `e1_detection.yaml` | `run_baselines.py` | RAGAS, MetaRAG, and MutOracle response-level detection |
| E2 | `e2_localization.yaml` | `run_mutoracle.py` | FITS fault-attribution accuracy |
| E2 calibrated | saved E2 + FITS validation | `run_calibrated_localization.py` | validation-trained centroid localizer |
| E3 | `e3_ablation.yaml` | `run_ablation.py` | Single-oracle and all-oracle ablations |
| E4 | `e4_separability.yaml` | `run_ablation.py` | Mutation/operator ablation and delta records |
| E5 | `e5_latency.yaml` | `run_latency.py` | Runtime latency audit artifact |
| E6 | `e6_weighted.yaml` | `run_ablation.py` | Uniform, weighted, and confidence-gated aggregation comparison |

## Analysis Outputs

Regenerate analysis assets from saved artifacts with:

```bash
make analysis
```

This writes local-only outputs to `.local/analysis-assets/` and materializes
DuckDB at `experiments/results/analysis.duckdb`. Use `make analysis-smoke` for
credential-free validation and `make analysis-dev` for development-scale runs.

Release summaries mainly rely on E1, calibrated E2, and E4 outputs. E3, E5,
and E6 are still part of the reproducibility record but are not the main
headline tables.

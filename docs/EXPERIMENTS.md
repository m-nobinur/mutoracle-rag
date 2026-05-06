# Experiments

Experiments run from YAML configs under `experiments/configs/`. Each script
writes the same artifact bundle beside its raw result file:

- raw JSONL records;
- summary CSV;
- config snapshot YAML;
- run manifest JSON;
- failure JSONL with per-example reasons.

Do not run experiments from notebooks, and do not type metrics manually.
Analysis assets are generated from saved artifacts by
`experiments/analyze_results.py`.

Use smoke mode to check plumbing, dev mode while changing code, and full mode
only when freezing release-facing artifacts.

## Smoke

```bash
uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode smoke
uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode smoke
uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode smoke
```

Smoke mode is credential-free where possible and is the right first command
after a clean clone.

## Development

```bash
make experiment-dev
make analysis-dev
```

`dev` mode writes separate `*_dev_*` artifacts, uses 20 queries and seed `13`,
and prints progress updates while each script is running.

## Full Artifact Runs

```bash
make experiment-full
uv run python experiments/run_calibrated_localization.py
make analysis
uv run mutoracle release-check --strict-full-results
```

Each config records seeds `13`, `42`, and `91` unless explicitly overridden.
Full mode is blocked until a smoke manifest exists, unless the operator passes
`--confirmed-smoke`.

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

## Analysis Assets

Analysis consumes saved artifacts and regenerates deterministic tables, figures,
and run-ID traceability assets without hand-entered metrics:

```bash
make analysis
```

`make analysis` writes outputs to a local-only `.local/analysis-assets/`
directory and materializes DuckDB at `experiments/results/analysis.duckdb`.
Use `make analysis-smoke` for credential-free workflow validation and
`make analysis-dev` for development-scale checks.

Release summaries primarily use E1, E2 calibrated, and E4 result tables. E3,
E5, and E6 assets are retained for reproducibility review but are not treated
as headline evidence. E4 reports both applied counts and rejection rates because
some mutation operators are intentionally guarded; a zero delta is valid, but it
may mean either a neutral applied mutation or a rejected mutation with no
behavioral score.

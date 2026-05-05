# Phase 8 Experiment Workflow

Canonical protocol: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

Phase 8 experiments run from YAML configs under `experiments/configs/`. Scripts
write a consistent artifact set beside each result file:

- raw JSONL records;
- summary CSV;
- config snapshot YAML;
- run manifest JSON;
- failure JSONL with per-example reasons.

## Smoke Commands

```bash
uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --mode smoke
uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --mode smoke
uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --mode smoke
uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --mode smoke
```

Each config records seeds `13`, `42`, and `91`. Full mode is blocked until a
smoke manifest exists, unless the operator explicitly passes
`--confirmed-smoke`. Estimated costs are checked before work starts; runs above
the configured cap require `--confirm-cost`.

Current state:

- Smoke outputs are available for E1-E6 and are suitable for workflow
 validation.
- Full-run outputs are required before final Phase 9 paper tables and figures.
- Planned dataset alignment still requires RGB-backed runs for E1, E3, E5, and
 E6 in addition to FITS-localization runs.

## Experiment Mapping

| Experiment | Config | Script | Purpose |
| --- | --- | --- | --- |
| E1 | `e1_detection.yaml` | `run_baselines.py` | RAGAS, MetaRAG, and MutOracle response-level detection |
| E2 | `e2_localization.yaml` | `run_mutoracle.py` | FITS fault-attribution accuracy |
| E3 | `e3_ablation.yaml` | `run_ablation.py` | Single-oracle and all-oracle ablations |
| E4 | `e4_separability.yaml` | `run_ablation.py` | Mutation/operator ablation and delta records |
| E5 | `e5_latency.yaml` | `run_latency.py` | Cost, latency, and overhead reporting |
| E6 | `e6_weighted.yaml` | `run_ablation.py` | Uniform, weighted, and confidence-gated aggregation comparison |

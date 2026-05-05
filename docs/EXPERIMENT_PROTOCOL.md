# Experiment Protocol

Phase 8 experiments are config-driven. Do not run paper experiments from
notebooks, and do not type paper metrics manually. Phase 9 analysis should
consume the saved result artifacts.

Each run writes:

- raw JSONL records;
- summary CSV;
- DuckDB import SQL;
- config snapshot YAML;
- run manifest JSON;
- failure JSONL with per-example reasons.

Manifests record run ID, git commit, seeds, dataset checksum, SDK versions,
model IDs, provider routing where available, latency, token-count fields, and
estimated cost.

## Smoke Suite

```bash
make experiment-smoke
```

Equivalent explicit commands:

```bash
uv run python experiments/run_baselines.py --experiment-config experiments/configs/e1_detection.yaml --smoke
uv run python experiments/run_mutoracle.py --config experiments/configs/e2_localization.yaml --smoke
uv run python experiments/run_ablation.py --config experiments/configs/e3_ablation.yaml --smoke
uv run python experiments/run_ablation.py --config experiments/configs/e4_separability.yaml --smoke
uv run python experiments/run_latency.py --config experiments/configs/e5_latency.yaml --smoke
uv run python experiments/run_ablation.py --config experiments/configs/e6_weighted.yaml --smoke
```

## Full Suite

Full mode requires a passing smoke manifest unless `--confirmed-smoke` is
provided. Runs above the configured or `OPENROUTER_DAILY_USD_CAP` cost cap
require `--confirm-cost`.

```bash
make experiment-full
```

If you intentionally bypass smoke handoff for a one-off debug run, call the
script directly with `--confirmed-smoke` and record that override in your notes.

## Experiment Matrix

| ID | Config | Script | Output target |
| --- | --- | --- | --- |
| E1 | `e1_detection.yaml` | `run_baselines.py` | detection table over RAGAS, MetaRAG, and MutOracle |
| E2 | `e2_localization.yaml` | `run_mutoracle.py` | FITS attribution table and confusion matrix input |
| E3 | `e3_ablation.yaml` | `run_ablation.py` | single-oracle and leave-one-out oracle ablations |
| E4 | `e4_separability.yaml` | `run_ablation.py` | mutation/operator ablation and delta records |
| E5 | `e5_latency.yaml` | `run_latency.py` | latency, cost, and overhead table |
| E6 | `e6_weighted.yaml` | `run_ablation.py` | uniform, weighted, and confidence-gated comparison |

## Seeds

All experiment configs record seeds `13`, `42`, and `91`. Summary records are
grouped by seed where applicable so Phase 9 can compute means, standard
deviations, confidence intervals, and significance tests from saved files.

## Readiness Checkpoint

- Current checked-in outputs are smoke artifacts that validate script and
 artifact behavior.
- Before paper-facing Phase 9 analysis, complete full E1-E6 runs and confirm
 the planned dataset matrix (RGB-driven E1/E3/E5/E6 plus FITS localization).
- Treat the current smoke outputs as plumbing validation, not final reported
 metrics.

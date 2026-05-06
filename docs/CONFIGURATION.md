# Configuration and Model Choices

MutOracle-RAG uses runtime configs plus experiment configs. Runtime configs
define model/oracle behavior. Experiment configs define datasets, splits,
seeds, and script-specific variants.

## Configuration Layers

| File | Purpose |
| --- | --- |
| `experiments/configs/dev.yaml` | Default development runtime config. |
| `experiments/configs/phase8_real.yaml` | Runtime config used by full-result experiment runs. |
| `experiments/configs/e*.yaml` | Experiment setup (query limits, seeds, dataset split, run mode behavior, and script sections). |

Common experiment configs include `e1_detection.yaml`, `e2_localization.yaml`,
`e3_ablation.yaml`, `e4_separability.yaml`, `e5_latency.yaml`, and
`e6_weighted.yaml`.

## Inspect and Validate

Inspect the resolved active configuration:

```bash
uv run mutoracle config show
```

Validate the resolved active configuration:

```bash
uv run mutoracle config validate
```

Validate a specific config file:

```bash
uv run mutoracle config validate --config experiments/configs/dev.yaml
```

## Models

Current development config:

| Component | Model |
| --- | --- |
| Generator | `openai/gpt-5-nano` |
| LLM judge | `minimax/minimax-m2.5` |
| NLI oracle | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Semantic oracle | `sentence-transformers/all-mpnet-base-v2` |

Current full-result experiment config:

| Component | Model |
| --- | --- |
| Generator | `morph/morph-v3-fast` |
| LLM judge | `google/gemini-3.1-flash-lite-preview` |
| NLI oracle | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Semantic oracle | `sentence-transformers/all-mpnet-base-v2` |

Some full artifacts also include deterministic fixture model IDs such as
`fixture-fits-generator`, `fixture-rgb-generator`, `fixture-nli`,
`fixture-semantic-similarity`, and `fixture-llm-judge`. Those are not external
models; they identify credential-free fixture paths used for reproducibility and
smoke validation.

## Aggregation

Default weighted localizer settings:

```yaml
weights:
  nli: 0.4
  semantic_similarity: 0.3
  llm_judge: 0.3
delta_threshold: 0.05
```

Calibrated E6 variant settings:

```yaml
weights:
  nli: 0.4
  semantic_similarity: 0.2
  llm_judge: 0.4
delta_threshold: 0.03
```

The calibrated config is stored in `experiments/configs/calibrated.yaml` and is
generated through:

```bash
uv run python experiments/run_weight_search.py --seed 2026
```

## Current Assessment

This configuration is good enough for reproducible artifact review. The
transparent max-delta rule remains an auditable baseline, but full FITS results
show retrieval mutations dominate raw stage deltas. Current release analyses
therefore report validation-calibrated localizers over eleven operator-delta
features: nearest-centroid reaches 86.7% held-out FITS accuracy and the small
logistic full-delta localizer reaches 90.0%.

The next configuration pass should evaluate stronger judge models, compound
faults, and external RAG benchmarks before claiming production-ready diagnosis.

## Recommended Next Pass

1. Freeze one generator/judge pair across all non-fixture full runs.
2. Evaluate stronger judge models such as Gemini 3 Pro or Claude Sonnet 4.5.
3. Expand prompt and generation mutation operators for non-factoid claims.
4. Validate the logistic localizer on external RAG benchmarks.
5. Re-run E1-E6 full artifacts and refresh local analysis assets before
  updating release summaries.

# Configuration

MutOracle-RAG uses two config layers:

- runtime configs for models, oracle behavior, cache settings, and aggregation;
- experiment configs for dataset split, seeds, run mode, and script-specific
  parameters.

## Files That Matter

| File | Purpose |
| --- | --- |
| `experiments/configs/dev.yaml` | Default runtime config for development and most local commands. |
| `experiments/configs/phase8_real.yaml` | Runtime config used for full-result artifact generation. |
| `experiments/configs/e*.yaml` | Experiment definitions for E1-E6 runs. |
| `experiments/configs/calibrated.yaml` | Generated calibrated localizer settings. |

## Inspect and Validate

```bash
uv run mutoracle config show
uv run mutoracle config validate
uv run mutoracle config validate --config experiments/configs/dev.yaml
```

## Environment Variables

- `OPENROUTER_API_KEY`: enables live generation and LLM judge calls.
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`: optional read-only access for local
  model downloads.

The application loads `.env` automatically and masks secrets in `config show`.

## Current Model Sets

Development config:

| Component | Model |
| --- | --- |
| Generator | `openai/gpt-5-nano` |
| LLM judge | `minimax/minimax-m2.5` |
| NLI oracle | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Semantic oracle | `sentence-transformers/all-mpnet-base-v2` |

Full-result config:

| Component | Model |
| --- | --- |
| Generator | `morph/morph-v3-fast` |
| LLM judge | `google/gemini-3.1-flash-lite-preview` |
| NLI oracle | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Semantic oracle | `sentence-transformers/all-mpnet-base-v2` |

Some artifacts also use fixture model IDs such as `fixture-fits-generator` or
`fixture-llm-judge`. Those labels refer to deterministic credential-free test
paths, not external providers.

## Local Oracle Dependencies

Install the extra packages only when you need model-backed NLI or
semantic-oracle inference:

```bash
uv sync --extra oracles --dev
```

## Aggregation Settings

Default weighted localizer:

```yaml
weights:
  nli: 0.4
  semantic_similarity: 0.3
  llm_judge: 0.3
delta_threshold: 0.05
```

Calibrated localizer:

```yaml
weights:
  nli: 0.4
  semantic_similarity: 0.2
  llm_judge: 0.4
delta_threshold: 0.03
```

Regenerate the calibrated config with:

```bash
uv run python experiments/run_weight_search.py --seed 2026
```

## Release Note

The release analysis keeps the transparent max-delta rule for audit, but
headline localization results use validation-calibrated localizers derived from
FITS validation data.

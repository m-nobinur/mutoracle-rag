# Oracle Layer

Phase 4 adds three normalized faithfulness signals for `RAGRun` objects:

| Oracle | Module | Score meaning |
| --- | --- | --- |
| NLI | `mutoracle.oracles.NLIOracle` | Entailment probability that the answer is supported by retrieved context. |
| Semantic similarity | `mutoracle.oracles.SemanticSimilarityOracle` | Context/answer cosine similarity mapped from `[-1, 1]` into `[0, 1]`. |
| LLM judge | `mutoracle.oracles.LLMJudgeOracle` | Strict JSON faithfulness verdict mapped to a normalized support score. |

All public `score(run)` methods return a float in `[0, 1]`, where higher means
more supported by retrieved context. Use `score_result(run)` when experiment
code needs audit metadata such as model name, input hash, prompt hash, raw
cosine, label probabilities, judge verdict, retry count, or cache status.

## Cache Contract

Oracle inputs are keyed by stable hashes of query, passages, and answer. Scores
are stored in the shared SQLite ledger through `oracle_scores`, separate from
provider completion cache rows.

The LLM judge has two cache layers:

- raw OpenRouter completions are cached by model, prompt hash, temperature, and
  seed;
- validated judge scores are cached by input hash, prompt hash, model, and
  schema version.

Cached reruns return the same score and mark `metadata["cache_hit"] = True`
without invoking the model backend or provider.

## Local Model Loading

The semantic and NLI oracles lazy-load their heavy libraries only when real
inference is requested:

- `sentence-transformers` for semantic similarity;
- `transformers` for NLI.

These dependencies are available through the optional oracle extra:

```bash
uv sync --extra oracles --dev
```

Unit tests inject tiny fake backends so the default suite stays credential-free
and does not download models.

## LLM Judge Schema

The locked judge output schema is:

```json
{
  "verdict": "faithful",
  "confidence": 0.92,
  "reason": "short reason"
}
```

Allowed verdicts are `faithful` and `hallucinated`. A faithful verdict maps to
`confidence`; a hallucinated verdict maps to `1 - confidence`.

Invalid JSON or schema failures retry once with a stricter prompt. If validation
still fails, the oracle records a structured `invalid_judge_response` failure
and returns `0.0` so downstream aggregation can continue deterministically.

## Known Limitations

- Semantic similarity is relatedness, not factual entailment; high similarity
  can still hide contradictions.
- The NLI oracle depends on the chosen checkpoint and context length behavior.
  Long contexts should be chunked or calibrated in later phases.
- The LLM judge is the only remote/paid oracle path and must remain behind the
  SQLite cache and cost budget.
- Provider-backed tests must use the `provider` pytest marker and are skipped by
  the default test command.

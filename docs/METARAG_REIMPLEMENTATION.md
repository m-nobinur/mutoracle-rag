# MetaRAG Reimplementation Notes

The project includes a transparent MetaRAG-style approximation for baseline
comparison. It is intentionally limited to the parts needed by MutOracle-RAG:
response-level hallucination detection on the same `RAGRun` objects used by the
localizer.

## Implemented Path

The local approximation is in `src/mutoracle/baselines/metarag_baseline.py`.

It performs:

- spaCy sentence extraction when `spacy` and `en_core_web_sm` are installed,
  with deterministic sentence splitting as the credential-free fallback;
- NLI-style verification of each claim against the concatenated retrieved
  passages;
- metamorphic synonym, antonym, and factoid variants using WordNet when
  available plus deterministic local fallback rules;
- violation scoring for variants: synonym variants are expected to remain
  supported, while antonym and factoid variants are expected not to be supported;
- faithfulness scoring as the share of extracted claims whose entailment score
  reaches the configured claim threshold, discounted by metamorphic variant
  violations;
- conversion to the shared baseline label using a validation-tuned
  faithfulness threshold.

The output schema is shared with RAGAS and includes:

- stable `run_id`;
- `baseline_name`;
- normalized `score`;
- threshold and predicted response-level label;
- latency, model IDs, per-metric scores, and metadata.

## Deviations

There is no official MetaRAG reference implementation vendored in this project.
This reimplementation therefore makes the following explicit deviations:

- claim extraction uses spaCy sentence boundaries plus deterministic fallback
  instead of an LLM claim extractor;
- claim verification uses the configured NLI backend rather than any
  unavailable MetaRAG-specific verifier;
- synonym and antonym generation uses WordNet only when it is available in the
  runtime environment, otherwise a small deterministic fallback map is used;
- the score is reported as faithfulness, where higher is better, so it can share
  the result schema with RAGAS;
- empty claim sets are treated as faithful with `score=1.0` and
  `metadata.empty_claim_set=true`, avoiding a false hallucination penalty for
  terse non-claim answers.

These deviations are recorded in result metadata with
`implementation=spacy_or_sentence_claims_plus_metamorphic_nli`.

## Threshold Calibration

Baseline thresholds must be selected with validation examples only. The helper
`tune_threshold_validation_only` rejects non-validation split records and
optimizes hallucination-label F1 over candidate faithfulness thresholds.

The test split must not be used for threshold selection. Experiment scripts
should load validation baseline scores first, persist the selected thresholds,
and then evaluate the frozen thresholds on the test split.

## RAGAS Compatibility

The RAGAS wrapper uses the official package through an adapter in
`src/mutoracle/baselines/ragas_baseline.py`. It records faithfulness, answer
relevancy, context precision, and context recall. The headline detector follows
the master-plan rule:
`hallucinated if faithfulness < tau_ragas`.

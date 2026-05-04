# Reference Notes

These notes keep Phase 0 grounded without expanding the bootstrap repository
into a literature-review archive.

## RAGAS

Role in this project:

- response-level RAG evaluation baseline;
- use official package where possible;
- tune faithfulness threshold only on validation data.

Risk:

- RAGAS detects likely unfaithfulness, but it is not intended to localize the
  failing pipeline stage.

## MetaRAG

Role in this project:

- metamorphic hallucination-detection baseline;
- implement a transparent local approximation if no directly reusable package is available;
- document deviations in `docs/METARAG_REIMPLEMENTATION.md` during Phase 7.

Risk:

- The project must not claim exact reproduction unless the exact implementation is actually used.

## RGB Benchmark

Role in this project:

- primary RAG-specific detection evaluation dataset;
- record source revision, license, checksum, and download date in data manifests.

## TriviaQA Multi-hop Subset

Role in this project:

- secondary generalization source;
- source material for FITS construction.

## OpenRouter

Role in this project:

- single external LLM provider boundary for v1;
- access through the OpenAI-compatible Python SDK;
- exact model IDs, provider routing, timestamps, prompts, token counts, latency, and cost belong in run manifests.

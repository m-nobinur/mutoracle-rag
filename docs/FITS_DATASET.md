# FITS Dataset

FITS is the fault-injection test split for MutOracle-RAG. It is designed for
stage-aware localization evaluation with single-stage labels:
`retrieval`, `prompt`, `generation`, and `no_fault`.

Build locally:

```bash
uv run mutoracle data build
```

Rebuild an existing frozen version only when intentional:

```bash
uv run mutoracle fits build --force
```

The build writes:

- `data/manifests/datasets.json`: RGB, TriviaQA, and Wikipedia/noise source
  manifests with URL, license, revision, checksum, and date
- `data/fits/manifest.json`: FITS build seed, source revisions, artifact
  checksums, and quality-gate summary
- `data/fits/fits_v1.0.0/*.jsonl`: generated validation/test/all/audit JSONL
  files
- `data/fits/fits_v1.0.0/BASELINE_LOCK.json`: immutable checksum lock for the
  baseline artifact bundle and tag pointer
- `data/FITS_DATASET.md`: generated dataset card for the exact local artifact

## JSONL Schema

Each row contains:

- `qid`: stable FITS ID
- `query`: injected or control query
- `gt_answer`: source ground-truth answer
- `fault_stage`: one of `retrieval`, `prompt`, `generation`, or `no_fault`
- `injection`: method-specific injection metadata and verifier note
- `source`: `rgb` or `triviaqa`
- `source_qid`: source example ID
- `split`: `validation` or `test`
- `build_seed`: deterministic build seed

Prompt-stage FITS records use one of two documented methods:

- `query_negation`: canonical QN transformation when query grammar supports it
- `controlled_prompt_perturbation`: deterministic fallback when QN rejects

## Quality Gates

The validator enforces:

- 75 examples per label for a 300-example v1.0.0 split
- 60 validation examples and 240 test examples, giving 15 validation and 60
  test examples per label
- balanced source provenance: 150 RGB-derived and 150 TriviaQA-derived examples
  overall, with 30/30 in validation and 120/120 in test
- fault-type stratification within +/- 5%
- no validation/test `qid` overlap
- no validation/test `(source, source_qid)` overlap
- deterministic SHA-256 checksums for JSONL artifacts
- 50-example audit sample with at least 95% label correctness
- query-length distribution alignment against source queries
- immutable frozen artifact behavior for `fits_v1.0.0` unless `--force` is used

The audit sample is a manual artifact-quality check rather than a multi-rater
annotation study, so it should not be interpreted as inter-annotator agreement.

The implementation uses deterministic schema-compatible local fixtures so the
build works without network credentials. The manifest preserves the intended
RGB, TriviaQA, and Wikipedia source identities for audit, even though the
current build uses deterministic local fixtures instead of staged raw
downloads.

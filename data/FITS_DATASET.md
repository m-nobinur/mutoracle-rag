# FITS v1.0.0 Dataset Card

FITS is the Phase 6 fault-injection test split for MutOracle-RAG.
It contains balanced single-stage labels for retrieval, prompt,
generation, and no-fault controls.

## Schema

`qid`, `query`, `gt_answer`, `fault_stage`, `injection`,
`source`, `source_qid`, `split`, and `build_seed`.

## Quality Gates

- Total examples: 300
- Label counts: {'retrieval': 75, 'prompt': 75, 'generation': 75, 'no_fault': 75}
- Split counts: {'validation': 60, 'test': 240}
- Validation/test overlap: 0
- Audit label correctness: 100.00%
- Query length means (source/fits): 10.20/10.45
- Query-length gate passed: True
- Passed: True

Raw versioned JSONL files are generated under `data/fits/` and
are intentionally ignored by git. Commit manifests and reports,
then publish frozen data as release assets when needed.
Rebuilding a frozen version requires `--force`.

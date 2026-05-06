# Mutation Taxonomy

MutOracle-RAG uses eleven canonical mutation operators. Operator IDs are stable
public identifiers and must stay consistent across code, configs, docs, plots,
and result records.

| ID | Operator | Stage | Deterministic behavior |
| --- | --- | --- | --- |
| CI | Context Injection | retrieval | Insert one unrelated control passage at a seeded position. |
| CR | Context Removal | retrieval | Remove one seeded passage; reject only when no passages exist. |
| CS | Context Shuffle | retrieval | Seed-shuffle retrieved passages; reject when fewer than two passages exist. |
| QP | Query Paraphrase | prompt | Apply a high-overlap rule paraphrase and reject below the similarity gate. |
| QN | Query Negation | prompt | Insert a grammatical negation and reject unsupported query shapes. |
| QD | Query Detail Drop | prompt | Drop a trailing query constraint when a similarity gate passes. |
| QI | Query Instruction Injection | prompt | Append an explicit support-only instruction. |
| FS | Factoid Synonym Substitution | generation | Replace one supported answer span with a near-synonym. |
| FA | Factoid Antonym Substitution | generation | Replace one supported answer span with an antonym. |
| FE | Factoid Entity Swap | generation | Replace a supported entity or domain phrase with a distractor. |
| GN | Answer Negation | generation | Add a local negation to the generated answer. |

## Metadata Contract

Each operator returns a `RAGRun` and appends mutation metadata:

```json
{
  "mutation": {
    "operator_id": "CI",
    "operator_name": "Context Injection",
    "stage": "retrieval",
    "rejected": false,
    "details": {}
  },
  "mutations": []
}
```

Rejected mutations keep the original field values and set `rejected` plus a
`rejection_reason`. This lets experiments run deterministically while allowing
later oracle and localization phases to filter unsupported cases.

## Before and After Examples

Baseline fixture run:

```text
Query:
How does MutOracle-RAG localize faults?

Passages:
1. retrieval context
2. prompt context
3. generation context

Answer:
MutOracle-RAG localizes hallucination faults by comparing oracle confidence
across the retrieval pipeline.
```

CI:

```text
Before passages:
retrieval context
prompt context
generation context

After passages:
retrieval context
Control distractor: SQLite stores relational data in a local file.
prompt context
generation context
```

CR:

```text
Before passages:
retrieval context
prompt context
generation context

After passages:
retrieval context
generation context
```

CS:

```text
Before passages:
retrieval context
prompt context
generation context

After passages:
prompt context
generation context
retrieval context
```

QP:

```text
Before query:
How does MutOracle-RAG localize faults?

After query:
Describe how MutOracle-RAG localize faults?
```

QN:

```text
Before query:
How does MutOracle-RAG localize faults?

After query:
How does MutOracle-RAG not localize faults?
```

FS:

```text
Before answer:
MutOracle-RAG localizes hallucination faults by comparing oracle confidence
across the retrieval pipeline.

After answer:
MutOracle-RAG localizes hallucination defects by comparing oracle confidence
across the retrieval pipeline.
```

FA:

```text
Before answer:
MutOracle-RAG localizes hallucination faults by comparing oracle confidence
across the retrieval pipeline.

After answer:
MutOracle-RAG obscures hallucination faults by comparing oracle confidence
across the retrieval pipeline.
```

FE:

```text
Before answer:
Elon Musk founded SpaceX.

After answer:
Jeff Bezos founded SpaceX.
```

GN:

```text
Before answer:
Elon Musk founded SpaceX.

After answer:
Not Elon Musk founded SpaceX.
```

## CLI

Run one mutation against the deterministic fixture RAG pipeline:

```bash
uv run mutoracle mutate --operator CI
```

The Makefile wrapper uses CI as the smoke example:

```bash
make mutate
```

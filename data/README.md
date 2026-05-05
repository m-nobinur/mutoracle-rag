# Data Manifests

Phase 6 records dataset provenance before any experiment consumes data. Each
manifest entry must include:

- `dataset_id`: stable local identifier such as `rgb`, `triviaqa`, or
  `wikipedia_noise`
- `name`: human-readable source name
- `url`: canonical source URL
- `license`: upstream license or redistribution note
- `revision`: upstream revision, tag, commit, or local fixture revision
- `checksum`: `sha256:<digest>` for the staged source preview or artifact
- `date`: manifest creation date in `YYYY-MM-DD` format
- `notes`: source-specific build or redistribution notes

The build command writes source manifests to `data/manifests/datasets.json` and
FITS artifact metadata to `data/fits/manifest.json`:

```bash
uv run mutoracle data build
```

The versioned raw FITS JSONL files live under `data/fits/fits_v1.0.0/` and are
gitignored so they can be published as release assets instead of committed.

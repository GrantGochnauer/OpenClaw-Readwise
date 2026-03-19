# Architecture Notes

## Main components

- `scripts/readwise_connector.py` — thin wrapper around the external `readwise` CLI
- `scripts/readwise_normalize.py` — normalizes varying CLI payload shapes
- `scripts/readwise_store.py` — SQLite cache, ranking logic, retrieval/evidence assembly, sync state
- `scripts/readwise_synthesis.py` — synthesis packet construction from evidence
- `scripts/readwise_export.py` — export download, extract, inspect, and ingest helpers
- `scripts/readwise_semantic.py` — semantic text prep and embedding-provider abstraction
- `scripts/readwise_cli.py` — user-facing command entry point

## Design stance

The repo is retrieval-first:

- normalize raw CLI output into stable internal shapes
- persist local mirror state in SQLite
- rank evidence with opinionated heuristics
- keep provenance visible
- treat synthesis as downstream of retrieval quality

## Data flow

1. `readwise_cli.py` issues a command
2. `readwise_connector.py` calls the external Readwise CLI
3. normalizers convert payloads into stable document/highlight/tag shapes
4. `readwise_store.py` stores and ranks local evidence
5. optional synthesis/eval/semantic flows build on top of cached evidence

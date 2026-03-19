# OpenClaw Readwise

A public, opinionated Readwise retrieval skill/tooling repo for assistants that need grounded answers from a user's saved reading.

It combines:

- Readwise Reader document search and inspection
- local SQLite mirroring for speed and reproducibility
- evidence-set construction for retrieval-first answers
- synthesis-packet generation from cached evidence
- export download/ingest flows
- semantic prep/embed scaffolding
- retrieval eval tooling

This repo is optimized for **personal knowledge retrieval**, not generic web search and not Reader write automation.

## Status

Useful now, but still evolving.

- Retrieval and local mirror flows are the strongest parts.
- Evidence assembly is solid enough for real use.
- Synthesis output is helpful, but still less polished than the retrieval layer.
- Semantic indexing and eval features are advanced capabilities.

## Repository layout

```text
.
├── SKILL.md
├── README.md
├── LICENSE
├── pyproject.toml
├── scripts/
│   ├── readwise_cli.py
│   ├── readwise_connector.py
│   ├── readwise_export.py
│   ├── readwise_normalize.py
│   ├── readwise_semantic.py
│   ├── readwise_store.py
│   └── readwise_synthesis.py
├── references/
│   ├── behavior-contract.md
│   └── eval-cases.example.json
└── data/
    └── readwise/   # runtime cache/output, gitignored
```

## Prerequisites

- Python 3.11+
- the `readwise` CLI installed and authenticated
- optional: `OPENAI_API_KEY` for embedding flows

This repo assumes the Readwise CLI is available on `PATH`.

## Data location

By default runtime data is stored under:

```text
data/readwise/
```

You can override that location with:

```bash
export READWISE_LOOKUP_DATA_DIR=/path/to/data/readwise
```

This affects the local SQLite cache and export artifacts.

## Quick start

### 1) Check the CLI

```bash
python3 scripts/readwise_cli.py --help
```

### 2) Initialize the local store

```bash
python3 scripts/readwise_cli.py init-store
```

### 3) Search Reader documents

```bash
python3 scripts/readwise_cli.py search-docs "tenant isolation" --json
```

### 4) Build an evidence set

```bash
python3 scripts/readwise_cli.py evidence-set "product strategy" --json
```

### 5) Build a synthesis packet

```bash
python3 scripts/readwise_cli.py synthesize "product strategy" --json
```

## Common commands

### Search documents

```bash
python3 scripts/readwise_cli.py search-docs "ai agents" --json
```

### List documents

```bash
python3 scripts/readwise_cli.py list-docs --location archive --limit 10 --json
```

### Fetch a document

```bash
python3 scripts/readwise_cli.py get-doc <document_id> --json
```

### Fetch highlights for a document

```bash
python3 scripts/readwise_cli.py get-doc-highlights <document_id> --json
```

### Search highlights

```bash
python3 scripts/readwise_cli.py search-highlights "deliberate practice" --json
```

### Cache tagged documents

```bash
python3 scripts/readwise_cli.py cache-tagged-docs --location archive --page-limit 3 --page-size 50 --detail-limit 10 --json
```

### Store stats

```bash
python3 scripts/readwise_cli.py store-stats --json
```

### Sync health

```bash
python3 scripts/readwise_cli.py sync-health --json
```

### Export and delta refresh flows

```bash
python3 scripts/readwise_cli.py trigger-export --json
python3 scripts/readwise_cli.py latest-export-anchor --json
python3 scripts/readwise_cli.py trigger-delta-export --json
python3 scripts/readwise_cli.py run-delta-refresh --json
```

### Semantic prep/embed

```bash
python3 scripts/readwise_cli.py semantic-prepare-tagged-docs --limit 50 --json
python3 scripts/readwise_cli.py semantic-embed-tagged-docs --limit 100 --json
```

### Eval flows

```bash
python3 scripts/readwise_cli.py eval-query "tenant isolation"
python3 scripts/readwise_cli.py eval-suite --json
```

## Opinionated defaults

This repo intentionally favors:

- manual tags as a strong signal
- cached/local evidence before expensive or noisy live retrieval
- precision safeguards for broad conceptual queries
- grounded evidence packets over unsupported synthesis

These are workflow defaults, not universal truths.

## What this repo does not focus on

- Reader write automation as a primary use case
- polished end-user UI
- generic vector database abstractions
- current-events retrieval

## Advanced notes

- `references/behavior-contract.md` describes the intended retrieval behavior.
- `references/eval-cases.example.json` provides a starter eval suite format.
- Semantic embedding uses direct HTTP calls and does not require a heavyweight SDK.

## Publishing / safety guidance

If you adapt this repo for your own corpus:

- do not commit exported reading data
- do not commit local cache databases
- do not commit auth material or environment files
- audit examples and docs for personal identifiers before publishing

## License

MIT

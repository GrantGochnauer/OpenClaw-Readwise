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

## What this is for

This repo is for the class of questions where a generic model answer is not enough and the real value comes from a user's own reading history.

Typical examples:

- "What have I saved about product strategy?"
- "Did I save anything on tenant isolation?"
- "Show me the strongest things I've read about leadership."
- "What else have I saved that's adjacent to this topic?"
- "Can you give me a grounded synthesis instead of just a generic answer?"

The goal is to help an assistant do more than keyword search:

- retrieve from a personal reading corpus
- weight stronger signals like tags and document quality
- build evidence sets from cached material
- produce answers that can point back to sources

## Status

Useful now, but still evolving.

- Retrieval and local mirror flows are the strongest parts.
- Evidence assembly is solid enough for real use.
- Synthesis output is helpful, but still less polished than the retrieval layer.
- Semantic indexing and eval features are advanced capabilities.

## Feature overview

### 1) Reader document retrieval

Search, list, and inspect Reader documents.

Useful when you want to answer questions like:

- "What have I read about AI agents?"
- "Show me archived docs tagged strategy."
- "Open this specific saved document and inspect the content."

Core commands:

```bash
python3 scripts/readwise_cli.py search-docs "ai agents" --json
python3 scripts/readwise_cli.py list-docs --location archive --limit 10 --json
python3 scripts/readwise_cli.py get-doc <document_id> --json
```

### 2) Highlight retrieval

Search highlights globally or fetch highlights for one document.

Useful when you want:

- specific passages instead of full documents
- quote-level evidence
- dense retrieval against what was actually highlighted

Core commands:

```bash
python3 scripts/readwise_cli.py search-highlights "deliberate practice" --json
python3 scripts/readwise_cli.py get-doc-highlights <document_id> --json
```

### 3) Local mirror / cache

Store Reader metadata, full document details, tags, highlights, and export-derived content in a local SQLite cache.

Why this matters:

- faster repeated retrieval
- more reproducible outputs
- better downstream ranking and synthesis
- reduced dependence on live API/CLI calls for every query

Core commands:

```bash
python3 scripts/readwise_cli.py init-store
python3 scripts/readwise_cli.py cache-tags --json
python3 scripts/readwise_cli.py cache-list-docs --location archive --limit 25 --json
python3 scripts/readwise_cli.py cache-doc <document_id> --with-highlights --json
python3 scripts/readwise_cli.py cache-tagged-docs --location archive --page-limit 3 --page-size 50 --detail-limit 10 --json
```

### 4) Evidence sets for grounded answers

Build a structured evidence set from cached documents and highlights before answering.

This is the main bridge between raw retrieval and assistant-quality responses.

Useful when you want to answer:

- "What are the strongest ideas I've saved on product strategy?"
- "What have I read about row-level access control?"
- "Use only tagged documents for this answer."

Core command:

```bash
python3 scripts/readwise_cli.py evidence-set "product strategy" --json
```

Helpful controls:

```bash
python3 scripts/readwise_cli.py evidence-set "product strategy" --strict --json
python3 scripts/readwise_cli.py evidence-set "ai agents" --tagged-only --json
python3 scripts/readwise_cli.py evidence-set "leadership" --broad --json
python3 scripts/readwise_cli.py evidence-set "org design" --counterpoint --json
```

### 5) Synthesis packets

Build a synthesis-oriented payload from cached evidence.

This is useful when you want:

- themes across multiple saved documents
- counterpoints and tensions
- source-backed synthesis instead of unsupported prose
- a downstream prompt/input for another assistant layer

Core command:

```bash
python3 scripts/readwise_cli.py synthesize "product strategy" --json
```

Useful variants:

```bash
python3 scripts/readwise_cli.py synthesize "product strategy" --strict --json
python3 scripts/readwise_cli.py synthesize "ai agents" --tagged-only --json
python3 scripts/readwise_cli.py synthesize "leadership" --counterpoint --json
```

### 6) Query expansion and adjacent discovery

Suggest broader related queries, then optionally run them, cache the results, and re-synthesize.

Useful when the first pass is too narrow and you want:

- adjacent ideas
- related tags
- broader coverage before synthesizing

Core commands:

```bash
python3 scripts/readwise_cli.py expand-query "openclaw" --json
python3 scripts/readwise_cli.py expand-and-cache "openclaw" --resynthesize --json
```

### 7) Export and delta refresh flows

Work with Reader export jobs so the local mirror can be refreshed in larger batches and kept current over time.

Useful when you want:

- a fuller local mirror
- incremental refreshes
- export-based ingestion instead of only live per-query fetches

Core commands:

```bash
python3 scripts/readwise_cli.py trigger-export --json
python3 scripts/readwise_cli.py export-status <export_id> --json
python3 scripts/readwise_cli.py wait-export-and-ingest <export_id> --json
python3 scripts/readwise_cli.py latest-export-anchor --json
python3 scripts/readwise_cli.py trigger-delta-export --json
python3 scripts/readwise_cli.py run-delta-refresh --json
```

### 8) Sync health / mirror inspection

Inspect how fresh the local mirror is and whether recent sync/export activity looks healthy.

Useful when you want to answer:

- "Is the cache stale?"
- "Did the last export ingest succeed?"
- "What should I run next to refresh the mirror?"

Core commands:

```bash
python3 scripts/readwise_cli.py store-stats --json
python3 scripts/readwise_cli.py sync-health --json
```

### 9) Semantic prep and embeddings

Prepare semantic representations for cached documents and optionally embed them.

Useful when you want:

- a stronger retrieval layer for broader semantic matching
- experimental ranking improvements
- embedding-backed retrieval scaffolding without adding a heavyweight framework

Core commands:

```bash
python3 scripts/readwise_cli.py semantic-prepare-tagged-docs --limit 50 --json
python3 scripts/readwise_cli.py semantic-embed-tagged-docs --limit 100 --json
python3 scripts/readwise_cli.py semantic-list-docs --status embedded --json
python3 scripts/readwise_cli.py semantic-stats --json
```

### 10) Retrieval evaluation

Evaluate retrieval behavior for a single query or against a labeled suite.

Useful when you want to ask:

- "Is the ranking behaving well for technical compound queries?"
- "Are broad conceptual searches drifting too much?"
- "Did my heuristic changes improve retrieval quality?"

Core commands:

```bash
python3 scripts/readwise_cli.py eval-query "tenant isolation"
python3 scripts/readwise_cli.py eval-suite --json
python3 scripts/readwise_cli.py eval-suite --mode specific_technical_compound --json
```

## Example workflows

## Workflow 1: answer "What have I saved about tenant isolation?"

1. Search or use the local mirror:

```bash
python3 scripts/readwise_cli.py search-docs "tenant isolation" --json
```

2. Build a grounded evidence set:

```bash
python3 scripts/readwise_cli.py evidence-set "tenant isolation" --strict --json
```

3. If needed, synthesize:

```bash
python3 scripts/readwise_cli.py synthesize "tenant isolation" --strict --json
```

Best fit:
- precise technical lookup
- source-backed answer
- low tolerance for topical drift

## Workflow 2: answer "What are the strongest ideas I've saved on product strategy?"

1. Start with cached evidence:

```bash
python3 scripts/readwise_cli.py evidence-set "product strategy" --strict --json
```

2. If coverage is thin, expand:

```bash
python3 scripts/readwise_cli.py expand-and-cache "product strategy" --resynthesize --strict --json
```

3. Build synthesis packet:

```bash
python3 scripts/readwise_cli.py synthesize "product strategy" --strict --json
```

Best fit:
- broad conceptual retrieval
- theme extraction
- grounded synthesis with precision safeguards

## Workflow 3: prioritize curated/tagged reading only

```bash
python3 scripts/readwise_cli.py evidence-set "ai agents" --tagged-only --json
python3 scripts/readwise_cli.py synthesize "ai agents" --tagged-only --json
```

Best fit:
- high-signal retrieval
- reducing noise
- using manual tags as a quality filter

## Workflow 4: build and refresh the local mirror

```bash
python3 scripts/readwise_cli.py init-store
python3 scripts/readwise_cli.py cache-tags --json
python3 scripts/readwise_cli.py cache-tagged-docs --location archive --page-limit 3 --page-size 50 --detail-limit 10 --json
python3 scripts/readwise_cli.py sync-health --json
```

Then, for larger refreshes:

```bash
python3 scripts/readwise_cli.py trigger-export --json
python3 scripts/readwise_cli.py wait-export-and-ingest <export_id> --json
```

Best fit:
- preparing the system for repeated use
- improving speed and reproducibility
- building a stronger local retrieval base

## Workflow 5: experiment with semantic indexing

```bash
python3 scripts/readwise_cli.py semantic-prepare-tagged-docs --limit 50 --json
python3 scripts/readwise_cli.py semantic-embed-tagged-docs --limit 100 --json
python3 scripts/readwise_cli.py semantic-stats --json
```

Best fit:
- advanced retrieval tuning
- embedding-backed experimentation
- evaluation and iteration

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

## How an assistant might use this repo

A good assistant experience usually looks like this:

1. detect that a question is really about the user's saved reading
2. check the local mirror first when it is likely fresh enough
3. retrieve documents/highlights
4. build an evidence set
5. optionally synthesize across evidence
6. answer naturally, with provenance when useful

That means the user should not always need to say:

- "check Readwise"
- "search Reader"
- "run the Python script"

The best use of this repo is often as a retrieval/evidence layer underneath a conversational assistant.

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

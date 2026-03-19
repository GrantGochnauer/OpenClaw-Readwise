---
name: readwise-lookup
description: Search Readwise Reader documents and highlights, build grounded evidence sets, manage a local mirror, inspect sync freshness, and prepare retrieval-oriented synthesis inputs. Use when the user asks what they have saved/read on a topic, wants evidence from their own reading corpus, wants tagged or related documents, or needs local mirror/eval/debug flows for Readwise retrieval.
---

# Readwise Lookup

Use this skill for **personal knowledge retrieval from Readwise/Reader**, especially when grounded evidence from saved reading is more valuable than a generic answer.

## When to use it

Use this skill when the user asks things like:

- “What have I saved about tenant isolation?”
- “What have I read about product strategy?”
- “Show me documents tagged OpenClaw.”
- “What else have I saved that’s adjacent to this?”
- “Check whether my local Readwise mirror is fresh.”
- “Evaluate or debug retrieval quality for this query.”

## What it does

- searches Reader documents
- lists Reader documents with filters and pagination
- fetches full document details/content by document id
- fetches highlights for a Reader document
- searches Readwise highlights
- lists Reader tags
- initializes and populates a local SQLite mirror under `data/readwise/` by default
- bulk-caches tagged Reader documents
- builds evidence sets from cached documents/highlights
- builds grounded synthesis packets from cached evidence
- suggests broader follow-up queries before re-synthesis
- triggers, downloads, inspects, and ingests Reader exports
- runs delta refresh flows from the latest export anchor
- reports local mirror freshness/health
- prepares semantic records for cached documents
- embeds semantic records into local SQLite storage
- evaluates retrieval behavior with single-query and suite-based evals

## Default retrieval stance

This skill is intentionally opinionated:

- manual tags are treated as a strong relevance signal
- cached evidence is preferred for speed and reproducibility
- live retrieval is still useful for freshness and expansion
- broad conceptual queries should favor precision safeguards to reduce topical drift
- grounded evidence is preferred over unsupported synthesis

## When not to use it

Do **not** default to this skill when:

- the question is generic and does not benefit from personal saved-reading context
- the answer likely lives in local files, source code, calendar, email, or another system
- the user needs live web/current-events coverage more than saved-reading evidence
- the task is to mutate Reader state; this repo is retrieval-first

## Response pattern

When Readwise materially informs the answer, the assistant should usually:

1. answer naturally
2. mention that saved reading was consulted when provenance matters
3. summarize the strongest patterns or evidence
4. include compact retrieval stats and source links/titles when helpful

## Files

- scripts live in `scripts/`
- deeper behavior notes live in `references/behavior-contract.md`
- example eval cases live in `references/eval-cases.example.json`

## Common entry point

```bash
python3 scripts/readwise_cli.py --help
```

## Notes

- This repo is **read-first** and retrieval-oriented.
- The main value is conversational retrieval and evidence grounding, not just raw CLI commands.
- Advanced semantic indexing and eval flows are included, but they should be treated as advanced capabilities rather than the only entry point.

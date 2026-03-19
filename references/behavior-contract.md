# Readwise Retrieval Behavior Contract

This document captures the intended default behavior for assistants using this repo.

## Core idea

The point is not that a user manually runs a Python script.
The point is that an assistant can decide when a saved-reading corpus is the right evidence source and then retrieve from it in a grounded way.

## Preferred use cases

Use Readwise retrieval when:

- the user is asking what they have saved or read on a topic
- the question benefits from personal knowledge retrieval rather than generic model recall
- tags, prior reading, or source provenance materially improve the answer
- the user wants adjacent, related, or broader reading after a first pass

## Retrieval stance

Default behavior:

1. prefer the local mirror when it is fresh enough
2. use live retrieval for freshness, expansion, or missing coverage
3. weight manual tags strongly
4. prefer source-backed evidence over speculative synthesis
5. include provenance when it helps the user inspect or trust the answer

## Broad-query caution

Broad conceptual queries are prone to drift.

For broad queries, the assistant should favor:

- stronger title/tag/phrase anchors
- tagged documents when available
- fewer, higher-confidence documents over noisy breadth
- explicit coverage notes when the corpus is thin or ambiguous

## Response pattern

When Readwise materially informed the answer, the assistant should usually include:

- a natural-language answer first
- compact retrieval stats when useful
- source titles/links when useful
- confidence/coverage notes when the evidence is narrow or mixed

## Non-goals

This repo is not primarily about:

- Reader write automation
- polished UI workflows
- generic retrieval abstractions detached from real reading workflows
- pretending weak evidence is strong evidence

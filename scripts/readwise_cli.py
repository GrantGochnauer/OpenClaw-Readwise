#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from readwise_connector import (
    ReadwiseCliMissingError,
    ReadwiseCommandError,
    ReadwiseConnector,
    ReadwiseJsonError,
)
from readwise_semantic import build_embedding_provider
from readwise_normalize import (
    normalize_document_details,
    normalize_document_highlights,
    normalize_document_list,
    normalize_document_search,
    normalize_highlight_search,
    normalize_tags,
)
from readwise_store import ReadwiseStore
from readwise_synthesis import build_synthesis_packet
from readwise_export import download_export_zip, extract_export_zip, inspect_extracted_export, ingest_extracted_export

DEFAULT_RESPONSE_FIELDS = [
    "title",
    "author",
    "source",
    "category",
    "location",
    "tags",
    "site_name",
    "summary",
    "source_url",
    "saved_at",
    "updated_at",
    "published_date",
    "reading_time",
    "word_count",
    "first_opened_at",
    "last_opened_at",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Readwise lookup helper for OpenClaw")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_docs = subparsers.add_parser("search-docs", help="Search Reader documents")
    search_docs.add_argument("query", help="Search query")
    search_docs.add_argument("--limit", type=int, default=5)
    search_docs.add_argument("--location")
    search_docs.add_argument("--category")
    search_docs.add_argument("--tag")
    search_docs.add_argument("--author")
    search_docs.add_argument("--title")
    search_docs.add_argument("--json", action="store_true")

    list_docs = subparsers.add_parser("list-docs", help="List Reader documents")
    list_docs.add_argument("--location")
    list_docs.add_argument("--tag")
    list_docs.add_argument("--seen", choices=["true", "false"])
    list_docs.add_argument("--limit", type=int, default=10)
    list_docs.add_argument("--page-cursor")
    list_docs.add_argument("--json", action="store_true")

    get_doc = subparsers.add_parser("get-doc", help="Get Reader document details")
    get_doc.add_argument("document_id", help="Reader document id")
    get_doc.add_argument("--chunk-size", type=int, default=2000)
    get_doc.add_argument("--max-chunks", type=int, default=8)
    get_doc.add_argument("--json", action="store_true")

    get_doc_highlights = subparsers.add_parser("get-doc-highlights", help="Get highlights for a Reader document")
    get_doc_highlights.add_argument("document_id", help="Reader document id")
    get_doc_highlights.add_argument("--json", action="store_true")

    search_highlights = subparsers.add_parser("search-highlights", help="Search Readwise highlights")
    search_highlights.add_argument("query", help="Vector search term")
    search_highlights.add_argument("--limit", type=int, default=5)
    search_highlights.add_argument("--json", action="store_true")

    list_tags = subparsers.add_parser("list-tags", help="List Reader tags")
    list_tags.add_argument("--json", action="store_true")

    init_store = subparsers.add_parser("init-store", help="Initialize the local Readwise cache store")
    init_store.add_argument("--json", action="store_true")

    cache_tags = subparsers.add_parser("cache-tags", help="Fetch and store Reader tags locally")
    cache_tags.add_argument("--json", action="store_true")

    cache_list_docs = subparsers.add_parser("cache-list-docs", help="Fetch one page of Reader documents and store them locally")
    cache_list_docs.add_argument("--location")
    cache_list_docs.add_argument("--tag")
    cache_list_docs.add_argument("--seen", choices=["true", "false"])
    cache_list_docs.add_argument("--limit", type=int, default=25)
    cache_list_docs.add_argument("--page-cursor")
    cache_list_docs.add_argument("--json", action="store_true")

    cache_doc = subparsers.add_parser("cache-doc", help="Fetch one document and store it locally")
    cache_doc.add_argument("document_id")
    cache_doc.add_argument("--chunk-size", type=int, default=2000)
    cache_doc.add_argument("--max-chunks", type=int, default=8)
    cache_doc.add_argument("--with-highlights", action="store_true")
    cache_doc.add_argument("--json", action="store_true")

    cache_tagged_docs = subparsers.add_parser("cache-tagged-docs", help="Page through live Reader docs, cache tagged items, and fetch details for the strongest tagged docs")
    cache_tagged_docs.add_argument("--location")
    cache_tagged_docs.add_argument("--seen", choices=["true", "false"])
    cache_tagged_docs.add_argument("--page-limit", type=int, default=3)
    cache_tagged_docs.add_argument("--page-size", type=int, default=50)
    cache_tagged_docs.add_argument("--detail-limit", type=int, default=10)
    cache_tagged_docs.add_argument("--with-highlights", action="store_true")
    cache_tagged_docs.add_argument("--json", action="store_true")

    evidence = subparsers.add_parser("evidence-set", help="Build an evidence set from the local cache")
    evidence.add_argument("query")
    evidence.add_argument("--doc-limit", type=int, default=4)
    evidence.add_argument("--highlight-limit", type=int, default=8)
    evidence.add_argument("--chunk-limit", type=int, default=2)
    evidence.add_argument("--strict", action="store_true", help="Prefer fewer, higher-confidence matches; especially useful for broad topics")
    evidence.add_argument("--tagged-only", action="store_true", help="Only include manually tagged documents/highlights in the evidence set")
    evidence.add_argument("--broad", action="store_true", help="Increase recall/diversity for exploratory retrieval before synthesis")
    evidence.add_argument("--counterpoint", action="store_true", help="Prefer evidence that contains explicit tradeoffs, tensions, or disagreements")
    evidence.add_argument("--preserve-strict", action="store_true", help="Carry strict filtering forward in downstream expansion/resynthesis steps")
    evidence.add_argument("--json", action="store_true")

    synthesize = subparsers.add_parser("synthesize", help="Build a grounded synthesis packet from cached evidence")
    synthesize.add_argument("query")
    synthesize.add_argument("--doc-limit", type=int, default=4)
    synthesize.add_argument("--highlight-limit", type=int, default=8)
    synthesize.add_argument("--chunk-limit", type=int, default=2)
    synthesize.add_argument("--strict", action="store_true", help="Prefer fewer, higher-confidence matches; especially useful for broad topics")
    synthesize.add_argument("--tagged-only", action="store_true", help="Only include manually tagged documents/highlights in the synthesis evidence")
    synthesize.add_argument("--broad", action="store_true", help="Increase recall/diversity for exploratory synthesis retrieval")
    synthesize.add_argument("--counterpoint", action="store_true", help="Prefer evidence that contains explicit tradeoffs, tensions, or disagreements")
    synthesize.add_argument("--preserve-strict", action="store_true", help="Carry strict filtering forward in downstream expansion/resynthesis steps")
    synthesize.add_argument("--json", action="store_true")

    expand = subparsers.add_parser("expand-query", help="Suggest broader related cached queries/tags before re-synthesizing")
    expand.add_argument("query")
    expand.add_argument("--limit", type=int, default=6)
    expand.add_argument("--json", action="store_true")

    expand_and_cache = subparsers.add_parser("expand-and-cache", help="Run broader live searches from suggested queries, cache results, and optionally re-synthesize")
    expand_and_cache.add_argument("query")
    expand_and_cache.add_argument("--query-limit", type=int, default=4)
    expand_and_cache.add_argument("--search-limit", type=int, default=5)
    expand_and_cache.add_argument("--detail-limit", type=int, default=3)
    expand_and_cache.add_argument("--with-highlights", action="store_true")
    expand_and_cache.add_argument("--resynthesize", action="store_true")
    expand_and_cache.add_argument("--strict", action="store_true", help="Apply strict filtering on the final resynthesis pass")
    expand_and_cache.add_argument("--tagged-only", action="store_true", help="Restrict final resynthesis evidence to tagged items only")
    expand_and_cache.add_argument("--broad", action="store_true", help="Use broader retrieval when building the final resynthesis evidence set")
    expand_and_cache.add_argument("--counterpoint", action="store_true", help="Prefer tension/tradeoff evidence on the final resynthesis pass")
    expand_and_cache.add_argument("--preserve-strict", action="store_true", help="Keep strict filtering active after expansion when resynthesizing")
    expand_and_cache.add_argument("--doc-limit", type=int, default=6)
    expand_and_cache.add_argument("--highlight-limit", type=int, default=10)
    expand_and_cache.add_argument("--chunk-limit", type=int, default=2)
    expand_and_cache.add_argument("--json", action="store_true")

    trigger_export = subparsers.add_parser("trigger-export", help="Trigger a Reader export job for full/delta mirror ingestion scaffolding")
    trigger_export.add_argument("--since-updated")
    trigger_export.add_argument("--json", action="store_true")

    latest_export_anchor = subparsers.add_parser("latest-export-anchor", help="Show the current canonical last_updated anchor for delta exports")
    latest_export_anchor.add_argument("--json", action="store_true")

    trigger_delta_export = subparsers.add_parser("trigger-delta-export", help="Trigger a delta export using the last successful export anchor")
    trigger_delta_export.add_argument("--since-updated")
    trigger_delta_export.add_argument("--json", action="store_true")

    run_delta_refresh = subparsers.add_parser("run-delta-refresh", help="Trigger a delta export from the current anchor, wait for completion, then download and ingest it")
    run_delta_refresh.add_argument("--since-updated")
    run_delta_refresh.add_argument("--poll-seconds", type=int, default=30)
    run_delta_refresh.add_argument("--max-waits", type=int, default=60)
    run_delta_refresh.add_argument("--json", action="store_true")

    sync_health = subparsers.add_parser("sync-health", help="Show freshness and recent sync/export health for the local Readwise mirror")
    sync_health.add_argument("--json", action="store_true")

    export_status = subparsers.add_parser("export-status", help="Check the status of a Reader export job")
    export_status.add_argument("export_id")
    export_status.add_argument("--json", action="store_true")

    wait_export = subparsers.add_parser("wait-export-and-ingest", help="Poll a Reader export until complete, then download, inspect, and ingest it")
    wait_export.add_argument("export_id")
    wait_export.add_argument("--poll-seconds", type=int, default=30)
    wait_export.add_argument("--max-waits", type=int, default=60)
    wait_export.add_argument("--json", action="store_true")

    download_export = subparsers.add_parser("download-export", help="Download and extract a completed Reader export ZIP")
    download_export.add_argument("export_id")
    download_export.add_argument("download_url")
    download_export.add_argument("--json", action="store_true")

    inspect_export = subparsers.add_parser("inspect-export", help="Inspect an extracted Reader export")
    inspect_export.add_argument("export_id")
    inspect_export.add_argument("--json", action="store_true")

    ingest_export = subparsers.add_parser("ingest-export", help="Ingest an extracted Reader export into the local cache")
    ingest_export.add_argument("export_id")
    ingest_export.add_argument("--json", action="store_true")

    semantic_prepare_tagged = subparsers.add_parser("semantic-prepare-tagged-docs", help="Prepare semantic text/embedding scaffolding for tagged cached documents")
    semantic_prepare_tagged.add_argument("--limit", type=int, default=50)
    semantic_prepare_tagged.add_argument("--chunk-limit", type=int, default=4)
    semantic_prepare_tagged.add_argument("--location")
    semantic_prepare_tagged.add_argument("--force", action="store_true")
    semantic_prepare_tagged.add_argument("--json", action="store_true")

    semantic_prepare_docs = subparsers.add_parser("semantic-prepare-docs", help="Prepare semantic text/embedding scaffolding for specific cached document ids")
    semantic_prepare_docs.add_argument("document_ids", nargs="+")
    semantic_prepare_docs.add_argument("--chunk-limit", type=int, default=4)
    semantic_prepare_docs.add_argument("--json", action="store_true")

    semantic_embed_tagged = subparsers.add_parser("semantic-embed-tagged-docs", help="Embed prepared semantic records for tagged cached documents")
    semantic_embed_tagged.add_argument("--limit", type=int, default=100)
    semantic_embed_tagged.add_argument("--batch-size", type=int, default=32)
    semantic_embed_tagged.add_argument("--provider", default="openai")
    semantic_embed_tagged.add_argument("--model", default="text-embedding-3-small")
    semantic_embed_tagged.add_argument("--base-url")
    semantic_embed_tagged.add_argument("--dimensions", type=int)
    semantic_embed_tagged.add_argument("--json", action="store_true")

    semantic_embed_docs = subparsers.add_parser("semantic-embed-docs", help="Embed prepared semantic records for specific cached document ids")
    semantic_embed_docs.add_argument("document_ids", nargs="+")
    semantic_embed_docs.add_argument("--batch-size", type=int, default=32)
    semantic_embed_docs.add_argument("--provider", default="openai")
    semantic_embed_docs.add_argument("--model", default="text-embedding-3-small")
    semantic_embed_docs.add_argument("--base-url")
    semantic_embed_docs.add_argument("--dimensions", type=int)
    semantic_embed_docs.add_argument("--json", action="store_true")

    semantic_list_docs = subparsers.add_parser("semantic-list-docs", help="List prepared/embedded semantic document status rows")
    semantic_list_docs.add_argument("--status", choices=["prepared", "embedded", "error", "pending"])
    semantic_list_docs.add_argument("--limit", type=int, default=20)
    semantic_list_docs.add_argument("--json", action="store_true")

    semantic_stats = subparsers.add_parser("semantic-stats", help="Show semantic indexing/embedding scaffolding stats")
    semantic_stats.add_argument("--json", action="store_true")

    store_stats = subparsers.add_parser("store-stats", help="Show local Readwise cache stats")
    store_stats.add_argument("--json", action="store_true")

    eval_query = subparsers.add_parser("eval-query", help="Run a lightweight retrieval evaluation for one query against the local cache")
    eval_query.add_argument("query")
    eval_query.add_argument("--doc-limit", type=int, default=4)
    eval_query.add_argument("--highlight-limit", type=int, default=6)
    eval_query.add_argument("--chunk-limit", type=int, default=2)
    eval_query.add_argument("--strict", action="store_true")
    eval_query.add_argument("--json", action="store_true")

    eval_suite = subparsers.add_parser("eval-suite", help="Run labeled evaluation cases from references/eval-cases.example.json")
    eval_suite.add_argument("--cases-path")
    eval_suite.add_argument("--mode", choices=["known_topic_lookup", "broad_conceptual_synthesis", "specific_technical_compound", "tag_constrained_retrieval", "exploratory_related_content"])
    eval_suite.add_argument("--doc-limit", type=int, default=4)
    eval_suite.add_argument("--highlight-limit", type=int, default=6)
    eval_suite.add_argument("--chunk-limit", type=int, default=2)
    eval_suite.add_argument("--strict", action="store_true")
    eval_suite.add_argument("--json", action="store_true")

    return parser


def _csv(values: List[str]) -> str:
    return ",".join(values)


def _fmt_tags(tags: List[str]) -> str:
    return ", ".join(tags) if tags else "none"


def _retrieval_mode(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "taggedOnly": bool(getattr(args, "tagged_only", False)),
        "broad": bool(getattr(args, "broad", False)),
        "counterpoint": bool(getattr(args, "counterpoint", False)),
        "preserveStrict": bool(getattr(args, "preserve_strict", False)),
    }


RETRIEVAL_MODE_LABELS = {
    "taggedOnly": "tagged-only",
    "broad": "broad recall",
    "counterpoint": "counterpoint-seeking",
    "preserveStrict": "preserve strictness",
}


def _active_retrieval_modes(payload: Dict[str, Any]) -> List[str]:
    retrieval_mode = payload.get("retrievalMode") or {}
    return [label for key, label in RETRIEVAL_MODE_LABELS.items() if retrieval_mode.get(key)]


def _truncate(text: str | None, limit: int = 220) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _list_docs_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        "reader-list-documents",
        "--limit",
        str(args.limit),
        "--response-fields",
        _csv(DEFAULT_RESPONSE_FIELDS),
    ]
    if args.location:
        cmd.extend(["--location", args.location])
    if args.tag is not None:
        cmd.extend(["--tag", args.tag])
    if args.seen is not None:
        cmd.extend(["--seen", args.seen])
    if args.page_cursor:
        cmd.extend(["--page-cursor", args.page_cursor])
    return cmd


def _extract_tag_query(query: str) -> str | None:
    if "tag:" not in query.lower():
        return None
    for token in query.split():
        if token.lower().startswith("tag:") and len(token) > 4:
            return token[4:]
    return None


def _default_eval_cases_path() -> Path:
    return Path(__file__).resolve().parents[1] / "references" / "eval-cases.example.json"


def _contains_any(text: str, needles: List[str]) -> bool:
    lowered = (text or "").lower()
    return any((needle or "").lower() in lowered for needle in (needles or []))


def _evaluate_case(case: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    documents = evidence.get("documents", []) or []
    titles = [doc.get("title") or "" for doc in documents]
    expected = case.get("expectAnyTitleContains") or []
    rejected = case.get("rejectTitleContains") or []
    min_docs = int(case.get("minDocumentCount") or 0)
    max_docs = int(case.get("maxDocumentCount") or 9999)

    hits_expected = sum(1 for title in titles if _contains_any(title, expected))
    hits_rejected = sum(1 for title in titles if _contains_any(title, rejected))
    count_ok = min_docs <= len(documents) <= max_docs

    top3 = titles[:3]
    top3_expected = sum(1 for title in top3 if _contains_any(title, expected))
    top3_rejected = sum(1 for title in top3 if _contains_any(title, rejected))
    confidence = evidence.get("confidence")
    strong_docs = sum(1 for doc in documents if (doc.get("selectionSignals") or {}).get("sourceQualityTier") == "strong")

    pass_case = hits_expected >= 1 and hits_rejected == 0 and count_ok and top3_rejected == 0

    return {
        "query": case.get("query"),
        "pass": pass_case,
        "documentCount": len(documents),
        "expectedMatches": hits_expected,
        "rejectedMatches": hits_rejected,
        "top3ExpectedMatches": top3_expected,
        "top3RejectedMatches": top3_rejected,
        "countOk": count_ok,
        "confidence": confidence,
        "strongDocuments": strong_docs,
        "titles": titles[:6],
        "selectionNotes": (evidence.get("selectionNotes") or [])[:10],
        "notes": case.get("notes"),
    }


def run_command(args: argparse.Namespace, connector: ReadwiseConnector) -> Dict[str, Any]:
    if args.command == "search-docs":
        cmd = ["reader-search-documents", "--query", args.query, "--limit", str(args.limit)]
        if args.location:
            cmd.extend(["--location-in", args.location])
        if args.category:
            cmd.extend(["--category-in", args.category])
        if args.tag:
            cmd.extend(["--tags-in", args.tag])
        if args.author:
            cmd.extend(["--author-search", args.author])
        if args.title:
            cmd.extend(["--title-search", args.title])
        result = connector.run_json(cmd)
        return normalize_document_search(result.data)

    if args.command == "list-docs":
        result = connector.run_json(_list_docs_command(args))
        return normalize_document_list(result.data)

    if args.command == "get-doc":
        result = connector.run_json(["reader-get-document-details", "--document-id", args.document_id])
        return normalize_document_details(result.data, chunk_size=args.chunk_size, max_chunks=args.max_chunks)

    if args.command == "get-doc-highlights":
        result = connector.run_json(["reader-get-document-highlights", "--document-id", args.document_id])
        return normalize_document_highlights(result.data, document_id=args.document_id)

    if args.command == "search-highlights":
        result = connector.run_json(
            ["readwise-search-highlights", "--vector-search-term", args.query, "--limit", str(args.limit)]
        )
        return normalize_highlight_search(result.data)

    if args.command == "list-tags":
        result = connector.run_json(["reader-list-tags"])
        return normalize_tags(result.data)

    if args.command == "init-store":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return {"kind": "storeInit", "ok": True, **store.stats()}
        finally:
            store.close()

    if args.command == "cache-tags":
        result = connector.run_json(["reader-list-tags"])
        payload = normalize_tags(result.data)
        store = ReadwiseStore()
        try:
            store.init_schema()
            saved = store.replace_tags(payload.get("results", []))
            return {"kind": "cacheTags", "saved": saved, **store.stats()}
        finally:
            store.close()

    if args.command == "cache-list-docs":
        result = connector.run_json(_list_docs_command(args))
        payload = normalize_document_list(result.data)
        store = ReadwiseStore()
        try:
            store.init_schema()
            saved = store.upsert_documents(payload.get("results", []))
            store.set_sync_state(
                "last_list_docs",
                {
                    "nextPageCursor": payload.get("nextPageCursor"),
                    "limit": args.limit,
                    "location": args.location,
                    "tag": args.tag,
                    "seen": args.seen,
                },
            )
            return {
                "kind": "cacheListDocs",
                "saved": saved,
                "nextPageCursor": payload.get("nextPageCursor"),
                **store.stats(),
            }
        finally:
            store.close()

    if args.command == "cache-doc":
        doc_result = connector.run_json(["reader-get-document-details", "--document-id", args.document_id])
        doc_payload = normalize_document_details(doc_result.data, chunk_size=args.chunk_size, max_chunks=args.max_chunks)

        store = ReadwiseStore()
        try:
            store.init_schema()
            store.upsert_document(doc_payload["document"], raw_json=doc_result.data)
            highlights_saved = 0
            if args.with_highlights:
                hl_result = connector.run_json(["reader-get-document-highlights", "--document-id", args.document_id])
                hl_payload = normalize_document_highlights(hl_result.data, document_id=args.document_id)
                highlights_saved = store.upsert_highlights(hl_payload.get("results", []))
            store.conn.commit()
            return {
                "kind": "cacheDoc",
                "documentId": args.document_id,
                "highlightsSaved": highlights_saved,
                **store.stats(),
            }
        finally:
            store.close()

    if args.command == "cache-tagged-docs":
        store = ReadwiseStore()
        try:
            store.init_schema()
            all_tagged = {}
            next_cursor = None
            pages_processed = 0
            metadata_cached = 0

            for _ in range(args.page_limit):
                list_args = argparse.Namespace(
                    limit=args.page_size,
                    location=args.location,
                    tag=None,
                    seen=args.seen,
                    page_cursor=next_cursor,
                )
                result = connector.run_json(_list_docs_command(list_args))
                payload = normalize_document_list(result.data)
                pages_processed += 1
                docs = payload.get("results", [])
                metadata_cached += store.upsert_documents(docs)
                for doc in docs:
                    if doc.get("tags"):
                        all_tagged[doc.get("documentId")] = doc
                next_cursor = payload.get("nextPageCursor")
                if not next_cursor:
                    break

            ranked_tagged = sorted(
                all_tagged.values(),
                key=lambda d: store._document_quality_score(d),
                reverse=True,
            )

            details_cached = 0
            highlights_saved = 0
            detail_ids = []
            for doc in ranked_tagged[: args.detail_limit]:
                document_id = doc.get("documentId")
                if not document_id:
                    continue
                detail = connector.run_json(["reader-get-document-details", "--document-id", document_id])
                detail_payload = normalize_document_details(detail.data)
                store.upsert_document(detail_payload["document"], raw_json=detail.data)
                detail_ids.append(document_id)
                details_cached += 1
                if args.with_highlights:
                    hl = connector.run_json(["reader-get-document-highlights", "--document-id", document_id])
                    hl_payload = normalize_document_highlights(hl.data, document_id=document_id)
                    highlights_saved += store.upsert_highlights(hl_payload.get("results", []))

            store.set_sync_state(
                "last_cache_tagged_docs",
                {
                    "location": args.location,
                    "seen": args.seen,
                    "pageLimit": args.page_limit,
                    "pageSize": args.page_size,
                    "detailLimit": args.detail_limit,
                    "taggedCandidates": len(all_tagged),
                },
            )

            return {
                "kind": "cacheTaggedDocs",
                "pagesProcessed": pages_processed,
                "metadataCached": metadata_cached,
                "taggedCandidates": len(all_tagged),
                "detailsCached": details_cached,
                "highlightsSaved": highlights_saved,
                "detailIds": detail_ids,
                "nextPageCursor": next_cursor,
                **store.stats(),
            }
        finally:
            store.close()

    if args.command == "evidence-set":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return store.build_evidence_set(
                args.query,
                doc_limit=args.doc_limit,
                highlight_limit=args.highlight_limit,
                chunk_limit=args.chunk_limit,
                strict_mode=bool(getattr(args, "strict", False)),
                retrieval_mode=_retrieval_mode(args),
            )
        finally:
            store.close()

    if args.command == "synthesize":
        store = ReadwiseStore()
        try:
            store.init_schema()
            evidence = store.build_evidence_set(
                args.query,
                doc_limit=args.doc_limit,
                highlight_limit=args.highlight_limit,
                chunk_limit=args.chunk_limit,
                strict_mode=bool(getattr(args, "strict", False)),
                retrieval_mode=_retrieval_mode(args),
            )
            packet = build_synthesis_packet(evidence)
            packet["expansionCandidates"] = store.expand_query_candidates(args.query)
            return packet
        finally:
            store.close()

    if args.command == "expand-query":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return store.expand_query_candidates(args.query, limit=args.limit)
        finally:
            store.close()

    if args.command == "expand-and-cache":
        store = ReadwiseStore()
        try:
            store.init_schema()
            expansions = store.expand_query_candidates(args.query, limit=args.query_limit)
            queries = expansions.get("suggestedQueries", [])[: args.query_limit]

            search_runs = []
            seen_doc_ids = set()
            cached_summaries = []

            for query in queries:
                tag_query = _extract_tag_query(query)
                if tag_query:
                    list_args = argparse.Namespace(
                        limit=args.search_limit,
                        location=None,
                        tag=tag_query,
                        seen=None,
                        page_cursor=None,
                    )
                    live = connector.run_json(_list_docs_command(list_args))
                    normalized = normalize_document_list(live.data)
                    results = normalized.get("results", [])
                else:
                    live = connector.run_json(["reader-search-documents", "--query", query, "--limit", str(args.search_limit)])
                    normalized = normalize_document_search(live.data)
                    results = normalized.get("results", [])
                store.upsert_documents(results)

                detail_cached = 0
                detail_ids = []
                for item in results[: args.detail_limit]:
                    document_id = item.get("documentId")
                    if not document_id or document_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(document_id)
                    detail = connector.run_json(["reader-get-document-details", "--document-id", document_id])
                    detail_payload = normalize_document_details(detail.data)
                    store.upsert_document(detail_payload["document"], raw_json=detail.data)
                    detail_ids.append(document_id)
                    detail_cached += 1
                    if args.with_highlights:
                        hl = connector.run_json(["reader-get-document-highlights", "--document-id", document_id])
                        hl_payload = normalize_document_highlights(hl.data, document_id=document_id)
                        store.upsert_highlights(hl_payload.get("results", []))

                cached_summaries.append({
                    "query": query,
                    "searchResults": len(results),
                    "detailCached": detail_cached,
                    "detailIds": detail_ids,
                })
                search_runs.append(normalized)

            store.set_sync_state(
                "last_expand_and_cache",
                {
                    "rootQuery": args.query,
                    "queries": queries,
                    "withHighlights": args.with_highlights,
                    "searchLimit": args.search_limit,
                    "detailLimit": args.detail_limit,
                },
            )

            response = {
                "kind": "expandAndCache",
                "rootQuery": args.query,
                "queriesRun": queries,
                "runs": cached_summaries,
                "expansionCandidates": expansions,
                **store.stats(),
            }

            if args.resynthesize:
                evidence = store.build_evidence_set(
                    args.query,
                    doc_limit=args.doc_limit,
                    highlight_limit=args.highlight_limit,
                    chunk_limit=args.chunk_limit,
                    strict_mode=bool(getattr(args, "strict", False) or getattr(args, "preserve_strict", False)),
                    retrieval_mode=_retrieval_mode(args),
                )
                packet = build_synthesis_packet(evidence)
                packet["expansionCandidates"] = store.expand_query_candidates(args.query)
                response["resynthesis"] = packet

            return response
        finally:
            store.close()

    if args.command == "trigger-export":
        cmd = ["reader-export-documents"]
        if args.since_updated:
            cmd.extend(["--since-updated", args.since_updated])
        result = connector.run_json(cmd)
        store = ReadwiseStore()
        try:
            store.init_schema()
            store.set_sync_state("last_export_trigger", result.data if isinstance(result.data, dict) else {"raw": result.data})
            return {"kind": "triggerExport", "result": result.data, **store.stats()}
        finally:
            store.close()

    if args.command == "latest-export-anchor":
        store = ReadwiseStore()
        try:
            store.init_schema()
            anchor = store.get_latest_export_anchor()
            return {"kind": "latestExportAnchor", "anchor": anchor, **store.stats()}
        finally:
            store.close()

    if args.command == "trigger-delta-export":
        store = ReadwiseStore()
        try:
            store.init_schema()
            anchor = store.get_latest_export_anchor()
            since_updated = args.since_updated or (anchor or {}).get("lastUpdated")
            if not since_updated:
                raise ValueError("No export anchor available. Run a full export first or pass --since-updated explicitly.")
        finally:
            store.close()

        cmd = ["reader-export-documents", "--since-updated", since_updated]
        result = connector.run_json(cmd)
        store = ReadwiseStore()
        try:
            store.init_schema()
            payload = result.data if isinstance(result.data, dict) else {"raw": result.data}
            payload["since_updated"] = since_updated
            store.set_sync_state("last_delta_export_trigger", payload)
            return {"kind": "triggerDeltaExport", "sinceUpdated": since_updated, "result": payload, **store.stats()}
        finally:
            store.close()

    if args.command == "run-delta-refresh":
        store = ReadwiseStore()
        try:
            store.init_schema()
            anchor = store.get_latest_export_anchor()
            since_updated = args.since_updated or (anchor or {}).get("lastUpdated")
            if not since_updated:
                raise ValueError("No export anchor available. Run a full export first or pass --since-updated explicitly.")
        finally:
            store.close()

        trigger = connector.run_json(["reader-export-documents", "--since-updated", since_updated])
        trigger_payload = trigger.data if isinstance(trigger.data, dict) else {"raw": trigger.data}
        export_id = trigger_payload.get("export_id")
        if not export_id:
            raise ValueError("Delta export trigger did not return an export_id.")

        last_result = None
        for attempt in range(args.max_waits):
            result = connector.run_json(["reader-get-export-documents-status", "--export-id", export_id])
            last_result = result.data
            status = result.data.get("status") if isinstance(result.data, dict) else None
            if status == "completed" and result.data.get("download_url"):
                downloaded = download_export_zip(result.data["download_url"], export_id)
                extracted = extract_export_zip(export_id)
                manifest = inspect_extracted_export(export_id)
                payload = ingest_extracted_export(export_id)
                store = ReadwiseStore()
                try:
                    store.init_schema()
                    saved = store.upsert_documents(payload.get("documents", []))
                    trigger_payload["since_updated"] = since_updated
                    store.set_sync_state("last_delta_export_trigger", trigger_payload)
                    store.set_sync_state(f"export_status:{export_id}", result.data)
                    store.set_sync_state(f"export_download:{export_id}", {**downloaded, **extracted})
                    store.set_sync_state(f"export_inspect:{export_id}", manifest)
                    store.set_sync_state(
                        f"export_ingest:{export_id}",
                        {
                            "count": payload.get("count"),
                            "manifestPath": payload.get("manifestPath"),
                            "delta": True,
                            "since_updated": since_updated,
                        },
                    )
                    return {
                        "kind": "runDeltaRefresh",
                        "exportId": export_id,
                        "sinceUpdated": since_updated,
                        "attempts": attempt + 1,
                        "status": result.data,
                        "ingested": saved,
                        "manifestPath": payload.get("manifestPath"),
                        **store.stats(),
                    }
                finally:
                    store.close()
            if attempt < args.max_waits - 1:
                time.sleep(args.poll_seconds)

        store = ReadwiseStore()
        try:
            store.init_schema()
            trigger_payload["since_updated"] = since_updated
            store.set_sync_state("last_delta_export_trigger", trigger_payload)
            if isinstance(last_result, dict):
                store.set_sync_state(f"export_status:{export_id}", last_result)
            return {
                "kind": "runDeltaRefresh",
                "exportId": export_id,
                "sinceUpdated": since_updated,
                "attempts": args.max_waits,
                "status": last_result,
                "timedOut": True,
                **store.stats(),
            }
        finally:
            store.close()

    if args.command == "sync-health":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return store.sync_health()
        finally:
            store.close()

    if args.command == "export-status":
        result = connector.run_json(["reader-get-export-documents-status", "--export-id", args.export_id])
        store = ReadwiseStore()
        try:
            store.init_schema()
            store.set_sync_state(f"export_status:{args.export_id}", result.data if isinstance(result.data, dict) else {"raw": result.data})
            return {"kind": "exportStatus", "exportId": args.export_id, "result": result.data, **store.stats()}
        finally:
            store.close()

    if args.command == "wait-export-and-ingest":
        last_result = None
        for attempt in range(args.max_waits):
            result = connector.run_json(["reader-get-export-documents-status", "--export-id", args.export_id])
            last_result = result.data
            status = result.data.get("status") if isinstance(result.data, dict) else None
            if status == "completed" and result.data.get("download_url"):
                downloaded = download_export_zip(result.data["download_url"], args.export_id)
                extracted = extract_export_zip(args.export_id)
                manifest = inspect_extracted_export(args.export_id)
                payload = ingest_extracted_export(args.export_id)
                store = ReadwiseStore()
                try:
                    store.init_schema()
                    saved = store.upsert_documents(payload.get("documents", []))
                    store.set_sync_state(f"export_status:{args.export_id}", result.data)
                    store.set_sync_state(f"export_download:{args.export_id}", {**downloaded, **extracted})
                    store.set_sync_state(f"export_inspect:{args.export_id}", manifest)
                    store.set_sync_state(
                        f"export_ingest:{args.export_id}",
                        {
                            "count": payload.get("count"),
                            "manifestPath": payload.get("manifestPath"),
                        },
                    )
                    return {
                        "kind": "waitExportAndIngest",
                        "exportId": args.export_id,
                        "attempts": attempt + 1,
                        "status": result.data,
                        "ingested": saved,
                        "manifestPath": payload.get("manifestPath"),
                        **store.stats(),
                    }
                finally:
                    store.close()
            if attempt < args.max_waits - 1:
                time.sleep(args.poll_seconds)

        store = ReadwiseStore()
        try:
            store.init_schema()
            if isinstance(last_result, dict):
                store.set_sync_state(f"export_status:{args.export_id}", last_result)
            return {
                "kind": "waitExportAndIngest",
                "exportId": args.export_id,
                "attempts": args.max_waits,
                "status": last_result,
                "timedOut": True,
                **store.stats(),
            }
        finally:
            store.close()

    if args.command == "download-export":
        downloaded = download_export_zip(args.download_url, args.export_id)
        extracted = extract_export_zip(args.export_id)
        store = ReadwiseStore()
        try:
            store.init_schema()
            store.set_sync_state(f"export_download:{args.export_id}", {**downloaded, **extracted})
            return {"kind": "downloadExport", "exportId": args.export_id, "download": downloaded, "extract": extracted, **store.stats()}
        finally:
            store.close()

    if args.command == "inspect-export":
        manifest = inspect_extracted_export(args.export_id)
        store = ReadwiseStore()
        try:
            store.init_schema()
            store.set_sync_state(f"export_inspect:{args.export_id}", manifest)
            return {"kind": "inspectExport", "exportId": args.export_id, "manifest": manifest, **store.stats()}
        finally:
            store.close()

    if args.command == "ingest-export":
        payload = ingest_extracted_export(args.export_id)
        store = ReadwiseStore()
        try:
            store.init_schema()
            saved = store.upsert_documents(payload.get("documents", []))
            store.set_sync_state(
                f"export_ingest:{args.export_id}",
                {
                    "count": payload.get("count"),
                    "manifestPath": payload.get("manifestPath"),
                },
            )
            return {
                "kind": "ingestExport",
                "exportId": args.export_id,
                "ingested": saved,
                "manifestPath": payload.get("manifestPath"),
                **store.stats(),
            }
        finally:
            store.close()

    if args.command == "semantic-prepare-tagged-docs":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return store.prepare_semantic_records_for_tagged_docs(
                limit=args.limit,
                chunk_limit=args.chunk_limit,
                location=args.location,
                force=bool(args.force),
            )
        finally:
            store.close()

    if args.command == "semantic-prepare-docs":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return store.prepare_semantic_records_for_documents(args.document_ids, chunk_limit=args.chunk_limit)
        finally:
            store.close()

    if args.command == "semantic-embed-tagged-docs":
        store = ReadwiseStore()
        try:
            store.init_schema()
            provider = build_embedding_provider(
                args.provider,
                model=args.model,
                base_url=args.base_url,
                dimensions=args.dimensions,
            )
            return store.embed_prepared_records_for_tagged_docs(
                provider,
                limit=args.limit,
                batch_size=args.batch_size,
            )
        finally:
            store.close()

    if args.command == "semantic-embed-docs":
        store = ReadwiseStore()
        try:
            store.init_schema()
            provider = build_embedding_provider(
                args.provider,
                model=args.model,
                base_url=args.base_url,
                dimensions=args.dimensions,
            )
            return store.embed_prepared_records(
                provider,
                document_ids=args.document_ids,
                batch_size=args.batch_size,
            )
        finally:
            store.close()

    if args.command == "semantic-list-docs":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return store.list_semantic_documents(status=args.status, limit=args.limit)
        finally:
            store.close()

    if args.command == "semantic-stats":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return {"kind": "semanticStats", **store.semantic_stats()}
        finally:
            store.close()

    if args.command == "store-stats":
        store = ReadwiseStore()
        try:
            store.init_schema()
            return {"kind": "storeStats", **store.stats()}
        finally:
            store.close()

    if args.command == "eval-query":
        store = ReadwiseStore()
        try:
            store.init_schema()
            evidence = store.build_evidence_set(
                args.query,
                doc_limit=args.doc_limit,
                highlight_limit=args.highlight_limit,
                chunk_limit=args.chunk_limit,
                strict_mode=bool(getattr(args, "strict", False)),
            )
            documents = evidence.get("documents", []) or []
            strong_docs = [d for d in documents if (d.get("selectionSignals") or {}).get("sourceQualityTier") == "strong"]
            weak_docs = [d for d in documents if (d.get("selectionSignals") or {}).get("sourceQualityTier") == "weak"]
            tagged_docs = [d for d in documents if (d.get("selectionSignals") or {}).get("hasManualTags")]
            broad = bool((evidence.get("queryProfile") or {}).get("isBroad"))
            query_profile = evidence.get("queryProfile") or {}
            vague_terms = set(query_profile.get("vagueTerms") or [])
            drift_titles = []
            weak_anchor_titles = []
            for d in documents[:4]:
                title = (d.get("title") or "").lower()
                title_tokens = set(title.replace(":", " ").replace("-", " ").split())
                strength = d.get("matchStrength") or {}
                signals = d.get("selectionSignals") or {}
                if vague_terms and not (title_tokens & vague_terms) and not (strength.get("tagSupport") or strength.get("phraseSupport")):
                    drift_titles.append(d.get("title"))
                if (strength.get("titleSupport", 0) + strength.get("tagSupport", 0) + strength.get("phraseSupport", 0)) < 2 and (strength.get("requestedTagHits", 0) == 0):
                    weak_anchor_titles.append(d.get("title"))
                if broad and not signals.get("hasManualTags") and (signals.get("titleDrift", 0) or signals.get("summaryDrift", 0)):
                    if d.get("title") not in drift_titles:
                        drift_titles.append(d.get("title"))
            precision = 3 if documents and all((d.get("matchStrength") or {}).get("titleSupport", 0) or (d.get("matchStrength") or {}).get("tagSupport", 0) for d in documents[:min(3, len(documents))]) else (2 if documents else 0)
            source_quality = 3 if strong_docs and not weak_docs else (2 if strong_docs else 1 if documents else 0)
            diversity = 3 if len({d.get("sourceDomain") or d.get("category") for d in documents}) >= min(3, len(documents)) else (2 if len(documents) >= 2 else 1 if documents else 0)
            synthesis_usefulness = 3 if len(documents) >= 3 and (len(tagged_docs) >= 1 or len(strong_docs) >= 2) else (2 if len(documents) >= 2 else 1 if documents else 0)
            if broad and weak_docs:
                source_quality = max(0, source_quality - 1)
            if broad and drift_titles:
                precision = max(0, precision - 1)
                synthesis_usefulness = max(0, synthesis_usefulness - 1)
            if broad and len(weak_anchor_titles) >= max(1, len(documents) // 2):
                precision = max(0, precision - 1)
                synthesis_usefulness = max(0, synthesis_usefulness - 1)
            return {
                "kind": "evalQuery",
                "query": args.query,
                "strictMode": bool(getattr(args, "strict", False)),
                "scores": {
                    "top3Precision": precision,
                    "sourceQuality": source_quality,
                    "evidenceDiversity": diversity,
                    "synthesisUsefulness": synthesis_usefulness,
                },
                "summary": {
                    "documents": len(documents),
                    "highlights": len(evidence.get("highlights", []) or []),
                    "strongDocuments": len(strong_docs),
                    "weakDocuments": len(weak_docs),
                    "taggedDocuments": len(tagged_docs),
                    "confidence": evidence.get("confidence"),
                    "isBroad": broad,
                    "mode": (evidence.get("queryProfile") or {}).get("mode"),
                    "driftTitles": [title for title in drift_titles if title][:4],
                    "weakAnchorTitles": [title for title in weak_anchor_titles if title][:4],
                },
                "topTitles": [d.get("title") for d in documents[:5] if d.get("title")],
                "selectionNotes": evidence.get("selectionNotes", [])[:10],
                "sourceEvidence": evidence,
            }
        finally:
            store.close()

    if args.command == "eval-suite":
        cases_path = Path(args.cases_path).expanduser() if args.cases_path else _default_eval_cases_path()
        raw = json.loads(cases_path.read_text())
        cases = raw.get("cases") or []
        if getattr(args, "mode", None):
            cases = [case for case in cases if case.get("mode") == args.mode]
        store = ReadwiseStore()
        try:
            store.init_schema()
            results = []
            for case in cases:
                evidence = store.build_evidence_set(
                    case.get("query") or "",
                    doc_limit=args.doc_limit,
                    highlight_limit=args.highlight_limit,
                    chunk_limit=args.chunk_limit,
                    strict_mode=bool(getattr(args, "strict", False)),
                )
                result = _evaluate_case(case, evidence)
                result["mode"] = case.get("mode")
                results.append(result)
            passed = sum(1 for item in results if item.get("pass"))
            return {
                "kind": "evalSuite",
                "casesPath": str(cases_path),
                "mode": getattr(args, "mode", None),
                "total": len(results),
                "passed": passed,
                "failed": len(results) - passed,
                "results": results,
            }
        finally:
            store.close()

    raise ValueError(f"Unsupported command: {args.command}")


def render_text(payload: Dict[str, Any]) -> str:
    kind = payload.get("kind")
    lines: List[str] = []

    if kind == "documentSearch":
        lines.append(f"Found {payload.get('count', 0)} document match(es).")
        for idx, item in enumerate(payload.get("results", []), start=1):
            lines.append(f"{idx}. {item.get('title') or '[untitled]'}")
            lines.append(f"   Author: {item.get('author') or 'unknown'}")
            lines.append(f"   Category: {item.get('category') or 'unknown'}")
            lines.append(f"   Location: {item.get('location') or 'unknown'}")
            lines.append(f"   Tags: {_fmt_tags(item.get('tags', []))}")
            if item.get("siteName"):
                lines.append(f"   Site: {item['siteName']}")
            if item.get("publishedDate"):
                lines.append(f"   Published: {item['publishedDate']}")
            if item.get("url"):
                lines.append(f"   URL: {item['url']}")
            if item.get("sourceUrl"):
                lines.append(f"   Source URL: {item['sourceUrl']}")
            if item.get("snippets"):
                lines.append(f"   Match: {item['snippets'][0][:280]}")
        return "\n".join(lines)

    if kind == "documentList":
        lines.append(
            f"Listed {len(payload.get('results', []))} document(s)"
            + (f" of {payload.get('count')} total." if payload.get("count") is not None else ".")
        )
        if payload.get("nextPageCursor"):
            lines.append(f"Next page cursor: {payload['nextPageCursor']}")
        for idx, item in enumerate(payload.get("results", []), start=1):
            lines.append(f"{idx}. {item.get('title') or '[untitled]'}")
            lines.append(f"   Author: {item.get('author') or 'unknown'}")
            lines.append(f"   Category: {item.get('category') or 'unknown'}")
            lines.append(f"   Location: {item.get('location') or 'unknown'}")
            lines.append(f"   Tags: {_fmt_tags(item.get('tags', []))}")
            if item.get("savedAt"):
                lines.append(f"   Saved: {item['savedAt']}")
            if item.get("updatedAt"):
                lines.append(f"   Updated: {item['updatedAt']}")
            if item.get("summary"):
                lines.append(f"   Summary: {item['summary'][:240]}")
        return "\n".join(lines)

    if kind == "documentDetails":
        doc = payload.get("document", {})
        content = (doc.get("content") or "").strip()
        preview = content[:1200]
        lines.extend(
            [
                f"Title: {doc.get('title') or '[untitled]'}",
                f"Author: {doc.get('author') or 'unknown'}",
                f"Category: {doc.get('category') or 'unknown'}",
                f"Location: {doc.get('location') or 'unknown'}",
                f"Tags: {_fmt_tags(doc.get('tags', []))}",
                f"Content length: {doc.get('contentLength', 0)} chars",
                f"Chunks prepared: {len(doc.get('contentChunks', []))}",
            ]
        )
        if doc.get("siteName"):
            lines.append(f"Site: {doc['siteName']}")
        if doc.get("publishedDate"):
            lines.append(f"Published: {doc['publishedDate']}")
        if doc.get("savedAt"):
            lines.append(f"Saved: {doc['savedAt']}")
        if doc.get("url"):
            lines.append(f"URL: {doc['url']}")
        if doc.get("sourceUrl"):
            lines.append(f"Source URL: {doc['sourceUrl']}")
        if doc.get("notes"):
            lines.append(f"Notes: {doc['notes']}")
        if preview:
            lines.append("")
            lines.append("Content preview:")
            lines.append(preview)
            if len(content) > len(preview):
                lines.append("\n[truncated]")
        return "\n".join(lines)

    if kind == "documentHighlights":
        lines.append(f"Found {payload.get('count', 0)} highlight(s) for document {payload.get('documentId')}.")
        for idx, item in enumerate(payload.get("results", []), start=1):
            lines.append(f"{idx}. {((item.get('text') or '').strip() or '[empty highlight]')[:280]}")
            lines.append(f"   Tags: {_fmt_tags(item.get('tags', []))}")
            if item.get("note"):
                lines.append(f"   Note: {item['note']}")
            if item.get("location"):
                lines.append(f"   Location: {item['location']}")
            if item.get("highlightedAt"):
                lines.append(f"   Highlighted: {item['highlightedAt']}")
        return "\n".join(lines)

    if kind == "highlightSearch":
        lines.append(f"Found {payload.get('count', 0)} highlight match(es).")
        for idx, item in enumerate(payload.get("results", []), start=1):
            lines.append(f"{idx}. {item.get('documentTitle') or '[untitled document]'}")
            lines.append(f"   Author: {item.get('documentAuthor') or 'unknown'}")
            lines.append(f"   Category: {item.get('documentCategory') or 'unknown'}")
            lines.append(f"   Document tags: {_fmt_tags(item.get('documentTags', []))}")
            lines.append(f"   Highlight tags: {_fmt_tags(item.get('highlightTags', []))}")
            if item.get("score") is not None:
                lines.append(f"   Score: {item['score']}")
            if item.get("highlightText"):
                lines.append(f"   Highlight: {item['highlightText'][:280]}")
            if item.get("highlightNote"):
                lines.append(f"   Note: {item['highlightNote']}")
            if item.get("url"):
                lines.append(f"   URL: {item['url']}")
        return "\n".join(lines)

    if kind == "tags":
        lines.append(f"Found {payload.get('count', 0)} tag(s).")
        for idx, item in enumerate(payload.get("results", []), start=1):
            lines.append(f"{idx}. {item.get('name') or item.get('key')}")
        return "\n".join(lines)

    if kind == "evidenceSet":
        lines.append(f"Evidence set for: {payload.get('query')}")
        lines.append(
            f"Coverage: {payload.get('documentCount', 0)} doc(s), {payload.get('highlightCount', 0)} highlight(s)"
        )
        if payload.get("confidence"):
            lines.append(f"Confidence: {payload.get('confidence')}")
        if payload.get("strictMode"):
            lines.append("Strict mode: on")
        active_modes = _active_retrieval_modes(payload)
        if active_modes:
            lines.append(f"Retrieval mode: {', '.join(active_modes)}")
        if payload.get("selectionNotes"):
            notes = payload.get('selectionNotes', [])[:4]
            cleaned = [note.replace(':', ': ', 1) if ':' in note else note for note in notes]
            lines.append(f"Selection notes: {'; '.join(cleaned)}")
        semantic = payload.get("semantic") or {}
        if semantic.get("rerankApplied"):
            lines.append(
                f"Semantic rerank: on ({semantic.get('provider')}/{semantic.get('model')}, matched {semantic.get('matchedDocuments', 0)}/{semantic.get('candidateDocuments', 0)} docs, weight {semantic.get('weight')})"
            )
        elif semantic.get("reason"):
            lines.append(f"Semantic rerank: off ({semantic.get('reason')})")
        lines.append("")
        lines.append("Documents:")
        for idx, doc in enumerate(payload.get("documents", []), start=1):
            why = doc.get("matchStrength") or {}
            signals = doc.get("selectionSignals") or {}
            tag_match = signals.get("tagMatch") or {}
            requested_terms = signals.get("requestedTagTerms") or []
            why_bits = []
            if tag_match.get("exactRequested"):
                if requested_terms:
                    why_bits.append(f"requested tags: {', '.join(requested_terms[:3])}")
                else:
                    why_bits.append("requested tag match")
            elif why.get("tagSupport"):
                why_bits.append("tag match")
            if signals.get("titleRequestedHits"):
                why_bits.append("title mentions requested tag")
            elif why.get("titleSupport"):
                why_bits.append("title match")
            if signals.get("summaryRequestedHits"):
                why_bits.append("summary mentions requested tag")
            if why.get("phraseSupport"):
                why_bits.append("phrase hit")
            if (doc.get("selectionSignals") or {}).get("contrastSignal"):
                why_bits.append("counterpoint signal")
            if doc.get("semanticScore") is not None:
                why_bits.append(f"semantic {doc.get('semanticScore')}")
            header = f"D{idx}. {doc.get('title') or '[untitled]'}"
            if doc.get("author"):
                header += f" — {doc.get('author')}"
            lines.append(header)
            meta = []
            if doc.get("sourceDomain"):
                meta.append(doc.get("sourceDomain"))
            if doc.get("tags"):
                meta.append(f"tags: {_fmt_tags(doc.get('tags', []))}")
            meta.append(f"score: {doc.get('qualityScore') or doc.get('cacheScore')}")
            if why_bits:
                meta.append(f"why: {', '.join(why_bits[:3])}")
            lines.append(f"   {' | '.join(meta)}")
            if doc.get("summary"):
                lines.append(f"   {_truncate(doc['summary'])}")
            for chunk in doc.get("chunks", [])[:2]:
                lines.append(f"   • {_truncate(chunk.get('text') or '')}")
        if payload.get("highlights"):
            lines.append("")
            lines.append("Highlights:")
            for idx, hl in enumerate(payload.get("highlights", []), start=1):
                why_bits = []
                if (hl.get("selectionSignals") or {}).get("hasTags"):
                    why_bits.append("tagged")
                if (hl.get("selectionSignals") or {}).get("contrastSignal"):
                    why_bits.append("contrast")
                lines.append(f"H{idx}. {hl.get('documentTitle') or '[untitled document]'}")
                lines.append(f"   {_truncate(hl.get('highlightText') or '')}")
                meta = [f"score: {hl.get('cacheScore')}"]
                if why_bits:
                    meta.append(f"why: {', '.join(why_bits)}")
                lines.append(f"   {' | '.join(meta)}")
        return "\n".join(lines)

    if kind == "synthesisPacket":
        lines.append(f"Synthesis packet for: {payload.get('query')}")
        lines.append(
            f"Coverage: {payload.get('coverage')} | confidence: {payload.get('confidence') or 'unknown'} | evidence: {payload.get('documentCount', 0)} doc(s), {payload.get('highlightCount', 0)} highlight(s)"
        )
        if payload.get("strictMode"):
            lines.append("Strict mode: on")
        active_modes = _active_retrieval_modes(payload)
        if active_modes:
            lines.append(f"Retrieval mode: {', '.join(active_modes)}")
        lines.append("")
        lines.append("Draft synthesis:")
        lines.append(payload.get("draftSynthesis") or "")
        if payload.get("themes"):
            lines.append("")
            lines.append("Themes:")
            for theme in payload.get("themes", [])[:5]:
                anchor = ", ".join(theme.get("documents", [])[:2])
                counts = f"{theme.get('documentCount')} doc(s), {theme.get('highlightCount')} highlight(s)"
                term_detail = f" | terms: {', '.join(theme.get('terms', [])[:3])}" if theme.get("terms") else ""
                source_detail = f" | label: {theme.get('labelSource')}" if theme.get("labelSource") else ""
                detail = f" — {anchor}" if anchor else ""
                lines.append(f"- {theme.get('theme')} [{counts}]{detail}{term_detail}{source_detail}")
                if theme.get("examples"):
                    lines.append(f"  Example: {_truncate(theme.get('examples', [''])[0], 160)}")
        if payload.get("counterpoints"):
            lines.append("")
            lines.append("Counterpoints:")
            for item in payload.get("counterpoints", [])[:3]:
                label = item.get("label") or "Counterpoint"
                title = item.get("title") or "[untitled]"
                theme_hint = f" | theme: {item.get('anchorTheme')}" if item.get("anchorTheme") else ""
                lines.append(f"- {label}")
                lines.append(f"  Source: {title}{theme_hint}")
                lines.append(f"  Claim: {_truncate(item.get('snippet') or '', 200)}")
                if item.get("whyItMatters"):
                    lines.append(f"  Why it matters: {_truncate(item.get('whyItMatters'), 180)}")
        if payload.get("recommendedExpansionQueries"):
            lines.append("")
            lines.append("Recommended expansion queries:")
            for query in payload.get("recommendedExpansionQueries", [])[:6]:
                lines.append(f"- {query}")
        if payload.get("coverageNotes"):
            lines.append("")
            lines.append("Coverage notes:")
            for note in payload.get("coverageNotes", [])[:4]:
                lines.append(f"- {note}")
        if payload.get("evidence"):
            lines.append("")
            lines.append("Evidence shortlist:")
            for item in payload.get("evidence", [])[:8]:
                if item.get("type") == "document":
                    lines.append(f"- doc: {item.get('title')} | score={item.get('qualityScore') or item.get('cacheScore')} | {item.get('whySelected')}")
                else:
                    lines.append(f"- highlight: {item.get('documentTitle')} | score={item.get('cacheScore')} | {item.get('whySelected')}")
        return "\n".join(lines)

    if kind == "expansionCandidates":
        lines.append(f"Expansion candidates for: {payload.get('query')}")
        if payload.get("relatedTerms"):
            lines.append("Related terms:")
            for term in payload.get("relatedTerms", [])[:8]:
                lines.append(f"- {term}")
        if payload.get("relatedTags"):
            lines.append("Related tags:")
            for tag in payload.get("relatedTags", [])[:8]:
                lines.append(f"- {tag}")
        if payload.get("suggestedQueries"):
            lines.append("Suggested queries:")
            for query in payload.get("suggestedQueries", [])[:8]:
                lines.append(f"- {query}")
        return "\n".join(lines)

    if kind == "expandAndCache":
        lines.append(f"Expanded and cached for: {payload.get('rootQuery')}")
        if payload.get("queriesRun"):
            lines.append("Queries run:")
            for query in payload.get("queriesRun", []):
                lines.append(f"- {query}")
        if payload.get("runs"):
            lines.append("Cache results:")
            for run in payload.get("runs", []):
                lines.append(
                    f"- {run.get('query')}: searchResults={run.get('searchResults')}, detailCached={run.get('detailCached')}"
                )
        lines.append(f"Documents: {payload.get('documents', 0)}")
        lines.append(f"Highlights: {payload.get('highlights', 0)}")
        lines.append(f"Tags: {payload.get('tags', 0)}")
        if payload.get("resynthesis"):
            lines.append("")
            lines.append("Resynthesis draft:")
            lines.append(payload["resynthesis"].get("draftSynthesis") or "")
        return "\n".join(lines)

    if kind == "cacheTaggedDocs":
        lines.append("Tagged-doc cache refresh complete.")
        lines.append(f"Pages processed: {payload.get('pagesProcessed', 0)}")
        lines.append(f"Metadata cached: {payload.get('metadataCached', 0)}")
        lines.append(f"Tagged candidates found: {payload.get('taggedCandidates', 0)}")
        lines.append(f"Details cached: {payload.get('detailsCached', 0)}")
        lines.append(f"Highlights saved: {payload.get('highlightsSaved', 0)}")
        if payload.get("nextPageCursor"):
            lines.append(f"Next page cursor: {payload['nextPageCursor']}")
        lines.append(f"Documents: {payload.get('documents', 0)}")
        lines.append(f"Highlights: {payload.get('highlights', 0)}")
        lines.append(f"Tags: {payload.get('tags', 0)}")
        return "\n".join(lines)

    if kind == "triggerExport":
        lines.append("Triggered Reader export job.")
        lines.append(f"Store: {payload.get('dbPath')}")
        lines.append(json.dumps(payload.get('result', {}), indent=2))
        return "\n".join(lines)

    if kind == "latestExportAnchor":
        lines.append("Latest export anchor:")
        lines.append(json.dumps(payload.get('anchor', {}), indent=2))
        return "\n".join(lines)

    if kind == "triggerDeltaExport":
        lines.append(f"Triggered delta export since: {payload.get('sinceUpdated')}")
        lines.append(json.dumps(payload.get('result', {}), indent=2))
        return "\n".join(lines)

    if kind == "runDeltaRefresh":
        lines.append(f"Ran delta refresh since: {payload.get('sinceUpdated')}")
        lines.append(f"Export id: {payload.get('exportId')}")
        lines.append(f"Attempts: {payload.get('attempts')}" )
        if payload.get('timedOut'):
            lines.append("Timed out waiting for delta export completion.")
        if payload.get('status') is not None:
            lines.append(json.dumps(payload.get('status', {}), indent=2))
        if payload.get('ingested') is not None:
            lines.append(f"Ingested documents: {payload.get('ingested')}")
            lines.append(f"Manifest path: {payload.get('manifestPath')}")
        return "\n".join(lines)

    if kind == "syncHealth":
        lines.append(f"Freshness: {payload.get('freshness')}")
        lines.append(f"Documents: {payload.get('documents', 0)}")
        lines.append(f"Tags: {payload.get('tags', 0)}")
        lines.append(f"Highlights: {payload.get('highlights', 0)}")
        operator = payload.get('operatorSummary') or {}
        if operator:
            lines.append("Operator summary:")
            lines.append(json.dumps(operator, indent=2))
        if payload.get('latestExportAnchor'):
            lines.append("Latest export anchor:")
            lines.append(json.dumps(payload.get('latestExportAnchor'), indent=2))
        if payload.get('latestIngest'):
            lines.append("Latest ingest:")
            lines.append(json.dumps(payload.get('latestIngest'), indent=2))
        if payload.get('lastDeltaExportTrigger'):
            lines.append("Last delta trigger:")
            lines.append(json.dumps(payload.get('lastDeltaExportTrigger'), indent=2))
        if payload.get('lastDeltaStatus'):
            lines.append("Last delta status:")
            lines.append(json.dumps(payload.get('lastDeltaStatus'), indent=2))
        if payload.get('failureSummary'):
            lines.append(f"Failure summary: {payload.get('failureSummary')}")
        if payload.get('recommendedActions'):
            lines.append("Recommended actions:")
            for action in payload.get('recommendedActions', []):
                lines.append(f"- {action}")
        if payload.get('recentEvents'):
            lines.append("Recent sync events:")
            for event in payload.get('recentEvents', [])[:6]:
                value = event.get('value') or {}
                summary_bits = []
                if isinstance(value, dict):
                    if value.get('status'):
                        summary_bits.append(f"status={value.get('status')}")
                    if value.get('export_id'):
                        summary_bits.append(f"export={value.get('export_id')}")
                    if value.get('count') is not None:
                        summary_bits.append(f"count={value.get('count')}")
                    if value.get('last_updated'):
                        summary_bits.append(f"last_updated={value.get('last_updated')}")
                suffix = f" | {'; '.join(summary_bits[:3])}" if summary_bits else ""
                lines.append(f"- {event.get('updatedAt')} | {event.get('key')}{suffix}")
        if payload.get('notes'):
            lines.append("Notes:")
            for note in payload.get('notes', []):
                lines.append(f"- {note}")
        return "\n".join(lines)

    if kind == "evalQuery":
        lines.append(f"Eval query: {payload.get('query')}")
        lines.append(f"Strict mode: {payload.get('strictMode')}")
        scores = payload.get('scores') or {}
        lines.append("Scores:")
        lines.append(f"- Top-3 precision: {scores.get('top3Precision')}/3")
        lines.append(f"- Source quality: {scores.get('sourceQuality')}/3")
        lines.append(f"- Evidence diversity: {scores.get('evidenceDiversity')}/3")
        lines.append(f"- Synthesis usefulness: {scores.get('synthesisUsefulness')}/3")
        summary = payload.get('summary') or {}
        lines.append("Summary:")
        lines.append(json.dumps(summary, indent=2))
        if payload.get('topTitles'):
            lines.append("Top titles:")
            for title in payload.get('topTitles', []):
                lines.append(f"- {title}")
        if payload.get('selectionNotes'):
            lines.append("Selection notes:")
            for note in payload.get('selectionNotes', []):
                lines.append(f"- {note}")
        return "\n".join(lines)

    if kind == "evalSuite":
        lines.append(f"Eval suite: {payload.get('casesPath')}")
        if payload.get('mode'):
            lines.append(f"Mode: {payload.get('mode')}")
        lines.append(f"Passed: {payload.get('passed')}/{payload.get('total')} | Failed: {payload.get('failed')}" )
        for item in payload.get('results', []):
            status = "PASS" if item.get('pass') else "FAIL"
            lines.append(f"- [{status}] {item.get('query')}")
            lines.append(
                f"  docs={item.get('documentCount')} expected={item.get('expectedMatches')} rejected={item.get('rejectedMatches')} top3_expected={item.get('top3ExpectedMatches')} top3_rejected={item.get('top3RejectedMatches')} countOk={item.get('countOk')} confidence={item.get('confidence')} strong={item.get('strongDocuments')}"
            )
            for title in item.get('titles', [])[:4]:
                lines.append(f"    • {title}")
        return "\n".join(lines)

    if kind == "exportStatus":
        lines.append(f"Export status for: {payload.get('exportId')}")
        lines.append(f"Store: {payload.get('dbPath')}")
        lines.append(json.dumps(payload.get('result', {}), indent=2))
        return "\n".join(lines)

    if kind == "waitExportAndIngest":
        lines.append(f"Waited on export: {payload.get('exportId')}")
        lines.append(f"Attempts: {payload.get('attempts')}" )
        if payload.get('timedOut'):
            lines.append("Timed out waiting for export completion.")
        if payload.get('status') is not None:
            lines.append(json.dumps(payload.get('status', {}), indent=2))
        if payload.get('ingested') is not None:
            lines.append(f"Ingested documents: {payload.get('ingested')}")
            lines.append(f"Manifest path: {payload.get('manifestPath')}")
        return "\n".join(lines)

    if kind == "downloadExport":
        lines.append(f"Downloaded export: {payload.get('exportId')}")
        lines.append(json.dumps(payload.get('download', {}), indent=2))
        lines.append(json.dumps(payload.get('extract', {}), indent=2))
        return "\n".join(lines)

    if kind == "inspectExport":
        lines.append(f"Inspected export: {payload.get('exportId')}")
        lines.append(json.dumps(payload.get('manifest', {}), indent=2))
        return "\n".join(lines)

    if kind == "ingestExport":
        lines.append(f"Ingested export: {payload.get('exportId')}")
        lines.append(f"Ingested documents: {payload.get('ingested', 0)}")
        lines.append(f"Manifest path: {payload.get('manifestPath')}")
        lines.append(f"Documents: {payload.get('documents', 0)}")
        lines.append(f"Highlights: {payload.get('highlights', 0)}")
        lines.append(f"Tags: {payload.get('tags', 0)}")
        return "\n".join(lines)

    if kind == "semanticPrepareTaggedDocs" or kind == "semanticPrepareDocuments":
        lines.append("Semantic preparation complete.")
        lines.append(f"Prepared documents: {payload.get('preparedDocuments', 0)}")
        lines.append(f"Skipped documents: {payload.get('skippedDocuments', 0)}")
        lines.append(f"Prepared records: {payload.get('recordsPrepared', 0)}")
        lines.append(f"Selected chunks: {payload.get('selectedChunks', 0)}")
        if payload.get("documentIds"):
            lines.append("Document ids:")
            for document_id in payload.get("documentIds", [])[:10]:
                lines.append(f"- {document_id}")
        return "\n".join(lines)

    if kind == "semanticEmbedRecords":
        lines.append("Semantic embedding complete.")
        lines.append(f"Provider/model: {payload.get('provider')}/{payload.get('model')}")
        lines.append(f"Embedded records: {payload.get('embeddedRecords', 0)}")
        lines.append(f"Embedded documents: {payload.get('embeddedDocuments', 0)}")
        lines.append(f"Skipped records: {payload.get('skippedRecords', 0)}")
        lines.append(f"Failed records: {payload.get('failedRecords', 0)}")
        if payload.get("documentIds"):
            lines.append("Document ids:")
            for document_id in payload.get("documentIds", [])[:10]:
                lines.append(f"- {document_id}")
        return "\n".join(lines)

    if kind == "semanticListDocuments":
        lines.append(f"Semantic documents: {payload.get('count', 0)}")
        for item in payload.get("results", []):
            lines.append(f"- {item.get('document_id')} [{item.get('embedding_status')}] {item.get('embedding_provider') or '-'} / {item.get('embedding_model') or '-'}")
            if item.get("embedding_error"):
                lines.append(f"  error: {_truncate(item.get('embedding_error'), 180)}")
        return "\n".join(lines)

    if kind == "semanticStats":
        lines.append(f"Store: {payload.get('dbPath')}")
        lines.append(f"Tagged documents: {payload.get('taggedDocuments', 0)}")
        lines.append(f"Semantic documents: {payload.get('semanticDocuments', 0)}")
        lines.append(f"Prepared semantic docs: {payload.get('semanticPreparedDocuments', 0)}")
        lines.append(f"Embedded semantic docs: {payload.get('semanticEmbeddedDocuments', 0)}")
        lines.append(f"Error semantic docs: {payload.get('semanticErrorDocuments', 0)}")
        lines.append(f"Prepared embedding records: {payload.get('preparedEmbeddingRecords', 0)}")
        lines.append(f"Embedded embedding records: {payload.get('embeddedEmbeddingRecords', 0)}")
        lines.append(f"Error embedding records: {payload.get('errorEmbeddingRecords', 0)}")
        if payload.get("semanticCoverage"):
            coverage = payload.get("semanticCoverage") or {}
            lines.append(
                f"Tagged coverage: prepared {coverage.get('taggedPreparedPct', 0)}% | embedded {coverage.get('taggedEmbeddedPct', 0)}%"
            )
        if payload.get("embeddingStatuses"):
            lines.append("Embedding statuses:")
            for kind_name, count in sorted((payload.get("embeddingStatuses") or {}).items()):
                lines.append(f"- {kind_name}: {count}")
        if payload.get("embeddingKinds"):
            lines.append("Embedding kinds:")
            for kind_name, count in sorted((payload.get("embeddingKinds") or {}).items()):
                lines.append(f"- {kind_name}: {count}")
        if payload.get("embeddedModels"):
            lines.append("Embedded models:")
            for item in payload.get("embeddedModels", []):
                lines.append(f"- {item.get('provider')}/{item.get('model')}: {item.get('count')} records")
        return "\n".join(lines)

    if kind in {"storeInit", "cacheTags", "cacheListDocs", "cacheDoc", "storeStats"}:
        lines.append(f"Store: {payload.get('dbPath')}")
        if payload.get("saved") is not None:
            lines.append(f"Saved: {payload['saved']}")
        if payload.get("highlightsSaved") is not None:
            lines.append(f"Highlights saved: {payload['highlightsSaved']}")
        if payload.get("nextPageCursor"):
            lines.append(f"Next page cursor: {payload['nextPageCursor']}")
        lines.append(f"Documents: {payload.get('documents', 0)}")
        lines.append(f"Highlights: {payload.get('highlights', 0)}")
        lines.append(f"Tags: {payload.get('tags', 0)}")
        lines.append(f"Semantic docs: {payload.get('document_semantic_index', 0)}")
        lines.append(f"Embedding records: {payload.get('document_embeddings', 0)}")
        lines.append(f"Sync state rows: {payload.get('sync_state', 0)}")
        return "\n".join(lines)

    return json.dumps(payload, indent=2)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    connector = ReadwiseConnector()

    try:
        payload = run_command(args, connector)
    except (ReadwiseCliMissingError, ReadwiseJsonError, ReadwiseCommandError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

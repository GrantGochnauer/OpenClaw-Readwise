#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize_tags(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        result: List[str] = []
        for item in raw:
            if isinstance(item, str) and item:
                result.append(item)
            elif isinstance(item, dict):
                value = item.get("name") or item.get("key") or item.get("value")
                if isinstance(value, str) and value:
                    result.append(value)
        return sorted(dict.fromkeys(result))
    if isinstance(raw, dict):
        result = []
        for key, value in raw.items():
            if isinstance(value, str) and value:
                result.append(value)
            elif isinstance(key, str) and key:
                result.append(key)
        return sorted(dict.fromkeys(result))
    if isinstance(raw, str):
        return [raw]
    return []


def _content_chunks(text: Optional[str], *, chunk_size: int = 2000, max_chunks: int = 8) -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        return []
    source = text.strip()
    if not source:
        return []

    chunks = []
    total = len(source)
    for index, start in enumerate(range(0, min(total, chunk_size * max_chunks), chunk_size), start=1):
        chunk_text = source[start : start + chunk_size]
        chunks.append(
            {
                "index": index,
                "start": start,
                "end": start + len(chunk_text),
                "text": chunk_text,
            }
        )
    return chunks


def _base_document_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "documentId": item.get("id") or item.get("document_id"),
        "title": item.get("title"),
        "author": item.get("author"),
        "source": item.get("source"),
        "category": item.get("category"),
        "location": item.get("location"),
        "url": item.get("url"),
        "siteName": item.get("site_name"),
        "summary": item.get("summary"),
        "sourceUrl": item.get("source_url"),
        "savedAt": item.get("saved_at"),
        "updatedAt": item.get("updated_at"),
        "publishedDate": item.get("published_date"),
        "wordCount": item.get("word_count"),
        "readingTime": item.get("reading_time"),
        "firstOpenedAt": item.get("first_opened_at"),
        "lastOpenedAt": item.get("last_opened_at"),
        "tags": _normalize_tags(item.get("tags")),
    }


def normalize_document_search(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, list):
        raise ValueError("Expected document search payload to be a list.")

    results = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        matches = item.get("matches")
        snippets = []
        if isinstance(matches, list):
            for match in matches:
                if isinstance(match, dict):
                    text = match.get("plaintext")
                    if isinstance(text, str) and text.strip():
                        snippets.append(text.strip())

        row = _base_document_metadata(item)
        row.update(
            {
                "matchCount": len(snippets),
                "snippets": snippets,
            }
        )
        results.append(row)

    return {"kind": "documentSearch", "count": len(results), "results": results}


def normalize_document_details(payload: Any, *, chunk_size: int = 2000, max_chunks: int = 8) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected document details payload to be an object.")

    content = payload.get("content")
    document = _base_document_metadata(payload)
    document.update(
        {
            "notes": payload.get("notes"),
            "content": content,
            "contentLength": len(content) if isinstance(content, str) else 0,
            "contentChunks": _content_chunks(content, chunk_size=chunk_size, max_chunks=max_chunks),
        }
    )
    return {"kind": "documentDetails", "document": document}


def normalize_document_list(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected document list payload to be an object.")

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Expected document list payload to contain results[].")

    results = []
    for item in raw_results:
        if isinstance(item, dict):
            results.append(_base_document_metadata(item))

    return {
        "kind": "documentList",
        "count": payload.get("count", len(results)),
        "nextPageCursor": payload.get("nextPageCursor"),
        "results": results,
    }


def normalize_highlight_search(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, list):
        raise ValueError("Expected highlight search payload to be a list.")

    results = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        attrs = item.get("attributes", {}) if isinstance(item.get("attributes"), dict) else {}
        results.append(
            {
                "highlightId": item.get("id"),
                "score": item.get("score"),
                "url": item.get("url"),
                "documentTitle": attrs.get("document_title"),
                "documentAuthor": attrs.get("document_author"),
                "documentCategory": attrs.get("document_category"),
                "documentTags": _normalize_tags(attrs.get("document_tags")),
                "highlightText": attrs.get("highlight_plaintext"),
                "highlightNote": attrs.get("highlight_note"),
                "highlightTags": _normalize_tags(attrs.get("highlight_tags")),
            }
        )

    return {"kind": "highlightSearch", "count": len(results), "results": results}


def normalize_document_highlights(payload: Any, *, document_id: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected document highlights payload to be an object.")

    raw_results = payload.get("result", [])
    if not isinstance(raw_results, list):
        raise ValueError("Expected document highlights payload to contain result[].")

    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "highlightId": item.get("id"),
                "documentId": document_id,
                "text": item.get("text") or item.get("highlighted_text") or item.get("content"),
                "note": item.get("note"),
                "location": item.get("location") or item.get("location_in_document"),
                "highlightedAt": item.get("highlighted_at") or item.get("created_at"),
                "updatedAt": item.get("updated_at"),
                "tags": _normalize_tags(item.get("tags")),
                "color": item.get("color"),
            }
        )

    return {"kind": "documentHighlights", "documentId": document_id, "count": len(results), "results": results}


def normalize_tags(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, list):
        raise ValueError("Expected tags payload to be a list.")

    tags = []
    for item in payload:
        if isinstance(item, dict):
            tags.append({"key": item.get("key"), "name": item.get("name")})

    tags.sort(key=lambda x: (x.get("name") or x.get("key") or "").lower())
    return {"kind": "tags", "count": len(tags), "results": tags}

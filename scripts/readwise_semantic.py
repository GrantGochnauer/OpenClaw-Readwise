#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


SEMANTIC_TEXT_VERSION = "v1"
EMBEDDING_KINDS = ("title", "summary", "tags", "doc_blend", "chunk")
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass
class SemanticEmbeddingProvider:
    name: str = "unconfigured"
    vector_dim: Optional[int] = None
    model: Optional[str] = None

    def is_configured(self) -> bool:
        return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("No embedding provider configured for Readwise semantic indexing yet.")


class NullEmbeddingProvider(SemanticEmbeddingProvider):
    pass


class OpenAIEmbeddingProvider(SemanticEmbeddingProvider):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout_seconds: int = 60,
    ):
        super().__init__(name="openai", model=model)
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.is_configured():
            raise RuntimeError("OPENAI_API_KEY is missing; cannot create Readwise embeddings.")
        if not texts:
            return []
        payload: Dict[str, Any] = {
            "model": self.model or DEFAULT_OPENAI_EMBEDDING_MODEL,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embeddings request failed ({exc.code}): {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI embeddings endpoint returned invalid JSON.") from exc

        items = data.get("data") or []
        items = sorted(items, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") or [] for item in items]
        if len(vectors) != len(texts):
            raise RuntimeError(f"OpenAI embeddings response length mismatch: expected {len(texts)}, got {len(vectors)}")
        if vectors and not self.vector_dim:
            self.vector_dim = len(vectors[0])
        return vectors


class SemanticPreparationError(ValueError):
    pass


LOW_SIGNAL_LINES = (
    "share this",
    "subscribe",
    "sign up",
    "advertisement",
    "sponsored",
    "cookie policy",
    "all rights reserved",
)


LOW_SIGNAL_TOKENS = {
    "http", "https", "www", "com", "org", "net", "readwise", "reader", "twitter", "x", "tweet",
    "tweets", "video", "podcast", "article", "articles", "document", "documents", "highlight", "highlights",
}


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def build_embedding_provider(
    provider_name: str = "openai",
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    dimensions: Optional[int] = None,
    timeout_seconds: int = 60,
) -> SemanticEmbeddingProvider:
    normalized = (provider_name or "openai").strip().lower()
    if normalized == "openai":
        return OpenAIEmbeddingProvider(
            model=model or DEFAULT_OPENAI_EMBEDDING_MODEL,
            base_url=base_url,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
        )
    if normalized in {"none", "null", "disabled"}:
        return NullEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {provider_name}")


def normalize_whitespace(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_semantic_text(text: Optional[str]) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in LOW_SIGNAL_LINES):
            continue
        if len(re.findall(r"https?://", lowered)) >= 2:
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"\[[^\]]+\]\([^)]*\)", lambda m: m.group(0).split("](")[0].lstrip("["), line)
        line = re.sub(r"`{1,3}", "", line)
        line = re.sub(r"^\s*[-*•]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        line = normalize_whitespace(line)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def semantic_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def token_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for match in TOKEN_RE.findall((text or "").lower()):
        token = match.strip("- ")
        if len(token) < 3 or token in LOW_SIGNAL_TOKENS:
            continue
        counts[token] = counts.get(token, 0) + 1
    return counts


def score_chunk_text(text: str, title_terms: Optional[Iterable[str]] = None, tag_terms: Optional[Iterable[str]] = None) -> int:
    norm = normalize_semantic_text(text)
    if not norm:
        return -999
    counts = token_counts(norm)
    if not counts:
        return -999
    word_count = sum(counts.values())
    unique_count = len(counts)
    score = min(word_count, 80) + min(unique_count * 2, 40)
    score -= norm.lower().count("http") * 12
    score -= len(re.findall(r"^\s*[-*•]\s+", text or "", flags=re.MULTILINE)) * 2

    title_term_set = {t.lower() for t in (title_terms or []) if t}
    tag_term_set = {t.lower() for t in (tag_terms or []) if t}
    overlap_title = len(title_term_set & set(counts))
    overlap_tags = len(tag_term_set & set(counts))
    score += overlap_title * 8
    score += overlap_tags * 6

    if word_count < 25:
        score -= 10
    if unique_count < 8:
        score -= 10
    return score


def select_semantic_chunks(document: Dict[str, Any], *, limit: int = 4) -> List[Dict[str, Any]]:
    chunks = document.get("contentChunks") or document.get("chunks") or []
    title_terms = TOKEN_RE.findall((document.get("title") or "").lower())
    tag_terms: List[str] = []
    for tag in document.get("tags") or []:
        tag_terms.extend(TOKEN_RE.findall((tag or "").lower()))

    ranked: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text") or ""
        prepared = normalize_semantic_text(text)
        score = score_chunk_text(prepared, title_terms=title_terms, tag_terms=tag_terms)
        if score < 20:
            continue
        ranked.append(
            {
                "chunkIndex": idx,
                "text": prepared,
                "charCount": len(prepared),
                "score": score,
                "source": chunk,
            }
        )

    ranked.sort(key=lambda item: (item["score"], item["charCount"]), reverse=True)
    return ranked[:limit]


def build_semantic_texts(document: Dict[str, Any], *, chunk_limit: int = 4) -> Dict[str, Any]:
    document_id = document.get("documentId")
    if not document_id:
        raise SemanticPreparationError("Document is missing documentId.")

    title = normalize_semantic_text(document.get("title"))
    summary = normalize_semantic_text(document.get("summary"))
    tags = [normalize_whitespace(tag) for tag in (document.get("tags") or []) if normalize_whitespace(tag)]
    notes = normalize_semantic_text(document.get("notes"))
    selected_chunks = select_semantic_chunks(document, limit=chunk_limit)

    tag_text = "tags: " + ", ".join(tags) if tags else ""
    blend_parts = []
    if title:
        blend_parts.append(f"title: {title}")
    if summary:
        blend_parts.append(f"summary: {summary}")
    if tag_text:
        blend_parts.append(tag_text)
    if notes:
        blend_parts.append(f"notes: {notes}")
    if selected_chunks:
        blend_parts.append(
            "key passages:\n" + "\n\n".join(
                f"chunk {item['chunkIndex'] + 1}: {item['text']}" for item in selected_chunks
            )
        )
    doc_blend = "\n\n".join(part for part in blend_parts if part).strip()

    records: List[Dict[str, Any]] = []
    if title:
        records.append({"embeddingKind": "title", "chunkIndex": None, "text": title, "textPreview": title[:240]})
    if summary:
        records.append({"embeddingKind": "summary", "chunkIndex": None, "text": summary, "textPreview": summary[:240]})
    if tag_text:
        records.append({"embeddingKind": "tags", "chunkIndex": None, "text": tag_text, "textPreview": tag_text[:240]})
    if doc_blend:
        records.append({"embeddingKind": "doc_blend", "chunkIndex": None, "text": doc_blend, "textPreview": doc_blend[:240]})
    for item in selected_chunks:
        records.append(
            {
                "embeddingKind": "chunk",
                "chunkIndex": item["chunkIndex"],
                "text": item["text"],
                "textPreview": item["text"][:240],
                "selectionScore": item["score"],
            }
        )

    basis_payload = {
        "version": SEMANTIC_TEXT_VERSION,
        "documentId": document_id,
        "title": title,
        "summary": summary,
        "tags": tags,
        "notes": notes,
        "chunkRecords": [
            {"chunkIndex": item["chunkIndex"], "textHash": text_hash(item["text"]), "score": item["score"]}
            for item in selected_chunks
        ],
    }

    return {
        "documentId": document_id,
        "textVersion": SEMANTIC_TEXT_VERSION,
        "basisHash": semantic_hash(basis_payload),
        "title": title,
        "summary": summary,
        "tagText": tag_text,
        "docBlend": doc_blend,
        "selectedChunks": [
            {"chunkIndex": item["chunkIndex"], "text": item["text"], "selectionScore": item["score"]}
            for item in selected_chunks
        ],
        "records": records,
    }

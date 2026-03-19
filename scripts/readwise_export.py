#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from readwise_normalize import normalize_document_details
from readwise_store import workspace_root


def exports_root() -> Path:
    return workspace_root() / "data" / "readwise" / "exports"


def export_paths(export_id: str) -> Dict[str, Path]:
    base = exports_root() / export_id
    return {
        "base": base,
        "zip": base / "export.zip",
        "extract": base / "extracted",
        "manifest": base / "manifest.json",
    }


def download_export_zip(download_url: str, export_id: str) -> Dict[str, Any]:
    paths = export_paths(export_id)
    paths["base"].mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(download_url) as response, open(paths["zip"], "wb") as out:
        shutil.copyfileobj(response, out)
    size = paths["zip"].stat().st_size
    return {"exportId": export_id, "zipPath": str(paths["zip"]), "bytes": size}


def extract_export_zip(export_id: str) -> Dict[str, Any]:
    paths = export_paths(export_id)
    paths["extract"].mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths["zip"], "r") as zf:
        zf.extractall(paths["extract"])
        names = zf.namelist()
    return {
        "exportId": export_id,
        "extractPath": str(paths["extract"]),
        "fileCount": len(names),
        "sample": names[:20],
    }


def _split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return {}, text
    _, rest = parts
    fm_block = text[4 : len(text) - len(rest) - 5]
    frontmatter = _parse_frontmatter_block(fm_block)
    return frontmatter, rest


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]
    if value.lower() == "null":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def _parse_frontmatter_block(block: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if re.match(r"^[A-Za-z0-9_\-]+:\s*", line):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                result[key] = []
                current_key = key
            else:
                result[key] = _parse_scalar(value)
                current_key = key
        elif line.lstrip().startswith("- ") and current_key:
            item = _parse_scalar(line.lstrip()[2:])
            existing = result.get(current_key)
            if not isinstance(existing, list):
                existing = []
                result[current_key] = existing
            existing.append(item)
        else:
            current_key = None
    return result


def inspect_extracted_export(export_id: str) -> Dict[str, Any]:
    paths = export_paths(export_id)
    md_files = sorted(paths["extract"].rglob("*.md"))
    sample_docs = []
    for path in md_files[:10]:
        text = path.read_text(encoding="utf-8", errors="replace")
        fm, body = _split_frontmatter(text)
        sample_docs.append(
            {
                "path": str(path.relative_to(paths["extract"])),
                "title": fm.get("title") or path.stem,
                "tags": fm.get("tags") if isinstance(fm.get("tags"), list) else [],
                "keys": sorted(fm.keys()),
                "bodyChars": len(body),
            }
        )
    manifest = {
        "exportId": export_id,
        "extractPath": str(paths["extract"]),
        "markdownFiles": len(md_files),
        "sampleDocs": sample_docs,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def ingest_extracted_export(export_id: str) -> Dict[str, Any]:
    paths = export_paths(export_id)
    md_files = sorted(paths["extract"].rglob("*.md"))
    documents: List[Dict[str, Any]] = []

    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        fm, body = _split_frontmatter(text)
        if not body.strip() and not fm:
            continue
        document_id = str(
            fm.get("id")
            or fm.get("document_id")
            or fm.get("source_id")
            or path.stem
        )
        tags = fm.get("tags") if isinstance(fm.get("tags"), list) else []
        details = normalize_document_details(
            {
                "id": document_id,
                "title": fm.get("title") or path.stem,
                "author": fm.get("author") or fm.get("creator"),
                "category": fm.get("category") or fm.get("type") or fm.get("source_type"),
                "tags": tags,
                "notes": fm.get("notes") or fm.get("summary"),
                "content": body,
                "location": fm.get("location"),
                "site_name": fm.get("site_name") or fm.get("site"),
                "source_url": fm.get("source_url") or fm.get("url") or fm.get("source"),
                "saved_at": fm.get("saved_at") or fm.get("saved_date"),
                "updated_at": fm.get("updated_at") or fm.get("last_updated"),
                "published_date": fm.get("published_date") or fm.get("published_at"),
                "word_count": fm.get("word_count"),
                "reading_time": fm.get("reading_time"),
            }
        )
        doc = details["document"]
        doc["raw"] = {"frontmatter": fm, "path": str(path.relative_to(paths["extract"]))}
        documents.append(doc)

    return {
        "exportId": export_id,
        "documents": documents,
        "count": len(documents),
        "manifestPath": str(paths["manifest"]),
    }

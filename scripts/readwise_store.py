#!/usr/bin/env python3
from __future__ import annotations

import array
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from readwise_semantic import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    SemanticEmbeddingProvider,
    build_embedding_provider,
    build_semantic_texts,
    normalize_semantic_text,
    text_hash,
)


LOW_SIGNAL_TOKENS = {
    "https", "http", "www", "com", "org", "net", "twitter", "xcom", "twimg", "github", "pbs", "profile",
    "images", "status", "posted", "browser", "support", "video", "tag", "retweeted", "comments",
}

STOP_TOKENS = {
    "about", "after", "also", "among", "and", "been", "being", "between", "could", "from", "have", "into",
    "just", "more", "most", "over", "some", "such", "than", "that", "their", "there", "these",
    "they", "this", "those", "through", "under", "using", "what", "when", "where", "which", "while",
    "with", "your", "readwise", "reader", "show", "docs", "doc", "tagged", "saved", "list", "tell",
}

BROAD_CATEGORY_PENALTIES = {
    "tweet": 10,
    "tweets": 10,
    "rss": 9,
    "feed": 9,
    "podcast": 3,
}

SOURCE_TYPE_CAPS = {
    "tweet": 1,
    "tweets": 1,
    "rss": 1,
    "feed": 1,
    "newsletter": 2,
    "podcast": 1,
}

VAGUE_TOPIC_TOKENS = {
    "strategy", "strategic", "founder", "founders", "mode", "management", "leadership",
    "product", "products", "company", "companies", "startup", "startups", "business",
    "teams", "team", "culture", "execution", "growth", "thinking", "decision", "decisions",
}

CONCEPT_ANCHOR_GROUPS = {
    "strategy": {"strategy", "strategic", "roadmap", "positioning", "portfolio", "planning"},
    "product": {"product", "products", "pm", "roadmap", "prototype", "customer", "feature"},
    "leadership": {"leadership", "manager", "managers", "management", "team", "teams", "culture"},
    "tenant": {"tenant", "tenancy", "multi-tenant", "multitenant", "isolation", "saas", "organization", "org"},
    "isolation": {"isolation", "boundary", "boundaries", "segmentation", "partition", "partitioning", "tenant", "multi-tenant", "multitenant", "sandbox", "separation"},
    "access": {"access", "authorization", "permissions", "policy", "policies", "entitlement"},
    "control": {"control", "controls", "access", "authorization", "permissions", "policy"},
    "level": {"level", "row", "record", "field", "table"},
    "agents": {"agent", "agents", "delegation", "tool", "tools", "autonomous", "automation"},
    "openclaw": {"openclaw", "claw", "gateway", "agent", "agents"},
}

CONCEPT_DRIFT_GROUPS = {
    "strategy": {"pdf", "extraction", "register", "registration", "save", "coupon", "hours", "left"},
    "product": {"register", "registration", "save", "coupon", "hours", "left", "promo", "webinar"},
    "leadership": {"coupon", "register", "promo", "hours", "left"},
    "tenant": {"covid", "cdc", "guidance", "days", "working", "people", "horizon"},
    "isolation": {"covid", "cdc", "guidance", "days", "working", "people", "horizon"},
    "access": {"email", "covid", "vaccine", "grandparents", "rescue"},
    "control": {"email", "covid", "vaccine", "grandparents", "rescue"},
    "openclaw": {"register", "registration", "workshop", "camp", "subscribers", "paid"},
}

WEAK_DOMAINS = {
    "x.com", "twitter.com", "mobile.twitter.com", "news.ycombinator.com", "linkedin.com", "www.linkedin.com",
    "youtube.com", "www.youtube.com", "m.youtube.com",
}

STRONG_DOMAIN_HINTS = {
    "substack.com", "medium.com", "stratechery.com", "martinfowler.com", "paulgraham.com", "amazon.com",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    override = os.getenv("READWISE_LOOKUP_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return repo_root() / "data" / "readwise"


def workspace_root() -> Path:
    return repo_root()


def default_db_path() -> Path:
    return data_root() / "readwise-cache.sqlite3"


class ReadwiseStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    @staticmethod
    def _coerce_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            flat = [ReadwiseStore._coerce_text(v) for v in value]
            flat = [v for v in flat if v not in (None, "")]
            return " | ".join(flat) if flat else None
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = re.findall(r"\d+", value)
            if digits:
                try:
                    return int(digits[0])
                except ValueError:
                    return None
        return None

    @staticmethod
    def _text_signal_score(text: Optional[str]) -> int:
        if not text:
            return 0
        text = text.strip()
        if not text:
            return 0
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
        if not words:
            return 0
        meaningful = [w for w in words if w not in LOW_SIGNAL_TOKENS]
        urls = len(re.findall(r"https?://|www\.", text.lower()))
        markdown_images = text.lower().count("![](")
        handles = text.count("@")
        bullets = len(re.findall(r"^\s*[-*•]\s+", text, flags=re.MULTILINE))
        numbered = len(re.findall(r"^\s*\d+[.)]\s+", text, flags=re.MULTILINE))
        score = len(meaningful) - (urls * 8) - (markdown_images * 6) - (handles * 2) - bullets - numbered
        return score

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = token.strip().lower()
        token = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", token)
        return token

    @classmethod
    def _query_terms(cls, query: str) -> List[str]:
        terms: List[str] = []
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_\-/+#.]{2,}", query.lower()):
            token = cls._normalize_token(raw)
            if not token or token in STOP_TOKENS or token in LOW_SIGNAL_TOKENS:
                continue
            if token.startswith("tag:") and len(token) > 4:
                token = token[4:]
            if token and token not in terms:
                terms.append(token)
        return terms

    @classmethod
    def _extract_tag_filters(cls, query: str) -> List[str]:
        tags: List[str] = []

        def add_tag(value: str) -> None:
            norm = cls._normalize_token(value)
            if norm and norm not in tags:
                tags.append(norm)

        lowered = query.lower()
        for raw in query.split():
            token = raw.strip()
            token_lower = token.lower()
            if token_lower.startswith("tag:") and len(token) > 4:
                add_tag(token[4:])
            elif token_lower.startswith("tags:") and len(token) > 5:
                for part in re.split(r"[,/|]", token[5:]):
                    add_tag(part)

        patterns = [
            r"\bdocs?\s+tagged\s+([a-z0-9_\-+/ ]{2,80})",
            r"\bdocuments?\s+tagged\s+([a-z0-9_\-+/ ]{2,80})",
            r"\bwhat\s+have\s+i\s+tagged\s+about\s+([a-z0-9_\-+/ ]{2,80})",
            r"\bwhat\s+have\s+i\s+saved\s+under\s+([a-z0-9_\-+/ ]{2,80})",
            r"\bwhat\s+in\s+my\s+([a-z0-9_\-+/ ]{2,80})\s+tags\b",
            r"\bin\s+my\s+([a-z0-9_\-+/ ]{2,80})\s+tags\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                phrase = (match.group(1) or "").strip(" ?!.,:;()[]{}")
                if not phrase:
                    continue
                parts = re.split(r"\s*(?:/|,|\band\b|\b&\b|\+)\s*", phrase)
                for part in parts:
                    cleaned = part.strip()
                    if cleaned:
                        add_tag(cleaned)
        return tags

    @classmethod
    def _token_counts(cls, text: Optional[str]) -> Counter[str]:
        counter: Counter[str] = Counter()
        if not text:
            return counter
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower()):
            token = cls._normalize_token(raw)
            if token and token not in STOP_TOKENS and token not in LOW_SIGNAL_TOKENS:
                counter[token] += 1
        return counter

    @classmethod
    def _text_overlap_score(cls, text: Optional[str], query_terms: List[str]) -> int:
        if not text or not query_terms:
            return 0
        counter = cls._token_counts(text)
        if not counter:
            return 0
        score = 0
        matched = 0
        for term in query_terms:
            if counter.get(term):
                matched += 1
                score += 6 + min(counter[term] - 1, 3)
        if matched:
            score += matched * matched
        if len(query_terms) >= 2 and matched == len(query_terms):
            score += 10
        elif len(query_terms) >= 3 and matched >= len(query_terms) - 1:
            score += 4
        return score

    @classmethod
    def _concept_anchor_score(cls, text: Optional[str], query_profile: Dict[str, Any]) -> int:
        if not text:
            return 0
        counter = cls._token_counts(text)
        if not counter:
            return 0
        score = 0
        groups = query_profile.get("conceptGroups") or {}
        for term, related in groups.items():
            if term in counter:
                score += 6
            overlap = len(set(related) & set(counter))
            if overlap:
                score += min(overlap * 3, 9)
        return score

    @classmethod
    def _concept_drift_score(cls, text: Optional[str], query_profile: Dict[str, Any]) -> int:
        if not text:
            return 0
        counter = cls._token_counts(text)
        if not counter:
            return 0
        score = 0
        groups = query_profile.get("conceptGroups") or {}
        for term in groups:
            drift_terms = CONCEPT_DRIFT_GROUPS.get(term) or set()
            overlap = len(set(drift_terms) & set(counter))
            if overlap:
                score += min(overlap * 4, 12)
        return score

    @classmethod
    def _concept_family_coverage(cls, text: Optional[str], query_profile: Dict[str, Any]) -> Dict[str, int]:
        counter = cls._token_counts(text)
        if not counter:
            return {}
        coverage: Dict[str, int] = {}
        groups = query_profile.get("conceptGroups") or {}
        token_set = set(counter)
        for term, related in groups.items():
            score = 0
            if term in token_set:
                score += 2
            score += len(set(related) & token_set)
            if score > 0:
                coverage[term] = score
        return coverage

    @classmethod
    def _technical_compound_bonus(cls, doc: Dict[str, Any], query_profile: Dict[str, Any]) -> int:
        mode = query_profile.get("mode")
        if mode != "specific_technical_compound":
            return 0
        text = " ".join(filter(None, [
            doc.get("title") or "",
            " ".join(doc.get("tags") or []),
            doc.get("summary") or "",
            doc.get("content") or "",
        ]))[:5000]
        token_set = set(cls._token_counts(text))
        if not token_set:
            return 0
        bonus = 0
        technical_terms = {"tenant", "multi-tenant", "multitenant", "authorization", "permissions", "policy", "policies", "security", "access", "control", "audit", "logging", "context", "identity", "boundary", "boundaries", "partition", "partitioning", "saas", "org", "organization"}
        technical_hits = len(token_set & technical_terms)
        bonus += min(technical_hits * 2, 12)
        if "tenant" in (query_profile.get("terms") or []) and technical_hits >= 2:
            bonus += 4
        if "isolation" in (query_profile.get("terms") or []) and ({"boundary", "boundaries", "partition", "partitioning", "separation", "sandbox"} & token_set):
            bonus += 4
        return bonus

    @classmethod
    def _phrase_score(cls, text: Optional[str], query: str) -> int:
        if not text or not query:
            return 0
        norm_text = re.sub(r"\s+", " ", text.lower())
        norm_query = re.sub(r"\s+", " ", query.lower()).strip()
        if len(norm_query) >= 4 and norm_query in norm_text:
            return 18
        return 0

    @classmethod
    def _looks_like_digest(cls, doc: Dict[str, Any]) -> bool:
        title = (doc.get("title") or "").lower()
        summary = (doc.get("summary") or "").lower()
        content = (doc.get("content") or "")[:5000].lower()
        haystack = "\n".join([title, summary, content])
        digest_patterns = [
            r"\b(link\s*dump|roundup|weekly\s+links|reading\s+list|bookmark\s+dump)\b",
            r"\b\d+\s+(links|articles|reads|things)\b",
            r"\btop\s+\d+\b",
        ]
        if any(re.search(pattern, haystack) for pattern in digest_patterns):
            return True
        bullet_lines = len(re.findall(r"^\s*[-*•]\s+", content, flags=re.MULTILINE))
        numbered_lines = len(re.findall(r"^\s*\d+[.)]\s+", content, flags=re.MULTILINE))
        return (bullet_lines + numbered_lines) >= 8

    @classmethod
    def _source_domain(cls, source_url: Optional[str]) -> str:
        if not source_url:
            return ""
        try:
            host = (urlparse(source_url).netloc or "").lower()
        except Exception:
            host = ""
        host = host.removeprefix("www.")
        return host

    @classmethod
    def _title_fingerprint(cls, title: Optional[str]) -> str:
        tokens = [
            cls._normalize_token(token)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", (title or "").lower())
            if cls._normalize_token(token) and cls._normalize_token(token) not in STOP_TOKENS and cls._normalize_token(token) not in LOW_SIGNAL_TOKENS
        ]
        return " ".join(tokens[:12])

    @classmethod
    def _doc_signature(cls, title: Optional[str], author: Optional[str], source_url: Optional[str]) -> str:
        base = "|".join([
            cls._title_fingerprint(title)[:120],
            cls._normalize_token(author or "")[:80],
            cls._source_domain(source_url)[:120],
        ])
        return base

    @classmethod
    def _source_quality_tier(cls, doc: Dict[str, Any]) -> str:
        category = (doc.get("category") or "").lower()
        domain = cls._source_domain(doc.get("sourceUrl"))
        has_tags = bool(doc.get("tags") or [])
        title = (doc.get("title") or "").lower()

        if category in {"book", "books", "pdf"}:
            return "strong"
        if category in {"article", "newsletter", "podcast", "video"}:
            if has_tags or (domain and domain not in WEAK_DOMAINS):
                return "strong"
            return "medium"
        if category in {"email"}:
            if has_tags or len(title) >= 24:
                return "medium"
            return "weak"
        if category in {"tweet", "tweets", "rss", "feed"}:
            return "medium" if has_tags else "weak"
        if has_tags and domain and domain not in WEAK_DOMAINS:
            return "strong"
        if has_tags:
            return "medium"
        return "medium" if domain and domain not in WEAK_DOMAINS else "weak"

    @classmethod
    def _source_quality_adjustment(cls, doc: Dict[str, Any], *, broad_query: bool = False) -> int:
        tier = cls._source_quality_tier(doc)
        category = (doc.get("category") or "").lower()
        adjustment = 0
        if tier == "strong":
            adjustment += 12 if broad_query else 8
        elif tier == "medium":
            adjustment += 3
        else:
            adjustment -= 10 if broad_query else 6
        if category in {"tweet", "tweets", "rss", "feed"}:
            adjustment -= 6 if broad_query else 3
        elif category == "email":
            adjustment -= 4 if broad_query else 2
        return adjustment

    @classmethod
    def _near_duplicate_score(cls, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        left_tokens = set(cls._title_fingerprint(left.get("title")).split())
        right_tokens = set(cls._title_fingerprint(right.get("title")).split())
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        same_domain = cls._source_domain(left.get("sourceUrl")) == cls._source_domain(right.get("sourceUrl"))
        same_author = cls._normalize_token(left.get("author") or "") == cls._normalize_token(right.get("author") or "")
        boost = 0.18 if same_domain else 0.0
        boost += 0.08 if same_author and left.get("author") and right.get("author") else 0.0
        return overlap + boost

    @classmethod
    def _query_profile(cls, query: str) -> Dict[str, Any]:
        terms = cls._query_terms(query)
        tag_filters = cls._extract_tag_filters(query)
        lowered = query.lower()
        vague_terms = [term for term in terms if term in VAGUE_TOPIC_TOKENS]
        specificity = 0
        if tag_filters:
            specificity += 3 * len(tag_filters)
        specificity += sum(2 if len(term) >= 8 else 1 for term in terms if term not in vague_terms)
        specificity -= len(vague_terms)
        specificity = max(specificity, 0)
        concept_groups: Dict[str, List[str]] = {}
        for term in terms:
            if term in CONCEPT_ANCHOR_GROUPS:
                concept_groups[term] = sorted(CONCEPT_ANCHOR_GROUPS[term])
        explicit_tag_intent = ("tag:" in lowered) or ("tags:" in lowered)
        implied_tag_intent = bool(
            tag_filters and (
                re.search(r"\btagged\b", lowered)
                or re.search(r"\bunder\b", lowered)
                or re.search(r"\bin\s+my\s+.+\s+tags\b", lowered)
                or re.search(r"\bwhat\s+have\s+i\s+tagged\s+about\b", lowered)
            )
        )
        tag_intent = "explicit" if explicit_tag_intent else ("implied" if implied_tag_intent else "none")
        tag_only_bias = bool(tag_filters and re.search(r"\b(show\s+me|what\s+have\s+i\s+saved\s+under|what\s+in\s+my|docs?\s+tagged|documents?\s+tagged)\b", lowered))
        tag_preference_strength = 0
        if tag_intent == "explicit" or tag_only_bias:
            tag_preference_strength = 2
        elif tag_intent == "implied" or tag_filters:
            tag_preference_strength = 1
        is_broad = len(terms) <= 2 and len(vague_terms) >= 1 and specificity <= 2 and tag_preference_strength == 0
        is_very_broad = len(terms) <= 2 and len(vague_terms) >= 1 and specificity == 0 and tag_preference_strength == 0
        mode = "known_topic_lookup"
        if tag_filters and tag_preference_strength > 0:
            mode = "tag_constrained_retrieval"
        elif len(terms) >= 2 and any(term in {"tenant", "isolation", "permissions", "policy", "access", "control", "security", "context", "audit", "logging", "row", "level"} for term in terms):
            mode = "specific_technical_compound"
        elif is_broad:
            mode = "broad_conceptual_synthesis"
        elif len(terms) >= 2 and concept_groups:
            mode = "specific_technical_compound"
        return {
            "terms": terms,
            "tagFilters": tag_filters,
            "tagRequestedTerms": list(tag_filters),
            "tagIntent": tag_intent,
            "tagPreferenceStrength": tag_preference_strength,
            "tagOnlyBias": tag_only_bias,
            "vagueTerms": vague_terms,
            "specificity": specificity,
            "isBroad": is_broad,
            "isVeryBroad": is_very_broad,
            "conceptGroups": concept_groups,
            "mode": mode,
        }

    @classmethod
    def _tag_match_score(cls, doc_tags: List[str], query_profile: Dict[str, Any]) -> Dict[str, Any]:
        requested = [cls._normalize_token(tag) for tag in (query_profile.get("tagRequestedTerms") or []) if cls._normalize_token(tag)]
        normalized_tags = [cls._normalize_token(tag) for tag in (doc_tags or []) if cls._normalize_token(tag)]
        requested_set = set(requested)
        doc_tag_set = set(normalized_tags)
        exact_requested = sum(1 for tag in requested if tag in doc_tag_set)
        overlap_ratio = (exact_requested / len(requested_set)) if requested_set else 0.0
        has_any_requested = exact_requested > 0
        score = 0
        if has_any_requested:
            score += exact_requested * 14
            if exact_requested >= 2:
                score += 8
            if requested_set and exact_requested == len(requested_set):
                score += 10
            if query_profile.get("tagIntent") == "implied":
                score += 8
            elif query_profile.get("tagIntent") == "explicit":
                score += 12
        elif requested_set and query_profile.get("tagPreferenceStrength", 0) >= 2:
            score -= 22
        elif requested_set:
            score -= 10
        return {
            "exactRequested": exact_requested,
            "requestedCount": len(requested_set),
            "overlapRatio": round(overlap_ratio, 3),
            "hasAnyRequested": has_any_requested,
            "strongMatch": bool(has_any_requested and (exact_requested >= 2 or overlap_ratio >= 0.99 or query_profile.get("tagPreferenceStrength", 0) >= 2)),
            "score": score,
        }

    def _document_quality_score(self, doc: Dict[str, Any], *, query: Optional[str] = None, query_terms: Optional[List[str]] = None) -> int:
        score = int(doc.get("cacheScore") or 0)
        title = doc.get("title") or ""
        summary = doc.get("summary") or ""
        notes = doc.get("notes") or ""
        source_url = doc.get("sourceUrl") or ""
        author = doc.get("author") or ""
        category = (doc.get("category") or "").lower()
        chunks = doc.get("contentChunks") or doc.get("chunks") or []
        chunk_text = "\n".join((c.get("text") or "") for c in chunks[:3] if isinstance(c, dict))
        tags = doc.get("tags") or []
        qterms = query_terms if query_terms is not None else self._query_terms(query or "")
        query_profile = self._query_profile(query or "")
        tag_filters = query_profile.get("tagFilters") or []
        tag_match = self._tag_match_score(tags, query_profile)

        score += min(self._text_signal_score(title), 20)
        score += min(self._text_signal_score(summary), 20)
        score += min(self._text_signal_score(chunk_text), 24)

        if source_url:
            score += 3
        if author and not author.startswith("http"):
            score += 2
        if summary:
            score += 3
        if tags:
            score += 20
            score += min(len(tags) * 3, 12)

        if category in {"article", "book", "books", "pdf", "newsletter", "podcast", "video"}:
            score += 5
        score += self._source_quality_adjustment(doc, broad_query=self._query_profile(query or "").get("isBroad", False))
        score -= BROAD_CATEGORY_PENALTIES.get(category, 0)
        if category in {"tweet", "rss", "feed"} and not tags:
            score -= 6

        domain = self._source_domain(source_url)
        if domain:
            score += 2
            if domain in WEAK_DOMAINS:
                score -= 8
            elif any(domain == hint or domain.endswith(f".{hint}") for hint in STRONG_DOMAIN_HINTS):
                score += 4
            elif "." in domain and domain not in WEAK_DOMAINS:
                score += 2

        tag_text = " ".join(tags)
        title_overlap = self._text_overlap_score(title, qterms)
        tag_overlap = self._text_overlap_score(tag_text, qterms)
        summary_overlap = self._text_overlap_score(summary, qterms)
        note_overlap = self._text_overlap_score(notes, qterms)
        chunk_overlap = self._text_overlap_score(chunk_text, qterms)
        requested_tag_terms = [tag for tag in (query_profile.get("tagRequestedTerms") or []) if tag]
        title_requested_hits = sum(1 for tag in requested_tag_terms if tag in title.lower())
        summary_requested_hits = sum(1 for tag in requested_tag_terms if tag in summary.lower())
        chunk_requested_hits = sum(1 for tag in requested_tag_terms if tag in chunk_text.lower())
        title_concept = self._concept_anchor_score(title, query_profile)
        tag_concept = self._concept_anchor_score(tag_text, query_profile)
        summary_concept = self._concept_anchor_score(summary, query_profile)
        chunk_concept = self._concept_anchor_score(chunk_text, query_profile)
        title_drift = self._concept_drift_score(title, query_profile)
        summary_drift = self._concept_drift_score(summary, query_profile)
        chunk_drift = self._concept_drift_score(chunk_text, query_profile)
        score += title_overlap * 5
        score += tag_overlap * 4
        score += summary_overlap * 2
        score += note_overlap
        score += chunk_overlap
        score += tag_match.get("score", 0)
        if requested_tag_terms and query_profile.get("tagPreferenceStrength", 0) > 0:
            score += title_requested_hits * 12
            score += summary_requested_hits * 5
            score += min(chunk_requested_hits * 3, 6)
        score += title_concept * 2
        score += tag_concept * 2
        score += summary_concept
        score += chunk_concept
        score += self._technical_compound_bonus(doc, query_profile)
        score -= title_drift * 2
        score -= summary_drift
        score -= chunk_drift
        score += self._phrase_score(title, query or "") * 2
        score += self._phrase_score(summary, query or "")
        if tag_filters:
            matched_tag_filters = tag_match.get("exactRequested", 0)
            score += matched_tag_filters * 18
            if matched_tag_filters < len(tag_filters):
                score -= (len(tag_filters) - matched_tag_filters) * (22 if query_profile.get("tagPreferenceStrength", 0) >= 2 else 14)

        topical_support = title_overlap + tag_overlap + summary_overlap + note_overlap
        if qterms and topical_support == 0 and chunk_overlap > 0:
            score -= 18
        if len(qterms) == 1 and topical_support == 0:
            score -= 22
        if len(qterms) >= 2 and topical_support < len(qterms):
            score -= (len(qterms) - min(topical_support, len(qterms))) * 6

        if len(qterms) >= 2:
            matched_fields = sum(
                1 for text in (title, tag_text, summary, chunk_text) if self._text_overlap_score(text, qterms) > 0
            )
            score += matched_fields * 3
            if query_profile.get("isBroad") and (title_concept + tag_concept + summary_concept + chunk_concept) == 0:
                score -= 16
        if self._looks_like_digest(doc):
            score -= 20
            if len(qterms) <= 2:
                score -= 10

        if len(title) < 12:
            score -= 3
        if author.startswith("http"):
            score -= 4
        if self._text_signal_score(chunk_text) < 0:
            score -= 12
        if self._text_signal_score(summary) < 0:
            score -= 8
        return score

    def _clean_chunks(self, chunks: List[Dict[str, Any]], *, limit: int, query_profile: Optional[Dict[str, Any]] = None, title: str = "", summary: str = "", tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        query_profile = query_profile or {}
        requested_tags = " ".join(query_profile.get("tagRequestedTerms") or [])
        title_terms = self._query_terms(title)
        summary_terms = self._query_terms(summary)
        anchor_terms = sorted(set(title_terms + summary_terms + list(query_profile.get("terms") or [])))
        ranked = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text") or ""
            signal = self._text_signal_score(text)
            if signal <= 0:
                continue
            score = signal
            if anchor_terms:
                score += self._text_overlap_score(text, anchor_terms)
            if requested_tags:
                score += self._text_overlap_score(text, self._query_terms(requested_tags)) * 2
            if tags:
                score += min(self._text_overlap_score(text, self._query_terms(" ".join(tags))), 8)
            if len(text.strip()) < 120:
                score -= 4
            ranked.append((score, chunk))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:limit]]

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                title TEXT,
                author TEXT,
                source TEXT,
                category TEXT,
                location TEXT,
                url TEXT,
                site_name TEXT,
                summary TEXT,
                source_url TEXT,
                saved_at TEXT,
                updated_at TEXT,
                published_date TEXT,
                word_count INTEGER,
                reading_time TEXT,
                first_opened_at TEXT,
                last_opened_at TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                notes TEXT,
                content TEXT,
                content_length INTEGER,
                content_chunks_json TEXT,
                raw_json TEXT,
                cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS highlights (
                highlight_id TEXT PRIMARY KEY,
                document_id TEXT,
                score REAL,
                url TEXT,
                document_title TEXT,
                document_author TEXT,
                document_category TEXT,
                document_tags_json TEXT NOT NULL DEFAULT '[]',
                text TEXT,
                note TEXT,
                highlight_tags_json TEXT NOT NULL DEFAULT '[]',
                location TEXT,
                highlighted_at TEXT,
                updated_at TEXT,
                color TEXT,
                raw_json TEXT,
                cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tags (
                tag_key TEXT PRIMARY KEY,
                name TEXT,
                raw_json TEXT,
                cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sync_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_semantic_index (
                document_id TEXT PRIMARY KEY,
                eligibility TEXT NOT NULL DEFAULT 'unknown',
                source_updated_at TEXT,
                content_updated_at TEXT,
                tag_count INTEGER NOT NULL DEFAULT 0,
                has_summary INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                selected_chunk_count INTEGER NOT NULL DEFAULT 0,
                text_version TEXT,
                basis_hash TEXT,
                title_text TEXT,
                summary_text TEXT,
                tag_text TEXT,
                doc_blend_text TEXT,
                selected_chunks_json TEXT NOT NULL DEFAULT '[]',
                last_prepared_at TEXT,
                last_embedded_at TEXT,
                embedding_status TEXT NOT NULL DEFAULT 'pending',
                embedding_provider TEXT,
                embedding_model TEXT,
                embedding_error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_embeddings (
                embedding_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                embedding_kind TEXT NOT NULL,
                chunk_index INTEGER,
                text_hash TEXT NOT NULL,
                text_preview TEXT,
                vector_provider TEXT,
                vector_model TEXT,
                vector_dim INTEGER,
                vector_blob BLOB,
                storage_ref TEXT,
                prepared_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                embedded_at TEXT,
                status TEXT NOT NULL DEFAULT 'prepared',
                error TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
            );

            CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);
            CREATE INDEX IF NOT EXISTS idx_documents_location ON documents(location);
            CREATE INDEX IF NOT EXISTS idx_highlights_document_id ON highlights(document_id);
            CREATE INDEX IF NOT EXISTS idx_highlights_document_title ON highlights(document_title);
            CREATE INDEX IF NOT EXISTS idx_semantic_embedding_status ON document_semantic_index(embedding_status);
            CREATE INDEX IF NOT EXISTS idx_semantic_eligibility ON document_semantic_index(eligibility);
            CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_id ON document_embeddings(document_id);
            CREATE INDEX IF NOT EXISTS idx_document_embeddings_status ON document_embeddings(status);
            """
        )
        self.conn.commit()

    def _row_to_document(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["tags"] = json.loads(data.pop("tags_json") or "[]")
        data["contentChunks"] = json.loads(data.pop("content_chunks_json") or "[]")
        data["raw"] = json.loads(data.pop("raw_json") or "{}") if data.get("raw_json") else None
        return {
            "documentId": data.get("document_id"),
            "title": data.get("title"),
            "author": data.get("author"),
            "source": data.get("source"),
            "category": data.get("category"),
            "location": data.get("location"),
            "url": data.get("url"),
            "siteName": data.get("site_name"),
            "summary": data.get("summary"),
            "sourceUrl": data.get("source_url"),
            "savedAt": data.get("saved_at"),
            "updatedAt": data.get("updated_at"),
            "publishedDate": data.get("published_date"),
            "wordCount": data.get("word_count"),
            "readingTime": data.get("reading_time"),
            "firstOpenedAt": data.get("first_opened_at"),
            "lastOpenedAt": data.get("last_opened_at"),
            "tags": data.get("tags", []),
            "notes": data.get("notes"),
            "content": data.get("content"),
            "contentLength": data.get("content_length"),
            "contentChunks": data.get("contentChunks", []),
            "cachedAt": data.get("cached_at"),
            "raw": data.get("raw"),
        }

    def _row_to_highlight(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["documentTags"] = json.loads(data.pop("document_tags_json") or "[]")
        data["highlightTags"] = json.loads(data.pop("highlight_tags_json") or "[]")
        data["raw"] = json.loads(data.pop("raw_json") or "{}") if data.get("raw_json") else None
        return {
            "highlightId": data.get("highlight_id"),
            "documentId": data.get("document_id"),
            "score": data.get("score"),
            "url": data.get("url"),
            "documentTitle": data.get("document_title"),
            "documentAuthor": data.get("document_author"),
            "documentCategory": data.get("document_category"),
            "documentTags": data.get("documentTags", []),
            "highlightText": data.get("text"),
            "highlightNote": data.get("note"),
            "highlightTags": data.get("highlightTags", []),
            "location": data.get("location"),
            "highlightedAt": data.get("highlighted_at"),
            "updatedAt": data.get("updated_at"),
            "color": data.get("color"),
            "cachedAt": data.get("cached_at"),
            "raw": data.get("raw"),
        }

    def upsert_document(self, doc: Dict[str, Any], *, raw_json: Optional[Dict[str, Any]] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO documents (
                document_id, title, author, source, category, location, url, site_name,
                summary, source_url, saved_at, updated_at, published_date, word_count,
                reading_time, first_opened_at, last_opened_at, tags_json, notes, content,
                content_length, content_chunks_json, raw_json, cached_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title,
                author=excluded.author,
                source=excluded.source,
                category=excluded.category,
                location=excluded.location,
                url=excluded.url,
                site_name=excluded.site_name,
                summary=excluded.summary,
                source_url=excluded.source_url,
                saved_at=excluded.saved_at,
                updated_at=excluded.updated_at,
                published_date=excluded.published_date,
                word_count=excluded.word_count,
                reading_time=excluded.reading_time,
                first_opened_at=excluded.first_opened_at,
                last_opened_at=excluded.last_opened_at,
                tags_json=excluded.tags_json,
                notes=excluded.notes,
                content=excluded.content,
                content_length=excluded.content_length,
                content_chunks_json=excluded.content_chunks_json,
                raw_json=excluded.raw_json,
                cached_at=CURRENT_TIMESTAMP
            """,
            (
                self._coerce_text(doc.get("documentId")),
                self._coerce_text(doc.get("title")),
                self._coerce_text(doc.get("author")),
                self._coerce_text(doc.get("source")),
                self._coerce_text(doc.get("category")),
                self._coerce_text(doc.get("location")),
                self._coerce_text(doc.get("url")),
                self._coerce_text(doc.get("siteName")),
                self._coerce_text(doc.get("summary")),
                self._coerce_text(doc.get("sourceUrl")),
                self._coerce_text(doc.get("savedAt")),
                self._coerce_text(doc.get("updatedAt")),
                self._coerce_text(doc.get("publishedDate")),
                self._coerce_int(doc.get("wordCount")),
                self._coerce_text(doc.get("readingTime")),
                self._coerce_text(doc.get("firstOpenedAt")),
                self._coerce_text(doc.get("lastOpenedAt")),
                json.dumps(doc.get("tags", []), ensure_ascii=False),
                self._coerce_text(doc.get("notes")),
                self._coerce_text(doc.get("content")),
                self._coerce_int(doc.get("contentLength")),
                json.dumps(doc.get("contentChunks", []), ensure_ascii=False),
                json.dumps(raw_json if raw_json is not None else doc, ensure_ascii=False),
            ),
        )

    def upsert_documents(self, docs: Iterable[Dict[str, Any]]) -> int:
        count = 0
        for doc in docs:
            self.upsert_document(doc)
            count += 1
        self.conn.commit()
        return count

    def upsert_highlight(self, item: Dict[str, Any], *, raw_json: Optional[Dict[str, Any]] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO highlights (
                highlight_id, document_id, score, url, document_title, document_author,
                document_category, document_tags_json, text, note, highlight_tags_json,
                location, highlighted_at, updated_at, color, raw_json, cached_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(highlight_id) DO UPDATE SET
                document_id=excluded.document_id,
                score=excluded.score,
                url=excluded.url,
                document_title=excluded.document_title,
                document_author=excluded.document_author,
                document_category=excluded.document_category,
                document_tags_json=excluded.document_tags_json,
                text=excluded.text,
                note=excluded.note,
                highlight_tags_json=excluded.highlight_tags_json,
                location=excluded.location,
                highlighted_at=excluded.highlighted_at,
                updated_at=excluded.updated_at,
                color=excluded.color,
                raw_json=excluded.raw_json,
                cached_at=CURRENT_TIMESTAMP
            """,
            (
                str(item.get("highlightId")),
                item.get("documentId"),
                item.get("score"),
                item.get("url"),
                item.get("documentTitle"),
                item.get("documentAuthor"),
                item.get("documentCategory"),
                json.dumps(item.get("documentTags", [])),
                item.get("highlightText") or item.get("text"),
                item.get("highlightNote") or item.get("note"),
                json.dumps(item.get("highlightTags", []) or item.get("tags", [])),
                item.get("location"),
                item.get("highlightedAt"),
                item.get("updatedAt"),
                item.get("color"),
                json.dumps(raw_json if raw_json is not None else item),
            ),
        )

    def upsert_highlights(self, items: Iterable[Dict[str, Any]]) -> int:
        count = 0
        for item in items:
            self.upsert_highlight(item)
            count += 1
        self.conn.commit()
        return count

    def replace_tags(self, tags: List[Dict[str, Any]]) -> int:
        self.conn.execute("DELETE FROM tags")
        for tag in tags:
            self.conn.execute(
                "INSERT INTO tags (tag_key, name, raw_json, cached_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (tag.get("key"), tag.get("name"), json.dumps(tag)),
            )
        self.conn.commit()
        return len(tags)

    def set_sync_state(self, key: str, value: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO sync_state (key, value_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=CURRENT_TIMESTAMP
            """,
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_sync_state(self, key: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT value_json FROM sync_state WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def get_recent_sync_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT key, value_json, updated_at
            FROM sync_state
            WHERE key LIKE 'export_status:%' OR key LIKE 'export_ingest:%' OR key IN ('last_export_trigger', 'last_delta_export_trigger')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        events: List[Dict[str, Any]] = []
        for key, value_json, updated_at in rows:
            try:
                value = json.loads(value_json)
            except Exception:
                value = {"raw": value_json}
            events.append({
                "key": key,
                "updatedAt": updated_at,
                "value": value,
            })
        return events

    def get_latest_export_anchor(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT key, value_json, updated_at
            FROM sync_state
            WHERE key LIKE 'export_status:%'
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[1])
        except Exception:
            value = None
        if not isinstance(value, dict):
            return None
        export_id = row[0].split(':', 1)[1] if ':' in row[0] else None
        last_updated = value.get('last_updated')
        if not last_updated:
            trigger = self.get_sync_state('last_export_trigger') or {}
            last_updated = trigger.get('last_updated')
        return {
            'exportId': export_id,
            'lastUpdated': last_updated,
            'status': value.get('status'),
            'documentsProcessed': value.get('documents_processed'),
            'documentsTotal': value.get('documents_total'),
            'downloadUrl': value.get('download_url'),
            'updatedAt': row[2],
        }

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value or not isinstance(value, str):
            return None
        try:
            if value.endswith('Z'):
                value = value[:-1] + '+00:00'
            return datetime.fromisoformat(value)
        except Exception:
            return None

    @staticmethod
    def _hours_since(value: Optional[str]) -> Optional[float]:
        dt = ReadwiseStore._parse_dt(value)
        if not dt:
            return None
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now - dt.astimezone(timezone.utc)).total_seconds() / 3600.0

    def sync_health(self) -> Dict[str, Any]:
        latest_anchor = self.get_latest_export_anchor()
        last_export_trigger = self.get_sync_state('last_export_trigger') or {}
        last_delta_trigger = self.get_sync_state('last_delta_export_trigger') or {}

        latest_ingest_row = self.conn.execute(
            """
            SELECT key, value_json, updated_at
            FROM sync_state
            WHERE key LIKE 'export_ingest:%'
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
        latest_ingest = None
        if latest_ingest_row:
            try:
                latest_ingest_value = json.loads(latest_ingest_row[1])
            except Exception:
                latest_ingest_value = {}
            latest_ingest = {
                'key': latest_ingest_row[0],
                'value': latest_ingest_value,
                'updatedAt': latest_ingest_row[2],
            }

        last_delta_export_id = last_delta_trigger.get('export_id')
        last_delta_status = None
        if last_delta_export_id:
            last_delta_status = self.get_sync_state(f'export_status:{last_delta_export_id}')

        anchor_age_hours = self._hours_since((latest_anchor or {}).get('lastUpdated'))
        latest_ingest_age_hours = self._hours_since((latest_ingest or {}).get('updatedAt'))

        notes: List[str] = []
        recommended_actions: List[str] = []
        failure_summary: Optional[str] = None
        has_anchor = bool(latest_anchor)
        has_ingest = bool(latest_ingest)
        last_delta_state = (last_delta_status or {}).get('status')
        delta_failed = bool(last_delta_state and last_delta_state not in {'completed'})

        if has_anchor and has_ingest and not delta_failed:
            if anchor_age_hours is not None and anchor_age_hours < 6 and latest_ingest_age_hours is not None and latest_ingest_age_hours < 6:
                freshness = 'fresh'
            elif anchor_age_hours is not None and anchor_age_hours < 24:
                freshness = 'acceptable'
            else:
                freshness = 'stale'
        elif has_anchor and not has_ingest:
            freshness = 'degraded'
            failure_summary = 'Anchor exists but the mirror has no corresponding ingest summary.'
            notes.append('Have export anchor but no recorded ingest summary.')
            recommended_actions.append('Run a delta refresh or re-ingest the latest export.')
        elif delta_failed:
            freshness = 'degraded'
            failure_summary = f'Latest delta export did not complete cleanly (status: {last_delta_state}).'
            notes.append(f"Latest delta export status is {last_delta_state}.")
            recommended_actions.append('Inspect the latest delta export and retry the refresh pipeline.')
        else:
            freshness = 'degraded'
            failure_summary = 'No export anchor exists yet, so the local mirror cannot be trusted for freshness.'
            notes.append('No export anchor found yet.')
            recommended_actions.append('Run a baseline export and ingest before relying on the local mirror.')

        latest_ingest_value = (latest_ingest or {}).get('value') if latest_ingest else None
        latest_ingest_key = (latest_ingest or {}).get('key') if latest_ingest else None
        if latest_ingest and isinstance(latest_ingest_value, dict) and latest_ingest_value.get('delta'):
            notes.append('Latest ingest was a delta refresh.')
        elif latest_ingest and latest_ingest_key and last_delta_export_id and latest_ingest_key == f'export_ingest:{last_delta_export_id}':
            notes.append('Latest ingest was a delta refresh.')
        elif latest_ingest:
            notes.append('Latest ingest was a full export baseline.')

        if freshness == 'stale':
            failure_summary = failure_summary or 'Mirror freshness has aged past the acceptable window.'
            recommended_actions.append('Run a delta refresh before answering freshness-sensitive questions.')
        elif freshness == 'acceptable':
            recommended_actions.append('Cache is usable for normal retrieval; prefer live fetch when recency matters.')
        elif freshness == 'fresh':
            recommended_actions.append('Cache is healthy for normal retrieval and synthesis.')

        operator_summary = {
            'status': freshness,
            'anchorAgeHours': round(anchor_age_hours, 2) if anchor_age_hours is not None else None,
            'latestIngestAgeHours': round(latest_ingest_age_hours, 2) if latest_ingest_age_hours is not None else None,
            'lastDeltaState': last_delta_state,
            'failureSummary': failure_summary,
            'recommendedActions': recommended_actions[:3],
        }

        return {
            'kind': 'syncHealth',
            'freshness': freshness,
            'documents': self.stats().get('documents', 0),
            'tags': self.stats().get('tags', 0),
            'highlights': self.stats().get('highlights', 0),
            'latestExportAnchor': latest_anchor,
            'lastExportTrigger': last_export_trigger,
            'lastDeltaExportTrigger': last_delta_trigger,
            'lastDeltaStatus': last_delta_status,
            'latestIngest': latest_ingest,
            'anchorAgeHours': anchor_age_hours,
            'latestIngestAgeHours': latest_ingest_age_hours,
            'operatorSummary': operator_summary,
            'failureSummary': failure_summary,
            'recentEvents': self.get_recent_sync_events(limit=6),
            'recommendedActions': recommended_actions[:3],
            'notes': notes,
        }

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def search_documents_cached(self, query: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        query_terms = self._query_terms(query)
        profile = self._query_profile(query)
        mode = profile.get("mode") or "known_topic_lookup"
        broad_multi_term = bool(mode == "broad_conceptual_synthesis" and len(query_terms) >= 2)
        specific_multi_term = bool(mode == "specific_technical_compound" and len(query_terms) >= 2)
        tag_filters = profile.get("tagFilters") or []
        tag_preference_strength = int(profile.get("tagPreferenceStrength") or 0)

        title_tag_clauses = []
        title_tag_params: List[Any] = []
        body_clauses = []
        body_params: List[Any] = []
        tag_clauses = []
        tag_params: List[Any] = []

        for term in ([query.lower()] + query_terms)[:6]:
            like = f"%{term}%"
            title_tag_clauses.append("lower(coalesce(title, '')) LIKE ?")
            title_tag_clauses.append("lower(coalesce(tags_json, '')) LIKE ?")
            title_tag_clauses.append("lower(coalesce(author, '')) LIKE ?")
            title_tag_params.extend([like, like, like])

            body_clauses.append("lower(coalesce(summary, '')) LIKE ?")
            body_clauses.append("lower(coalesce(content, '')) LIKE ?")
            body_params.extend([like, like])

        for tag in tag_filters[:6]:
            like = f'%"{tag}"%'
            tag_clauses.append("lower(coalesce(tags_json, '')) LIKE ?")
            tag_params.append(like)

        title_tag_where = " OR ".join(title_tag_clauses) if title_tag_clauses else "1=1"
        body_where = " OR ".join(body_clauses) if body_clauses else "1=1"
        tag_where = " OR ".join(tag_clauses) if tag_clauses else "0=1"
        candidate_limit = max(limit * 8, 40)

        tag_rows = []
        if tag_clauses:
            tag_rows = self.conn.execute(
                f"""
                SELECT *
                FROM documents
                WHERE {tag_where}
                ORDER BY updated_at DESC, saved_at DESC
                LIMIT ?
                """,
                (*tag_params, candidate_limit),
            ).fetchall()

        primary_rows = self.conn.execute(
            f"""
            SELECT *
            FROM documents
            WHERE {title_tag_where}
            ORDER BY updated_at DESC, saved_at DESC
            LIMIT ?
            """,
            (*title_tag_params, candidate_limit),
        ).fetchall()

        primary_ids = {row[0] for row in list(tag_rows) + list(primary_rows)}
        seen_ids = set()
        secondary_rows = []
        enough_tag_primary_candidates = len(primary_ids) >= max(limit * 4, 16)
        skip_body_pass = bool(tag_preference_strength >= 2 and enough_tag_primary_candidates)
        if not broad_multi_term and not skip_body_pass:
            secondary_rows = self.conn.execute(
                f"""
                SELECT *
                FROM documents
                WHERE {body_where}
                ORDER BY updated_at DESC, saved_at DESC
                LIMIT ?
                """,
                (*body_params, candidate_limit),
            ).fetchall()
        elif broad_multi_term and len(primary_rows) < max(limit * 3, 18):
            secondary_rows = self.conn.execute(
                f"""
                SELECT *
                FROM documents
                WHERE ({body_where}) AND ({title_tag_where})
                ORDER BY updated_at DESC, saved_at DESC
                LIMIT ?
                """,
                (*body_params, *title_tag_params, candidate_limit),
            ).fetchall()

        results = []
        for row in list(tag_rows) + list(primary_rows) + list(secondary_rows):
            document_id = row[0]
            if document_id in seen_ids:
                continue
            if row in secondary_rows and document_id in primary_ids:
                continue
            seen_ids.add(document_id)
            doc = self._row_to_document(row)
            tag_match = self._tag_match_score(doc.get("tags") or [], profile)
            doc["tagMatch"] = tag_match
            doc["cacheScore"] = self._document_quality_score(doc, query=query, query_terms=query_terms)
            if tag_preference_strength > 0:
                doc["cacheScore"] += tag_match.get("score", 0)
                if tag_match.get("exactRequested", 0) == 0 and tag_preference_strength >= 2:
                    doc["cacheScore"] -= 18
            doc["qualityScore"] = doc["cacheScore"]
            title_tag_text = " ".join(filter(None, [doc.get("title") or "", " ".join(doc.get("tags") or []), doc.get("author") or ""]))
            if tag_preference_strength >= 2 and tag_filters and tag_match.get("exactRequested", 0) == 0:
                gate_text = " ".join(filter(None, [title_tag_text, doc.get("summary") or "", (doc.get("content") or "")[:1200]]))
                if self._text_overlap_score(gate_text, query_terms) < max(1, len(query_terms) - 1):
                    continue
            if broad_multi_term:
                if self._text_overlap_score(title_tag_text, query_terms) == 0 and self._concept_anchor_score(title_tag_text, profile) == 0 and not (doc.get("tags") or []):
                    continue
            elif specific_multi_term:
                summary_text = doc.get("summary") or ""
                content_text = doc.get("content") or ""
                gate_text = " ".join(filter(None, [title_tag_text, summary_text, content_text[:4000]]))
                gate_overlap = self._text_overlap_score(gate_text, query_terms)
                gate_concept = self._concept_anchor_score(gate_text, profile)
                gate_family_coverage = self._concept_family_coverage(gate_text, profile)
                gate_drift = self._concept_drift_score(gate_text, profile)
                covered_families = sum(1 for value in gate_family_coverage.values() if value > 0)
                if gate_drift >= 8 and gate_overlap < len(query_terms):
                    continue
                if gate_overlap < len(query_terms):
                    if len(profile.get("conceptGroups") or {}) >= 2:
                        if covered_families < 2:
                            continue
                    elif gate_concept == 0:
                        continue
                if "tenant" in query_terms and "isolation" in query_terms:
                    tenant_cov = gate_family_coverage.get("tenant", 0)
                    isolation_cov = gate_family_coverage.get("isolation", 0)
                    technical_tokens = {"tenant", "multi-tenant", "multitenant", "boundary", "boundaries", "partition", "partitioning", "saas", "organization", "org", "security"}
                    tech_hits = len(set(self._token_counts(gate_text)) & technical_tokens)
                    if tenant_cov == 0 or isolation_cov == 0:
                        continue
                    if tech_hits < 2:
                        continue
            results.append(doc)
        results.sort(key=lambda d: (d.get("qualityScore", 0), d.get("updatedAt") or "", d.get("savedAt") or ""), reverse=True)
        return results[:limit]

    def search_highlights_cached(self, query: str, *, limit: int = 8) -> List[Dict[str, Any]]:
        query_terms = self._query_terms(query)
        clauses = []
        params: List[Any] = []
        for term in ([query.lower()] + query_terms)[:6]:
            like = f"%{term}%"
            clauses.append("lower(coalesce(text, '')) LIKE ?")
            clauses.append("lower(coalesce(note, '')) LIKE ?")
            clauses.append("lower(coalesce(document_title, '')) LIKE ?")
            clauses.append("lower(coalesce(document_tags_json, '')) LIKE ?")
            clauses.append("lower(coalesce(highlight_tags_json, '')) LIKE ?")
            params.extend([like, like, like, like, like])
        where_sql = " OR ".join(clauses) if clauses else "1=1"
        candidate_limit = max(limit * 10, 50)
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM highlights
            WHERE {where_sql}
            ORDER BY updated_at DESC, highlighted_at DESC
            LIMIT ?
            """,
            (*params, candidate_limit),
        ).fetchall()

        results = []
        for row in rows:
            hl = self._row_to_highlight(row)
            text = hl.get("highlightText") or ""
            note = hl.get("highlightNote") or ""
            tag_text = " ".join((hl.get("documentTags") or []) + (hl.get("highlightTags") or []))
            title = hl.get("documentTitle") or ""
            category = (hl.get("documentCategory") or "").lower()
            score = 0
            text_overlap = self._text_overlap_score(text, query_terms)
            note_overlap = self._text_overlap_score(note, query_terms)
            title_overlap = self._text_overlap_score(title, query_terms)
            tag_overlap = self._text_overlap_score(tag_text, query_terms)
            score += text_overlap * 3
            score += note_overlap * 2
            score += title_overlap * 2
            score += tag_overlap * 2
            score += self._phrase_score(text, query)
            score += self._phrase_score(title, query)
            score += max(min(self._text_signal_score(text), 18), -12)
            score -= BROAD_CATEGORY_PENALTIES.get(category, 0)
            if len(query_terms) == 1 and (title_overlap + tag_overlap + note_overlap) == 0:
                score -= 10
            if len(query_terms) >= 2 and (text_overlap + note_overlap + title_overlap + tag_overlap) < len(query_terms):
                score -= 6
            if self._looks_like_digest({"title": title, "summary": note, "content": text}):
                score -= 8
            hl["cacheScore"] = score
            results.append(hl)
        results.sort(key=lambda h: (h.get("cacheScore", 0), h.get("highlightedAt") or "", h.get("updatedAt") or ""), reverse=True)
        return results[:limit]

    def _has_contrast_signal(self, text: str) -> bool:
        text = (text or "").lower()
        if not text:
            return False
        markers = (" but ", " however", " although", " though", " yet ", " instead", " whereas", " while ", " unless", " despite", " tradeoff", " trade-off", " tension", " risk", " risks", " vs ", " versus ")
        return any(marker in f" {text} " for marker in markers)

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> Optional[float]:
        if not left or not right or len(left) != len(right):
            return None
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = sum(float(a) * float(a) for a in left) ** 0.5
        right_norm = sum(float(b) * float(b) for b in right) ** 0.5
        if left_norm <= 0 or right_norm <= 0:
            return None
        return dot / (left_norm * right_norm)

    def _best_effort_query_embedding(self, query: str) -> Tuple[Optional[List[float]], Dict[str, Any]]:
        provider = build_embedding_provider("openai", model=DEFAULT_OPENAI_EMBEDDING_MODEL)
        if not provider.is_configured():
            return None, {
                "active": False,
                "reason": "provider-unconfigured",
                "provider": provider.name,
                "model": provider.model,
            }
        prepared_query = normalize_semantic_text(query)
        if not prepared_query:
            return None, {
                "active": False,
                "reason": "empty-query",
                "provider": provider.name,
                "model": provider.model,
            }
        try:
            vectors = provider.embed([prepared_query])
        except Exception as exc:
            return None, {
                "active": False,
                "reason": "embedding-error",
                "provider": provider.name,
                "model": provider.model,
                "error": str(exc),
            }
        vector = vectors[0] if vectors else []
        if not vector:
            return None, {
                "active": False,
                "reason": "empty-vector",
                "provider": provider.name,
                "model": provider.model,
            }
        return vector, {
            "active": True,
            "provider": provider.name,
            "model": provider.model,
            "queryText": prepared_query,
            "queryVectorDim": len(vector),
        }

    def _semantic_scores_for_documents(self, document_ids: List[str], query_vector: Optional[List[float]]) -> Dict[str, Dict[str, Any]]:
        if not query_vector or not document_ids:
            return {}
        placeholders = ", ".join("?" for _ in document_ids)
        rows = self.conn.execute(
            f"""
            SELECT document_id, embedding_kind, chunk_index, vector_provider, vector_model, vector_dim, vector_blob
            FROM document_embeddings
            WHERE status = 'embedded' AND document_id IN ({placeholders})
            """,
            document_ids,
        ).fetchall()
        scores: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            vector = self._blob_to_vector(row["vector_blob"])
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity is None:
                continue
            doc_id = row["document_id"]
            current = scores.get(doc_id)
            candidate = {
                "score": similarity,
                "kind": row["embedding_kind"],
                "chunkIndex": row["chunk_index"],
                "provider": row["vector_provider"],
                "model": row["vector_model"],
                "vectorDim": row["vector_dim"],
            }
            if current is None or similarity > current["score"]:
                scores[doc_id] = candidate
        return scores

    def build_evidence_set(self, query: str, *, doc_limit: int = 4, highlight_limit: int = 8, chunk_limit: int = 2, strict_mode: bool = False, retrieval_mode: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        retrieval_mode = dict(retrieval_mode or {})
        tagged_only = bool(retrieval_mode.get("taggedOnly"))
        broad_mode = bool(retrieval_mode.get("broad"))
        counterpoint_mode = bool(retrieval_mode.get("counterpoint"))
        preserve_strict = bool(retrieval_mode.get("preserveStrict"))

        profile = self._query_profile(query)
        query_terms = profile["terms"]
        tag_filters = profile.get("tagFilters") or []
        effective_doc_limit = min(doc_limit, 2) if (strict_mode and profile["isBroad"] and not broad_mode) else doc_limit
        effective_highlight_limit = min(highlight_limit, 4) if (strict_mode and profile["isBroad"] and not broad_mode) else highlight_limit
        if broad_mode:
            effective_doc_limit = max(effective_doc_limit, min(doc_limit + 2, 8))
            effective_highlight_limit = max(effective_highlight_limit, min(highlight_limit + 2, 12))
        documents = self.search_documents_cached(query, limit=max(effective_doc_limit * (12 if broad_mode else 8), 48 if broad_mode else 32))
        highlights = self.search_highlights_cached(query, limit=max(effective_highlight_limit * (8 if broad_mode else 6), 48 if broad_mode else 36))
        query_vector, semantic_info = self._best_effort_query_embedding(query)
        semantic_scores = self._semantic_scores_for_documents(
            [doc.get("documentId") for doc in documents if doc.get("documentId")],
            query_vector,
        )
        semantic_hits = len(semantic_scores)
        semantic_weight = 0.12 if strict_mode else 0.22
        if broad_mode:
            semantic_weight += 0.04
        if tagged_only:
            semantic_weight += 0.02
        semantic_weight = min(max(semantic_weight, 0.0), 0.35)
        lexical_weight = 1.0 - semantic_weight
        if semantic_info.get("active") and semantic_hits:
            for doc in documents:
                doc_id = doc.get("documentId")
                semantic = semantic_scores.get(doc_id)
                base_quality = float(doc.get("qualityScore", doc.get("cacheScore", 0)) or 0)
                rerank_bonus = 0.0
                if semantic:
                    rerank_bonus = max(0.0, (float(semantic["score"]) - 0.15) * 100.0 * semantic_weight)
                hybrid_score = (base_quality * lexical_weight) + rerank_bonus
                doc["semanticScore"] = round(float(semantic["score"]), 4) if semantic else None
                doc["semanticMatch"] = semantic
                doc["hybridScore"] = round(hybrid_score, 3)
            documents.sort(
                key=lambda d: (
                    d.get("hybridScore", d.get("qualityScore", d.get("cacheScore", 0))),
                    d.get("qualityScore", d.get("cacheScore", 0)),
                    d.get("updatedAt") or "",
                    d.get("savedAt") or "",
                ),
                reverse=True,
            )
            semantic_info.update({
                "active": True,
                "rerankApplied": True,
                "candidateDocuments": len(documents),
                "matchedDocuments": semantic_hits,
                "weight": round(semantic_weight, 3),
                "lexicalWeight": round(lexical_weight, 3),
            })
        else:
            semantic_info.update({
                "active": bool(semantic_info.get("active") and semantic_hits),
                "rerankApplied": False,
                "candidateDocuments": len(documents),
                "matchedDocuments": semantic_hits,
                "weight": round(semantic_weight, 3),
                "lexicalWeight": round(lexical_weight, 3),
            })

        evidence_docs = []
        seen_signatures = set()
        category_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        rejection_notes: List[str] = []
        for doc in documents:
            quality_score = doc.get("qualityScore", self._document_quality_score(doc, query=query, query_terms=query_terms))
            hybrid_score = float(doc.get("hybridScore", quality_score) or quality_score)
            tags = doc.get("tags") or []
            tag_match = doc.get("tagMatch") or self._tag_match_score(tags, profile)
            title_summary_tags = " ".join(filter(None, [doc.get("title"), doc.get("summary"), " ".join(tags)]))
            direct_support = self._text_overlap_score(title_summary_tags, query_terms)
            title_support = self._text_overlap_score(doc.get("title") or "", query_terms)
            tag_support = self._text_overlap_score(" ".join(tags), query_terms)
            phrase_support = self._phrase_score(title_summary_tags, query)
            has_manual_tags = bool(tags)
            domain = self._source_domain(doc.get("sourceUrl"))
            source_tier = self._source_quality_tier(doc)
            title_concept = self._concept_anchor_score(doc.get("title") or "", profile)
            tag_concept = self._concept_anchor_score(" ".join(tags), profile)
            summary_concept = self._concept_anchor_score(doc.get("summary") or "", profile)
            family_coverage = self._concept_family_coverage(" ".join(filter(None, [doc.get("title") or "", " ".join(tags), doc.get("summary") or ""])), profile)
            title_drift = self._concept_drift_score(doc.get("title") or "", profile)
            summary_drift = self._concept_drift_score(doc.get("summary") or "", profile)
            contrast_signal = self._has_contrast_signal(" ".join(filter(None, [doc.get("title"), doc.get("summary"), " ".join((chunk.get("text") or "") for chunk in (doc.get("contentChunks") or doc.get("chunks") or [])[:2]) ])))

            min_quality = 24 if has_manual_tags else 38
            if strict_mode:
                min_quality += 10
            if profile["isBroad"]:
                min_quality += 4
            if profile["isBroad"] and strict_mode:
                min_quality += 8
            if broad_mode:
                min_quality -= 6
            if tagged_only and not has_manual_tags:
                rejection_notes.append(f"untagged:{doc.get('title') or '[untitled]'}")
                continue
            if profile.get("tagOnlyBias") and (tag_match.get("exactRequested", 0) == 0):
                literal_support = title_support + summary_concept + phrase_support
                if literal_support < max(2, len(query_terms)):
                    rejection_notes.append(f"tag-only-bias-no-tag-hit:{doc.get('title') or '[untitled]'}")
                    continue
            elif profile.get("tagPreferenceStrength", 0) > 0 and tag_filters and (tag_match.get("exactRequested", 0) == 0) and not has_manual_tags:
                rejection_notes.append(f"tag-intent-no-match:{doc.get('title') or '[untitled]'}")
                continue
            if counterpoint_mode and contrast_signal:
                quality_score += 8
            if profile["isBroad"] and (title_drift + summary_drift) >= 8 and not has_manual_tags:
                rejection_notes.append(f"broad-topic-drift:{doc.get('title') or '[untitled]'}")
                continue
            if profile["isBroad"] and len(query_terms) >= 2:
                covered_families = sum(1 for value in family_coverage.values() if value > 0)
                literal_anchor = title_support + tag_support + phrase_support
                concept_anchor = title_concept + tag_concept + summary_concept
                if len(profile.get("conceptGroups") or {}) >= 2 and not has_manual_tags:
                    if covered_families < 2 and literal_anchor < 2:
                        rejection_notes.append(f"broad-topic-family-miss:{doc.get('title') or '[untitled]'}")
                        continue
                if not has_manual_tags and source_tier != "strong":
                    if literal_anchor < 2 and concept_anchor < 6:
                        rejection_notes.append(f"broad-topic-preselection-gate:{doc.get('title') or '[untitled]'}")
                        continue
                if not has_manual_tags and literal_anchor == 0 and title_concept == 0 and tag_concept == 0:
                    rejection_notes.append(f"broad-topic-no-title-tag-anchor:{doc.get('title') or '[untitled]'}")
                    continue
                if "openclaw" in query_terms and not has_manual_tags and phrase_support == 0 and title_support == 0 and tag_support == 0:
                    rejection_notes.append(f"openclaw-no-anchor:{doc.get('title') or '[untitled]'}")
                    continue
            if source_tier == "weak" and profile["isBroad"] and not has_manual_tags and title_support == 0 and tag_support == 0:
                rejection_notes.append(f"weak-source-broad-query:{doc.get('title') or '[untitled]'}")
                continue
            if quality_score < min_quality:
                rejection_notes.append(f"low-quality:{doc.get('title') or '[untitled]'}")
                continue
            if len(query_terms) >= 2 and not has_manual_tags and direct_support < len(query_terms) and phrase_support == 0 and quality_score < 60:
                rejection_notes.append(f"weak-multi-term-support:{doc.get('title') or '[untitled]'}")
                continue
            if len(query_terms) >= 2 and profile["isBroad"] and not has_manual_tags and (title_support + tag_support + phrase_support + title_concept + tag_concept + summary_concept) == 0:
                rejection_notes.append(f"broad-multi-term-no-anchor:{doc.get('title') or '[untitled]'}")
                continue
            if len(query_terms) >= 2 and profile["isBroad"] and not has_manual_tags and direct_support == 0 and title_support == 0 and tag_support == 0:
                rejection_notes.append(f"broad-multi-term-no-literal-anchor:{doc.get('title') or '[untitled]'}")
                continue
            if len(query_terms) == 1 and not has_manual_tags and direct_support == 0 and phrase_support == 0 and quality_score < 55:
                rejection_notes.append(f"weak-single-term-support:{doc.get('title') or '[untitled]'}")
                continue
            if profile["isBroad"]:
                strong_tag_match = tag_support > 0 or any(term in self._token_counts(" ".join(tags)) for term in profile["vagueTerms"])
                strong_title_match = title_support > 0 or phrase_support > 0
                if strict_mode and not (strong_tag_match or strong_title_match):
                    rejection_notes.append(f"strict-broad-topic-miss:{doc.get('title') or '[untitled]'}")
                    continue
                if strict_mode and quality_score < 62 and not strong_tag_match:
                    rejection_notes.append(f"strict-broad-topic-low-confidence:{doc.get('title') or '[untitled]'}")
                    continue
                if not strict_mode and not has_manual_tags and not strong_title_match and quality_score < 64:
                    rejection_notes.append(f"broad-topic-weak-anchor:{doc.get('title') or '[untitled]'}")
                    continue
                if not strict_mode and not has_manual_tags and source_tier != "strong" and (title_support + tag_support + phrase_support + title_concept + tag_concept + summary_concept) < 2:
                    rejection_notes.append(f"broad-topic-low-anchor:{doc.get('title') or '[untitled]'}")
                    continue
            signature = self._doc_signature(doc.get("title"), doc.get("author"), doc.get("sourceUrl"))
            if signature and signature in seen_signatures:
                rejection_notes.append(f"duplicate-signature:{doc.get('title') or '[untitled]'}")
                continue
            if any(self._near_duplicate_score(doc, chosen) >= 0.72 for chosen in evidence_docs):
                rejection_notes.append(f"near-duplicate:{doc.get('title') or '[untitled]'}")
                continue
            category = (doc.get("category") or "unknown").lower()
            cap = SOURCE_TYPE_CAPS.get(category, max(1, effective_doc_limit // 2 or 1))
            if broad_mode:
                cap += 1
            if profile.get("tagPreferenceStrength", 0) > 0 and tag_match.get("hasAnyRequested"):
                cap += 1
            if profile.get("tagOnlyBias") and tag_match.get("strongMatch"):
                cap += 1
            if source_tier == "weak" and len(evidence_docs) < max(1, effective_doc_limit - 1):
                cap = min(cap, 1)
            if category_counts[category] >= cap:
                rejection_notes.append(f"category-cap:{doc.get('title') or '[untitled]'}")
                continue
            domain_cap = ((2 if broad_mode else 1) if effective_doc_limit <= 3 else (3 if broad_mode else 2))
            if profile.get("tagPreferenceStrength", 0) > 0 and tag_match.get("hasAnyRequested"):
                domain_cap += 1
            if source_tier == "weak":
                domain_cap = 1
            if domain and domain_counts[domain] >= domain_cap:
                rejection_notes.append(f"domain-cap:{doc.get('title') or '[untitled]'}")
                continue
            if source_tier == "weak" and any(self._source_quality_tier(chosen) == "strong" and chosen.get("matchStrength", {}).get("titleSupport", 0) >= title_support for chosen in evidence_docs):
                rejection_notes.append(f"weaker-source-displaced:{doc.get('title') or '[untitled]'}")
                continue
            chunks = self._clean_chunks(
                doc.get("contentChunks", []),
                limit=chunk_limit,
                query_profile=profile,
                title=doc.get("title") or "",
                summary=doc.get("summary") or "",
                tags=tags,
            )
            evidence_docs.append(
                {
                    "documentId": doc.get("documentId"),
                    "title": doc.get("title"),
                    "author": doc.get("author"),
                    "category": doc.get("category"),
                    "location": doc.get("location"),
                    "tags": doc.get("tags", []),
                    "summary": doc.get("summary"),
                    "sourceUrl": doc.get("sourceUrl"),
                    "sourceDomain": domain,
                    "cacheScore": doc.get("cacheScore"),
                    "qualityScore": quality_score,
                    "hybridScore": hybrid_score,
                    "semanticScore": doc.get("semanticScore"),
                    "semanticMatch": doc.get("semanticMatch"),
                    "matchStrength": {
                        "directSupport": direct_support,
                        "titleSupport": title_support,
                        "tagSupport": tag_support,
                        "phraseSupport": phrase_support,
                        "titleConcept": title_concept,
                        "tagConcept": tag_concept,
                        "summaryConcept": summary_concept,
                        "requestedTagHits": tag_match.get("exactRequested", 0),
                    },
                    "selectionSignals": {
                        "hasManualTags": has_manual_tags,
                        "contrastSignal": contrast_signal,
                        "sourceDiversityKept": True,
                        "sourceQualityTier": source_tier,
                        "conceptFamilyCoverage": family_coverage,
                        "titleDrift": title_drift,
                        "summaryDrift": summary_drift,
                        "tagMatch": tag_match,
                        "requestedTagTerms": profile.get("tagRequestedTerms") or [],
                        "titleRequestedHits": sum(1 for tag in (profile.get("tagRequestedTerms") or []) if tag and tag in (doc.get("title") or "").lower()),
                        "summaryRequestedHits": sum(1 for tag in (profile.get("tagRequestedTerms") or []) if tag and tag in (doc.get("summary") or "").lower()),
                    },
                    "chunks": chunks,
                }
            )
            if signature:
                seen_signatures.add(signature)
            category_counts[category] += 1
            if domain:
                domain_counts[domain] += 1
            if len(evidence_docs) >= effective_doc_limit:
                break

        if profile["isBroad"] and len(query_terms) >= 2:
            filtered_docs = []
            removed_titles: List[str] = []
            for chosen in evidence_docs:
                strength = chosen.get("matchStrength") or {}
                signals = chosen.get("selectionSignals") or {}
                concept_cov = signals.get("conceptFamilyCoverage") or {}
                covered_families = sum(1 for value in concept_cov.values() if value > 0)
                literal_anchor = (strength.get("titleSupport", 0) + strength.get("tagSupport", 0) + strength.get("phraseSupport", 0))
                concept_anchor = (strength.get("titleConcept", 0) + strength.get("tagConcept", 0) + strength.get("summaryConcept", 0))
                drift = int(signals.get("titleDrift") or 0) + int(signals.get("summaryDrift") or 0)
                keep = True
                if not signals.get("hasManualTags"):
                    if literal_anchor < 2 and concept_anchor < 6:
                        keep = False
                    if len(profile.get("conceptGroups") or {}) >= 2 and covered_families < 2:
                        keep = False
                    if drift >= 8:
                        keep = False
                if keep:
                    filtered_docs.append(chosen)
                else:
                    removed_titles.append(chosen.get("title") or "[untitled]")
            evidence_docs = filtered_docs
            rejection_notes.extend([f"post-filter-broad-topic:{title}" for title in removed_titles[:6]])

        evidence_highlights = []
        highlight_doc_counts: Counter[str] = Counter()
        seen_highlight_fingerprints = set()
        doc_ids_in_evidence = {doc.get("documentId") for doc in evidence_docs if doc.get("documentId")}
        for hl in highlights:
            text = (hl.get("highlightText") or "").strip()
            if self._text_signal_score(text) <= 0:
                continue
            doc_key = hl.get("documentId") or hl.get("documentTitle") or "unknown"
            if hl.get("documentId") and hl.get("documentId") not in doc_ids_in_evidence and len(doc_ids_in_evidence) >= effective_doc_limit:
                continue
            title_support = self._text_overlap_score(hl.get("documentTitle") or "", query_terms)
            tag_support = self._text_overlap_score(" ".join((hl.get("documentTags") or []) + (hl.get("highlightTags") or [])), query_terms)
            highlight_has_tags = bool((hl.get("documentTags") or []) or (hl.get("highlightTags") or []))
            contrast_signal = self._has_contrast_signal(" ".join(filter(None, [hl.get("highlightText"), hl.get("highlightNote") or "", hl.get("documentTitle") or ""])))
            if tagged_only and not highlight_has_tags:
                continue
            if counterpoint_mode and contrast_signal:
                hl["cacheScore"] = (hl.get("cacheScore") or 0) + 6
            if strict_mode and profile["isBroad"] and title_support == 0 and tag_support == 0 and (hl.get("cacheScore") or 0) < 20:
                continue
            fingerprint = self._normalize_token((hl.get("documentTitle") or "") + " " + text[:180])
            if fingerprint and fingerprint in seen_highlight_fingerprints:
                continue
            if highlight_doc_counts[doc_key] >= (1 if strict_mode and profile["isBroad"] else 2):
                continue
            evidence_highlights.append(
                {
                    "highlightId": hl.get("highlightId"),
                    "documentId": hl.get("documentId"),
                    "documentTitle": hl.get("documentTitle"),
                    "documentAuthor": hl.get("documentAuthor"),
                    "documentTags": hl.get("documentTags", []),
                    "highlightText": hl.get("highlightText"),
                    "highlightNote": hl.get("highlightNote"),
                    "highlightTags": hl.get("highlightTags", []),
                    "url": hl.get("url"),
                    "cacheScore": hl.get("cacheScore"),
                    "selectionSignals": {
                        "hasTags": highlight_has_tags,
                        "contrastSignal": contrast_signal,
                    },
                }
            )
            seen_highlight_fingerprints.add(fingerprint)
            highlight_doc_counts[doc_key] += 1
            if len(evidence_highlights) >= effective_highlight_limit:
                break

        if evidence_docs:
            top_scores = [int(doc.get("qualityScore") or 0) for doc in evidence_docs[:2]]
            top_score = top_scores[0]
            second_score = top_scores[1] if len(top_scores) > 1 else 0
            confidence = "high" if len(evidence_docs) >= min(2, effective_doc_limit) and top_score >= 70 else "medium"
            if top_score < 60 or (len(evidence_docs) == 1 and profile["isBroad"]):
                confidence = "low"
            elif second_score and top_score - second_score >= 18 and len(evidence_docs) <= 2:
                confidence = "medium"
            if profile["isBroad"]:
                non_tagged = [doc for doc in evidence_docs if not (doc.get("selectionSignals") or {}).get("hasManualTags")]
                weak_anchor_docs = [
                    doc for doc in evidence_docs
                    if ((doc.get("matchStrength") or {}).get("titleSupport", 0) + (doc.get("matchStrength") or {}).get("tagSupport", 0) + (doc.get("matchStrength") or {}).get("phraseSupport", 0)) < 2
                ]
                if len(evidence_docs) < min(3, effective_doc_limit):
                    confidence = "low"
                elif len(weak_anchor_docs) >= max(1, len(evidence_docs) // 2):
                    confidence = "medium" if confidence == "high" else confidence
                elif len(non_tagged) == len(evidence_docs):
                    confidence = "medium" if confidence == "high" else confidence
        else:
            confidence = "low"
        if strict_mode and profile["isBroad"] and len(evidence_docs) <= 1:
            confidence = "low"

        return {
            "kind": "evidenceSet",
            "query": query,
            "strictMode": strict_mode,
            "queryProfile": profile,
            "confidence": confidence,
            "documentCount": len(evidence_docs),
            "highlightCount": len(evidence_highlights),
            "documents": evidence_docs,
            "highlights": evidence_highlights,
            "selectionNotes": rejection_notes[:12],
            "retrievalMode": {
                "taggedOnly": tagged_only,
                "broad": broad_mode,
                "counterpoint": counterpoint_mode,
                "preserveStrict": preserve_strict,
            },
            "semantic": semantic_info,
        }

    def expand_query_candidates(self, query: str, *, limit: int = 6) -> Dict[str, Any]:
        evidence = self.build_evidence_set(query, doc_limit=max(limit, 4), highlight_limit=max(limit, 6), chunk_limit=1)
        docs = evidence.get("documents", [])
        highlights = evidence.get("highlights", [])

        tag_counts: Counter[str] = Counter()
        term_counts: Counter[str] = Counter()

        for doc in docs:
            for tag in doc.get("tags", []) or []:
                if tag:
                    tag_counts[tag] += 1
            corpus = " ".join(filter(None, [doc.get("title"), doc.get("summary")]))
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", corpus.lower()):
                if token not in {"readwise", "reader", "openclaw", "with", "from", "that", "this", "your", "have", "more", "only", "into", "using"}:
                    term_counts[token] += 1

        for hl in highlights:
            for tag in (hl.get("highlightTags", []) or []) + (hl.get("documentTags", []) or []):
                if tag:
                    tag_counts[tag] += 1
            corpus = " ".join(filter(None, [hl.get("documentTitle"), hl.get("highlightText")]))
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", corpus.lower()):
                if token not in {"readwise", "reader", "openclaw", "with", "from", "that", "this", "your", "have", "more", "only", "into", "using"}:
                    term_counts[token] += 1

        related_terms = [t for t, _ in term_counts.most_common(limit) if t not in query.lower()][:limit]
        tags = [t for t, _ in tag_counts.most_common(limit)][:limit]
        suggested_queries = [query]
        suggested_queries.extend([f"{query} tag:{tag}" for tag in tags[:3]])
        suggested_queries.extend([f"{query} {term}" for term in related_terms[:3]])

        deduped_queries = []
        seen = set()
        for item in suggested_queries:
            normalized = item.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped_queries.append(item)

        return {
            "kind": "expansionCandidates",
            "query": query,
            "relatedTerms": related_terms,
            "relatedTags": tags,
            "suggestedQueries": deduped_queries,
        }

    def prepare_semantic_records_for_documents(self, document_ids: List[str], *, chunk_limit: int = 4, eligibility: str = "manual") -> Dict[str, Any]:
        prepared = 0
        skipped = 0
        records_upserted = 0
        selected_chunks = 0
        processed_ids: List[str] = []

        for document_id in document_ids:
            doc = self.get_document(document_id)
            if not doc:
                skipped += 1
                continue
            semantic = build_semantic_texts(doc, chunk_limit=chunk_limit)
            chunks = semantic.get("selectedChunks", []) or []
            records = semantic.get("records", []) or []
            self.conn.execute(
                """
                INSERT INTO document_semantic_index (
                    document_id, eligibility, source_updated_at, content_updated_at, tag_count, has_summary,
                    chunk_count, selected_chunk_count, text_version, basis_hash, title_text, summary_text,
                    tag_text, doc_blend_text, selected_chunks_json, last_prepared_at, embedding_status,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'prepared', CURRENT_TIMESTAMP)
                ON CONFLICT(document_id) DO UPDATE SET
                    eligibility=excluded.eligibility,
                    source_updated_at=excluded.source_updated_at,
                    content_updated_at=excluded.content_updated_at,
                    tag_count=excluded.tag_count,
                    has_summary=excluded.has_summary,
                    chunk_count=excluded.chunk_count,
                    selected_chunk_count=excluded.selected_chunk_count,
                    text_version=excluded.text_version,
                    basis_hash=excluded.basis_hash,
                    title_text=excluded.title_text,
                    summary_text=excluded.summary_text,
                    tag_text=excluded.tag_text,
                    doc_blend_text=excluded.doc_blend_text,
                    selected_chunks_json=excluded.selected_chunks_json,
                    last_prepared_at=CURRENT_TIMESTAMP,
                    embedding_status='prepared',
                    embedding_error=NULL,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    document_id,
                    eligibility,
                    doc.get("updatedAt"),
                    doc.get("updatedAt") or doc.get("savedAt"),
                    len(doc.get("tags") or []),
                    1 if semantic.get("summary") else 0,
                    len(doc.get("contentChunks") or doc.get("chunks") or []),
                    len(chunks),
                    semantic.get("textVersion"),
                    semantic.get("basisHash"),
                    semantic.get("title"),
                    semantic.get("summary"),
                    semantic.get("tagText"),
                    semantic.get("docBlend"),
                    json.dumps(chunks, ensure_ascii=False),
                ),
            )
            self.conn.execute("DELETE FROM document_embeddings WHERE document_id = ?", (document_id,))
            for record in records:
                embedding_id = f"{document_id}:{record.get('embeddingKind')}:{record.get('chunkIndex') if record.get('chunkIndex') is not None else 'doc'}"
                self.conn.execute(
                    """
                    INSERT INTO document_embeddings (
                        embedding_id, document_id, embedding_kind, chunk_index, text_hash, text_preview,
                        prepared_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'prepared')
                    ON CONFLICT(embedding_id) DO UPDATE SET
                        text_hash=excluded.text_hash,
                        text_preview=excluded.text_preview,
                        prepared_at=CURRENT_TIMESTAMP,
                        embedded_at=NULL,
                        vector_provider=NULL,
                        vector_model=NULL,
                        vector_dim=NULL,
                        vector_blob=NULL,
                        storage_ref=NULL,
                        status='prepared',
                        error=NULL
                    """,
                    (
                        embedding_id,
                        document_id,
                        record.get("embeddingKind"),
                        record.get("chunkIndex"),
                        text_hash(record.get("text") or ""),
                        record.get("textPreview"),
                    ),
                )
                records_upserted += 1
            prepared += 1
            selected_chunks += len(chunks)
            processed_ids.append(document_id)

        self.conn.commit()
        return {
            "kind": "semanticPrepareDocuments",
            "preparedDocuments": prepared,
            "skippedDocuments": skipped,
            "recordsPrepared": records_upserted,
            "selectedChunks": selected_chunks,
            "documentIds": processed_ids,
        }

    def prepare_semantic_records_for_tagged_docs(self, *, limit: int = 50, chunk_limit: int = 4, location: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        where = ["tags_json IS NOT NULL", "tags_json != '[]'"]
        params: List[Any] = []
        if location:
            where.append("location = ?")
            params.append(location)
        if not force:
            where.append("document_id NOT IN (SELECT document_id FROM document_semantic_index WHERE embedding_status IN ('prepared', 'embedded'))")
        rows = self.conn.execute(
            f"""
            SELECT document_id
            FROM documents
            WHERE {' AND '.join(where)}
            ORDER BY updated_at DESC, saved_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        document_ids = [row[0] for row in rows]
        payload = self.prepare_semantic_records_for_documents(document_ids, chunk_limit=chunk_limit, eligibility="tagged")
        payload.update({
            "kind": "semanticPrepareTaggedDocs",
            "force": force,
            "limit": limit,
            "location": location,
        })
        return payload

    @staticmethod
    def _vector_to_blob(vector: List[float]) -> bytes:
        return array.array("f", [float(v) for v in vector]).tobytes()

    @staticmethod
    def _blob_to_vector(blob: Optional[bytes]) -> List[float]:
        if not blob:
            return []
        values = array.array("f")
        values.frombytes(blob)
        return list(values)

    @staticmethod
    def _selected_chunks_lookup(selected_chunks_json: Optional[str]) -> Dict[int, str]:
        try:
            items = json.loads(selected_chunks_json or "[]")
        except json.JSONDecodeError:
            return {}
        lookup: Dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            chunk_index = item.get("chunkIndex")
            text = item.get("text") or ""
            if isinstance(chunk_index, int) and text:
                lookup[chunk_index] = text
        return lookup

    def _semantic_embedding_text_from_row(self, row: sqlite3.Row) -> str:
        kind = row["embedding_kind"]
        if kind == "title":
            return row["title_text"] or ""
        if kind == "summary":
            return row["summary_text"] or ""
        if kind == "tags":
            return row["tag_text"] or ""
        if kind == "doc_blend":
            return row["doc_blend_text"] or ""
        if kind == "chunk":
            return self._selected_chunks_lookup(row["selected_chunks_json"]).get(row["chunk_index"], "")
        return ""

    def _refresh_semantic_document_statuses(self, document_ids: List[str]) -> None:
        seen = []
        for document_id in document_ids:
            if document_id and document_id not in seen:
                seen.append(document_id)
        for document_id in seen:
            counts = self.conn.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'embedded' THEN 1 ELSE 0 END) AS embedded_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
                    COUNT(*) AS total_count,
                    MAX(embedded_at) AS last_embedded_at,
                    MAX(vector_provider) AS provider,
                    MAX(vector_model) AS model,
                    MAX(error) AS error_text
                FROM document_embeddings
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
            total = counts["total_count"] or 0
            embedded = counts["embedded_count"] or 0
            errors = counts["error_count"] or 0
            if total and embedded == total:
                status = "embedded"
            elif errors:
                status = "error"
            elif total:
                status = "prepared"
            else:
                status = "pending"
            self.conn.execute(
                """
                UPDATE document_semantic_index
                SET embedding_status = ?,
                    last_embedded_at = ?,
                    embedding_provider = ?,
                    embedding_model = ?,
                    embedding_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (
                    status,
                    counts["last_embedded_at"] if status == "embedded" else None,
                    counts["provider"] if status == "embedded" else None,
                    counts["model"] if status == "embedded" else None,
                    counts["error_text"] if status == "error" else None,
                    document_id,
                ),
            )

    def embed_prepared_records(
        self,
        provider: SemanticEmbeddingProvider,
        *,
        document_ids: Optional[List[str]] = None,
        eligibility: Optional[str] = None,
        limit: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        if not provider.is_configured():
            raise ValueError(f"Embedding provider '{provider.name}' is not configured.")
        where = ["e.status = 'prepared'"]
        params: List[Any] = []
        if document_ids:
            placeholders = ", ".join("?" for _ in document_ids)
            where.append(f"e.document_id IN ({placeholders})")
            params.extend(document_ids)
        if eligibility:
            where.append("s.eligibility = ?")
            params.append(eligibility)
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT
                e.embedding_id,
                e.document_id,
                e.embedding_kind,
                e.chunk_index,
                e.text_hash,
                s.title_text,
                s.summary_text,
                s.tag_text,
                s.doc_blend_text,
                s.selected_chunks_json,
                s.eligibility
            FROM document_embeddings e
            JOIN document_semantic_index s ON s.document_id = e.document_id
            WHERE {' AND '.join(where)}
            ORDER BY e.document_id, e.embedding_kind, e.chunk_index
            {limit_sql}
            """,
            params,
        ).fetchall()
        if not rows:
            return {
                "kind": "semanticEmbedRecords",
                "provider": provider.name,
                "model": provider.model,
                "requestedDocuments": len(document_ids or []),
                "embeddedRecords": 0,
                "embeddedDocuments": 0,
                "skippedRecords": 0,
                "failedRecords": 0,
                "documentIds": [],
            }

        prepared_items = []
        skipped_records = 0
        failed_records = 0
        failed_documents: List[str] = []
        for row in rows:
            text = self._semantic_embedding_text_from_row(row)
            if not text:
                skipped_records += 1
                failed_records += 1
                failed_documents.append(row["document_id"])
                self.conn.execute(
                    "UPDATE document_embeddings SET status = 'error', error = ?, embedded_at = NULL WHERE embedding_id = ?",
                    ("Prepared semantic text missing for embedding row.", row["embedding_id"]),
                )
                continue
            if text_hash(text) != row["text_hash"]:
                skipped_records += 1
                failed_records += 1
                failed_documents.append(row["document_id"])
                self.conn.execute(
                    "UPDATE document_embeddings SET status = 'error', error = ?, embedded_at = NULL WHERE embedding_id = ?",
                    ("Prepared semantic text hash mismatch; re-run semantic preparation.", row["embedding_id"]),
                )
                continue
            prepared_items.append((row, text))

        embedded_records = 0
        embedded_documents: List[str] = []
        for start in range(0, len(prepared_items), max(1, batch_size)):
            batch = prepared_items[start:start + max(1, batch_size)]
            batch_rows = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]
            try:
                vectors = provider.embed(batch_texts)
            except Exception as exc:
                message = str(exc)
                failed_records += len(batch_rows)
                failed_documents.extend([row["document_id"] for row in batch_rows])
                for row in batch_rows:
                    self.conn.execute(
                        """
                        UPDATE document_embeddings
                        SET status = 'error', error = ?, embedded_at = NULL,
                            vector_provider = ?, vector_model = ?, vector_dim = NULL, vector_blob = NULL
                        WHERE embedding_id = ?
                        """,
                        (message, provider.name, provider.model, row["embedding_id"]),
                    )
                continue

            for row, vector in zip(batch_rows, vectors):
                embedded_records += 1
                embedded_documents.append(row["document_id"])
                self.conn.execute(
                    """
                    UPDATE document_embeddings
                    SET status = 'embedded', error = NULL, embedded_at = CURRENT_TIMESTAMP,
                        vector_provider = ?, vector_model = ?, vector_dim = ?, vector_blob = ?, storage_ref = NULL
                    WHERE embedding_id = ?
                    """,
                    (provider.name, provider.model, len(vector), self._vector_to_blob(vector), row["embedding_id"]),
                )

        self._refresh_semantic_document_statuses(
            [row["document_id"] for row in rows] + failed_documents + embedded_documents
        )
        self.conn.commit()
        embedded_doc_ids = []
        for doc_id in embedded_documents:
            if doc_id not in embedded_doc_ids:
                embedded_doc_ids.append(doc_id)
        return {
            "kind": "semanticEmbedRecords",
            "provider": provider.name,
            "model": provider.model,
            "requestedDocuments": len(document_ids or []),
            "embeddedRecords": embedded_records,
            "embeddedDocuments": len(embedded_doc_ids),
            "skippedRecords": skipped_records,
            "failedRecords": failed_records,
            "documentIds": embedded_doc_ids,
        }

    def embed_prepared_records_for_tagged_docs(
        self,
        provider: SemanticEmbeddingProvider,
        *,
        limit: int = 25,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT document_id
            FROM document_semantic_index
            WHERE eligibility = 'tagged' AND embedding_status IN ('prepared', 'error')
            ORDER BY COALESCE(last_prepared_at, updated_at) DESC, document_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        document_ids = [row[0] for row in rows]
        payload = self.embed_prepared_records(
            provider,
            document_ids=document_ids,
            batch_size=batch_size,
        )
        payload.update({
            "kind": "semanticEmbedRecords",
            "requestedDocuments": len(document_ids),
            "limit": limit,
        })
        return payload

    def list_semantic_documents(self, *, status: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        where = []
        params: List[Any] = []
        if status:
            where.append("embedding_status = ?")
            params.append(status)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self.conn.execute(
            f"""
            SELECT document_id, eligibility, embedding_status, embedding_provider, embedding_model,
                   selected_chunk_count, last_prepared_at, last_embedded_at, embedding_error
            FROM document_semantic_index
            {where_sql}
            ORDER BY COALESCE(last_embedded_at, last_prepared_at, updated_at) DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return {
            "kind": "semanticListDocuments",
            "count": len(rows),
            "results": [dict(row) for row in rows],
            "status": status,
        }

    def semantic_stats(self) -> Dict[str, Any]:
        stats = self.stats()
        semantic_docs = self.conn.execute("SELECT COUNT(*) FROM document_semantic_index").fetchone()[0]
        prepared_docs = self.conn.execute("SELECT COUNT(*) FROM document_semantic_index WHERE embedding_status = 'prepared'").fetchone()[0]
        embedded_docs = self.conn.execute("SELECT COUNT(*) FROM document_semantic_index WHERE embedding_status = 'embedded'").fetchone()[0]
        error_docs = self.conn.execute("SELECT COUNT(*) FROM document_semantic_index WHERE embedding_status = 'error'").fetchone()[0]
        tagged_docs = self.conn.execute("SELECT COUNT(*) FROM documents WHERE tags_json IS NOT NULL AND tags_json != '[]'").fetchone()[0]
        total_records = self.conn.execute("SELECT COUNT(*) FROM document_embeddings").fetchone()[0]
        embedded_records = self.conn.execute("SELECT COUNT(*) FROM document_embeddings WHERE status = 'embedded'").fetchone()[0]
        error_records = self.conn.execute("SELECT COUNT(*) FROM document_embeddings WHERE status = 'error'").fetchone()[0]
        by_kind_rows = self.conn.execute(
            "SELECT embedding_kind, COUNT(*) AS count FROM document_embeddings GROUP BY embedding_kind ORDER BY embedding_kind"
        ).fetchall()
        by_status_rows = self.conn.execute(
            "SELECT status, COUNT(*) AS count FROM document_embeddings GROUP BY status ORDER BY status"
        ).fetchall()
        model_rows = self.conn.execute(
            """
            SELECT COALESCE(vector_provider, 'unknown') AS provider, COALESCE(vector_model, 'unknown') AS model, COUNT(*) AS count
            FROM document_embeddings
            WHERE status = 'embedded'
            GROUP BY COALESCE(vector_provider, 'unknown'), COALESCE(vector_model, 'unknown')
            ORDER BY count DESC, provider, model
            """
        ).fetchall()
        stats.update(
            {
                "semanticDocuments": semantic_docs,
                "semanticPreparedDocuments": prepared_docs,
                "semanticEmbeddedDocuments": embedded_docs,
                "semanticErrorDocuments": error_docs,
                "taggedDocuments": tagged_docs,
                "semanticCoverage": {
                    "taggedPreparedPct": round((semantic_docs / tagged_docs) * 100, 1) if tagged_docs else 0.0,
                    "taggedEmbeddedPct": round((embedded_docs / tagged_docs) * 100, 1) if tagged_docs else 0.0,
                },
                "preparedEmbeddingRecords": total_records,
                "totalEmbeddingRecords": total_records,
                "embeddedEmbeddingRecords": embedded_records,
                "errorEmbeddingRecords": error_records,
                "embeddingKinds": {row[0]: row[1] for row in by_kind_rows},
                "embeddingStatuses": {row[0]: row[1] for row in by_status_rows},
                "embeddedModels": [
                    {"provider": row[0], "model": row[1], "count": row[2]}
                    for row in model_rows
                ],
            }
        )
        return stats

    def stats(self) -> Dict[str, Any]:
        counts = {}
        for table in ("documents", "highlights", "tags", "sync_state", "document_semantic_index", "document_embeddings"):
            counts[table] = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {
            "dbPath": str(self.db_path),
            **counts,
        }

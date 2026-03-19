#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "your", "about", "into", "have", "what",
    "when", "where", "which", "they", "them", "their", "there", "here", "than", "then", "just",
    "also", "over", "under", "using", "used", "use", "onto", "been", "being", "will", "would",
    "could", "should", "after", "before", "because", "while", "through", "across", "much", "many",
    "more", "most", "some", "only", "like", "very", "does", "doesn", "dont", "cant", "can",
    "one", "two", "three", "four", "five", "first", "second", "third", "new", "day", "around",
    "every", "daily", "good", "cool", "stuff", "built", "build", "readwise", "reader", "claw",
    "document", "documents", "highlight", "highlights", "article", "articles", "tweet", "tweets",
    "note", "notes", "item", "items", "https", "http", "www", "com", "org", "net", "twitter",
    "github", "twimg", "profile", "images", "browser", "support", "video", "posted", "status",
    "retweeted", "comments", "tool", "tools", "agent", "agents", "internal", "company", "development",
    "need", "work", "tell", "call", "data", "context", "system", "spec", "using", "used", "prompt",
}

CONTRAST_MARKERS = {
    "but", "however", "although", "though", "yet", "instead", "whereas", "while", "except", "unless",
    "tradeoff", "trade-off", "tension", "vs", "versus", "despite", "problem", "risk", "risks",
}

COUNTERPOINT_PATTERNS: List[Tuple[str, str]] = [
    ("tradeoff", r"\b(trade[- ]?offs?|balance|tension)\b"),
    ("objection", r"\b(but|however|although|though|yet|whereas|instead)\b"),
    ("risk", r"\b(risk|risks|danger|failure|fragile|break|cost)\b"),
    ("limit", r"\b(limit|limits|constraint|constraints|cap|bottleneck|downside)\b"),
]



def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text or "")]



def _normalize_theme_token(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token



def _meaningful_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _tokenize(text):
        token = _normalize_theme_token(token)
        if token in STOPWORDS or len(token) < 4:
            continue
        tokens.append(token)
    return tokens



def _doc_corpus(doc: Dict[str, Any]) -> str:
    return " ".join([
        doc.get("title") or "",
        doc.get("summary") or "",
        " ".join(doc.get("tags", []) or []),
        " ".join((chunk.get("text") or "") for chunk in (doc.get("chunks") or [])[:2]),
    ]).strip()



def _highlight_corpus(hl: Dict[str, Any]) -> str:
    return " ".join([
        hl.get("documentTitle") or "",
        hl.get("highlightText") or "",
        hl.get("highlightNote") or "",
        " ".join(hl.get("highlightTags", []) or []),
        " ".join(hl.get("documentTags", []) or []),
    ]).strip()



def _weighted_terms(documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]], *, limit: int = 12) -> List[str]:
    counts: Counter[str] = Counter()
    for doc in documents:
        for token in _meaningful_tokens(doc.get("title") or ""):
            counts[token] += 9
        for token in _meaningful_tokens(doc.get("summary") or ""):
            counts[token] += 4
        for tag in doc.get("tags", []) or []:
            for token in _meaningful_tokens(tag):
                counts[token] += 8
        for chunk in doc.get("chunks", [])[:2]:
            for token in _meaningful_tokens(chunk.get("text") or ""):
                counts[token] += 2
    for hl in highlights:
        for token in _meaningful_tokens(hl.get("highlightText") or ""):
            counts[token] += 2
        for token in _meaningful_tokens(hl.get("documentTitle") or ""):
            counts[token] += 2
        for tag in (hl.get("highlightTags") or []) + (hl.get("documentTags") or []):
            for token in _meaningful_tokens(tag):
                counts[token] += 4
    return [term for term, _ in counts.most_common(limit)]



def _coverage_label(document_count: int, highlight_count: int, confidence: str) -> str:
    total = document_count + highlight_count
    if confidence == "low":
        return "thin"
    if total <= 3:
        return "narrow"
    if total <= 7:
        return "moderate"
    return "broad"



def _display_label(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9+&'\- ]+", " ", (text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = [w for w in re.split(r"\s+", cleaned) if w]
    return " ".join(word[:1].upper() + word[1:] for word in words[:6])



def _looks_generic_theme_label(label: str) -> bool:
    lowered = (label or "").strip().lower()
    if not lowered:
        return True
    generic = {
        "task", "tasks", "message", "messages", "check", "default", "hour", "model",
        "install", "assistant", "email", "system", "prompt", "prompts", "tool", "tools",
    }
    tokens = set(_meaningful_tokens(lowered))
    return bool(tokens) and tokens.issubset(generic)



def _extract_phrases(text: str, *, sizes: Iterable[int] = (4, 3, 2)) -> List[str]:
    phrases: List[str] = []
    tokens = _meaningful_tokens(text)
    for size in sizes:
        for idx in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[idx: idx + size])
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases



def _theme_label_candidates(documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]]) -> List[Tuple[str, set[str], int, str]]:
    candidates: Counter[Tuple[str, Tuple[str, ...], str]] = Counter()

    def add(label: str, weight: int, source: str) -> None:
        cleaned = re.sub(r"\s+", " ", (label or "").strip(" -–—:|/"))
        tokens = tuple(sorted(set(_meaningful_tokens(cleaned))))
        if not cleaned or not tokens:
            return
        candidates[(cleaned, tokens, source)] += weight

    for doc in documents:
        for tag in doc.get("tags", []) or []:
            add(tag, 12, "tag")
        title = doc.get("title") or ""
        add(title, 5, "title")
        for phrase in _extract_phrases(title, sizes=(4, 3, 2))[:8]:
            add(phrase, 3, "title-phrase")

    for hl in highlights:
        for tag in (hl.get("highlightTags") or []) + (hl.get("documentTags") or []):
            add(tag, 8, "tag")
        title = hl.get("documentTitle") or ""
        for phrase in _extract_phrases(title, sizes=(4, 3, 2))[:5]:
            add(phrase, 2, "title-phrase")

    ranked = [
        (label, set(tokens), score, source)
        for (label, tokens, source), score in candidates.items()
    ]
    ranked.sort(key=lambda item: (item[2], len(item[1]), item[3] == "tag"), reverse=True)
    return ranked



def _best_theme_label(tokens: set[str], documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]]) -> Tuple[str, str]:
    preferred: Counter[Tuple[str, str]] = Counter()
    token_set = {t.lower() for t in tokens}

    for label, label_tokens, base_score, source in _theme_label_candidates(documents, highlights):
        overlap = len(token_set & label_tokens)
        if not overlap:
            continue
        score = 0
        if token_set.issubset(label_tokens):
            score += base_score + 12 + len(label_tokens)
        elif overlap >= max(1, min(len(token_set), 2)):
            score += base_score + 4 + overlap
        if source == "tag":
            score += 10
        elif source == "title":
            score += 6
        elif source == "title-phrase":
            score += 5
        if len(label_tokens) >= 2:
            score += 3
        if _looks_generic_theme_label(label):
            score -= 12
        preferred[(label, source)] += score

    if preferred:
        ranked = sorted(preferred.items(), key=lambda item: item[1], reverse=True)
        for (best_label, best_source), _score in ranked:
            if not _looks_generic_theme_label(best_label):
                return _display_label(best_label), best_source
        best_label, best_source = ranked[0][0]
        return _display_label(best_label), best_source

    ordered = sorted(tokens)
    if len(ordered) == 1:
        return _display_label(ordered[0]), "term"
    return _display_label(" ".join(ordered[:3])), "term"



def _candidate_theme_labels(documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]], top_terms: List[str]) -> List[Tuple[str, set[str]]]:
    corpora = [_doc_corpus(doc).lower() for doc in documents] + [_highlight_corpus(hl).lower() for hl in highlights]
    labels: List[Tuple[str, set[str]]] = []
    seen: set[Tuple[str, ...]] = set()

    for label, tokens, score, source in _theme_label_candidates(documents, highlights):
        token_key = tuple(sorted(tokens))
        if len(tokens) > 4 or token_key in seen:
            continue
        if _looks_generic_theme_label(label) and source not in {"tag", "title", "title-phrase"}:
            continue
        support = sum(1 for corpus in corpora if all(token in corpus for token in tokens))
        if source == "tag" and support >= 1:
            labels.append((label, tokens))
            seen.add(token_key)
        elif support >= 2:
            labels.append((label, tokens))
            seen.add(token_key)

    for term in top_terms[:8]:
        token_key = (term,)
        if token_key not in seen:
            labels.append((term, {term}))
            seen.add(token_key)
    for i, left in enumerate(top_terms[:6]):
        for right in top_terms[i + 1:8]:
            token_key = tuple(sorted((left, right)))
            if token_key in seen:
                continue
            support = 0
            for corpus in corpora:
                if left in corpus and right in corpus:
                    support += 1
            if support >= 2:
                labels.append((f"{left} + {right}", {left, right}))
                seen.add(token_key)
    labels.sort(key=lambda item: (len(item[1]), sum(1 for corpus in corpora if all(t in corpus for t in item[1]))), reverse=True)
    return labels



def _build_theme_clusters(documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]], top_terms: List[str], query: str = "", query_terms: List[str] | None = None) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    used_terms: set[str] = set()
    used_labels: set[str] = set()
    query_tokens = set((query_terms or []) or _meaningful_tokens(query or ""))
    if query_tokens:
        doc_matches = [doc for doc in documents if all(token in _doc_corpus(doc).lower() for token in query_tokens if token)]
        hl_matches = [hl for hl in highlights if all(token in _highlight_corpus(hl).lower() for token in query_tokens if token)]
        if doc_matches or hl_matches:
            query_label = _display_label(query)
            if query_label:
                clusters.append(
                    {
                        "theme": query_label,
                        "rawTheme": query,
                        "terms": sorted(query_tokens),
                        "documentCount": len(doc_matches),
                        "highlightCount": len(hl_matches),
                        "documents": [d.get("title") for d in doc_matches[:3] if d.get("title")],
                        "examples": [
                            (hl.get("highlightText") or "").strip()[:180]
                            for hl in hl_matches[:2]
                            if (hl.get("highlightText") or "").strip()
                        ],
                        "labelSource": "query",
                    }
                )
                used_terms |= query_tokens
                used_labels.add(query_label.lower().strip())
    for raw_label, tokens in _candidate_theme_labels(documents, highlights, top_terms):
        if tokens & used_terms and len(tokens) == 1:
            continue
        doc_matches = [doc for doc in documents if all(token in _doc_corpus(doc).lower() for token in tokens)]
        hl_matches = [hl for hl in highlights if all(token in _highlight_corpus(hl).lower() for token in tokens)]
        if len(doc_matches) + len(hl_matches) < 2:
            continue
        label, label_source = _best_theme_label(tokens, doc_matches, hl_matches)
        if label_source == "term" and len(tokens) > 1 and len(doc_matches) < 2 and len(hl_matches) < 2:
            continue
        if label_source in {"title", "title-phrase"}:
            continue
        normalized_label = label.lower().strip()
        if normalized_label in used_labels:
            continue
        clusters.append(
            {
                "theme": label,
                "rawTheme": raw_label,
                "terms": sorted(tokens),
                "documentCount": len(doc_matches),
                "highlightCount": len(hl_matches),
                "documents": [d.get("title") for d in doc_matches[:3] if d.get("title")],
                "examples": [
                    (hl.get("highlightText") or "").strip()[:180]
                    for hl in hl_matches[:2]
                    if (hl.get("highlightText") or "").strip()
                ],
                "labelSource": label_source,
            }
        )
        used_terms |= tokens
        used_labels.add(normalized_label)
        if len(clusters) >= 5:
            break
    return clusters



def _document_contributions(documents: List[Dict[str, Any]], themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for doc in documents[:6]:
        corpus = _doc_corpus(doc).lower()
        matched = [theme["theme"] for theme in themes if all(term in corpus for term in theme.get("terms", []))]
        strength = doc.get("matchStrength") or {}
        reasons = []
        if strength.get("tagSupport"):
            reasons.append("manual tags align")
        if strength.get("titleSupport"):
            reasons.append("title matches")
        if strength.get("phraseSupport"):
            reasons.append("exact phrase hit")
        if not reasons and strength.get("directSupport"):
            reasons.append("summary/body overlap")
        groups.append(
            {
                "documentId": doc.get("documentId"),
                "title": doc.get("title"),
                "author": doc.get("author"),
                "matchedThemes": matched[:4],
                "cacheScore": doc.get("cacheScore"),
                "qualityScore": doc.get("qualityScore"),
                "whyIncluded": reasons[:3],
                "whySelected": "; ".join(reasons[:3]) or "strong lexical overlap",
                "sourceDomain": doc.get("sourceDomain"),
                "selectionSignals": doc.get("selectionSignals") or {},
            }
        )
    return groups



def _counterpoint_kind(text: str) -> str:
    lowered = (text or "").lower()
    for label, pattern in COUNTERPOINT_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return "contrast"



def _counterpoint_label(kind: str, title: str, text: str, anchor_theme: str | None = None) -> str:
    if kind == "tradeoff":
        base = "Tradeoff worth carrying forward"
    elif kind == "risk":
        base = "Risk or failure mode"
    elif kind == "limit":
        base = "Boundary or limit case"
    elif kind == "objection":
        base = "Important objection"
    else:
        base = f"Counterpoint from {title or '[untitled]'}"
    return f"{base} — {anchor_theme}" if anchor_theme else base



def _counterpoint_explanation(kind: str, source: Dict[str, Any], sentence: str, anchor_theme: str | None = None) -> str:
    strength = source.get("matchStrength") or {}
    anchors = []
    if anchor_theme:
        anchors.append(f"it pushes against the '{anchor_theme}' cluster")
    if strength.get("tagSupport"):
        anchors.append("it is tag-aligned")
    if strength.get("titleSupport"):
        anchors.append("the title is on-topic")
    if strength.get("phraseSupport"):
        anchors.append("it contains a direct phrase hit")

    if kind == "tradeoff":
        reason = "it adds an explicit tradeoff instead of just more agreement"
    elif kind == "risk":
        reason = "it names a risk or failure mode the synthesis should not flatten away"
    elif kind == "limit":
        reason = "it marks where the main pattern may stop holding"
    else:
        reason = "it introduces visible disagreement or qualification"

    if anchors:
        return reason + "; " + ", ".join(anchors[:3]) + "."
    return reason + "."



def _extract_counterpoints(documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]], themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    seen = set()
    theme_terms = [(theme.get("theme"), set(theme.get("terms") or [])) for theme in themes]
    for source in list(documents) + list(highlights):
        text = (source.get("summary") or "") + "\n" + "\n".join((chunk.get("text") or "") for chunk in (source.get("chunks") or [])[:2])
        text = (text if text.strip() else source.get("highlightText") or source.get("highlightNote") or "").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        sentence = next((s for s in sentences if any(re.search(rf"\b{re.escape(marker)}\b", s.lower()) for marker in CONTRAST_MARKERS)), "")
        if not sentence or len(sentence) < 50:
            continue
        if len(re.findall(r"[A-Za-z]{3,}", sentence)) < 8:
            continue
        if sentence.count("【") > 1:
            continue
        title = source.get("title") or source.get("documentTitle") or "[untitled]"
        fingerprint = f"{title}|{sentence[:120].lower()}"
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kind = _counterpoint_kind(sentence)
        sentence_tokens = set(_meaningful_tokens(" ".join([title, sentence])))
        best_theme = ""
        best_overlap = 0
        for theme_name, terms in theme_terms:
            overlap = len(sentence_tokens & terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_theme = theme_name or ""
        selection = source.get("selectionSignals") or {}
        strength = source.get("matchStrength") or {}
        source_tier = selection.get("sourceQualityTier") or "medium"
        if source_tier == "weak" and not strength.get("tagSupport") and best_overlap < 2:
            continue
        if not strength.get("tagSupport") and not strength.get("titleSupport") and best_overlap < 2:
            continue
        relevance = best_overlap * 5
        if selection.get("contrastSignal"):
            relevance += 6
        if strength.get("tagSupport"):
            relevance += 3
        if strength.get("titleSupport"):
            relevance += 2
        if source_tier == "strong":
            relevance += 2
        elif source_tier == "weak":
            relevance -= 4
        findings.append({
            "title": title,
            "author": source.get("author") or source.get("documentAuthor"),
            "snippet": sentence[:240],
            "kind": kind,
            "label": _counterpoint_label(kind, title, sentence, best_theme or None),
            "whyItMatters": _counterpoint_explanation(kind, source, sentence, best_theme or None),
            "sourceType": "document" if source.get("title") else "highlight",
            "anchorTheme": best_theme or None,
            "relevanceScore": relevance,
        })
    findings.sort(key=lambda item: (item.get("relevanceScore", 0), item.get("kind") == "tradeoff", item.get("kind") == "risk", len(item.get("snippet") or "")), reverse=True)
    return findings[:3]



def _recommended_expansion_queries(query: str, themes: List[Dict[str, Any]], common_tags: List[str]) -> List[str]:
    expansions: List[str] = []
    for theme in themes[:3]:
        for term in theme.get("terms", [])[:2]:
            if term and term.lower() not in query.lower():
                expansions.append(f"{query} {term}")
    for tag in common_tags[:2]:
        if tag and tag.lower() not in query.lower():
            expansions.append(f"{query} tag:{tag}")
    deduped: List[str] = []
    seen = set()
    for item in expansions:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped[:5]



def _coverage_sentence(query: str, coverage: str, confidence: str, strict_mode: bool, profile: Dict[str, Any]) -> str:
    if confidence == "low":
        if strict_mode and profile.get("isBroad"):
            return f"This is a deliberately narrow read on '{query}': the cache only had a small number of high-confidence matches, so recall was traded for precision."
        return f"The saved corpus only gives a thin read on '{query}', so treat this as directional rather than complete."
    if coverage == "broad":
        return f"The saved corpus is broad enough to show a real pattern around '{query}', though it still reflects what was worth saving rather than the whole literature."
    if coverage == "moderate":
        return f"There is enough overlap to say something useful about '{query}', but the picture is still partial."
    return f"The evidence for '{query}' is concentrated in a small set of documents that appear genuinely on-topic."



def _theme_sentences(themes: List[Dict[str, Any]]) -> List[str]:
    sentences: List[str] = []
    for theme in themes[:3]:
        docs = ", ".join(theme.get("documents", [])[:2])
        if docs:
            sentences.append(f"One clear cluster is {theme['theme']}, anchored most clearly by {docs}.")
        else:
            sentences.append(f"One clear cluster is {theme['theme']}.")
    return sentences



def _lead_document_sentence(contributions: List[Dict[str, Any]]) -> str:
    if not contributions:
        return ""
    parts = []
    for doc in contributions[:3]:
        why = ", ".join(doc.get("whyIncluded", [])[:2]) or "strong lexical overlap"
        label = doc.get("title") or "[untitled]"
        if doc.get("author"):
            label += f" by {doc['author']}"
        parts.append(f"{label} ({why})")
    return "The center of gravity here is " + "; ".join(parts) + "."



def _counterpoint_sentence(counterpoints: List[Dict[str, Any]]) -> str:
    if not counterpoints:
        return ""
    lead = counterpoints[0]
    label = lead.get("title") or "[untitled]"
    if lead.get("author"):
        label += f" by {lead['author']}"
    return f"The strongest internal tension comes from {label}: \"{lead.get('snippet')}\" This matters because {lead.get('whyItMatters') or 'it complicates the main pattern.'}"



def _caveat_sentences(documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]], profile: Dict[str, Any], strict_mode: bool) -> List[str]:
    caveats: List[str] = []
    if strict_mode and profile.get("isBroad"):
        caveats.append("Strict filtering was active for a broad topic, so weak adjacent matches were intentionally excluded.")
    if not highlights:
        caveats.append("This pass is driven mostly by document titles, tags, summaries, and chunks rather than quote-level highlights.")
    elif len(highlights) < len(documents):
        caveats.append("Quote-level evidence exists, but the synthesis is still document-led rather than highlight-led.")
    if any((doc.get("category") or "").lower() in {"tweet", "tweets", "rss", "feed"} for doc in documents):
        caveats.append("Some lighter-weight sources are present, so the stronger claims should be grounded in the fuller documents.")
    return caveats[:3]



def _draft_synthesis(query: str, documents: List[Dict[str, Any]], highlights: List[Dict[str, Any]], themes: List[Dict[str, Any]], common_tags: List[str], coverage: str, confidence: str, strict_mode: bool, profile: Dict[str, Any], contributions: List[Dict[str, Any]], counterpoints: List[Dict[str, Any]]) -> str:
    paragraphs: List[str] = [_coverage_sentence(query, coverage, confidence, strict_mode, profile)]
    lead = _lead_document_sentence(contributions)
    if lead:
        paragraphs.append(lead)
    theme_lines = _theme_sentences(themes)
    if theme_lines:
        paragraphs.append(" ".join(theme_lines))
    counter = _counterpoint_sentence(counterpoints)
    if counter:
        paragraphs.append(counter)
    if common_tags:
        paragraphs.append(f"The tag pattern points in the same direction: {', '.join(common_tags[:5])} recur most often in the selected evidence.")
    caveats = _caveat_sentences(documents, highlights, profile, strict_mode)
    if caveats:
        paragraphs.append("Caveats: " + " ".join(caveats))
    return "\n\n".join(paragraphs)



def build_synthesis_packet(evidence: Dict[str, Any]) -> Dict[str, Any]:
    documents = evidence.get("documents", []) or []
    highlights = evidence.get("highlights", []) or []
    query = evidence.get("query")
    strict_mode = bool(evidence.get("strictMode"))
    profile = evidence.get("queryProfile") or {}
    confidence = evidence.get("confidence") or "medium"
    retrieval_mode = evidence.get("retrievalMode") or {}

    top_terms = _weighted_terms(documents, highlights, limit=12)
    tag_counts = Counter(
        [tag for d in documents for tag in d.get("tags", [])]
        + [tag for h in highlights for tag in (h.get("highlightTags", []) or [])]
        + [tag for h in highlights for tag in (h.get("documentTags", []) or [])]
    )
    common_tags = [tag for tag, _ in tag_counts.most_common(6)]
    themes = _build_theme_clusters(documents, highlights, top_terms, query or "", (profile.get("terms") or []))
    coverage = _coverage_label(len(documents), len(highlights), confidence)
    contribution_groups = _document_contributions(documents, themes)
    counterpoints = _extract_counterpoints(documents, highlights, themes)

    evidence_list = []
    for doc in documents:
        reasons = []
        strength = doc.get("matchStrength") or {}
        if strength.get("tagSupport"):
            reasons.append("manual tags")
        if strength.get("titleSupport"):
            reasons.append("title match")
        if strength.get("phraseSupport"):
            reasons.append("phrase hit")
        if (doc.get("selectionSignals") or {}).get("contrastSignal"):
            reasons.append("counterpoint signal")
        evidence_list.append(
            {
                "type": "document",
                "id": doc.get("documentId"),
                "title": doc.get("title"),
                "author": doc.get("author"),
                "sourceUrl": doc.get("sourceUrl"),
                "sourceDomain": doc.get("sourceDomain"),
                "tags": doc.get("tags", []),
                "cacheScore": doc.get("cacheScore"),
                "qualityScore": doc.get("qualityScore"),
                "matchStrength": doc.get("matchStrength"),
                "whySelected": ", ".join(reasons[:3]) or "high cache overlap",
            }
        )
    for hl in highlights:
        reasons = []
        if (hl.get("selectionSignals") or {}).get("hasTags"):
            reasons.append("tagged")
        if (hl.get("selectionSignals") or {}).get("contrastSignal"):
            reasons.append("contrast marker")
        evidence_list.append(
            {
                "type": "highlight",
                "id": hl.get("highlightId"),
                "documentId": hl.get("documentId"),
                "documentTitle": hl.get("documentTitle"),
                "url": hl.get("url"),
                "tags": hl.get("highlightTags", []),
                "cacheScore": hl.get("cacheScore"),
                "whySelected": ", ".join(reasons[:2]) or "quote-level support",
            }
        )

    expansion_options = [
        "broaden to more cached documents for the same query",
        "fetch and cache more documents before synthesizing again",
        "expand by tag-aligned documents",
        "expand to adjacent/related queries",
        "include more highlighted documents for quote-level support",
        "narrow to the strongest few documents only",
    ]

    return {
        "kind": "synthesisPacket",
        "query": query,
        "strictMode": strict_mode,
        "retrievalMode": retrieval_mode,
        "coverage": coverage,
        "confidence": confidence,
        "documentCount": len(documents),
        "highlightCount": len(highlights),
        "topTerms": top_terms,
        "commonTags": common_tags,
        "themes": themes,
        "counterpoints": counterpoints,
        "documentContributions": contribution_groups,
        "draftSynthesis": _draft_synthesis(
            query or "",
            documents,
            highlights,
            themes,
            common_tags,
            coverage,
            confidence,
            strict_mode,
            profile,
            contribution_groups,
            counterpoints,
        ),
        "coverageNotes": _caveat_sentences(documents, highlights, profile, strict_mode),
        "evidence": evidence_list,
        "expansionOptions": expansion_options,
        "recommendedExpansionQueries": _recommended_expansion_queries(query or "", themes, common_tags),
        "sourceEvidence": evidence,
    }

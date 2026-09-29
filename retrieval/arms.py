"""Retrieval arms. Each ranks the same chunks against the same query.

Everything except the ranking function is held fixed -- same corpus, same chunker,
same query, same k. Causal attribution of any measured delta is therefore
unambiguous, which is the property most retrieval comparisons quietly lack because
they change the index, the chunker and the query together.

Arms are deliberately boring. Novel retrieval is not the point; measuring which
cheap thing actually works is.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Sequence

from .chunk import Chunk

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> List[str]:
    """Split identifiers on snake_case and camelCase so `find_gaps` matches `gaps`."""
    out: List[str] = []
    for tok in TOKEN.findall(text):
        low = tok.lower()
        out.append(low)
        parts = [p for p in tok.split("_") if p]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
        for sub in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", tok):
            if sub.lower() != low:
                out.append(sub.lower())
    return out


# --------------------------------------------------------------------- arms

def arm_none(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A0 baseline: corpus order, no ranking at all. What unranked grep gives you."""
    return [c.uid for c in chunks]


def arm_lexical(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A1: rank by raw count of query tokens present. The naive `grep | sort` tier."""
    q = set(tokenize(query))
    scored = [(sum(1 for t in set(tokenize(c.text)) if t in q), c.uid) for c in chunks]
    return [uid for _, uid in sorted(scored, key=lambda s: (-s[0], s[1]))]


def arm_bm25(chunks: Sequence[Chunk], query: str) -> List[str]:
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(c.text) for c in chunks]
    bm = BM25Okapi(corpus)
    scores = bm.get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i].uid))
    return [chunks[i].uid for i in order]


def arm_tfidf(chunks: Sequence[Chunk], query: str) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    X = vec.fit_transform([c.text for c in chunks])
    sims = cosine_similarity(vec.transform([query]), X)[0]
    order = sorted(range(len(chunks)), key=lambda i: (-sims[i], chunks[i].uid))
    return [chunks[i].uid for i in order]


def arm_lsa(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A4: TF-IDF projected by SVD. A dense arm with no neural dependency."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    X = vec.fit_transform([c.text for c in chunks])
    k = max(2, min(64, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = svd.fit_transform(X)
    zq = svd.transform(vec.transform([query]))
    sims = cosine_similarity(zq, Z)[0]
    order = sorted(range(len(chunks)), key=lambda i: (-sims[i], chunks[i].uid))
    return [chunks[i].uid for i in order]


_EMBED_CACHE: Dict[str, object] = {}


def arm_embed(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A5: neural sentence embeddings, cosine similarity. The 'vector search' arm."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    if "m" not in _EMBED_CACHE:
        _EMBED_CACHE["m"] = SentenceTransformer("all-MiniLM-L6-v2")
    m = _EMBED_CACHE["m"]
    # Chunk embeddings are identical across observation levels, so encoding them once
    # per corpus turns a 4x cost into a 1x cost.
    key = hash(tuple(c.text for c in chunks))
    if key not in _EMBED_CACHE:
        _EMBED_CACHE[key] = m.encode([c.text for c in chunks], show_progress_bar=False)  # type: ignore[attr-defined]
    E = _EMBED_CACHE[key]
    q = m.encode([query], show_progress_bar=False)                      # type: ignore[attr-defined]
    sims = cosine_similarity(q, E)[0]
    order = sorted(range(len(chunks)), key=lambda i: (-sims[i], chunks[i].uid))
    return [chunks[i].uid for i in order]


def rrf(rankings: Sequence[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion. k=60 is the value from the original Cormack et al. paper."""
    score: Dict[str, float] = {}
    for ranking in rankings:
        for rank, uid in enumerate(ranking, start=1):
            score[uid] = score.get(uid, 0.0) + 1.0 / (k + rank)
    return [uid for uid, _ in sorted(score.items(), key=lambda s: (-s[1], s[0]))]


def arm_hybrid(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A6: RRF over BM25 + embeddings -- lexical and semantic fused.

    ai-harness-research/README.md records the finding this tests: on symptom->root-cause
    retrieval a structural retriever beat dense by 3.3x MRR, dense won on aggregate, and
    RRF fusion beat both by +18.1%. Whether that holds on injected faults is the question.
    """
    return rrf([arm_bm25(chunks, query), arm_embed(chunks, query)])


def arm_hybrid_nonneural(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A6b: RRF over BM25 + TF-IDF. The same fusion idea with zero model dependency."""
    return rrf([arm_bm25(chunks, query), arm_tfidf(chunks, query)])


ARMS: Dict[str, Callable[[Sequence[Chunk], str], List[str]]] = {
    "A0_none": arm_none,
    "A1_lexical": arm_lexical,
    "A2_bm25": arm_bm25,
    "A3_tfidf": arm_tfidf,
    "A4_lsa": arm_lsa,
    "A5_embed": arm_embed,
    "A6_hybrid_rrf": arm_hybrid,
    "A6b_hybrid_lexonly": arm_hybrid_nonneural,
}


# ------------------------------------------------------- hosted (Cohere) arms

def arm_cohere_embed(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A7: Cohere Embed v3, asymmetric search_document / search_query encoding.

    The asymmetry is the point of a retrieval-tuned embedder and is what separates
    it from encoding both sides with one generic sentence model (A5).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    from .providers import embed_into_memo, memo_vec
    embed_into_memo([c.text for c in chunks], "search_document")
    E = [memo_vec(c.text, "search_document") for c in chunks]
    q = [memo_vec(query, "search_query")]
    sims = cosine_similarity(q, E)[0]
    order = sorted(range(len(chunks)), key=lambda i: (-sims[i], chunks[i].uid))
    return [chunks[i].uid for i in order]


def arm_cohere_rerank(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A8: Cohere Rerank 3.5, a cross-encoder scoring query against each chunk.

    Cross-encoders see both sides together instead of comparing two independently
    produced vectors, which is why they usually beat bi-encoder retrieval -- and why
    they cost a call per query rather than a cached vector.
    """
    from .providers import cohere_rerank
    order = cohere_rerank(query, [c.text for c in chunks])
    return [chunks[i].uid for i in order]


def arm_cohere_hybrid(chunks: Sequence[Chunk], query: str) -> List[str]:
    """A9: BM25 retrieve, then Cohere Rerank -- the standard production pattern."""
    return rrf([arm_bm25(chunks, query), arm_cohere_rerank(chunks, query)])


ARMS["A7_cohere_embed"] = arm_cohere_embed
ARMS["A8_cohere_rerank"] = arm_cohere_rerank
ARMS["A9_bm25_then_cohere_rerank"] = arm_cohere_hybrid

from __future__ import annotations

import re
from typing import Any

import faiss
import numpy as np
import pandas as pd

from .parsing import build_pipeline_handoff, first_resolved_reference

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
}


def tokenize_for_bm25(text: Any) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms for sparse retrieval."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def tokenize_query(text: Any) -> list[str]:
    """Tokenize a user query while dropping common stopwords when possible."""
    base_tokens = tokenize_for_bm25(text)
    tokens = [token for token in base_tokens if len(token) > 1 and token not in QUERY_STOPWORDS]
    return tokens or [token for token in base_tokens if len(token) > 1]


def top_k_indices_from_scores(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return the indices of the highest-scoring entries without full sorting."""
    if len(scores) == 0:
        return np.array([], dtype=np.int32)
    top_k = min(int(top_k), len(scores))
    if top_k <= 0:
        return np.array([], dtype=np.int32)
    candidate_idx = np.argpartition(scores, -top_k)[-top_k:]
    return candidate_idx[np.argsort(scores[candidate_idx])[::-1]]


def field_overlap_boost(query_tokens: list[str], row: Any) -> float:
    """Compute a lightweight lexical bonus from title, category, and store overlap."""
    query_token_set = set(query_tokens)
    title_tokens = set(tokenize_for_bm25(getattr(row, "title", "")))
    category_tokens = set(tokenize_for_bm25(getattr(row, "main_category", "")))
    store_tokens = set(tokenize_for_bm25(getattr(row, "store", "")))
    return (
        3.0 * len(query_token_set & title_tokens)
        + 1.5 * len(query_token_set & category_tokens)
        + 1.0 * len(query_token_set & store_tokens)
    )


def hybrid_search(
    query_text: str,
    artifacts: dict[str, Any],
    *,
    top_k: int = 30,
    source_filter: str | None = None,
    pool_size: int | None = None,
) -> pd.DataFrame:
    """Run hybrid sparse+dense retrieval over the candidate pool."""
    candidate_pool = artifacts["candidate_pool"]
    bm25_index = artifacts["bm25_index"]
    faiss_index = artifacts["faiss_index"]
    item_embeddings = artifacts["item_embeddings"]
    encoder = artifacts["encoder"]
    source_to_row_ids = artifacts["source_to_row_ids"]

    pool_size = int(pool_size or artifacts["index_metadata"].get("hybrid_pool_size", 500))
    query_tokens = tokenize_query(query_text)
    if not query_tokens:
        return candidate_pool.iloc[:0].assign(retrieval_score=pd.Series(dtype=float))

    if source_filter:
        eligible_idx = source_to_row_ids.get(str(source_filter).lower(), np.array([], dtype=np.int32))
    else:
        eligible_idx = np.arange(len(candidate_pool), dtype=np.int32)
    if len(eligible_idx) == 0:
        return candidate_pool.iloc[:0].assign(retrieval_score=pd.Series(dtype=float))

    bm25_scores = bm25_index.get_scores(query_tokens)
    eligible_bm25 = bm25_scores[eligible_idx]
    bm25_top_local = top_k_indices_from_scores(eligible_bm25, pool_size)
    bm25_top = eligible_idx[bm25_top_local]

    q_vec = encoder.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    faiss.normalize_L2(q_vec)

    if source_filter:
        dense_scores = item_embeddings[eligible_idx] @ q_vec[0]
        dense_top_local = top_k_indices_from_scores(dense_scores, pool_size)
        dense_top = eligible_idx[dense_top_local]
    else:
        _, dense_top = faiss_index.search(q_vec, pool_size)
        dense_top = dense_top[0]

    shortlist = np.unique(np.concatenate([bm25_top, dense_top])).astype(np.int32)
    if len(shortlist) == 0:
        return candidate_pool.iloc[:0].assign(retrieval_score=pd.Series(dtype=float))

    rrf_k = int(artifacts["index_metadata"].get("rrf_k", 60))
    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_top):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(dense_top):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

    out = candidate_pool.iloc[shortlist].copy()
    out["lexical_overlap"] = [field_overlap_boost(query_tokens, row) for row in out.itertuples(index=False)]
    if (out["lexical_overlap"] > 0).any():
        out = out[out["lexical_overlap"] > 0].copy()
    if out.empty:
        return candidate_pool.iloc[:0].assign(retrieval_score=pd.Series(dtype=float))

    out["retrieval_score"] = out.index.to_series().map(rrf_scores).fillna(0.0) + 0.03 * out["lexical_overlap"]
    out = out.sort_values(
        ["retrieval_score", "lexical_overlap", "average_rating", "rating_number"],
        ascending=[False, False, False, False],
    )
    return out.head(top_k).reset_index(drop=True)


def search_similar(
    anchor_asin: str,
    artifacts: dict[str, Any],
    *,
    top_k: int = 30,
    source_filter: str | None = None,
) -> pd.DataFrame:
    """Find items similar to a reference item using dense embeddings."""
    candidate_pool = artifacts["candidate_pool"]
    faiss_index = artifacts["faiss_index"]
    item_embeddings = artifacts["item_embeddings"]
    asin_to_row = artifacts["asin_to_row"]
    source_to_row_ids = artifacts["source_to_row_ids"]

    if anchor_asin not in asin_to_row:
        raise KeyError(f"Unknown anchor ASIN: {anchor_asin}")

    row_idx = asin_to_row[anchor_asin]
    anchor_vec = item_embeddings[row_idx : row_idx + 1].copy().astype(np.float32)
    anchor_vec_1d = anchor_vec[0]

    if source_filter:
        eligible_idx = source_to_row_ids.get(str(source_filter).lower(), np.array([], dtype=np.int32))
        eligible_idx = eligible_idx[eligible_idx != row_idx]
        if len(eligible_idx) == 0:
            return candidate_pool.iloc[:0].assign(retrieval_score=pd.Series(dtype=float))
        dense_scores = item_embeddings[eligible_idx] @ anchor_vec_1d
        top_local = top_k_indices_from_scores(dense_scores, top_k)
        dense_top = eligible_idx[top_local]
        top_scores = dense_scores[top_local]
    else:
        _, dense_top = faiss_index.search(anchor_vec, top_k + 1)
        dense_top = dense_top[0]
        dense_top = dense_top[dense_top != row_idx][:top_k]
        top_scores = np.array(
            [float(anchor_vec_1d @ item_embeddings[i]) for i in dense_top],
            dtype=np.float32,
        )

    out = candidate_pool.iloc[dense_top].copy()
    out["retrieval_score"] = top_scores
    return out.reset_index(drop=True)


def retrieve_candidates_for_request(
    parsed_request: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    top_k: int = 30,
) -> pd.DataFrame:
    """Choose the right candidate-generation strategy for a parsed request."""
    handoff = build_pipeline_handoff(parsed_request)
    resolved_reference = first_resolved_reference(parsed_request)

    if handoff["candidate_generation_mode"] == "reference_similarity" and resolved_reference is not None:
        return search_similar(
            resolved_reference["resolved_parent_asin"],
            artifacts,
            top_k=top_k,
            source_filter=(
                handoff["source_filter"] if handoff["same_source_only"] else handoff["source_filter"]
            ),
        )

    return hybrid_search(
        handoff["retrieval_query"] or parsed_request["original_query"],
        artifacts,
        top_k=top_k,
        source_filter=handoff["source_filter"],
    )

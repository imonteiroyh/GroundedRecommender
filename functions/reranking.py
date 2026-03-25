from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .parsing import first_resolved_reference
from .retrieval import retrieve_candidates_for_request


def normalize_text_list(values: list[Any]) -> list[str]:
    """Normalize a list of text values for case-insensitive comparisons."""
    return [str(value).strip().lower() for value in values if str(value).strip()]


def extract_need_terms(parsed_request: dict[str, Any]) -> list[str]:
    """Extract lightweight lexical signals for use-case and preference matching."""
    hard_constraints = parsed_request.get("hard_constraints", {})
    tokens: list[str] = []
    tokens.extend(hard_constraints.get("must_include_terms", []))
    use_case = hard_constraints.get("use_case")
    if use_case:
        tokens.extend(str(use_case).lower().replace("/", " ").replace("-", " ").split())

    for preference in parsed_request.get("soft_preferences", []):
        if preference not in {"highly_rated", "budget_friendly"}:
            tokens.extend(str(preference).lower().replace("-", " ").split())

    stop_terms = {"for", "and", "the", "with", "good", "great", "best", "item", "something"}
    return sorted({token for token in tokens if len(token) > 2 and token not in stop_terms})


def apply_hard_constraints(
    candidates: pd.DataFrame,
    parsed_request: dict[str, Any],
    *,
    reference_price: float | None = None,
) -> pd.Series:
    """Build the boolean mask for hard constraints and exclusions."""
    hard_constraints = parsed_request.get("hard_constraints", {})
    mask = pd.Series(True, index=candidates.index)

    max_price = hard_constraints.get("max_price")
    if max_price is not None:
        mask &= candidates["price"].isna() | (candidates["price"] <= max_price)

    if hard_constraints.get("cheaper_than_reference") and reference_price is not None:
        mask &= candidates["price"].isna() | (candidates["price"] < reference_price)

    min_rating = hard_constraints.get("min_rating")
    if min_rating is not None:
        mask &= candidates["average_rating"].isna() | (candidates["average_rating"] >= min_rating)

    text_lower = candidates["retrieval_text"].fillna("").str.lower()
    title_lower = candidates["title"].fillna("").str.lower()
    store_lower = candidates["store"].fillna("").str.lower()

    for term in normalize_text_list(hard_constraints.get("must_include_terms", [])):
        mask &= text_lower.str.contains(term, regex=False)

    for brand in normalize_text_list(parsed_request.get("excluded_brands", [])):
        mask &= ~title_lower.str.contains(brand, regex=False)
        mask &= ~store_lower.str.contains(brand, regex=False)

    for term in normalize_text_list(parsed_request.get("excluded_terms", [])):
        mask &= ~text_lower.str.contains(term, regex=False)

    domain_hint = parsed_request.get("domain_hint")
    if domain_hint:
        mask &= candidates["source"].str.lower().eq(str(domain_hint).lower())

    return mask


def min_max_normalize(series: pd.Series, neutral: float = 0.0) -> pd.Series:
    """Normalize a score series into the [0, 1] range."""
    values = series.fillna(0.0).astype(float)
    span = values.max() - values.min()
    if span <= 0:
        return pd.Series(neutral, index=series.index, dtype=float)
    return (values - values.min()) / span


def signal_retrieval(candidates: pd.DataFrame) -> pd.Series:
    return min_max_normalize(candidates["retrieval_score"], neutral=1.0)


def signal_rating(candidates: pd.DataFrame) -> pd.Series:
    rating = candidates["average_rating"].clip(1, 5)
    return ((rating - 1.0) / 4.0).fillna(0.0)


def signal_popularity(candidates: pd.DataFrame) -> pd.Series:
    counts = np.log1p(candidates["rating_number"].fillna(0))
    return min_max_normalize(counts, neutral=0.0)


def signal_price_fit(
    candidates: pd.DataFrame,
    parsed_request: dict[str, Any],
    *,
    reference_price: float | None = None,
) -> pd.Series:
    """Score how well each candidate fits the budget-related request signals."""
    prices = candidates["price"]
    hard_constraints = parsed_request.get("hard_constraints", {})
    max_price = hard_constraints.get("max_price")

    if max_price is not None and max_price > 0:
        score = pd.Series(0.4, index=candidates.index, dtype=float)
        valid = prices.notna() & (prices > 0)
        score.loc[valid] = ((max_price - prices.loc[valid]) / max_price).clip(0, 1)
        return score

    if hard_constraints.get("cheaper_than_reference") and reference_price is not None and reference_price > 0:
        score = pd.Series(0.0, index=candidates.index, dtype=float)
        valid = prices.notna() & (prices > 0)
        score.loc[valid] = ((reference_price - prices.loc[valid]) / reference_price).clip(0, 1)
        return score

    return pd.Series(0.0, index=candidates.index, dtype=float)


def signal_reference_similarity(
    candidates: pd.DataFrame,
    artifacts: dict[str, Any],
    *,
    anchor_asin: str | None = None,
) -> pd.Series:
    """Score candidates by dense similarity to a resolved reference item."""
    if not anchor_asin:
        return pd.Series(0.0, index=candidates.index, dtype=float)

    asin_to_row = artifacts["asin_to_row"]
    item_embeddings = artifacts["item_embeddings"]
    anchor_idx = asin_to_row.get(anchor_asin)
    if anchor_idx is None:
        return pd.Series(0.0, index=candidates.index, dtype=float)

    anchor_vec = item_embeddings[anchor_idx].copy().astype(np.float32)
    candidate_rows = candidates["parent_asin"].map(asin_to_row).values
    candidate_vectors = np.stack([item_embeddings[row] for row in candidate_rows]).astype(np.float32)
    similarities = candidate_vectors @ anchor_vec
    return min_max_normalize(pd.Series(similarities, index=candidates.index), neutral=0.0)


def signal_need_match(candidates: pd.DataFrame, parsed_request: dict[str, Any]) -> pd.Series:
    """Score candidates by lexical coverage of extracted need terms."""
    need_terms = extract_need_terms(parsed_request)
    if not need_terms:
        return pd.Series(0.0, index=candidates.index, dtype=float)

    text = candidates["retrieval_text"].fillna("").str.lower()
    hits = pd.Series(0.0, index=candidates.index, dtype=float)
    for term in need_terms:
        hits += text.str.contains(term, regex=False).astype(float)
    return (hits / len(need_terms)).clip(0, 1)


def resolve_weights(parsed_request: dict[str, Any]) -> dict[str, float]:
    """Allocate reranking weights based on the structure of the request."""
    weights = {
        "retrieval": 0.35,
        "rating": 0.20,
        "popularity": 0.10,
        "price_fit": 0.00,
        "reference_similarity": 0.00,
        "need_match": 0.10,
        "_pool": 0.25,
    }

    hard_constraints = parsed_request.get("hard_constraints", {})
    soft_preferences = set(parsed_request.get("soft_preferences", []))
    reference = first_resolved_reference(parsed_request)

    if hard_constraints.get("max_price") is not None or hard_constraints.get("cheaper_than_reference"):
        take = min(0.15, weights["_pool"])
        weights["price_fit"] += take
        weights["_pool"] -= take

    if reference is not None:
        take = min(0.15, weights["_pool"])
        weights["reference_similarity"] += take
        weights["_pool"] -= take

    if extract_need_terms(parsed_request):
        take = min(0.10, weights["_pool"])
        weights["need_match"] += take
        weights["_pool"] -= take

    if "highly_rated" in soft_preferences:
        take = min(0.10, weights["_pool"] + 0.05)
        weights["rating"] += take
        weights["popularity"] = max(0.05, weights["popularity"] - 0.05)
        weights["_pool"] = max(0.0, weights["_pool"] - max(0.0, take - 0.05))

    weights["retrieval"] += weights["_pool"]
    del weights["_pool"]

    total = sum(weights.values())
    return {name: round(value / total, 4) for name, value in weights.items()}


def retrieve_candidates(
    parsed_request: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    top_k: int = 30,
) -> pd.DataFrame:
    """Convenience wrapper around request-aware candidate retrieval."""
    return retrieve_candidates_for_request(parsed_request, artifacts, top_k=top_k)


def rerank_candidates(
    parsed_request: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    top_k_retrieval: int = 30,
    top_k_final: int = 5,
) -> dict[str, Any]:
    """Filter and rerank retrieved candidates using explicit scoring signals."""
    baseline = retrieve_candidates(parsed_request, artifacts, top_k=top_k_retrieval).copy()
    baseline["baseline_rank"] = np.arange(1, len(baseline) + 1)

    reference = first_resolved_reference(parsed_request)
    reference_price = reference.get("price") if reference else None
    anchor_asin = reference.get("resolved_parent_asin") if reference else None

    keep_mask = apply_hard_constraints(baseline, parsed_request, reference_price=reference_price)
    filtered = baseline.loc[keep_mask].copy()
    filtered["passes_constraints"] = True

    if filtered.empty:
        return {
            "baseline_candidates": baseline.reset_index(drop=True),
            "filtered_candidates": filtered.reset_index(drop=True),
            "reranked_candidates": filtered.reset_index(drop=True),
            "weights": resolve_weights(parsed_request),
        }

    filtered["s_retrieval"] = signal_retrieval(filtered)
    filtered["s_rating"] = signal_rating(filtered)
    filtered["s_popularity"] = signal_popularity(filtered)
    filtered["s_price_fit"] = signal_price_fit(
        filtered,
        parsed_request,
        reference_price=reference_price,
    )
    filtered["s_reference_similarity"] = signal_reference_similarity(
        filtered,
        artifacts,
        anchor_asin=anchor_asin,
    )
    filtered["s_need_match"] = signal_need_match(filtered, parsed_request)

    weights = resolve_weights(parsed_request)
    for name, weight in weights.items():
        filtered[f"w_{name}"] = weight
        filtered[f"c_{name}"] = filtered[f"s_{name}"] * weight

    contribution_columns = [column for column in filtered.columns if column.startswith("c_")]
    filtered["final_score"] = filtered[contribution_columns].sum(axis=1)
    filtered = filtered.sort_values(
        ["final_score", "retrieval_score", "average_rating"],
        ascending=False,
    ).reset_index(drop=True)
    filtered["reranked_rank"] = np.arange(1, len(filtered) + 1)
    filtered["rank_shift"] = filtered["baseline_rank"] - filtered["reranked_rank"]

    return {
        "baseline_candidates": baseline.reset_index(drop=True),
        "filtered_candidates": filtered.reset_index(drop=True),
        "reranked_candidates": filtered.head(top_k_final).reset_index(drop=True),
        "weights": weights,
    }

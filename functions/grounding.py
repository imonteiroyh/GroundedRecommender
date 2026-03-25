from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import re

from .io import sql
from .parsing import first_resolved_reference

EVIDENCE_SOURCE_METADATA = "metadata"
EVIDENCE_SOURCE_REVIEW = "review"

SOFT_PREFERENCE_TERMS = {
    "highly_rated": "highly rated review",
    "lightweight": "lightweight portable compact",
    "budget_friendly": "affordable price cheaper",
    "giftable": "gift gifting present",
}

GROUNDING_NEED_TERMS = {
    "price": "price affordable cost cheaper",
    "rating": "rating highly rated review",
    "use_case_fit": "use case practical experience",
    "giftability": "gift gifting present",
    "portability": "lightweight portable compact",
    "reference_comparison": "similar compare alternative",
    "brand_constraint": "brand manufacturer",
}


def _make_evidence_id(parent_asin: str, source: str, rank: int | None = None) -> str:
    if source == EVIDENCE_SOURCE_METADATA:
        return f"{parent_asin}::metadata"
    if source == EVIDENCE_SOURCE_REVIEW and rank is not None:
        return f"{parent_asin}::review::{int(rank)}"
    return f"{parent_asin}::{source}"


def clean_listish_text(text: Any) -> str:
    """Clean serialized list-like metadata fields into readable plain text."""
    text = "" if text is None else str(text)
    if text.strip() == "[]":
        return ""
    text = re.sub(r'[\[\]"]', " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: Any, max_chars: int = 500) -> str:
    """Trim long evidence strings while preserving readability."""
    text = "" if text is None else str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_retrieval_query(parsed_request: dict[str, Any]) -> str:
    """Assemble a retrieval-oriented query string from parsed request fields."""
    terms = []
    hard_constraints = parsed_request.get("hard_constraints", {})
    terms.extend(hard_constraints.get("must_include_terms", []))
    if hard_constraints.get("use_case"):
        terms.append(hard_constraints["use_case"])
    for preference in parsed_request.get("soft_preferences", []):
        if preference in SOFT_PREFERENCE_TERMS:
            terms.append(SOFT_PREFERENCE_TERMS[preference])
    if parsed_request.get("domain_hint"):
        terms.append(str(parsed_request["domain_hint"]).replace("_", " "))
    terms.append(parsed_request.get("original_query", ""))
    return " ".join(part for part in terms if part).strip()


def build_grounding_query(parsed_request: dict[str, Any]) -> str:
    """Build the evidence-scoring query used by the grounding stage."""
    terms = [parsed_request.get("original_query", ""), build_retrieval_query(parsed_request)]
    for need in parsed_request.get("grounding_needs", []):
        if need in GROUNDING_NEED_TERMS:
            terms.append(GROUNDING_NEED_TERMS[need])
    return " ".join(part for part in terms if part).strip()


def fetch_metadata_evidence(
    con,
    candidate_frame: pd.DataFrame,
    metadata_cache: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch metadata-based evidence rows for the candidate shortlist."""
    if candidate_frame.empty:
        return pd.DataFrame(
            columns=[
                "parent_asin",
                "candidate_title",
                "evidence_source",
                "evidence_id",
                "evidence_text",
                "review_title",
                "review_rating",
                "helpful_vote",
                "verified_purchase",
            ]
        )

    parent_asins = candidate_frame["parent_asin"].tolist()
    if metadata_cache is None:
        candidate_ids = pd.DataFrame({"parent_asin": parent_asins})
        con.register("candidate_ids", candidate_ids)
        rows = sql(
            con,
            """
            SELECT
                i.parent_asin,
                i.title AS candidate_title,
                i.description_text,
                i.features_text,
                i.price,
                i.average_rating,
                i.rating_number
            FROM items i
            INNER JOIN candidate_ids c USING (parent_asin)
            ORDER BY i.parent_asin
            """,
        )
        con.unregister("candidate_ids")
    else:
        rows = metadata_cache.loc[metadata_cache["parent_asin"].isin(parent_asins)].copy()

    rows["description_text"] = rows["description_text"].fillna("").apply(clean_listish_text)
    rows["features_text"] = rows["features_text"].fillna("").apply(clean_listish_text)

    def compose_metadata_text(row):
        parts = [f"Title: {row['candidate_title']}"]
        if row["description_text"]:
            parts.append(f"Description: {truncate_text(row['description_text'], 350)}")
        if row["features_text"]:
            parts.append(f"Features: {truncate_text(row['features_text'], 350)}")
        if pd.notna(row["price"]):
            parts.append(f"Price: ${row['price']:.2f}")
        if pd.notna(row["average_rating"]):
            rating_text = f"Average rating: {row['average_rating']:.1f}"
            if pd.notna(row["rating_number"]):
                rating_text += f" from {int(row['rating_number'])} ratings"
            parts.append(rating_text)
        return " ".join(parts)

    rows["evidence_source"] = EVIDENCE_SOURCE_METADATA
    rows["evidence_id"] = rows["parent_asin"].apply(
        lambda parent_asin: _make_evidence_id(parent_asin, EVIDENCE_SOURCE_METADATA)
    )
    rows["evidence_text"] = rows.apply(compose_metadata_text, axis=1)
    rows["review_title"] = None
    rows["review_rating"] = np.nan
    rows["helpful_vote"] = np.nan
    rows["verified_purchase"] = None

    return rows[
        [
            "parent_asin",
            "candidate_title",
            "evidence_source",
            "evidence_id",
            "evidence_text",
            "review_title",
            "review_rating",
            "helpful_vote",
            "verified_purchase",
        ]
    ]


def fetch_review_evidence(
    con,
    candidate_frame: pd.DataFrame,
    review_cache: pd.DataFrame | None = None,
    *,
    max_reviews_per_item: int = 20,
    candidate_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch and rank review-based evidence rows for the candidate shortlist."""
    if candidate_frame.empty:
        return pd.DataFrame(
            columns=[
                "parent_asin",
                "candidate_title",
                "evidence_source",
                "evidence_id",
                "evidence_text",
                "review_title",
                "review_rating",
                "helpful_vote",
                "verified_purchase",
            ]
        )

    parent_asins = candidate_frame["parent_asin"].tolist()
    if review_cache is None:
        candidate_ids = pd.DataFrame({"parent_asin": parent_asins})
        con.register("candidate_ids", candidate_ids)
        rows = sql(
            con,
            f"""
            WITH cleaned_reviews AS (
                SELECT
                    r.parent_asin,
                    r.title AS review_title,
                    coalesce(r.text, '') AS review_text,
                    r.rating AS review_rating,
                    r.helpful_vote,
                    r.verified_purchase
                FROM reviews r
                INNER JOIN candidate_ids c USING (parent_asin)
            ),
            ranked_reviews AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY parent_asin
                        ORDER BY coalesce(helpful_vote, 0) DESC, length(review_text) DESC, coalesce(review_rating, 0) DESC
                    ) AS review_rank
                FROM cleaned_reviews
                WHERE length(trim(coalesce(review_text, ''))) >= 40
            )
            SELECT *
            FROM ranked_reviews
            WHERE review_rank <= {int(max_reviews_per_item)}
            ORDER BY parent_asin, review_rank
            """,
        )
        con.unregister("candidate_ids")
    else:
        rows = review_cache.loc[review_cache["parent_asin"].isin(parent_asins)].copy()
        if "review_rank" in rows.columns:
            rows = rows.loc[rows["review_rank"] <= int(max_reviews_per_item)].copy()

    if rows.empty:
        return pd.DataFrame(
            columns=[
                "parent_asin",
                "candidate_title",
                "evidence_source",
                "evidence_id",
                "evidence_text",
                "review_title",
                "review_rating",
                "helpful_vote",
                "verified_purchase",
            ]
        )

    rows["review_text"] = (
        rows["review_text"]
        .fillna("")
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    rows = rows.loc[rows["review_text"].str.len() >= 40].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "parent_asin",
                "candidate_title",
                "evidence_source",
                "evidence_id",
                "evidence_text",
                "review_title",
                "review_rating",
                "helpful_vote",
                "verified_purchase",
            ]
        )

    title_lookup = (
        candidate_pool.set_index("parent_asin")["title"].to_dict()
        if candidate_pool is not None
        else candidate_frame.set_index("parent_asin")["title"].to_dict()
    )
    rows["candidate_title"] = rows["parent_asin"].map(title_lookup)
    rows["review_text"] = rows["review_text"].apply(lambda text: truncate_text(text, 450))
    rows["evidence_source"] = EVIDENCE_SOURCE_REVIEW
    rows["evidence_id"] = rows.apply(
        lambda row: _make_evidence_id(row["parent_asin"], EVIDENCE_SOURCE_REVIEW, row.get("review_rank")),
        axis=1,
    )
    rows["evidence_text"] = rows.apply(
        lambda row: (
            f"Review title: {row['review_title']}. Review: {row['review_text']}"
            if row["review_title"]
            else f"Review: {row['review_text']}"
        ),
        axis=1,
    )

    return rows[
        [
            "parent_asin",
            "candidate_title",
            "evidence_source",
            "evidence_id",
            "evidence_text",
            "review_title",
            "review_rating",
            "helpful_vote",
            "verified_purchase",
        ]
    ]


def matched_terms(query_text: str, evidence_text: str, *, top_k: int = 6) -> list[str]:
    """Return the most salient overlapping terms between query and evidence."""
    query_terms = set(re.findall(r"[a-z0-9]+", str(query_text).lower()))
    evidence_terms = re.findall(r"[a-z0-9]+", str(evidence_text).lower())
    overlap = [term for term in evidence_terms if term in query_terms]
    deduped = []
    seen = set()
    for term in overlap:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped[:top_k]


def matched_topic(parsed_request: dict[str, Any], evidence_row: pd.Series) -> str | None:
    """Map an evidence row to the grounding topic it most directly supports."""
    needs = parsed_request.get("grounding_needs", [])
    overlap_terms = set(evidence_row.get("matched_terms", []))

    if evidence_row["evidence_source"] == EVIDENCE_SOURCE_METADATA and "price" in needs:
        if {"price", "cheaper", "affordable", "cost"} & overlap_terms:
            return "price"
    if "rating" in needs and {"rating", "rated", "review"} & overlap_terms:
        return "rating"
    if "use_case_fit" in needs and parsed_request.get("hard_constraints", {}).get("use_case"):
        return "use_case_fit"
    if "giftability" in needs:
        return "giftability"
    if "portability" in needs:
        return "portability"
    if "reference_comparison" in needs:
        return "reference_comparison"
    return needs[0] if needs else None


def assemble_grounding_evidence(
    parsed_request: dict[str, Any],
    candidate_frame: pd.DataFrame,
    con,
    candidate_pool: pd.DataFrame,
    cross_encoder,
    *,
    metadata_cache: pd.DataFrame | None = None,
    review_cache: pd.DataFrame | None = None,
    max_reviews_per_item: int = 20,
    top_review_evidence_per_item: int = 2,
) -> pd.DataFrame:
    """Assemble and score evidence units for a reranked candidate set."""
    if candidate_frame.empty:
        return pd.DataFrame(
            columns=[
                "parent_asin",
                "candidate_title",
                "evidence_source",
                "evidence_id",
                "evidence_text",
                "relevance_score",
                "matched_terms",
                "matched_topic",
                "grounding_need",
                "review_title",
                "review_rating",
                "helpful_vote",
                "verified_purchase",
                "retrieval_score",
            ]
        )

    metadata_units = fetch_metadata_evidence(
        con,
        candidate_frame,
        metadata_cache=metadata_cache,
    )
    review_units = fetch_review_evidence(
        con,
        candidate_frame,
        review_cache=review_cache,
        max_reviews_per_item=max_reviews_per_item,
        candidate_pool=candidate_pool,
    )

    evidence_units = pd.concat([metadata_units, review_units], ignore_index=True)
    evidence_units = evidence_units.merge(
        candidate_frame[["parent_asin", "retrieval_score"]],
        on="parent_asin",
        how="left",
    )

    grounding_query = build_grounding_query(parsed_request)
    pairs = [(grounding_query, text) for text in evidence_units["evidence_text"].fillna("").tolist()]
    evidence_units["relevance_score"] = cross_encoder.predict(pairs).astype(float)
    evidence_units["matched_terms"] = evidence_units["evidence_text"].apply(
        lambda text: matched_terms(grounding_query, text)
    )
    evidence_units["matched_topic"] = evidence_units.apply(
        lambda row: matched_topic(parsed_request, row),
        axis=1,
    )
    evidence_units["grounding_need"] = ", ".join(parsed_request.get("grounding_needs", []))

    metadata_rows = evidence_units[evidence_units["evidence_source"] == EVIDENCE_SOURCE_METADATA]
    review_rows = evidence_units[evidence_units["evidence_source"] == EVIDENCE_SOURCE_REVIEW]

    if not review_rows.empty:
        review_rows = (
            review_rows.sort_values(
                ["parent_asin", "relevance_score", "helpful_vote"],
                ascending=[True, False, False],
            )
            .groupby("parent_asin", group_keys=False)
            .head(top_review_evidence_per_item)
        )

    final_evidence = pd.concat([metadata_rows, review_rows], ignore_index=True)
    final_evidence = final_evidence.sort_values(
        ["retrieval_score", "parent_asin", "evidence_source", "relevance_score"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    return final_evidence[
        [
            "parent_asin",
            "candidate_title",
            "evidence_source",
            "evidence_id",
            "evidence_text",
            "relevance_score",
            "matched_terms",
            "matched_topic",
            "grounding_need",
            "review_title",
            "review_rating",
            "helpful_vote",
            "verified_purchase",
            "retrieval_score",
        ]
    ]


def build_candidate_frame_for_grounding(
    parsed_request: dict[str, Any],
    retrieval_result: pd.DataFrame,
) -> pd.DataFrame:
    """Drop the reference item from grounding candidates when needed."""
    reference = first_resolved_reference(parsed_request)
    if reference is not None and not retrieval_result.empty:
        return retrieval_result.loc[
            retrieval_result["parent_asin"] != reference["resolved_parent_asin"]
        ].reset_index(drop=True)
    return retrieval_result.reset_index(drop=True)

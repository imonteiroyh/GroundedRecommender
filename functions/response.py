from __future__ import annotations

from typing import Any

import pandas as pd


FINAL_RESPONSE_TEMPLATE = {
    "original_query": None,
    "parsed_request": None,
    "retrieved_candidates": [],
    "reranked_finalists": [],
    "supporting_evidence": [],
    "explanation_text": "",
    "tradeoff_notes": [],
    "clarification": {
        "needed": False,
        "questions": [],
    },
}


def _candidate_to_record(row: pd.Series, include_scores: bool = True) -> dict[str, Any]:
    record = {
        "parent_asin": row.get("parent_asin"),
        "source": row.get("source"),
        "title": row.get("title"),
        "store": row.get("store"),
        "price": None if pd.isna(row.get("price")) else float(row.get("price")),
        "average_rating": None if pd.isna(row.get("average_rating")) else float(row.get("average_rating")),
        "rating_number": None if pd.isna(row.get("rating_number")) else int(row.get("rating_number")),
    }
    if include_scores:
        for column in ["retrieval_score", "final_score", "baseline_rank", "reranked_rank", "rank_shift"]:
            if column in row.index and pd.notna(row[column]):
                record[column] = float(row[column]) if "score" in column else int(row[column])
    return record


def _evidence_to_record(row: pd.Series) -> dict[str, Any]:
    return {
        "parent_asin": row.get("parent_asin"),
        "candidate_title": row.get("candidate_title"),
        "evidence_source": row.get("evidence_source"),
        "matched_topic": row.get("matched_topic"),
        "matched_terms": row.get("matched_terms"),
        "relevance_score": None if pd.isna(row.get("relevance_score")) else float(row.get("relevance_score")),
        "evidence_text": row.get("evidence_text"),
    }


def _pick_support_snippets(evidence_frame: pd.DataFrame, finalists: pd.DataFrame) -> pd.DataFrame:
    if evidence_frame.empty or finalists.empty:
        return evidence_frame.iloc[:0].copy()
    keep_ids = finalists["parent_asin"].tolist()
    subset = evidence_frame.loc[evidence_frame["parent_asin"].isin(keep_ids)].copy()
    subset = (
        subset.sort_values(["parent_asin", "relevance_score"], ascending=[True, False])
        .groupby("parent_asin", group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )
    return subset


def _build_tradeoff_notes(
    parsed_request: dict[str, Any],
    finalists: pd.DataFrame,
    evidence_frame: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    if finalists.empty:
        notes.append("No finalists survived the current hard constraints.")
        return notes

    if finalists["price"].isna().any():
        notes.append("Some finalists have missing price fields, so budget comparisons remain conservative.")
    if (
        "highly_rated" in parsed_request.get("soft_preferences", [])
        and finalists["average_rating"].isna().any()
    ):
        notes.append("Some finalists have incomplete rating metadata, so the quality signal is only partial.")
    if not evidence_frame.empty:
        unsupported = evidence_frame["matched_topic"].isna().all()
        if unsupported:
            notes.append(
                "The retrieved evidence is only loosely aligned with the intended explanation topics."
            )
    if parsed_request.get("clarification_needed"):
        notes.append("The request remains partly underspecified, so the response stays conservative.")
    return notes


def _build_explanation_text(
    parsed_request: dict[str, Any],
    finalists: pd.DataFrame,
    support_snippets: pd.DataFrame,
) -> str:
    if finalists.empty:
        questions = parsed_request.get("clarification_questions", [])
        if questions:
            return "I need one more detail before I can recommend confidently: " + " ".join(questions)
        return "I could not produce a confident recommendation from the current request and constraints."

    top = finalists.iloc[0]
    clauses = [f"I would start with {top['title']}."]

    if pd.notna(top.get("price")) and parsed_request.get("hard_constraints", {}).get("max_price") is not None:
        clauses.append(f"It is priced at ${float(top['price']):.2f}, which stays within the stated budget.")

    if pd.notna(top.get("average_rating")):
        rating_text = f"It carries an average rating of {float(top['average_rating']):.1f}"
        if pd.notna(top.get("rating_number")):
            rating_text += f" across {int(top['rating_number'])} ratings."
        else:
            rating_text += "."
        clauses.append(rating_text)

    top_support = support_snippets.loc[support_snippets["parent_asin"] == top["parent_asin"]]
    matched_topics = [topic for topic in top_support["matched_topic"].dropna().tolist() if topic]
    matched_topics = list(dict.fromkeys(matched_topics))
    if matched_topics:
        clauses.append("The strongest supporting evidence covers: " + ", ".join(matched_topics) + ".")

    if parsed_request.get("excluded_brands"):
        clauses.append("Excluded brands were filtered out before the final ranking.")

    if parsed_request.get("clarification_needed"):
        clauses.append(
            "The request still has some ambiguity, so this recommendation should be treated as a careful first pass."
        )

    return " ".join(clauses)


def build_final_response(
    parsed_request: dict[str, Any],
    retrieved_candidates: pd.DataFrame,
    reranked_finalists: pd.DataFrame,
    evidence_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Assemble the final structured recommendation payload."""
    response = FINAL_RESPONSE_TEMPLATE.copy()
    response["original_query"] = parsed_request.get("original_query")
    response["parsed_request"] = parsed_request
    response["retrieved_candidates"] = [
        _candidate_to_record(row) for _, row in retrieved_candidates.head(5).iterrows()
    ]
    response["reranked_finalists"] = [
        _candidate_to_record(row) for _, row in reranked_finalists.head(3).iterrows()
    ]

    support_snippets = _pick_support_snippets(evidence_frame, reranked_finalists)
    response["supporting_evidence"] = [_evidence_to_record(row) for _, row in support_snippets.iterrows()]
    response["explanation_text"] = _build_explanation_text(
        parsed_request,
        reranked_finalists,
        support_snippets,
    )
    response["tradeoff_notes"] = _build_tradeoff_notes(
        parsed_request,
        reranked_finalists,
        support_snippets,
    )
    response["clarification"] = {
        "needed": bool(parsed_request.get("clarification_needed")),
        "questions": parsed_request.get("clarification_questions", []),
    }
    return response

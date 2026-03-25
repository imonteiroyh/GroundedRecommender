from __future__ import annotations

from typing import Any

import pandas as pd


def keyword_hit_count(frame: pd.DataFrame, keywords: list[str]) -> int:
    """Count how many rows mention at least one of the expected keywords."""
    text = (frame["title"].fillna("") + " " + frame["main_category"].fillna("")).str.lower()
    return int(text.map(lambda value: any(keyword in value for keyword in keywords)).sum())


def normalize_list(values: list[Any]) -> list[str]:
    """Normalize a list of labels for stable notebook-side comparisons."""
    return sorted({str(value).strip().lower() for value in values if str(value).strip()})


def grounding_rubric(
    parsed_request: dict[str, Any],
    finalists: pd.DataFrame,
    evidence: pd.DataFrame,
) -> dict[str, str]:
    """Score grounding quality with a small qualitative rubric."""
    if finalists.empty:
        return {
            "relevance": "weak",
            "supportiveness": "weak",
            "specificity": "weak",
            "overclaim_risk": "high",
        }

    relevance = "strong" if not evidence.empty and evidence["matched_terms"].map(bool).any() else "mixed"
    supportiveness = "strong" if not evidence.empty and evidence["matched_topic"].notna().any() else "mixed"
    specificity = (
        "strong" if not evidence.empty and evidence["evidence_text"].str.len().mean() > 80 else "mixed"
    )
    overclaim_risk = "high" if parsed_request.get("clarification_needed") else "medium"
    if (
        not evidence.empty
        and evidence["matched_topic"].notna().any()
        and not parsed_request.get("clarification_needed")
    ):
        overclaim_risk = "low"

    return {
        "relevance": relevance,
        "supportiveness": supportiveness,
        "specificity": specificity,
        "overclaim_risk": overclaim_risk,
    }

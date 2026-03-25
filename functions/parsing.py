from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Any

import pandas as pd

REQUEST_SCHEMA_TEMPLATE = {
    "original_query": None,
    "user_intent": None,
    "domain_hint": None,
    "hard_constraints": {
        "max_price": None,
        "min_rating": None,
        "must_include_terms": [],
        "use_case": None,
        "cheaper_than_reference": False,
        "same_source_as_reference": False,
    },
    "soft_preferences": [],
    "reference_items": [],
    "excluded_brands": [],
    "excluded_terms": [],
    "grounding_needs": [],
    "clarification_needed": False,
    "clarification_questions": [],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "another",
    "best",
    "but",
    "for",
    "from",
    "good",
    "i",
    "if",
    "in",
    "is",
    "it",
    "like",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "recommend",
    "show",
    "something",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
}

PRODUCT_TERM_RULES = [
    ("noise cancelling headphones", "electronics", "headphones"),
    ("wireless headphones", "electronics", "headphones"),
    ("bluetooth headphones", "electronics", "headphones"),
    ("headphones", "electronics", "headphones"),
    ("headset", "electronics", "headset"),
    ("earbuds", "electronics", "earbuds"),
    ("laptop", "electronics", "laptop"),
    ("notebook", "electronics", "laptop"),
    ("monitor", "electronics", "monitor"),
    ("keyboard", "electronics", "keyboard"),
    ("speaker", "electronics", "speaker"),
    ("router", "electronics", "router"),
    ("tablet", "electronics", "tablet"),
    ("frying pan", "home_and_kitchen", "frying pan"),
    ("induction pan", "home_and_kitchen", "frying pan"),
    ("skillet", "home_and_kitchen", "skillet"),
    ("cookware", "home_and_kitchen", "cookware"),
    ("knife", "home_and_kitchen", "knife"),
    ("chef knife", "home_and_kitchen", "knife"),
    ("blender", "home_and_kitchen", "blender"),
    ("coffee maker", "home_and_kitchen", "coffee maker"),
    ("dumbbells", "sports_and_outdoors", "dumbbells"),
    ("dumbbell", "sports_and_outdoors", "dumbbell"),
    ("adjustable dumbbells", "sports_and_outdoors", "dumbbells"),
    ("weights", "sports_and_outdoors", "weights"),
    ("water bottle", "sports_and_outdoors", "water bottle"),
    ("yoga mat", "sports_and_outdoors", "yoga mat"),
    ("tent", "sports_and_outdoors", "tent"),
    ("backpack", "sports_and_outdoors", "backpack"),
]

USE_CASE_HINTS = {
    "programming": "electronics",
    "coding": "electronics",
    "commuting": "electronics",
    "travel": "electronics",
    "induction": "home_and_kitchen",
    "cooking": "home_and_kitchen",
    "baking": "home_and_kitchen",
    "kitchen": "home_and_kitchen",
    "gym": "sports_and_outdoors",
    "workout": "sports_and_outdoors",
    "training": "sports_and_outdoors",
    "home gym": "sports_and_outdoors",
    "hiking": "sports_and_outdoors",
    "camping": "sports_and_outdoors",
    "gift": None,
    "gifting": None,
}

SOFT_PREFERENCE_PATTERNS = {
    "lightweight": ["lightweight", "portable", "compact"],
    "highly_rated": ["highly rated", "top rated", "well reviewed", "best reviewed"],
    "budget_friendly": ["budget", "affordable", "cheap", "inexpensive", "not too expensive"],
    "giftable": ["gift", "gifting"],
}

SIMILARITY_HINTS = ["similar", "something like", "like this", "this item", "alternative to"]


def normalize_text(text: Any) -> str:
    """Normalize free text into a lowercase alphanumeric form for matching."""
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9$ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> list[str]:
    """Split normalized text into lightweight parser tokens."""
    normalized = normalize_text(text)
    return [token for token in normalized.split() if token and token not in STOPWORDS]


def dedupe_preserve_order(values: list[Any]) -> list[Any]:
    """Remove duplicates while keeping the first occurrence of each value."""
    seen = set()
    output = []
    for value in values:
        if value in (None, "", []):
            continue
        key = value if isinstance(value, (str, int, float, bool, tuple)) else str(value)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def build_parser_catalog(candidate_pool: pd.DataFrame) -> dict[str, Any]:
    """Create lookup tables used to resolve brands and reference mentions."""
    catalog = candidate_pool.copy()
    catalog["title_norm"] = catalog["title"].map(normalize_text)
    catalog["store_norm"] = catalog["store"].map(normalize_text)
    catalog["title_tokens"] = catalog["title_norm"].map(lambda text: set(tokenize(text)))

    store_lookup = (
        catalog[["store", "store_norm", "rating_number"]]
        .dropna(subset=["store"])
        .sort_values("rating_number", ascending=False)
        .drop_duplicates(subset=["store_norm"])
        .reset_index(drop=True)
    )
    return {"candidate_pool": catalog, "store_lookup": store_lookup}


def make_reference_record(
    mention: str,
    row: pd.Series | dict[str, Any] | None = None,
    resolution_status: str = "resolved",
) -> dict[str, Any]:
    """Build a normalized reference-item record from a row or mention."""
    if row is None:
        return {
            "mention": mention,
            "resolved_parent_asin": None,
            "resolved_title": None,
            "source": None,
            "store": None,
            "price": None,
            "resolution_status": resolution_status,
        }

    series = row if isinstance(row, pd.Series) else pd.Series(row)
    price = series.get("price")
    return {
        "mention": mention,
        "resolved_parent_asin": series.get("parent_asin") or series.get("resolved_parent_asin"),
        "resolved_title": series.get("title") or series.get("resolved_title"),
        "source": series.get("source"),
        "store": series.get("store"),
        "price": None if pd.isna(price) else float(price),
        "resolution_status": resolution_status,
    }


def first_resolved_reference(parsed_request: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first successfully resolved reference item, if any."""
    return next(
        (item for item in parsed_request.get("reference_items", []) if item.get("resolved_parent_asin")),
        None,
    )


def resolve_brand_name(raw_phrase: str, store_lookup: pd.DataFrame) -> str | None:
    """Map a raw brand phrase to the closest known store or brand string."""
    phrase_norm = normalize_text(raw_phrase)
    if not phrase_norm:
        return None

    exact = store_lookup.loc[store_lookup["store_norm"] == phrase_norm]
    if not exact.empty:
        return exact.iloc[0]["store"]

    partial = store_lookup.loc[
        store_lookup["store_norm"].str.contains(re.escape(phrase_norm), regex=True, na=False)
    ]
    if not partial.empty:
        return partial.iloc[0]["store"]

    return raw_phrase.strip(" '")


def extract_product_terms(query_norm: str) -> tuple[list[str], list[str]]:
    """Extract canonical product terms and their implied domains from a query."""
    detected_terms = []
    detected_domains = []
    for phrase, domain, canonical_term in PRODUCT_TERM_RULES:
        if phrase in query_norm:
            detected_terms.append(canonical_term)
            detected_domains.append(domain)
    return dedupe_preserve_order(detected_terms), detected_domains


def extract_use_case(query_norm: str) -> str | None:
    """Extract a simple use-case phrase introduced by `for ...` patterns."""
    match = re.search(
        r"\bfor\s+([a-z0-9$ ]+?)(?=\b(?:but|and|under|below|with|without|avoid|not|that)\b|[,.!?]|$)",
        query_norm,
    )
    if match:
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        if phrase and phrase not in {"me", "it", "this", "that"}:
            return "gifting" if phrase in {"gift", "gifting"} else phrase
    if "gift" in query_norm or "gifting" in query_norm:
        return "gifting"
    return None


def extract_max_price(query_norm: str) -> float | None:
    """Extract an upper price bound from common budget expressions."""
    patterns = [
        r"(?:under|below|less than|up to|max(?:imum)? of)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|max)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query_norm)
        if match:
            return float(match.group(1))
    return None


def extract_min_rating(query_norm: str) -> float | None:
    """Extract a minimum star rating requirement from the query."""
    patterns = [
        r"(?:at least|minimum|min)\s*(\d(?:\.\d)?)\s*(?:stars?|/5)",
        r"rated\s*(\d(?:\.\d)?)\s*(?:stars?|/5)",
        r"(\d(?:\.\d)?)\+\s*stars?",
    ]
    for pattern in patterns:
        match = re.search(pattern, query_norm)
        if match:
            return float(match.group(1))
    return None


def extract_soft_preferences(query_norm: str) -> list[str]:
    """Extract soft-preference tags from lightweight phrase patterns."""
    preferences = []
    for label, patterns in SOFT_PREFERENCE_PATTERNS.items():
        if any(pattern in query_norm for pattern in patterns):
            preferences.append(label)
    return dedupe_preserve_order(preferences)


def extract_excluded_terms(query_norm: str) -> list[str]:
    """Extract excluded terms from lightweight negative patterns."""
    matches = re.findall(
        r"(?:without|no)\s+([a-z0-9&'\- ]+?)(?=\s+(?:for|but|under|below|with|avoid)\b|[,.!?]|$)",
        query_norm,
    )
    return dedupe_preserve_order([match.strip() for match in matches if match.strip()])


def resolve_reference_mention(mention: str, candidate_pool: pd.DataFrame) -> dict[str, Any] | None:
    """Resolve a free-text item mention against the candidate pool."""
    mention = mention.strip()
    if not mention:
        return None

    if re.fullmatch(r"[A-Z0-9]{10}", mention.upper()):
        match = candidate_pool.loc[candidate_pool["parent_asin"] == mention.upper()]
        if not match.empty:
            return make_reference_record(mention, match.iloc[0])

    mention_norm = normalize_text(mention)
    if mention_norm in {"this item", "this", "it", "that item"}:
        return None

    contains_matches = candidate_pool.loc[
        candidate_pool["title_norm"].str.contains(re.escape(mention_norm), regex=True, na=False)
    ]
    if not contains_matches.empty:
        best = contains_matches.sort_values(
            ["rating_number", "average_rating"],
            ascending=[False, False],
        ).iloc[0]
        return make_reference_record(mention, best)

    mention_tokens = set(tokenize(mention_norm))
    if not mention_tokens:
        return None

    overlap_scores = candidate_pool["title_tokens"].map(
        lambda title_tokens: len(title_tokens & mention_tokens) / max(len(mention_tokens), 1)
    )
    best_idx = overlap_scores.idxmax()
    best_score = overlap_scores.loc[best_idx]
    if best_score >= 0.6:
        return make_reference_record(mention, candidate_pool.loc[best_idx])
    return None


def extract_reference_mentions(query_text: str) -> list[str]:
    """Extract candidate item mentions that likely refer to a known product."""
    mentions: list[str] = []
    mentions.extend(re.findall(r'"([^"]+)"', query_text))
    mentions.extend(re.findall(r"'([^']+)'", query_text))

    query_norm = normalize_text(query_text)
    patterns = [
        r"(?:similar to|something like|alternative to|like)\s+([^,.;]+?)(?=\s+(?:but|and|for|under|below|avoid|without)\b|[,.!?]|$)"
    ]
    for pattern in patterns:
        match = re.search(pattern, query_norm)
        if match:
            mentions.append(match.group(1).strip())

    cleaned = []
    for mention in mentions:
        mention = mention.strip(" '")
        if mention and mention not in {"this item", "this", "it", "that item"}:
            cleaned.append(mention)
    return dedupe_preserve_order(cleaned)


def infer_domain_hint(
    query_norm: str,
    reference_items: list[dict[str, Any]],
    product_domains: list[str],
    use_case: str | None,
) -> str | None:
    """Infer the most likely product domain for the request."""
    resolved_reference = next(
        (item for item in reference_items if item.get("resolved_parent_asin")),
        None,
    )
    if resolved_reference is not None:
        return resolved_reference.get("source")

    if "home and kitchen" in query_norm or "home_and_kitchen" in query_norm:
        return "home_and_kitchen"
    if "sports and outdoors" in query_norm or "sports_and_outdoors" in query_norm:
        return "sports_and_outdoors"
    if "electronics" in query_norm:
        return "electronics"

    if product_domains:
        return Counter(product_domains).most_common(1)[0][0]

    if use_case:
        for phrase, hinted_domain in USE_CASE_HINTS.items():
            if phrase in use_case:
                return hinted_domain
    return None


def blank_request(query_text: str) -> dict[str, Any]:
    """Return an empty request object seeded with the original query."""
    request = deepcopy(REQUEST_SCHEMA_TEMPLATE)
    request["original_query"] = query_text
    return request


def parse_user_request(
    query_text: str,
    parser_catalog: dict[str, Any],
    reference_item_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse a conversational request into the shared structured contract."""
    candidate_pool = parser_catalog["candidate_pool"]
    store_lookup = parser_catalog["store_lookup"]

    parsed = blank_request(query_text)
    query_norm = normalize_text(query_text)

    parsed["soft_preferences"] = extract_soft_preferences(query_norm)
    parsed["hard_constraints"]["max_price"] = extract_max_price(query_norm)
    parsed["hard_constraints"]["min_rating"] = extract_min_rating(query_norm)
    parsed["hard_constraints"]["use_case"] = extract_use_case(query_norm)

    product_terms, product_domains = extract_product_terms(query_norm)
    parsed["hard_constraints"]["must_include_terms"] = product_terms

    explicit_reference_mentions = extract_reference_mentions(query_text)
    reference_items = []
    for mention in explicit_reference_mentions:
        resolved = resolve_reference_mention(mention, candidate_pool)
        if resolved is not None:
            reference_items.append(resolved)
        else:
            reference_items.append(make_reference_record(mention, row=None, resolution_status="unresolved"))

    uses_context_reference = (
        any(phrase in query_norm for phrase in ["this item", "this brand", "similar", "alternative"])
        and not explicit_reference_mentions
    )
    if uses_context_reference:
        if reference_item_context is not None:
            reference_items.append(
                make_reference_record("this item", reference_item_context, resolution_status="from_context")
            )
        else:
            parsed["clarification_needed"] = True
            parsed["clarification_questions"].append("Which item should the recommendation be similar to?")

    parsed["reference_items"] = reference_items
    resolved_reference = first_resolved_reference(parsed)

    if any(phrase in query_norm for phrase in SIMILARITY_HINTS):
        parsed["hard_constraints"]["same_source_as_reference"] = resolved_reference is not None

    if "cheaper" in query_norm or "less expensive" in query_norm:
        parsed["hard_constraints"]["cheaper_than_reference"] = True
        parsed["soft_preferences"].append("budget_friendly")
        if resolved_reference is not None and resolved_reference.get("price") is not None:
            reference_price = float(resolved_reference["price"])
            current_max_price = parsed["hard_constraints"]["max_price"]
            parsed["hard_constraints"]["max_price"] = (
                reference_price if current_max_price is None else min(current_max_price, reference_price)
            )

    if (
        "avoid this brand" in query_norm
        and resolved_reference is not None
        and resolved_reference.get("store")
    ):
        parsed["excluded_brands"].append(resolved_reference["store"])

    explicit_avoid_matches = re.findall(
        r"avoid\s+(?:brand\s+)?([a-z0-9&'\- ]+?)(?=\s+(?:for|but|under|below|with|without)\b|[,.!?]|$)",
        query_norm,
    )
    for match in explicit_avoid_matches:
        if match.strip() != "this brand":
            resolved_brand = resolve_brand_name(match, store_lookup)
            if resolved_brand:
                parsed["excluded_brands"].append(resolved_brand)

    parsed["excluded_terms"] = extract_excluded_terms(query_norm)

    parsed["soft_preferences"] = dedupe_preserve_order(parsed["soft_preferences"])
    parsed["excluded_brands"] = dedupe_preserve_order(parsed["excluded_brands"])
    parsed["excluded_terms"] = dedupe_preserve_order(parsed["excluded_terms"])

    parsed["domain_hint"] = infer_domain_hint(
        query_norm=query_norm,
        reference_items=reference_items,
        product_domains=product_domains,
        use_case=parsed["hard_constraints"]["use_case"],
    )

    if any(phrase in query_norm for phrase in SIMILARITY_HINTS):
        parsed["user_intent"] = "similar_item_refinement"
    elif "gift" in query_norm:
        parsed["user_intent"] = "gift_search"
    elif (
        parsed["hard_constraints"]["use_case"]
        or parsed["hard_constraints"]["must_include_terms"]
        or parsed["domain_hint"]
    ):
        parsed["user_intent"] = "constrained_search"
    else:
        parsed["user_intent"] = "open_search"

    grounding_needs = []
    if (
        parsed["hard_constraints"]["max_price"] is not None
        or parsed["hard_constraints"]["cheaper_than_reference"]
    ):
        grounding_needs.append("price")
    if parsed["hard_constraints"]["min_rating"] is not None or "highly_rated" in parsed["soft_preferences"]:
        grounding_needs.append("rating")
    if parsed["hard_constraints"]["use_case"]:
        grounding_needs.append("use_case_fit")
    if "lightweight" in parsed["soft_preferences"]:
        grounding_needs.append("portability")
    if "giftable" in parsed["soft_preferences"]:
        grounding_needs.append("giftability")
    if resolved_reference is not None:
        grounding_needs.append("reference_comparison")
    if parsed["excluded_brands"]:
        grounding_needs.append("brand_constraint")
    parsed["grounding_needs"] = dedupe_preserve_order(grounding_needs)

    if (
        parsed["user_intent"] == "gift_search"
        and not parsed["domain_hint"]
        and not parsed["hard_constraints"]["must_include_terms"]
    ):
        parsed["clarification_needed"] = True
        parsed["clarification_questions"].append("What kind of gift or product category do you have in mind?")

    unresolved_references = [
        item for item in reference_items if item.get("resolution_status") == "unresolved"
    ]
    if unresolved_references:
        parsed["clarification_needed"] = True
        parsed["clarification_questions"].append("Can you name the reference item more explicitly?")

    if (
        parsed["user_intent"] == "open_search"
        and not parsed["hard_constraints"]["must_include_terms"]
        and not parsed["soft_preferences"]
        and not parsed["domain_hint"]
    ):
        parsed["clarification_needed"] = True
        parsed["clarification_questions"].append(
            "What kind of item or use case should the recommendation focus on?"
        )

    parsed["clarification_questions"] = dedupe_preserve_order(parsed["clarification_questions"])
    return parsed


def build_pipeline_handoff(parsed_request: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed request into the smaller handoff used by later stages."""
    resolved_reference = first_resolved_reference(parsed_request)

    lexical_terms: list[str] = []
    lexical_terms.extend(parsed_request["hard_constraints"]["must_include_terms"])
    if parsed_request["hard_constraints"]["use_case"]:
        lexical_terms.append(parsed_request["hard_constraints"]["use_case"])
    if "lightweight" in parsed_request["soft_preferences"]:
        lexical_terms.append("lightweight")
    if "highly_rated" in parsed_request["soft_preferences"]:
        lexical_terms.append("highly rated")
    if "giftable" in parsed_request["soft_preferences"]:
        lexical_terms.append("gift")
    lexical_terms = dedupe_preserve_order(lexical_terms)

    retrieval_mode = "reference_similarity" if resolved_reference is not None else "text_query"
    retrieval_query = None if retrieval_mode == "reference_similarity" else " ".join(lexical_terms)
    if retrieval_mode == "text_query" and not retrieval_query:
        retrieval_query = parsed_request["original_query"]

    return {
        "original_query": parsed_request["original_query"],
        "candidate_generation_mode": retrieval_mode,
        "reference_parent_asin": (resolved_reference["resolved_parent_asin"] if resolved_reference else None),
        "source_filter": parsed_request["domain_hint"],
        "same_source_only": parsed_request["hard_constraints"]["same_source_as_reference"],
        "retrieval_query": retrieval_query,
        "hard_filters": {
            "max_price": parsed_request["hard_constraints"]["max_price"],
            "min_rating": parsed_request["hard_constraints"]["min_rating"],
            "excluded_brands": parsed_request["excluded_brands"],
            "excluded_terms": parsed_request["excluded_terms"],
        },
        "soft_preferences": parsed_request["soft_preferences"],
        "grounding_needs": parsed_request["grounding_needs"],
        "clarification_needed": parsed_request["clarification_needed"],
        "clarification_questions": parsed_request["clarification_questions"],
    }


def flatten_parsed_output(label: str, parsed: dict[str, Any]) -> dict[str, Any]:
    """Flatten a parsed request for tabular inspection in notebooks."""
    first_reference = parsed["reference_items"][0] if parsed["reference_items"] else {}
    return {
        "label": label,
        "user_intent": parsed["user_intent"],
        "domain_hint": parsed["domain_hint"],
        "must_include_terms": ", ".join(parsed["hard_constraints"]["must_include_terms"]),
        "use_case": parsed["hard_constraints"]["use_case"],
        "max_price": parsed["hard_constraints"]["max_price"],
        "min_rating": parsed["hard_constraints"]["min_rating"],
        "cheaper_than_reference": parsed["hard_constraints"]["cheaper_than_reference"],
        "excluded_brands": ", ".join(map(str, parsed["excluded_brands"])),
        "soft_preferences": ", ".join(parsed["soft_preferences"]),
        "grounding_needs": ", ".join(parsed["grounding_needs"]),
        "reference_title": first_reference.get("resolved_title"),
        "clarification_needed": parsed["clarification_needed"],
    }


def normalize_expected_list(values: list[Any]) -> list[str]:
    """Normalize expected labels for notebook-side parser comparisons."""
    return [normalize_text(value) for value in values if normalize_text(value)]

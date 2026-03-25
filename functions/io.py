from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

import duckdb
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path
    baseline_dir: Path
    candidate_reranking_dir: Path
    rag_grounding_dir: Path
    curated_dir: Path
    items_path: Path
    reviews_path: Path
    candidate_pool_path: Path
    bm25_index_path: Path
    faiss_index_path: Path
    embeddings_path: Path
    index_metadata_path: Path
    scenario_suite_path: Path


def find_data_dir(require_baseline: bool = True) -> Path:
    """Locate the project data directory under either `../data` or `data`."""
    candidates = [Path("../data").resolve(), Path("data").resolve()]
    required = ["items.parquet", "reviews.parquet"]
    if require_baseline:
        required.append("baseline_retrieval/candidate_pool.parquet")

    for candidate in candidates:
        if all((candidate / rel_path).exists() for rel_path in required):
            return candidate

    missing = ", ".join(required)
    raise FileNotFoundError(f"Could not find project data directory containing: {missing}")


def get_project_paths(require_baseline: bool = True) -> ProjectPaths:
    """Build the canonical filesystem layout used by the project."""
    data_dir = find_data_dir(require_baseline=require_baseline)
    baseline_dir = data_dir / "baseline_retrieval"
    candidate_reranking_dir = data_dir / "candidate_reranking"
    rag_grounding_dir = data_dir / "rag_grounding"
    curated_dir = data_dir / "curated"

    return ProjectPaths(
        data_dir=data_dir,
        baseline_dir=baseline_dir,
        candidate_reranking_dir=candidate_reranking_dir,
        rag_grounding_dir=rag_grounding_dir,
        curated_dir=curated_dir,
        items_path=data_dir / "items.parquet",
        reviews_path=data_dir / "reviews.parquet",
        candidate_pool_path=baseline_dir / "candidate_pool.parquet",
        bm25_index_path=baseline_dir / "bm25_index.pkl",
        faiss_index_path=baseline_dir / "faiss_index.bin",
        embeddings_path=baseline_dir / "item_embeddings.npy",
        index_metadata_path=baseline_dir / "index_metadata.json",
        scenario_suite_path=curated_dir / "scenario_suite.json",
    )


def sql(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a pandas DataFrame."""
    return con.execute(dedent(query).strip()).fetchdf()


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> Any:
    """Execute a SQL query and return the first scalar value."""
    return con.execute(dedent(query).strip()).fetchone()[0]


def safe_preview(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    limit: int = 5,
    *,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    """Return a bounded preview of a DuckDB view or table."""
    if limit > max_rows:
        raise ValueError(f"Preview limit {limit} exceeds the {max_rows:,} row safety cap.")
    return sql(con, f"SELECT * FROM {view_name} LIMIT {limit}")


def cols_of(con: duckdb.DuckDBPyConnection, path: str | Path) -> set[str]:
    """Inspect a JSON source and return the available top-level columns."""
    rows = con.execute(f"DESCRIBE SELECT * FROM read_json_auto('{Path(path).as_posix()}')").fetchall()
    return {row[0] for row in rows}


def select_with_nulls(col_list: list[str] | tuple[str, ...] | Any, present_cols: set[str]) -> str:
    """Build a SELECT list that preserves missing columns as explicit NULLs."""
    return ", ".join(f"{col} AS {col}" if col in present_cols else f"NULL AS {col}" for col in col_list)


def connect_catalog_views(
    paths: ProjectPaths | None = None,
    include_reviews: bool = True,
    memory_limit: str = "8GB",
) -> duckdb.DuckDBPyConnection:
    """Create DuckDB views over the cleaned item and review parquet files."""
    paths = get_project_paths(require_baseline=False) if paths is None else paths
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(
        f"CREATE OR REPLACE VIEW items AS SELECT * FROM read_parquet('{paths.items_path.as_posix()}')"
    )
    if include_reviews:
        con.execute(
            f"CREATE OR REPLACE VIEW reviews AS SELECT * FROM read_parquet('{paths.reviews_path.as_posix()}')"
        )
    return con


def load_encoder(
    model_name: str = DEFAULT_ENCODER_MODEL,
    local_files_only: bool = True,
) -> SentenceTransformer:
    """Load the sentence-transformer encoder used for retrieval."""
    return SentenceTransformer(model_name, local_files_only=local_files_only)


def load_cross_encoder(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    local_files_only: bool = True,
) -> CrossEncoder:
    """Load the cross-encoder used to score grounding evidence."""
    return CrossEncoder(model_name, local_files_only=local_files_only)


def load_retrieval_artifacts(
    paths: ProjectPaths | None = None,
    load_encoder_model: bool = False,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the persisted retrieval artifacts and align their shared indexes."""
    paths = get_project_paths(require_baseline=True) if paths is None else paths

    candidate_pool = (
        pd.read_parquet(paths.candidate_pool_path).sort_values("retrieval_row_id").reset_index(drop=True)
    )

    with open(paths.bm25_index_path, "rb") as handle:
        bm25_index = pickle.load(handle)

    faiss_index = faiss.read_index(str(paths.faiss_index_path))
    item_embeddings = np.load(paths.embeddings_path, mmap_mode="r")
    index_metadata = json.loads(paths.index_metadata_path.read_text())

    artifacts = {
        "paths": paths,
        "candidate_pool": candidate_pool,
        "bm25_index": bm25_index,
        "faiss_index": faiss_index,
        "item_embeddings": item_embeddings,
        "index_metadata": index_metadata,
        "asin_to_row": dict(zip(candidate_pool["parent_asin"], candidate_pool.index)),
        "source_to_row_ids": {
            str(source).lower(): frame.index.to_numpy(dtype=np.int32)
            for source, frame in candidate_pool.groupby("source", sort=False)
        },
    }

    if load_encoder_model:
        encoder_model = index_metadata.get("encoder_model", DEFAULT_ENCODER_MODEL)
        artifacts["encoder"] = load_encoder(
            model_name=encoder_model,
            local_files_only=local_files_only,
        )

    return artifacts


def build_reference_item_from_row(
    row: pd.Series,
    mention: str = "this item",
    resolution_status: str = "from_context",
) -> dict[str, Any]:
    """Convert a catalog row into the reference-item schema used by parsing."""
    return {
        "mention": mention,
        "resolved_parent_asin": row.get("parent_asin"),
        "resolved_title": row.get("title"),
        "source": row.get("source"),
        "store": row.get("store"),
        "price": None if pd.isna(row.get("price")) else float(row.get("price")),
        "resolution_status": resolution_status,
    }


def _select_anchor_row(
    candidate_pool: pd.DataFrame,
    *,
    source: str,
    title_pattern: str,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating_number: int = 0,
) -> pd.Series:
    """Pick a strong anchor item for a scenario from a constrained source slice."""
    mask = candidate_pool["source"].eq(source)
    mask &= candidate_pool["title"].fillna("").str.contains(title_pattern, case=False, regex=True)
    if min_price is not None:
        mask &= candidate_pool["price"].fillna(-np.inf) >= min_price
    if max_price is not None:
        mask &= candidate_pool["price"].fillna(np.inf) <= max_price
    mask &= candidate_pool["rating_number"].fillna(0) >= min_rating_number

    matches = candidate_pool.loc[mask].sort_values(
        ["rating_number", "average_rating"],
        ascending=[False, False],
    )
    if matches.empty:
        matches = candidate_pool.loc[candidate_pool["source"].eq(source)].sort_values(
            ["rating_number", "average_rating"],
            ascending=[False, False],
        )
    if matches.empty:
        raise ValueError(f"Could not resolve anchor row for source={source}.")
    return matches.iloc[0]


def resolve_reference_strategy(strategy: str, candidate_pool: pd.DataFrame) -> dict[str, Any]:
    """Resolve a named evaluation strategy to a concrete reference item."""
    strategy = str(strategy)
    if strategy == "electronics_headphones_anchor":
        row = _select_anchor_row(
            candidate_pool,
            source="electronics",
            title_pattern=r"headphones|headset|earbuds",
            min_price=20.0,
            max_price=220.0,
            min_rating_number=100,
        )
    elif strategy == "home_cookware_anchor":
        row = _select_anchor_row(
            candidate_pool,
            source="home_and_kitchen",
            title_pattern=r"frying pan|skillet|cookware",
            min_price=15.0,
            max_price=150.0,
            min_rating_number=50,
        )
    elif strategy == "sports_dumbbells_anchor":
        row = _select_anchor_row(
            candidate_pool,
            source="sports_and_outdoors",
            title_pattern=r"dumbbell|weights",
            min_price=20.0,
            max_price=400.0,
            min_rating_number=25,
        )
    else:
        raise KeyError(f"Unknown reference strategy: {strategy}")

    return build_reference_item_from_row(row)


def load_curated_scenarios(paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Load the curated scenario suite used by demo and evaluation notebooks."""
    paths = get_project_paths(require_baseline=True) if paths is None else paths
    return json.loads(paths.scenario_suite_path.read_text())


def materialize_scenarios(
    raw_scenarios: list[dict[str, Any]],
    candidate_pool: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Attach concrete reference-item context to scenario templates when needed."""
    scenarios: list[dict[str, Any]] = []
    for scenario in raw_scenarios:
        enriched = dict(scenario)
        strategy = enriched.get("reference_strategy")
        enriched["reference_item_context"] = (
            resolve_reference_strategy(strategy, candidate_pool) if strategy else None
        )
        scenarios.append(enriched)
    return scenarios

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import shorten
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from functions.grounding import assemble_grounding_evidence
from functions.io import (
    build_reference_item_from_row,
    connect_catalog_views,
    load_cross_encoder,
    load_retrieval_artifacts,
)
from functions.parsing import build_parser_catalog, parse_user_request
from functions.response import build_final_response
from functions.reranking import rerank_candidates

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def discover_notebooks() -> list[Path]:
    """Return the project notebooks sorted by their numeric prefix."""
    return sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb"))


def run_notebook(notebook_path: Path, *, timeout: int) -> None:
    """Execute a notebook in place using the notebook's own directory as cwd."""
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)


def run_notebook_chain(start: int, end: int, timeout: int) -> int:
    """Execute the notebook sequence from the requested start to end index."""
    selected = [path for path in discover_notebooks() if start <= int(path.name[:2]) <= end]
    if not selected:
        print(f"No notebooks found in the requested range {start:02d} to {end:02d}.")
        return 1

    print("Executing notebooks:")
    for path in selected:
        print(f"  - {path.name}")

    for path in selected:
        print(f"\n[{path.name}] starting")
        try:
            run_notebook(path, timeout=timeout)
        except CellExecutionError as exc:
            print(f"[{path.name}] failed")
            print(exc)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{path.name}] failed with an unexpected error")
            print(exc)
            return 1
        print(f"[{path.name}] done")

    print("\nNotebook execution completed.")
    return 0


def fmt_money(value: Any) -> str:
    """Format a possibly-missing price value for terminal output."""
    if value is None:
        return "n/a"
    return f"${float(value):.2f}"


def fmt_rating(value: Any, count: Any) -> str:
    """Format rating metadata for terminal output."""
    if value is None:
        return "n/a"
    if count is None:
        return f"{float(value):.1f}"
    return f"{float(value):.1f} ({int(count)} ratings)"


class RecommendationChat:
    """Interactive terminal chat over the project recommendation pipeline."""

    def __init__(
        self,
        *,
        top_k_retrieval: int = 20,
        top_k_final: int = 3,
        top_review_evidence_per_item: int = 1,
    ) -> None:
        print("Loading retrieval artifacts and models...")
        self.artifacts = load_retrieval_artifacts(load_encoder_model=True)
        self.candidate_pool = self.artifacts["candidate_pool"]
        self.parser_catalog = build_parser_catalog(self.candidate_pool)
        self.con = connect_catalog_views(include_reviews=True)
        self.cross_encoder = load_cross_encoder()
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.top_review_evidence_per_item = top_review_evidence_per_item
        self.reference_context: dict[str, Any] | None = None
        self.last_reranked = None
        print("Chat ready. Type /help for commands.")

    def reset(self) -> None:
        """Reset conversational reference state."""
        self.reference_context = None
        self.last_reranked = None

    def set_reference_from_rank(self, rank: int) -> bool:
        """Select one of the last finalists as the current reference item."""
        if self.last_reranked is None or self.last_reranked.empty:
            return False
        if rank < 1 or rank > len(self.last_reranked):
            return False
        row = self.last_reranked.iloc[rank - 1]
        self.reference_context = build_reference_item_from_row(
            row,
            mention="this item",
            resolution_status="from_context",
        )
        return True

    def current_reference_label(self) -> str | None:
        """Return a human-readable description of the active reference context."""
        if not self.reference_context:
            return None
        title = self.reference_context.get("resolved_title") or "unknown item"
        store = self.reference_context.get("store")
        return f"{title} [{store}]" if store else str(title)

    def ask(self, query: str) -> dict[str, Any]:
        """Run the full recommendation pipeline for a single user query."""
        parsed = parse_user_request(
            query,
            self.parser_catalog,
            reference_item_context=self.reference_context,
        )
        rerank_result = rerank_candidates(
            parsed,
            self.artifacts,
            top_k_retrieval=self.top_k_retrieval,
            top_k_final=self.top_k_final,
        )
        evidence = assemble_grounding_evidence(
            parsed,
            rerank_result["reranked_candidates"],
            self.con,
            self.candidate_pool,
            self.cross_encoder,
            top_review_evidence_per_item=self.top_review_evidence_per_item,
        )
        final_response = build_final_response(
            parsed,
            rerank_result["baseline_candidates"],
            rerank_result["reranked_candidates"],
            evidence,
        )

        self.last_reranked = rerank_result["reranked_candidates"]
        if self.last_reranked is not None and not self.last_reranked.empty:
            self.reference_context = build_reference_item_from_row(
                self.last_reranked.iloc[0],
                mention="this item",
                resolution_status="from_context",
            )

        return {
            "parsed_request": parsed,
            "rerank_result": rerank_result,
            "evidence": evidence,
            "final_response": final_response,
        }


def print_chat_help() -> None:
    """Print the available interactive chat commands."""
    print(
        "\nCommands:\n"
        "  /help       Show this help message\n"
        "  /reset      Clear the current reference context\n"
        "  /context    Show the active reference item\n"
        "  /use N      Use finalist N from the last answer as the reference item\n"
        "  /quit       Exit the chat\n"
    )


def render_response(result: dict[str, Any], *, show_json: bool = False) -> None:
    """Render a structured recommendation response for terminal use."""
    response = result["final_response"]
    clarification = response["clarification"]

    print()
    print(response["explanation_text"])

    if clarification["needed"] and clarification["questions"]:
        print("\nClarification:")
        for question in clarification["questions"]:
            print(f"  - {question}")

    finalists = response["reranked_finalists"]
    if finalists:
        print("\nFinalists:")
        for idx, finalist in enumerate(finalists, start=1):
            title = finalist.get("title") or "unknown item"
            store = finalist.get("store") or "unknown store"
            price = fmt_money(finalist.get("price"))
            rating = fmt_rating(finalist.get("average_rating"), finalist.get("rating_number"))
            print(f"  {idx}. {title}")
            print(f"     store={store} price={price} rating={rating}")

    evidence_rows = response["supporting_evidence"]
    if evidence_rows:
        print("\nEvidence:")
        for evidence in evidence_rows[:4]:
            topic = evidence.get("matched_topic") or "general_support"
            snippet = shorten(str(evidence.get("evidence_text") or ""), width=160, placeholder="...")
            print(f"  - {topic}: {snippet}")

    if response["tradeoff_notes"]:
        print("\nTrade-offs:")
        for note in response["tradeoff_notes"]:
            print(f"  - {note}")

    if show_json:
        print("\nRaw response JSON:")
        print(json.dumps(response, indent=2, ensure_ascii=False))


def run_chat(args: argparse.Namespace) -> int:
    """Start the interactive recommendation chat loop."""
    chat = RecommendationChat(
        top_k_retrieval=args.top_k_retrieval,
        top_k_final=args.top_k_final,
        top_review_evidence_per_item=args.top_review_evidence_per_item,
    )
    print_chat_help()

    while True:
        try:
            raw = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            return 0

        if not raw:
            continue
        if raw in {"/quit", "/exit"}:
            print("Exiting chat.")
            return 0
        if raw == "/help":
            print_chat_help()
            continue
        if raw == "/reset":
            chat.reset()
            print("Reference context cleared.")
            continue
        if raw == "/context":
            label = chat.current_reference_label()
            print(label or "No active reference item.")
            continue
        if raw.startswith("/use "):
            try:
                rank = int(raw.split(maxsplit=1)[1])
            except ValueError:
                print("Usage: /use N")
                continue
            if chat.set_reference_from_rank(rank):
                print(f"Reference item set to finalist {rank}.")
            else:
                print("Could not set the reference item from that finalist rank.")
            continue

        result = chat.ask(raw)
        render_response(result, show_json=args.show_json)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for notebook execution and chat."""
    parser = argparse.ArgumentParser(
        description="CLI entry point for the grounded recommender project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    notebooks_parser = subparsers.add_parser(
        "notebooks",
        help="Execute the project notebooks from 00 to 07.",
    )
    notebooks_parser.add_argument("--start", type=int, default=0, help="First notebook index to run.")
    notebooks_parser.add_argument("--end", type=int, default=7, help="Last notebook index to run.")
    notebooks_parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-notebook execution timeout in seconds.",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive terminal chat over the recommendation pipeline.",
    )
    chat_parser.add_argument("--top-k-retrieval", type=int, default=20)
    chat_parser.add_argument("--top-k-final", type=int, default=3)
    chat_parser.add_argument("--top-review-evidence-per-item", type=int, default=1)
    chat_parser.add_argument("--show-json", action="store_true")

    return parser


def main() -> int:
    """Dispatch the selected CLI subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "notebooks":
        return run_notebook_chain(args.start, args.end, args.timeout)
    if args.command == "chat":
        return run_chat(args)

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

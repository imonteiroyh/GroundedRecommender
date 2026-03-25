# Conversational Recommendation Pipeline with Grounded Reranking

This project studies how to add a conversational layer to a product recommender system without handing the full decision process to an LLM.

The core idea is simple:

- the user speaks in natural language
- a parser converts that request into structured constraints and preferences
- a retrieval layer finds plausible candidates
- a grounding layer attaches metadata and review evidence to those candidates
- a reranker combines retrieval, constraints, price/rating signals, and reference-item similarity
- a response layer produces an explainable recommendation package

The repository is organized as an end-to-end experimental pipeline, not as an agent that improvises recommendations from scratch.

## Results Summary

The current results show that the system already behaves like a coherent conversational recommender rather than a disconnected collection of components. Across a set of curated product-search scenarios, the pipeline consistently retrieves plausible candidates, reduces them to a small final shortlist, and attaches supporting evidence to the response. It also handles ambiguity in a controlled way: when the request is underspecified, the system can surface clarification needs instead of forcing an overconfident recommendation.

The strongest outcome is not just that recommendations are returned, but that the final shortlist changes in meaningful ways after reranking. Once explicit constraints and preferences are applied, candidates can move substantially from their original retrieval positions, which shows that the system is doing more than rephrasing the baseline search results. The final responses are also grounded in concrete signals such as price, rating, use-case fit, and similarity to reference items, making the recommendations easier to inspect and defend.

Evaluation results reinforce that pattern. Parsing is currently the most reliable stage, with the structured interpretation of user intent matching the expected behavior in the curated checks. Reranking is also stable, consistently preserving hard constraints in the final candidates, while grounding provides evidence that is specific enough to support the generated explanations. Retrieval is the weakest stage at the moment: it performs well overall, but still shows limitations in some item-to-item cases, which makes it the clearest area for future improvement.

## Why This Design

This repository is trying to answer a specific question:

**How much conversational flexibility can we add to recommender systems while still keeping ranking behavior explicit, auditable, and testable?**

Because of that, the project avoids a fully opaque "LLM decides everything" approach. The LLM-facing part is mainly used as an interpretation and explanation layer around a more classical retrieval-and-ranking pipeline.

## What The Project Does

In practical terms, the system follows this order:

1. **Prepare the data**
   `00_data_acquisition.ipynb` downloads and organizes the raw data into clean item and review tables.
2. **Inspect the dataset**
   `01_catalog_and_reviews_eda.ipynb` checks catalog coverage, review quality, and grounding readiness.
3. **Build the retrieval layer**
   `02_baseline_retrieval.ipynb` creates the candidate pool, BM25 index, FAISS index, embeddings, and retrieval metadata.
4. **Interpret the user request**
   `03_preference_parsing.ipynb` converts natural language into a structured representation with budget, preferences, exclusions, intended use, and reference items.
5. **Collect evidence to support recommendations**
   `04_rag_grounding.ipynb` retrieves metadata and review evidence and scores the relevance of that evidence.
6. **Rerank the candidates**
   `05_candidate_reranking.ipynb` applies hard constraints and combines explicit ranking signals.
7. **Run the full pipeline**
   `06_conversational_recommendation_demo.ipynb` connects parsing, retrieval, grounding, reranking, and final response generation.
8. **Evaluate system behavior**
   `07_evaluation.ipynb` measures retrieval, parsing, reranking, and grounding performance, and analyzes failure cases.

In summary, the operational flow is:

`user request -> parsing -> retrieval -> grounding -> reranking -> response package`

If you want to see the system working first, the best entry point is `06_conversational_recommendation_demo.ipynb`. If you want to follow the full build step by step, start from `00`.

## CLI Usage

The repository also includes a simple command-line entry point in `main.py`.

To run the full notebook sequence from `00` to `07`:

```bash
python main.py notebooks
```

To start the terminal chat over the recommendation pipeline:

```bash
python main.py chat
```

## Next Steps

If someone were to continue this project from here, the first practical extension would be to scale the catalog beyond the local sample that was used for safety and notebook feasibility. The current retrieval stage is already strong and coherent on the evaluated scenarios, but the project was built under a local processing constraint: there were simply too many items to handle comfortably end to end on a full snapshot. With a larger working sample, or with a more scalable execution setup, a richer and more diverse recommendation space would be expected.

The next major direction would be to make parsing and grounding less rigid while still keeping the downstream ranking explicit. The current parser is useful because it produces a stable contract, but it is still deterministic and therefore narrow in the range of requests it can absorb gracefully. A stronger version would be able to capture softer preferences and optional desirables, not just hard needs. The grounding layer could also react more directly to the user query itself and surface signals such as “waterproof would be nice to have” even when that preference is not a strict requirement. Once extracted, those additional signals could flow explicitly into reranking rather than staying buried in explanation-only text.

The chat interface is also only a minimal proof of interaction. It works, but it is slow and far from an ideal user experience. There is clear room for lower-latency execution, a more iterative conversational design, and a richer handoff layer between recommendation logic and user-facing language. One natural direction would be to place an LLM in front of the recommendation tool as an agent that collects missing context, calls the pipeline when needed, and rewrites the structured output into a more natural response. That was not implemented here mainly because the realistic options were either running models locally, with the associated compute cost, or calling external APIs, with their own monetary cost.

## Reference

Hou, Yupeng et al. "Bridging Language and Items for Retrieval and Recommendation." *arXiv preprint arXiv:2403.03952* (2024).

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import (
    load_hybrid_retriever,
    search as hybrid_search,
    tokenize,
)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scifact" / "scifact"
QRELS_PATH = DATA_DIR / "qrels" / "test.tsv"
QUERIES_PATH = DATA_DIR / "queries.jsonl"

TOP_K = 5


def calculate_metrics(retrieved_ids, relevant_ids):
    """Calculate Recall@5, MRR@5, and nDCG@5 for one query."""
    retrieved_ids = [str(doc_id) for doc_id in retrieved_ids]
    relevant_ids = set(str(doc_id) for doc_id in relevant_ids)

    hits = [
        1 if doc_id in relevant_ids else 0
        for doc_id in retrieved_ids[:TOP_K]
    ]

    recall = (
        sum(hits) / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    reciprocal_rank = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / rank
            break

    ideal_hits = min(len(relevant_ids), TOP_K)
    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    actual_dcg = sum(
        hit / np.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )

    ndcg = actual_dcg / ideal_dcg if ideal_dcg else 0.0

    return recall, reciprocal_rank, ndcg


def main():
    print("Loading evaluation data...")

    queries = {}
    with open(QUERIES_PATH, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            queries[str(record["_id"])] = record["text"]

    qrels = pd.read_csv(QRELS_PATH, sep="\t", dtype=str)
    qrels = qrels[qrels["score"].astype(int) > 0]

    relevant_by_query = (
        qrels.groupby("query-id")["corpus-id"]
        .apply(list)
        .to_dict()
    )

    query_ids = sorted(
        set(relevant_by_query) & set(queries),
        key=lambda value: int(value),
    )

    print("Queries to evaluate:", len(query_ids))
    print("Loading retrieval systems...")

    bm25, model, documents, index = load_hybrid_retriever()

    document_ids = [str(doc["id"]) for doc in documents]
    dense_results = []
    hybrid_results = []

    print("\nEvaluating retrieval methods...")

    for count, query_id in enumerate(query_ids, start=1):
        query = queries[query_id]
        relevant_ids = relevant_by_query[query_id]

        # Dense retrieval: rank all indexed documents by similarity.
        embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        _, indices = index.search(embedding, TOP_K)
        dense_ids = [
            document_ids[int(idx)]
            for idx in indices[0]
            if idx >= 0
        ]

        dense_results.append(
            calculate_metrics(dense_ids, relevant_ids)
        )

        # Hybrid retrieval: use the existing RRF implementation.
        results = hybrid_search(
            query,
            bm25,
            model,
            documents,
            index,
            top_k=TOP_K,
        )
        hybrid_ids = [str(result["id"]) for result in results]

        hybrid_results.append(
            calculate_metrics(hybrid_ids, relevant_ids)
        )

        if count % 50 == 0:
            print(f"Evaluated {count}/{len(query_ids)} queries")

    methods = {
        "Dense": dense_results,
        "Hybrid (RRF)": hybrid_results,
    }

    summary = []

    for method, values in methods.items():
        metrics = np.array(values)

        summary.append({
            "Method": method,
            "Recall@5": metrics[:, 0].mean(),
            "MRR@5": metrics[:, 1].mean(),
            "nDCG@5": metrics[:, 2].mean(),
        })

    results_df = pd.DataFrame(summary)

    print("\n=== DENSE VS HYBRID RESULTS ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "dense_vs_hybrid_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()

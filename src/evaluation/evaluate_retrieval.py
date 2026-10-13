
from pathlib import Path
import csv
import json
from collections import defaultdict

import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import (
    load_hybrid_retriever,
    search,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scifact" / "scifact"
QRELS_PATH = DATA_DIR / "qrels" / "test.tsv"
QUERIES_PATH = DATA_DIR / "queries.jsonl"

TOP_K = 5


def load_qrels():
    relevance = defaultdict(set)

    with open(QRELS_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            if int(row["score"]) > 0:
                relevance[str(row["query-id"])].add(
                    str(row["corpus-id"])
                )

    return relevance


def load_queries():
    queries = {}

    with open(QUERIES_PATH, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            queries[str(item["_id"])] = item["text"]

    return queries


def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0

    hits = len(set(retrieved_ids[:TOP_K]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank_at_k(retrieved_ids, relevant_ids):
    for rank, document_id in enumerate(
        retrieved_ids[:TOP_K], start=1
    ):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids):
    dcg = 0.0

    for rank, document_id in enumerate(
        retrieved_ids[:TOP_K], start=1
    ):
        if document_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), TOP_K)

    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


def main():
    queries = load_queries()
    relevance = load_qrels()

    bm25, model, documents, index = load_hybrid_retriever()

    query_ids = sorted(relevance.keys(), key=int)

    recall_scores = []
    reciprocal_rank_scores = []
    ndcg_scores = []

    print(f"\nEvaluating hybrid retrieval on {len(query_ids)} queries...")

    for query_id in query_ids:
        results = search(
            queries[query_id],
            bm25,
            model,
            documents,
            index,
            top_k=TOP_K,
        )

        retrieved_ids = [result["id"] for result in results]
        relevant_ids = relevance[query_id]

        recall_scores.append(
            recall_at_k(retrieved_ids, relevant_ids)
        )

        reciprocal_rank_scores.append(
            reciprocal_rank_at_k(retrieved_ids, relevant_ids)
        )

        ndcg_scores.append(
            ndcg_at_k(retrieved_ids, relevant_ids)
        )

    print("\n=== HYBRID RETRIEVAL EVALUATION ===")
    print("Queries evaluated:", len(query_ids))
    print(f"Recall@{TOP_K}: {np.mean(recall_scores):.4f}")
    print(f"MRR@{TOP_K}:    {np.mean(reciprocal_rank_scores):.4f}")
    print(f"nDCG@{TOP_K}:   {np.mean(ndcg_scores):.4f}")


if __name__ == "__main__":
    main()
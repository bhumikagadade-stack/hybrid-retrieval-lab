
from pathlib import Path
import csv
import json
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import (
    load_hybrid_retriever,
    search,
)


DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scifact" / "scifact"
QRELS_PATH = DATA_DIR / "qrels" / "test.tsv"
QUERIES_PATH = DATA_DIR / "queries.jsonl"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hybrid_failure_analysis.csv"
)

TOP_K = 5


def load_queries():
    queries = {}

    with open(QUERIES_PATH, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            queries[str(item["_id"])] = item["text"]

    return queries


def load_relevance_judgments():
    relevance = defaultdict(set)

    with open(QRELS_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            if int(row["score"]) > 0:
                relevance[str(row["query-id"])].add(
                    str(row["corpus-id"])
                )

    return relevance


def main():
    print("Loading queries and relevance judgments...")

    queries = load_queries()
    relevance = load_relevance_judgments()

    bm25, model, documents, index = load_hybrid_retriever()

    failures = []
    partial_hits = []

    query_ids = sorted(relevance.keys(), key=int)

    print(f"\nAnalyzing {len(query_ids)} queries...")

    for query_id in query_ids:
        query_text = queries[query_id]
        relevant_ids = relevance[query_id]

        results = search(
            query_text,
            bm25,
            model,
            documents,
            index,
            top_k=TOP_K,
        )

        retrieved_ids = {result["id"] for result in results}
        hits = retrieved_ids.intersection(relevant_ids)

        if not hits:
            failures.append({
                "query_id": query_id,
                "query": query_text,
                "relevant_document_ids": ";".join(
                    sorted(relevant_ids)
                ),
                "retrieved_document_ids": ";".join(
                    result["id"] for result in results
                ),
                "retrieved_titles": " | ".join(
                    result["title"] for result in results
                ),
                "failure_type": "No relevant result in top 5",
            })

        elif len(hits) < len(relevant_ids):
            partial_hits.append(query_id)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id",
        "query",
        "relevant_document_ids",
        "retrieved_document_ids",
        "retrieved_titles",
        "failure_type",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)

    print("\n=== HYBRID RETRIEVAL ERROR ANALYSIS ===")
    print("Queries evaluated:", len(query_ids))
    print("Queries with no relevant result in top 5:", len(failures))
    print("Queries with at least one relevant result:", len(query_ids) - len(failures))
    print("Queries with some, but not all, relevant documents retrieved:", len(partial_hits))
    print("\nFailure report saved to:")
    print(OUTPUT_PATH)

    print("\n=== FIRST 5 FAILURES ===")

    for failure in failures[:5]:
        print("\nQuery ID:", failure["query_id"])
        print("Query:", failure["query"])
        print("Relevant document IDs:", failure["relevant_document_ids"])
        print("Retrieved titles:", failure["retrieved_titles"])
        print("-" * 70)


if __name__ == "__main__":
    main()
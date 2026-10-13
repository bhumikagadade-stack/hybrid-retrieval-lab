
from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scifact"
    / "scifact"
    / "corpus.jsonl"
)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def load_documents() -> tuple[list[str], list[str]]:
    """Load document IDs and title-plus-text content."""
    document_ids = []
    documents = []

    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            document_ids.append(str(record["_id"]))

            title = record.get("title", "")
            text = record.get("text", "")
            documents.append(f"{title} {text}".strip())

    return document_ids, documents


if __name__ == "__main__":
    document_ids, documents = load_documents()

    print("Preparing tokens...")
    tokenized_documents = [tokenize(doc) for doc in documents]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_documents)

    query = "0-dimensional biomaterials show inductive properties"
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)
        
    relevant_document_id = "31715818"

    relevant_index = document_ids.index(relevant_document_id)
    relevant_score = scores[relevant_index]

    # Rank 1 means the document has the highest score.
    relevant_rank = int((scores > relevant_score).sum()) + 1

    print("\n=== RELEVANT DOCUMENT DIAGNOSTICS ===")
    print("Relevant document ID:", relevant_document_id)
    print(f"BM25 score: {relevant_score:.4f}")
    print("Rank among all documents:", relevant_rank)
    print("Total documents searched:", len(documents))
    top_indices = scores.argsort()[::-1][:5]

    
    relevant_document_id = "31715818"

    retrieved_document_ids = [
        document_ids[index] for index in top_indices
]

    print("\nGround-truth relevant document:", relevant_document_id)
    print("Relevant document found in top 5:", relevant_document_id in retrieved_document_ids)

    print(f"\nSearch query: {query}")
    print("\nTop 5 results:")

    for rank, index in enumerate(top_indices, start=1):
        print(f"\nRank: {rank}")
        print(f"Document ID: {document_ids[index]}")
        print(f"BM25 score: {scores[index]:.4f}")
        print(f"Text: {documents[index][:250]}...")
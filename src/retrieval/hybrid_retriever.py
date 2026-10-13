
from pathlib import Path
import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scifact" / "scifact"
INDEX_DIR = PROJECT_ROOT / "indexes"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
DOCUMENTS_PATH = INDEX_DIR / "documents.json"
FAISS_INDEX_PATH = INDEX_DIR / "scifact.faiss"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RRF_K = 60
CANDIDATE_K = 100
FINAL_K = 5


# --------------------------------------------------
# 2. Tokenization for BM25
# --------------------------------------------------

def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# 3. Load the retrieval systems
# --------------------------------------------------

def load_hybrid_retriever():
    print("Loading scientific documents...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        corpus = [json.loads(line) for line in file]

    # BM25 searches document titles and full text.
    tokenized_corpus = [
        tokenize(doc["title"] + " " + doc["text"])
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # Document IDs and titles must match FAISS vector positions.
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    if len(documents) != index.ntotal:
        raise ValueError(
            "The document list and FAISS index have different sizes."
        )

    if len(corpus) != len(documents):
        raise ValueError(
            "The corpus and indexed document list have different sizes."
        )

    # Check that BM25 and FAISS use the same document order.
    for corpus_doc, indexed_doc in zip(corpus, documents):
        if str(corpus_doc["_id"]) != str(indexed_doc["id"]):
            raise ValueError(
                "Document order mismatch between corpus and FAISS data."
            )

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Documents loaded:", len(documents))
    print("FAISS vectors:", index.ntotal)
    print("Hybrid retriever is ready.")

    return bm25, model, documents, index


# --------------------------------------------------
# 4. Retrieve candidates using both methods
# --------------------------------------------------

def search(query, bm25, model, documents, index, top_k=FINAL_K):
    if not query.strip():
        return []

    candidate_k = min(CANDIDATE_K, len(documents))

    # BM25 ranking
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

    bm25_ranks = {
        int(doc_idx): rank
        for rank, doc_idx in enumerate(bm25_indices, start=1)
    }

    # Dense semantic ranking
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    _, dense_indices = index.search(query_embedding, candidate_k)

    dense_ranks = {
        int(doc_idx): rank
        for rank, doc_idx in enumerate(dense_indices[0], start=1)
        if doc_idx != -1
    }

    # Reciprocal Rank Fusion
    candidate_indices = set(bm25_ranks) | set(dense_ranks)
    fused_results = []

    for doc_idx in candidate_indices:
        score = 0.0

        if doc_idx in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[doc_idx])

        if doc_idx in dense_ranks:
            score += 1.0 / (RRF_K + dense_ranks[doc_idx])

        fused_results.append({
            "id": documents[doc_idx]["id"],
            "title": documents[doc_idx]["title"],
            "score": score,
            "bm25_rank": bm25_ranks.get(doc_idx),
            "dense_rank": dense_ranks.get(doc_idx),
        })

    fused_results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    # Return the highest-ranked hybrid results.
    results = []

    for rank, result in enumerate(fused_results[:top_k], start=1):
        results.append({
            "rank": rank,
            **result
        })

    return results


# --------------------------------------------------
# 5. Run a sample search
# --------------------------------------------------

if __name__ == "__main__":
    bm25, model, documents, index = load_hybrid_retriever()

    query = "0-dimensional biomaterials show inductive properties"

    print("\nSearch query:", query)
    print("\n=== HYBRID SEARCH RESULTS ===")

    results = search(
        query,
        bm25,
        model,
        documents,
        index,
        top_k=FINAL_K
    )

    for result in results:
        print(
            f"\nRank {result['rank']} | "
            f"RRF score: {result['score']:.5f}"
        )
        print("Document ID:", result["id"])
        print("Title:", result["title"])
        print("BM25 rank:", result["bm25_rank"])
        print("Dense rank:", result["dense_rank"])
        print("-" * 70)
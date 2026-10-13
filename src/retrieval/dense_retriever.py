
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "indexes"

DOCUMENTS_PATH = INDEX_DIR / "documents.json"
FAISS_INDEX_PATH = INDEX_DIR / "scifact.faiss"

TARGET_DOCUMENT_ID = "31715818"
QUERY = "0-dimensional biomaterials show inductive properties"


def load_retriever():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    return model, documents, index


def search(query, model, documents, index, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "rank": len(results) + 1,
            "id": documents[idx]["id"],
            "title": documents[idx]["title"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":
    model, documents, index = load_retriever()

    print("Search query:", QUERY)

    # Search the full index to find the target paper's rank
    results = search(
        QUERY, model, documents, index, top_k=index.ntotal
    )

    print("\nTop 5 results:\n")

    for result in results[:5]:
        print(
            f"Rank {result['rank']} | "
            f"ID: {result['id']} | "
            f"Score: {result['score']:.4f}"
        )
        print("Title:", result["title"])
        print("-" * 70)

    target_result = next(
        (result for result in results
         if result["id"] == TARGET_DOCUMENT_ID),
        None
    )

    print("\n=== RELEVANT DOCUMENT DIAGNOSTIC ===")

    if target_result:
        print("Target document ID:", TARGET_DOCUMENT_ID)
        print("Rank:", target_result["rank"])
        print("Cosine similarity:", round(target_result["score"], 4))
        print("Found in top 5:", target_result["rank"] <= 5)
    else:
        print("Target document not found in the index.")
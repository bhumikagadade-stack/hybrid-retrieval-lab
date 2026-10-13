
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import load_hybrid_retriever, tokenize

QUERY = "0-dimensional biomaterials show inductive properties."
TARGET_ID = "31715818"

# Load the existing retrieval components.
bm25, model, documents, index = load_hybrid_retriever()

# Calculate the target paper's BM25 rank.
query_tokens = tokenize(QUERY)
bm25_scores = bm25.get_scores(query_tokens)

bm25_target_index = next(
    i for i, doc in enumerate(documents)
    if str(doc["id"]) == TARGET_ID
)

bm25_target_score = float(bm25_scores[bm25_target_index])
bm25_rank = int((bm25_scores > bm25_target_score).sum()) + 1

# Calculate the target paper's dense retrieval rank.
query_embedding = model.encode(
    [QUERY],
    normalize_embeddings=True
).astype("float32")

dense_scores, dense_indices = index.search(
    query_embedding, len(documents)
)

dense_rank = next(
    (
        rank
        for rank, idx in enumerate(dense_indices[0], start=1)
        if int(idx) >= 0
        and str(documents[int(idx)]["id"]) == TARGET_ID
    ),
    None,
)

print("\n=== RETRIEVAL DIAGNOSTIC ===")
print("Query:", QUERY)
print("Target document ID:", TARGET_ID)
print("BM25 score:", round(bm25_target_score, 6))
print("BM25 rank:", bm25_rank)
print("Dense retrieval rank:", dense_rank)
print("Total documents:", len(documents))

from src.retrieval.hybrid_retriever import search

hybrid_results = search(
    QUERY,
    bm25,
    model,
    documents,
    index,
    top_k=5,
)

print("\n=== HYBRID TOP 5 ===")

for result in hybrid_results:
    print(
        f"Rank: {result['rank']} | "
        f"ID: {result['id']} | "
        f"Title: {result['title']} | "
        f"BM25 rank: {result['bm25_rank']} | "
        f"Dense rank: {result['dense_rank']}"
    )

print(
    "\nTarget paper in hybrid top 5:",
    any(str(result["id"]) == TARGET_ID for result in hybrid_results),
)

# Compare the target paper's RRF score with the hybrid top 5.
target_dense_rank = dense_rank

target_rrf_score = 0.0

if target_dense_rank is not None and target_dense_rank <= 100:
    target_rrf_score += 1.0 / (60 + target_dense_rank)

if bm25_rank <= 100:
    target_rrf_score += 1.0 / (60 + bm25_rank)

print("\n=== RRF SCORE COMPARISON ===")
print(f"Target paper RRF score: {target_rrf_score:.6f}")

for result in hybrid_results:
    print(
        f"ID: {result['id']} | "
        f"RRF score: {result['score']:.6f}"
    )

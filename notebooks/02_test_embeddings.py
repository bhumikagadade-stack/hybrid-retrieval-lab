
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The patient suffered a heart attack.",
    "The patient experienced a myocardial infarction.",
    "The weather is sunny today."
]

embeddings = model.encode(sentences)

# Measure the similarity between every pair of sentences
similarity_matrix = cosine_similarity(embeddings)

print("Cosine Similarity Matrix:\n")
print(similarity_matrix)

print("\nComparison of sentence pairs:")
print("Heart attack vs. myocardial infarction:",
      round(similarity_matrix[0][1], 3))

print("Heart attack vs. sunny weather:",
      round(similarity_matrix[0][2], 3))
# Retrieval Experiments

## 1. Dense Retrieval vs. Hybrid Retrieval

### Objective

Compare dense semantic retrieval against hybrid retrieval combining BM25 keyword search and dense retrieval using Reciprocal Rank Fusion (RRF).

### Dataset

* **Dataset:** SciFact
* **Evaluation queries:** 300
* **Documents:** 5,183
* **Evaluation depth:** Top 5 results
* **Metrics:** Recall@5, MRR@5, nDCG@5

### Results

| Method                 | Recall@5 |  MRR@5 | nDCG@5 |
| ---------------------- | -------: | -----: | -----: |
| Dense Retrieval        |   0.7379 | 0.5997 | 0.6293 |
| Hybrid Retrieval (RRF) |   0.7395 | 0.6375 | 0.6579 |

### Observations

* Hybrid retrieval achieved a small improvement in Recall@5.
* MRR@5 improved from 0.5997 to 0.6375, indicating better ranking of the first relevant result on average.
* nDCG@5 improved from 0.6293 to 0.6579, indicating better ranking quality near the top of the results list.
* These aggregate results do not imply that hybrid retrieval improves every individual query.

### Error Analysis

A diagnostic query, "0-dimensional biomaterials show inductive properties.", exposed a limitation in the hybrid ranking.

The known relevant document (ID `31715818`) ranked fifth in dense retrieval but outside BM25's top 100 candidates. It therefore received only a dense contribution to its RRF score and did not appear in the hybrid top five.

This example illustrates why aggregate evaluation should be complemented by query-level failure analysis.

### Limitations

* SciFact relevance judgments may be incomplete.
* Results depend on the embedding model, candidate cutoff, and RRF configuration.
* Further experiments are needed before concluding that one retrieval configuration is best for all queries.

### Reproducibility

Run the dense-versus-hybrid evaluation with:

```bash
python src/evaluation/compare_dense_hybrid.py
```

The results are saved to `data/processed/dense_vs_hybrid_results.csv`.


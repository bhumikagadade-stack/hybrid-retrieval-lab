# Hybrid Retrieval Lab: Dense vs. Hybrid Search

## Overview

Hybrid Retrieval Lab is an information retrieval project that compares **semantic search, keyword-based search, and hybrid search** on the SciFact scientific research dataset.

The project combines **BM25 keyword retrieval** with **dense vector retrieval** using Sentence Transformers. It then merges their rankings using Reciprocal Rank Fusion (RRF) to investigate whether combining different retrieval methods improves the relevance of retrieved documents.

The goal is to build a reproducible retrieval pipeline, evaluate its performance using standard information retrieval metrics, and analyze cases where retrieval methods succeed or fail.

## Key Results

Evaluation was performed on 300 queries using the SciFact test relevance judgments, with the top 5 retrieved documents evaluated for each query.

| Retrieval Method | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| Dense Retrieval | 0.7379 | 0.5997 | 0.6293 |
| BM25 Retrieval | 0.7284 | 0.6132 | 0.6360 |
| **Hybrid Retrieval (RRF)** | **0.7395** | **0.6375** | **0.6579** |

**Key finding:** Hybrid retrieval achieved the strongest overall results across the three reported metrics. Compared with dense retrieval, it produced a modest improvement in Recall@5 and clearer improvements in MRR@5 and nDCG@5, indicating better ranking of relevant documents near the top of the results.

These results apply to this dataset and evaluation setup; they do not establish that hybrid retrieval will outperform other methods in every setting.

## How It Works

1. **BM25 keyword retrieval:** Finds documents based on lexical overlap between a query and document text.
2. **Dense retrieval:** Encodes queries and documents into numerical vectors using `sentence-transformers/all-MiniLM-L6-v2`, allowing semantically similar text to match even when wording differs.
3. **Vector search:** Uses FAISS with normalized embeddings and inner-product similarity for efficient nearest-neighbor retrieval.
4. **Hybrid ranking:** Combines the two ranked candidate lists using Reciprocal Rank Fusion (RRF).
5. **Evaluation:** Compares retrieval quality using Recall@5, Mean Reciprocal Rank (MRR@5), and normalized Discounted Cumulative Gain (nDCG@5).
6. **Failure analysis:** Investigates queries where relevant documents are not retrieved in the top five.

## Dataset

This project uses the [SciFact dataset](https://github.com/allenai/scifact), a scientific claim verification dataset containing scientific abstracts, claims, and relevance judgments.

- **Corpus documents:** 5,183
- **Evaluation queries:** 300 unique query IDs
- **Retrieval depth:** Top 5 results
- **Evaluation labels:** SciFact test relevance judgments

The evaluation labels may not cover every document that could be relevant to a query, so some apparently unsuccessful retrievals may reflect incomplete judgments.

## Tech Stack

- **Language:** Python
- **Keyword retrieval:** BM25 (`rank-bm25`)
- **Semantic embeddings:** Sentence Transformers
- **Vector search:** FAISS
- **Data analysis:** NumPy, Pandas
- **Evaluation:** Recall@K, MRR@K, nDCG@K
- **Development:** VS Code, Jupyter Notebook, Git, GitHub

## Project Structure

```text
hybrid-retrieval-lab/
├── data/
│   ├── raw/                  # Downloaded dataset (not tracked by Git)
│   └── processed/            # Generated evaluation results
├── indexes/                  # Generated vector index and embeddings
├── models/                   # Saved model artifacts, if applicable
├── notebooks/                # Exploration and experimentation
├── src/
│   ├── retrieval/            # BM25, dense and hybrid retrieval
│   ├── evaluation/           # Evaluation and failure analysis
│   └── api/                  # API components
├── tests/                    # Tests
├── .gitignore
├── requirements.txt
├── EXPERIMENTS.md
└── README.md
```

## Running the Evaluation

### 1. Clone the repository

```bash
git clone https://github.com/bhumikagadade-stack/hybrid-retrieval-lab.git
cd hybrid-retrieval-lab
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download and prepare the dataset

Download SciFact from the [official repository](https://github.com/allenai/scifact) and place the extracted corpus, queries, and relevance judgments in the expected `data/raw/scifact/scifact/` directory structure.

Generate the document embeddings and FAISS index using the project's indexing workflow before running retrieval evaluation.

### 5. Compare dense and hybrid retrieval

```powershell
python src/evaluation/compare_dense_hybrid.py
```

The comparison results are saved to:

```text
data/processed/dense_vs_hybrid_results.csv
```

See `EXPERIMENTS.md` for the recorded experiment results and limitations.

## Failure Analysis

The project also examines cases where different retrieval methods disagree.

For example, one SciFact query about “0-dimensional biomaterials” retrieved a relevant document at rank 5 with dense search, while BM25 ranked it much lower because the query and document had limited lexical overlap.

This illustrates a key retrieval challenge: **semantic similarity and exact word matching capture different kinds of relevance**. Hybrid ranking can combine their strengths, although it may still miss relevant documents.

## Limitations and Future Improvements

- Evaluate on additional datasets and more complete relevance judgments.
- Experiment with the number of candidates retrieved by each method.
- Tune the RRF ranking constant using a validation set.
- Add reranking with a cross-encoder model.
- Build an interactive search interface and expose retrieval through an API.
- Add automated tests and reproducible setup instructions.

## Author

**Bhumika Gadade**

Aspiring AI/ML and Data Analytics professional interested in information retrieval, semantic search, machine learning, and evaluation-driven system development.

GitHub: [bhumikagadade-stack](https://github.com/bhumikagadade-stack)
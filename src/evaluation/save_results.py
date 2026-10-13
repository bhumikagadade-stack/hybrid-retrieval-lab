
from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "retrieval_results.csv"


def main():
    results = [
        {
            "method": "Dense",
            "recall_at_5": 0.7379,
            "mrr_at_5": 0.5997,
            "ndcg_at_5": 0.6293,
        },
        {
            "method": "BM25",
            "recall_at_5": 0.7284,
            "mrr_at_5": 0.6132,
            "ndcg_at_5": 0.6360,
        },
        {
            "method": "Hybrid (RRF)",
            "recall_at_5": 0.7395,
            "mrr_at_5": 0.6375,
            "ndcg_at_5": 0.6579,
        },
    ]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "recall_at_5",
                "mrr_at_5",
                "ndcg_at_5",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("Evaluation results saved successfully.")
    print("File:", RESULTS_PATH)
    print("\nResults summary:\n")

    for result in results:
        print(
            f"{result['method']:<15} "
            f"Recall@5: {result['recall_at_5']:.4f} | "
            f"MRR@5: {result['mrr_at_5']:.4f} | "
            f"nDCG@5: {result['ndcg_at_5']:.4f}"
        )


if __name__ == "__main__":
    main()
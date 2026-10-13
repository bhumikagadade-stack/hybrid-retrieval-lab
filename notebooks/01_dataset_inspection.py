
from pathlib import Path
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scifact"
    / "scifact"
)

CORPUS_PATH = DATA_DIR / "corpus.jsonl"

query = "0-dimensional biomaterials show inductive properties"
query_tokens = re.findall(r"\b\w+\b", query.lower())

target_document_id = "31715818"

with open(CORPUS_PATH, "r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line)

        if str(document["_id"]) == target_document_id:
            document_text = (
                document.get("title", "")
                + " "
                + document.get("text", "")
            ).lower()

            document_tokens = set(
                re.findall(r"\b\w+\b", document_text)
            )

            print("Query tokens:", query_tokens)
            print("\nToken matches in relevant paper:")

            for token in query_tokens:
                print(f"{token}: {token in document_tokens}")

            break
    else:
        print("Document not found.")
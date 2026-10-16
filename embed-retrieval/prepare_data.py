"""Cache BeIR/nfcorpus corpus, queries, and qrels locally as jsonl.

NFCorpus (Boteva et al. 2016, via the BEIR benchmark): 3,633 medical
documents, 3,237 queries (a mix of short "PLAIN-*" lay queries and longer
"NFCORPUS-*"/natural-language queries), graded relevance judgments split
into train/validation/test by query. We don't redistribute this data —
this script just caches HF's existing BeIR/nfcorpus(+qrels) locally so the
rest of the pipeline doesn't re-download it.
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "data"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    corpus = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    queries = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    qrels = load_dataset("BeIR/nfcorpus-qrels")

    with (OUT_DIR / "corpus.jsonl").open("w") as f:
        for row in corpus:
            text = (row["title"] + ". " + row["text"]).strip() if row["title"] else row["text"]
            f.write(json.dumps({"id": row["_id"], "text": text}) + "\n")

    with (OUT_DIR / "queries.jsonl").open("w") as f:
        for row in queries:
            f.write(json.dumps({"id": row["_id"], "text": row["text"]}) + "\n")

    for split in ("train", "validation", "test"):
        rows = qrels[split]
        with (OUT_DIR / f"qrels_{split}.jsonl").open("w") as f:
            for qid, cid, score in zip(rows["query-id"], rows["corpus-id"], rows["score"]):
                f.write(json.dumps({"query_id": qid, "corpus_id": cid, "score": score}) + "\n")

    meta = {
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "n_qrels": {split: len(qrels[split]) for split in ("train", "validation", "test")},
        "n_queries_with_qrels": {
            split: len(set(qrels[split]["query-id"])) for split in ("train", "validation", "test")
        },
        "source": "BeIR/nfcorpus + BeIR/nfcorpus-qrels on the Hugging Face Hub",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

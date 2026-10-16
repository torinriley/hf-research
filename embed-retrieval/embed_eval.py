"""Evaluate a sentence-transformers model (zero-shot or fine-tuned) as a retriever."""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from metrics import evaluate_rankings, load_jsonl, load_qrels

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / "cache"


def embed_corpus_and_queries(model: SentenceTransformer, cache_tag: str):
    cache_path = CACHE_DIR / f"{cache_tag}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["corpus_ids"], cached["corpus_emb"], cached["query_ids"], cached["query_emb"]

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    corpus_ids = np.array([r["id"] for r in corpus])
    query_ids = np.array([r["id"] for r in queries])

    corpus_emb = model.encode([r["text"] for r in corpus], batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    query_emb = model.encode([r["text"] for r in queries], batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache_path, corpus_ids=corpus_ids, corpus_emb=corpus_emb, query_ids=query_ids, query_emb=query_emb)
    return corpus_ids, corpus_emb, query_ids, query_emb


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF model id or local path")
    ap.add_argument("--cache-tag", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    model = SentenceTransformer(args.model_path)
    corpus_ids, corpus_emb, query_ids, query_emb = embed_corpus_and_queries(model, args.cache_tag)

    test_qrels = load_qrels(args.split)
    test_mask = np.array([qid in test_qrels for qid in query_ids])
    sims = query_emb[test_mask] @ corpus_emb.T  # already normalized -> cosine

    rankings = {}
    eval_qids = query_ids[test_mask]
    for i, qid in enumerate(eval_qids):
        order = np.argsort(-sims[i])[:100]
        rankings[qid] = list(corpus_ids[order])

    results = evaluate_rankings(rankings, test_qrels)
    results["model"] = args.model_path
    results["embedding_dim"] = int(corpus_emb.shape[1])
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()

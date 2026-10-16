"""Benchmark VecStream's HNSW index against its own exact search, using
real sentence-embedding vectors instead of synthetic Gaussian ones.

VecStream's own committed benchmark (benchmarks/ann_benchmark.py in the
VecStream repo) uses synthetic Gaussian vectors. Real embeddings have
different geometry (anisotropic, clustered by topic) which can change the
recall/latency frontier — this is a genuine extension of that benchmark,
not a repeat of it. VecStream is run directly from its own repository (not
published as a package), so we add its path to sys.path rather than
vendoring its code.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

VECSTREAM_REPO = Path.home() / "Downloads" / "Code" / "VecStream"
sys.path.insert(0, str(VECSTREAM_REPO))

from vecstream import HNSWIndex, VectorStore  # noqa: E402

CACHE_DIR = Path(__file__).parent / "cache"


def load_embeddings(cache_tag: str):
    cached = np.load(CACHE_DIR / f"{cache_tag}.npz", allow_pickle=True)
    return cached["corpus_ids"], cached["corpus_emb"], cached["query_ids"], cached["query_emb"]


def benchmark(cache_tag: str, ef_search_values: list[int], k: int = 10, seed: int = 42) -> dict:
    corpus_ids, corpus_emb, query_ids, query_emb = load_embeddings(cache_tag)
    dim = corpus_emb.shape[1]

    exact = VectorStore()
    for cid, vec in zip(corpus_ids, corpus_emb):
        exact.add_vector(str(cid), vec.tolist())

    ann = HNSWIndex(dim=dim, M=16, ef_construction=100, seed=seed)
    t0 = time.perf_counter()
    for cid, vec in zip(corpus_ids, corpus_emb):
        ann.add_item(str(cid), vec.tolist())
    build_time = time.perf_counter() - t0
    ann.validate()

    # exact ground truth top-k for every query, once
    exact_topk = {}
    exact_latencies = []
    for qid, qvec in zip(query_ids, query_emb):
        t0 = time.perf_counter()
        result = exact.exact_search(qvec.tolist(), k=k)
        exact_latencies.append(time.perf_counter() - t0)
        exact_topk[str(qid)] = {cid for cid, _ in result}

    frontier = []
    for ef in ef_search_values:
        recalls, latencies = [], []
        for qid, qvec in zip(query_ids, query_emb):
            t0 = time.perf_counter()
            result = ann.search(qvec.tolist(), k=k, ef_search=ef)
            latencies.append(time.perf_counter() - t0)
            got = {cid for cid, _ in result}
            truth = exact_topk[str(qid)]
            recalls.append(len(got & truth) / len(truth) if truth else float("nan"))
        valid_recalls = [r for r in recalls if r == r]  # drop nan
        frontier.append(
            {
                "ef_search": ef,
                "recall_at_k_vs_exact": sum(valid_recalls) / len(valid_recalls),
                "p50_ms": 1000 * float(np.median(latencies)),
                "p99_ms": 1000 * float(np.percentile(latencies, 99)),
                "qps": 1 / (sum(latencies) / len(latencies)),
            }
        )

    return {
        "cache_tag": cache_tag,
        "n_vectors": len(corpus_ids),
        "dim": dim,
        "k": k,
        "hnsw_build_time_sec": build_time,
        "exact_search_p50_ms": 1000 * float(np.median(exact_latencies)),
        "exact_search_qps": 1 / (sum(exact_latencies) / len(exact_latencies)),
        "frontier": frontier,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-tag", required=True)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    args = ap.parse_args()

    results = benchmark(args.cache_tag, args.ef_search)
    print(json.dumps(results, indent=2))
    out_path = Path(__file__).parent / f"vecstream_benchmark_{args.cache_tag}.json"
    out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

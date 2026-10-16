"""Standard IR metrics: Recall@k and NDCG@k against graded qrels."""
import json
import math
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_qrels(split: str) -> dict[str, dict[str, int]]:
    """query_id -> {corpus_id: relevance_score}"""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in load_jsonl(DATA_DIR / f"qrels_{split}.jsonl"):
        qrels[row["query_id"]][row["corpus_id"]] = row["score"]
    return qrels


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    hit = len(set(ranked_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[str], qrel: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        rel = qrel.get(cid, 0)
        if rel:
            dcg += (2**rel - 1) / math.log2(i + 2)
    ideal_rels = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    if idcg == 0:
        return float("nan")
    return dcg / idcg


def evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks=(10, 100)) -> dict:
    """rankings: query_id -> ranked list of corpus_id (best first)."""
    per_metric = {f"recall@{k}": [] for k in ks}
    per_metric.update({f"ndcg@{k}": [] for k in ks})
    for qid, ranked in rankings.items():
        qrel = qrels.get(qid)
        if not qrel:
            continue
        relevant = {cid for cid, s in qrel.items() if s > 0}
        for k in ks:
            r = recall_at_k(ranked, relevant, k)
            if not math.isnan(r):
                per_metric[f"recall@{k}"].append(r)
            n = ndcg_at_k(ranked, qrel, k)
            if not math.isnan(n):
                per_metric[f"ndcg@{k}"].append(n)
    return {name: sum(vals) / len(vals) for name, vals in per_metric.items() if vals} | {
        "n_queries_evaluated": sum(1 for q in rankings if q in qrels)
    }

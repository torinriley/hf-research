"""Classic lexical IR baseline: TF-IDF + cosine similarity.

Not neural, not embedding-based — the point of comparison that tells us
whether the embedding models are earning their complexity at all.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from metrics import evaluate_rankings, load_jsonl, load_qrels

DATA_DIR = Path(__file__).parent / "data"


def main() -> dict:
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    test_qrels = load_qrels("test")

    corpus_ids = [r["id"] for r in corpus]
    corpus_texts = [r["text"] for r in corpus]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    corpus_matrix = vectorizer.fit_transform(corpus_texts)

    test_queries = [q for q in queries if q["id"] in test_qrels]
    query_matrix = vectorizer.transform([q["text"] for q in test_queries])

    sims = cosine_similarity(query_matrix, corpus_matrix)
    rankings = {}
    for i, q in enumerate(test_queries):
        order = sims[i].argsort()[::-1][:100]
        rankings[q["id"]] = [corpus_ids[j] for j in order]

    results = evaluate_rankings(rankings, test_qrels)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()

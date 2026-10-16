from metrics import ndcg_at_k, recall_at_k


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_respects_k():
    # relevant doc is ranked 3rd; recall@2 should miss it
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0
    assert recall_at_k(["x", "y", "a"], {"a"}, k=3) == 1.0


def test_ndcg_perfect_ranking_is_one():
    qrel = {"a": 2, "b": 1, "c": 0}
    assert abs(ndcg_at_k(["a", "b", "c"], qrel, k=3) - 1.0) < 1e-9


def test_ndcg_worse_ranking_is_lower():
    qrel = {"a": 2, "b": 1, "c": 0}
    perfect = ndcg_at_k(["a", "b", "c"], qrel, k=3)
    worse = ndcg_at_k(["c", "b", "a"], qrel, k=3)
    assert worse < perfect


def test_ndcg_no_relevant_at_all_is_nan():
    import math

    assert math.isnan(ndcg_at_k(["x", "y"], {}, k=2))

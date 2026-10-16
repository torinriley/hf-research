"""Baselines for single-word spelling correction.

- identity: output = input (lower bound / sanity check)
- nn_edit_distance: nearest word in the training vocabulary by Levenshtein
  distance (classic, non-neural baseline a lot of real spell checkers use).
"""
import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def load_vocab(split: str = "train") -> list[str]:
    words = set()
    with (DATA_DIR / f"{split}.jsonl").open() as f:
        for line in f:
            words.add(json.loads(line)["correct"])
    return sorted(words)


class NearestNeighborCorrector:
    """O(|vocab|) edit-distance lookup. Fine at this vocab size (~40k)."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        # bucket by first char + length to prune candidates
        self._buckets: dict[tuple[str, int], list[str]] = {}
        for w in vocab:
            key = (w[0] if w else "", len(w))
            self._buckets.setdefault(key, []).append(w)

    @lru_cache(maxsize=None)
    def correct(self, word: str) -> str:
        candidates = []
        for dl in (-1, 0, 1):
            for first in {word[0] if word else "", ""}:
                candidates.extend(self._buckets.get((first, len(word) + dl), []))
        if not candidates:
            candidates = self.vocab
        best, best_dist = word, levenshtein(word, word) if word in self.vocab else 10**9
        for cand in candidates:
            d = levenshtein(word, cand)
            if d < best_dist:
                best, best_dist = cand, d
        return best


def identity_correct(word: str) -> str:
    return word

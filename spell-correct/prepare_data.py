"""Download torinriley/spell-correction and build leakage-free splits.

Split by *correct* word, not by pair: every misspelling of a given target
word lands in the same split. This means the val/test sets measure
generalization to words the model's decoder has never produced during
training, not just memorization of a fixed vocabulary.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 1234
OUT_DIR = Path(__file__).parent / "data"


def main() -> None:
    random.seed(SEED)
    ds = load_dataset("torinriley/spell-correction")["train"]

    pairs = sorted(set(zip(ds["misspelled"], ds["correct"])))  # dedupe, deterministic order

    by_word: dict[str, list[str]] = {}
    for mis, cor in pairs:
        by_word.setdefault(cor, []).append(mis)

    words = sorted(by_word)
    random.shuffle(words)

    n = len(words)
    n_test = max(1, int(n * 0.05))
    n_val = max(1, int(n * 0.05))
    test_words = set(words[:n_test])
    val_words = set(words[n_test : n_test + n_val])
    train_words = set(words[n_test + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for cor, mis_list in by_word.items():
        bucket = "test" if cor in test_words else "val" if cor in val_words else "train"
        for mis in mis_list:
            splits[bucket].append({"misspelled": mis, "correct": cor})

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(rows)} pairs, {len({r['correct'] for r in rows})} unique words")

    meta = {
        "seed": SEED,
        "total_pairs_after_dedup": len(pairs),
        "unique_words": n,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_by": "correct_word (no word's target spelling appears in more than one split)",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

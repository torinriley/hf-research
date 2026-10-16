"""In-domain contrastive fine-tuning of a small pretrained embedding model.

Uses sentence-transformers' MultipleNegativesRankingLoss (in-batch
negatives, standard InfoNCE-style contrastive objective) on (query,
relevant-document) pairs built from the NFCorpus train qrels. We don't
reimplement in-batch contrastive loss from scratch: it's a well-established
building block, and the research question here is whether in-domain
fine-tuning helps retrieval on this dataset, not whether we can reproduce
InfoNCE by hand.
"""
import argparse
import json
import random
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from metrics import load_jsonl, load_qrels

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"
SEED = 1234
MAX_POSITIVES_PER_QUERY = 3


def build_training_examples() -> list[InputExample]:
    random.seed(SEED)
    corpus = {r["id"]: r["text"] for r in load_jsonl(DATA_DIR / "corpus.jsonl")}
    queries = {r["id"]: r["text"] for r in load_jsonl(DATA_DIR / "queries.jsonl")}
    train_qrels = load_qrels("train")

    examples = []
    for qid, qrel in train_qrels.items():
        if qid not in queries:
            continue
        positives = [cid for cid, score in qrel.items() if score > 0 and cid in corpus]
        random.shuffle(positives)
        for cid in positives[:MAX_POSITIVES_PER_QUERY]:
            examples.append(InputExample(texts=[queries[qid], corpus[cid]]))
    random.shuffle(examples)
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default=str(CKPT_DIR / "nfcorpus-finetuned-minilm"))
    args = ap.parse_args()

    examples = build_training_examples()
    print(f"training pairs: {len(examples)}")

    model = SentenceTransformer(args.base_model)
    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_loader) * args.epochs),
        show_progress_bar=True,
        output_path=args.out_dir,
    )

    (Path(args.out_dir) / "finetune_meta.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "n_training_pairs": len(examples),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": SEED,
                "max_positives_per_query": MAX_POSITIVES_PER_QUERY,
                "loss": "MultipleNegativesRankingLoss (in-batch negatives)",
            },
            indent=2,
        )
    )
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()

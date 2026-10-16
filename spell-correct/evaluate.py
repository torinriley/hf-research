import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from baseline import NearestNeighborCorrector, identity_correct, levenshtein, load_vocab
from data import DATA_DIR
from model import ModelConfig, SpellCorrectorTransformer
from train import get_device
from vocab import BOS, EOS, PAD, decode, encode

CKPT_DIR = Path(__file__).parent / "checkpoints"


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, src_mask: torch.Tensor, device, max_len: int = 24) -> list[str]:
    model.eval()
    batch = src.size(0)
    ys = torch.full((batch, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    for _ in range(max_len):
        tgt_mask = ys == PAD
        logits = model(src, ys, src_mask, tgt_mask)
        next_token = logits[:, -1, :].argmax(-1)
        next_token = torch.where(finished, torch.full_like(next_token, PAD), next_token)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == EOS
        if finished.all():
            break
    return [decode(row.tolist()[1:]) for row in ys]


def load_rows(split: str) -> list[dict]:
    rows = []
    with (DATA_DIR / f"{split}.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalized_edit_distance(a: str, b: str) -> float:
    if max(len(a), len(b)) == 0:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def score(rows: list[dict], predictions: list[str]) -> dict:
    exact = sum(p == r["correct"] for p, r in zip(predictions, rows)) / len(rows)
    ned = sum(normalized_edit_distance(p, r["correct"]) for p, r in zip(predictions, rows)) / len(rows)
    ned_before = sum(normalized_edit_distance(r["misspelled"], r["correct"]) for r in rows) / len(rows)
    return {
        "exact_match": exact,
        "mean_normalized_edit_distance": ned,
        "mean_normalized_edit_distance_before_correction": ned_before,
        "n": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    rows = load_rows(args.split)
    device = get_device()

    results = {}

    t0 = time.time()
    identity_preds = [identity_correct(r["misspelled"]) for r in rows]
    results["identity_baseline"] = score(rows, identity_preds) | {"latency_sec_per_example": (time.time() - t0) / len(rows)}

    # Fair setting: the NN baseline only gets the train vocabulary, exactly like the model's
    # training signal. Because splits are by correct-word, this means the lookup table
    # structurally cannot contain the right answer for any test-split example.
    t0 = time.time()
    train_vocab = load_vocab("train")
    nn_corrector = NearestNeighborCorrector(train_vocab)
    nn_preds = [nn_corrector.correct(r["misspelled"]) for r in rows]
    results["nearest_neighbor_train_vocab_only"] = score(rows, nn_preds) | {
        "latency_sec_per_example": (time.time() - t0) / len(rows),
        "vocab_size": len(train_vocab),
        "note": "lookup table cannot contain the target word by construction of the split",
    }

    # Best case for a closed-vocabulary lookup: give it every word in the dataset
    # (train+val+test), so the correct answer is guaranteed to be in its table.
    # This isolates "can edit-distance lookup do the matching" from "does it know the word".
    t0 = time.time()
    oracle_vocab = sorted(set(train_vocab) | set(load_vocab("val")) | set(load_vocab("test")))
    oracle_nn = NearestNeighborCorrector(oracle_vocab)
    oracle_preds = [oracle_nn.correct(r["misspelled"]) for r in rows]
    results["nearest_neighbor_oracle_vocab"] = score(rows, oracle_preds) | {
        "latency_sec_per_example": (time.time() - t0) / len(rows),
        "vocab_size": len(oracle_vocab),
        "note": "upper bound for closed-vocabulary lookup; target word always in table",
    }

    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = SpellCorrectorTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    t0 = time.time()
    model_preds = []
    batch_size = 256
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        srcs = [torch.tensor(encode(r["misspelled"])) for r in batch_rows]
        maxlen = max(len(s) for s in srcs)
        src = torch.full((len(srcs), maxlen), PAD, dtype=torch.long)
        for j, s in enumerate(srcs):
            src[j, : len(s)] = s
        src = src.to(device)
        src_mask = (src == PAD).to(device)
        model_preds.extend(greedy_decode(model, src, src_mask, device))
    results["transformer_model"] = score(rows, model_preds) | {
        "latency_sec_per_example": (time.time() - t0) / len(rows),
        "num_parameters": model.num_parameters(),
        "trained_epoch": ckpt["epoch"],
        "val_loss_at_checkpoint": ckpt["val_loss"],
    }

    print(json.dumps(results, indent=2))

    out_path = Path(__file__).parent / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2))

    # a few qualitative examples for the model card
    examples = []
    for r, p in list(zip(rows, model_preds))[:15]:
        examples.append({"misspelled": r["misspelled"], "correct": r["correct"], "predicted": p})
    (Path(__file__).parent / f"examples_{args.split}.json").write_text(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()

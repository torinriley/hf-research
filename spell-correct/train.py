import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # nn.Transformer's padding-mask fast path isn't on MPS yet

import torch
from torch.utils.data import DataLoader

from data import SpellPairs, collate
from model import ModelConfig, SpellCorrectorTransformer
from vocab import PAD

CKPT_DIR = Path(__file__).parent / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    for src, tgt, src_mask, tgt_mask, *_ in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        decoder_in = tgt[:, :-1]
        decoder_out_mask = tgt_mask[:, :-1]
        target = tgt[:, 1:]

        with torch.set_grad_enabled(train):
            logits = model(src, decoder_in, src_mask, decoder_out_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            n_tokens = (target != PAD).sum().item()
            if train:
                optimizer.zero_grad()
                (loss / max(n_tokens, 1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device={device}")

    train_ds = SpellPairs("train")
    val_ds = SpellPairs("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    cfg = ModelConfig()
    model = SpellCorrectorTransformer(cfg).to(device)
    print(f"params={model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(exist_ok=True)
    history = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(), "config": cfg.__dict__, "epoch": epoch, "val_loss": val_loss},
                CKPT_DIR / "best.pt",
            )

    elapsed = time.time() - start
    (CKPT_DIR / "train_history.json").write_text(
        json.dumps({"history": history, "elapsed_sec": elapsed, "args": vars(args), "device": str(device)}, indent=2)
    )
    print(f"done in {elapsed:.1f}s, best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()

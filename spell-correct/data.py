import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vocab import BOS, PAD, encode

DATA_DIR = Path(__file__).parent / "data"


class SpellPairs(Dataset):
    def __init__(self, split: str):
        self.rows = []
        with (DATA_DIR / f"{split}.jsonl").open() as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        src = encode(row["misspelled"])  # <bos> ... <eos>
        tgt = encode(row["correct"])
        return torch.tensor(src), torch.tensor(tgt), row["misspelled"], row["correct"]


def collate(batch, max_len: int = 32):
    srcs, tgts, mis, cor = zip(*batch)

    def pad_stack(seqs):
        maxlen = min(max(len(s) for s in seqs), max_len)
        out = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            s = s[:maxlen]
            out[i, : len(s)] = s
        return out

    src = pad_stack(srcs)
    tgt = pad_stack(tgts)
    src_mask = src == PAD
    tgt_mask = tgt == PAD
    return src, tgt, src_mask, tgt_mask, mis, cor

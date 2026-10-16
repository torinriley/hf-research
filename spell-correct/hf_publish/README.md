---
license: mit
datasets:
- torinriley/spell-correction
language:
- en
tags:
- spelling-correction
- character-level
- transformer
- from-scratch
pipeline_tag: text-generation
---

# spell-correct-transformer

A 1,001,757-parameter character-level encoder-decoder Transformer, trained
**from scratch** (no pretrained weights) to correct single-word English
spelling errors. Architecture, training, and evaluation code:
[torinriley/hf-research](https://github.com/torinriley/hf-research/tree/main/spell-correct).

## Headline result

On a held-out test set where **no correct spelling in the test set ever
appears as a training target** (split by target word, not by example pair
— see Methodology), this model gets:

| Method | Exact match | Mean normalized edit distance |
|---|---|---|
| Identity (no correction) | 0.0% | 0.135 |
| Edit-distance nearest-neighbor (fair baseline, same vocabulary as training) | 0.0% | 0.376 |
| **This model** | **40.0%** | **0.099** |
| Edit-distance nearest-neighbor, oracle vocabulary (upper bound — cheats by having the answer in its table) | 84.1% | 0.071 |

The model was never shown these words as correction targets during
training, so this is genuine generalization to novel vocabulary, not
memorization. The dictionary-lookup baseline cannot do this by
construction — it can only ever return a word already in its table.

## Architecture

`nn.Transformer` encoder-decoder, character-level, 29-token vocabulary
(a–z + pad/bos/eos): `d_model=128`, 4 attention heads, 3 encoder layers, 3
decoder layers, feedforward dim 256, dropout 0.1.

## Training

- Data: [torinriley/spell-correction](https://huggingface.co/datasets/torinriley/spell-correction), 49,531 unique (misspelled, correct) pairs after dedup.
- Split: by correct word, seed 1234 — 40,260 train words, 2,236 val words, 2,236 test words, no overlap.
- Optimizer: AdamW, lr=3e-4, no LR schedule, batch size 256, 15 epochs.
- Hardware: single Apple M5 (MPS backend), 181s total wall-clock.
- Val loss was still decreasing at epoch 15 (0.2665) — not pushed further to keep this first iteration cheap; a longer run is the obvious next step.

## Usage

```python
import json, torch
from safetensors.torch import load_file
# model.py / vocab.py from https://github.com/torinriley/hf-research/tree/main/spell-correct
from model import ModelConfig, SpellCorrectorTransformer
from vocab import encode, decode, PAD, BOS, EOS

cfg = ModelConfig(**json.load(open("config.json")))
model = SpellCorrectorTransformer(cfg)
model.load_state_dict(load_file("model.safetensors"))
model.eval()

def correct(word: str, max_len: int = 24) -> str:
    src = torch.tensor([encode(word)])
    src_mask = torch.zeros_like(src, dtype=torch.bool)
    ys = torch.tensor([[BOS]])
    for _ in range(max_len):
        tgt_mask = ys == PAD
        logits = model(src, ys, src_mask, tgt_mask)
        nxt = logits[:, -1, :].argmax(-1)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        if nxt.item() == EOS:
            break
    return decode(ys[0].tolist()[1:])

print(correct("accetnuation"))  # -> "accentuation"
```

## Limitations

- Single lowercase word only (a–z), no sentence context — cannot fix
  real-word errors ("there"/"their") or use surrounding context.
- One training run, one seed, no hyperparameter sweep: these are point
  estimates, not confidence intervals.
- The underlying dataset's vocabulary looks skewed toward
  scientific/Latinate terms; general-English performance is untested.

## Reproducibility

Full methodology, baseline design, and discussion of why the
dictionary-lookup baseline does *worse* than doing nothing on this split:
[REPORT.md](https://github.com/torinriley/hf-research/blob/main/spell-correct/REPORT.md).

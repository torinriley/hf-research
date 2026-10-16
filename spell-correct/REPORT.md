# Char-level spelling correction: a from-scratch Transformer vs. edit-distance lookup

## Hypothesis

A small character-level encoder-decoder Transformer trained from scratch on
`torinriley/spell-correction` can correct spellings of words it never saw as
a *target* during training (true generalization), whereas a classic
closed-vocabulary edit-distance lookup structurally cannot, because it can
only ever output a word that is already in its table.

## Data and split

- Source: [torinriley/spell-correction](https://huggingface.co/datasets/torinriley/spell-correction), 49,625 (misspelled, correct) pairs, lowercase a–z only.
- Deduplicated to 49,531 unique pairs.
- **Split by correct word, not by pair** (seed 1234): every misspelling of a
  given target word is assigned to exactly one split. 40,260 words / 44,600
  pairs train, 2,236 words / 2,468 pairs val, 2,236 words / 2,463 pairs test.
  This is deliberately the harder, more honest split — a pair-level random
  split would let the model (or a lookup table) see the same target word
  with a different typo at train time, which overstates real generalization.
  See `prepare_data.py`.

## Model

`SpellCorrectorTransformer` (`model.py`): a standard `nn.Transformer`
encoder-decoder over a 29-token vocabulary (a–z + pad/bos/eos),
`d_model=128`, 4 heads, 3 encoder + 3 decoder layers, 1,001,757 parameters.
Trained from scratch (no pretrained weights) — at this vocabulary size and
task, a pretrained LM would add nothing a from-scratch model can't learn
from ~45k examples, so training from scratch is the more honest artifact.

- Optimizer: AdamW, lr=3e-4, no schedule.
- Loss: token cross-entropy, teacher forcing, padding ignored.
- 15 epochs, batch size 256, seed 1234. Trained on an Apple M5 (MPS backend,
  with CPU fallback for one unsupported op — `PYTORCH_ENABLE_MPS_FALLBACK=1`).
  Wall clock: 241s total (`checkpoints/train_history.json`).
- Val loss was still decreasing at epoch 15 (0.2661) — a longer run would
  likely improve results further; this was not pushed further to keep the
  first iteration cheap.

## Baselines

1. **Identity**: output = input. Lower bound / sanity check.
2. **Nearest-neighbor edit distance, train vocabulary only**: looks up the
   closest word (Levenshtein) in the 40,260-word train vocabulary. Because
   of the correct-word split, the right answer is *never* in this table for
   test-split examples — this measures what a dictionary-based corrector
   does when it meets a word outside its dictionary.
3. **Nearest-neighbor edit distance, oracle vocabulary**: same method, but
   the lookup table is the union of train+val+test vocabularies (44,732
   words), so the correct answer is always present. This is the ceiling for
   what pure edit-distance lookup can do, independent of vocabulary
   coverage — useful to isolate "is nearest-match-by-edit-distance a good
   matching rule" from "does the dictionary contain the word".

## Results (held-out test split, 2,463 pairs, none of whose correct spelling appears in train/val)

| Method | Exact match | Mean normalized edit distance | Notes |
|---|---|---|---|
| Identity (no correction) | 0.0% | 0.135 | baseline NED of the raw typos |
| NN edit-distance, train vocab only | 0.0% | 0.376 | **worse than doing nothing** — forced to pick a wrong word from a table that can't contain the answer |
| NN edit-distance, oracle vocab | 84.1% | 0.071 | ceiling when the answer is guaranteed to be in the table |
| **Transformer (ours)** | **40.0%** | **0.099** | never saw these words' correct spelling during training |

Full numbers: `eval_test.json`. Qualitative examples: `examples_test.json`.

## What this shows

- The oracle-vocab baseline (84.1%) is not a fair comparison to the model —
  it's the ceiling for a method that is handed the answer set in advance.
  The fair comparison is to the train-vocab-only baseline, which the
  Transformer beats by a wide margin (40.0% vs. 0.0% exact match, 0.099 vs.
  0.376 mean NED) because it learned general character-level correction
  patterns (common typo shapes: transpositions, doubled/dropped letters,
  adjacent-key substitutions) rather than memorizing a word list.
- The train-vocab-only baseline being *worse than doing nothing* (0.376 vs.
  0.135 NED) is a real, if unsurprising, negative result: forcing an
  edit-distance match against a vocabulary that cannot contain the right
  answer actively makes typos worse on average, because it confidently
  substitutes a wrong-but-close word for a typo that may have already been
  closer to the truth.
- The gap to the oracle ceiling (40.0% vs. 84.1%) shows real headroom:
  longer training, a larger model, or char-level data augmentation
  (synthetic typos over a larger word list) would be the natural next
  experiments, not architecture changes.

## Reproduction

```bash
cd spell-correct
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
python prepare_data.py        # writes data/{train,val,test}.jsonl, seed=1234
python train.py --epochs 15 --batch-size 256 --seed 1234
python evaluate.py --split test
```

## Limitations

- Single-word correction only; no sentence-level context, so it cannot
  disambiguate real-word errors ("there" vs "their").
- Vocabulary is lowercase a–z only; no digits, punctuation, or casing.
- English-only, and the underlying dataset's domain/source vocabulary
  (looks skewed toward scientific/Latinate terms, e.g. "arachidonic",
  "azogallein") is not characterized here — a known limitation of the
  source dataset, not of this model.
- One training run, no hyperparameter sweep or multiple seeds — point
  estimates, not confidence intervals.

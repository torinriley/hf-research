# Spell correction: from-scratch char Transformer vs. edit-distance lookup

Closes a gap in the `torinriley/spell-correction` dataset (it had downloads
and likes but no model trained against it) and runs a small, honest
experiment: does a tiny generative model generalize to misspellings of
words it never saw as a training target, where a classic dictionary-lookup
corrector structurally cannot?

See [REPORT.md](REPORT.md) for the full write-up, methodology, and results.
Headline result: on words held out by target-word (not just by pair), the
1M-parameter Transformer gets 40.0% exact-match / 0.099 mean normalized
edit distance, vs. 0.0% / 0.376 for the only fair (non-oracle) baseline.

```
spell-correct/
├── prepare_data.py   # leakage-free train/val/test split
├── vocab.py           # fixed a-z + special-token vocabulary
├── model.py           # SpellCorrectorTransformer (from scratch, ~1M params)
├── data.py            # Dataset/collate
├── baseline.py         # identity + edit-distance nearest-neighbor baselines
├── train.py            # training loop, checkpointing, seed=1234
├── evaluate.py         # scores model + baselines on held-out split
├── REPORT.md           # methodology, results, limitations
├── checkpoints/        # best.pt + train_history.json (gitignored, see below)
└── data/                # generated splits (gitignored, regenerate with prepare_data.py)
```

Reproduce: see "Reproduction" in REPORT.md.

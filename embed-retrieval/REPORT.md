# Embedding retrieval: fine-tuning gains and a real-embedding ANN benchmark

## Hypotheses

1. Lightly fine-tuning a small pretrained sentence-embedding model with
   in-domain contrastive pairs improves retrieval quality over the
   off-the-shelf model on NFCorpus, and both beat a lexical (TF-IDF)
   baseline.
2. [VecStream](https://github.com/torinriley/VecStream)'s from-scratch HNSW
   index, previously only benchmarked on synthetic Gaussian vectors,
   behaves differently (and should recover recall faster at low
   `ef_search`) on real, topically-clustered sentence embeddings.

## Data

[BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) +
[BeIR/nfcorpus-qrels](https://huggingface.co/datasets/BeIR/nfcorpus-qrels)
(Boteva et al., 2016, via the BEIR benchmark): 3,633 medical documents,
3,237 queries, graded relevance (0/1/2) split by query into train (2,590
queries, 110,575 judgments) / validation (324 queries) / test (323
queries). Not redistributed — `prepare_data.py` just caches the existing
HF dataset locally. Test-split documents and relevance judgments are never
used for fine-tuning.

## Part 1: does in-domain fine-tuning help retrieval?

### Methods

- **TF-IDF**: `sklearn.TfidfVectorizer` (English stopwords, 50k max
  features) + cosine similarity. Classic lexical baseline, no training.
- **Zero-shot embedding**: `sentence-transformers/all-MiniLM-L6-v2`
  (22.7M params), used as-is, no fine-tuning.
- **Fine-tuned embedding**: same base model, fine-tuned on NFCorpus train
  qrels with `MultipleNegativesRankingLoss` (in-batch-negative contrastive
  loss) over (query, relevant-document) pairs — up to 3 positives sampled
  per query (seed 1234) to avoid overweighting queries with unusually many
  judged-relevant documents, giving 7,253 training pairs. 3 epochs, batch
  size 32, default sentence-transformers optimizer settings. Training
  wall-clock: 404s on CPU. Final train loss 2.576 (`finetune.py`,
  `checkpoints/nfcorpus-finetuned-minilm/finetune_meta.json`).

All three are evaluated on the same held-out **test** split (323 queries),
Recall@10/100 and NDCG@10/100, standard IR metrics (`metrics.py`).

### Results (test split, n=323 queries)

| Method | Recall@10 | Recall@100 | NDCG@10 | NDCG@100 |
|---|---|---|---|---|
| TF-IDF (lexical) | 0.141 | 0.234 | 0.291 | 0.255 |
| Zero-shot MiniLM | 0.159 | 0.315 | 0.319 | 0.302 |
| **Fine-tuned MiniLM** | **0.164** | **0.350** | **0.332** | **0.327** |

Full JSON: `tfidf_results.json`, `zeroshot_results.json`, `finetuned_results.json`.

**Sanity check against published numbers**: the zero-shot MiniLM NDCG@10
(0.319) lands within normal run-to-run variance of the number reported
for `all-MiniLM-L6-v2` on NFCorpus in the BEIR/MTEB leaderboard ecosystem
(~0.317), which gives confidence the eval pipeline (encoding, qrels
handling, NDCG implementation) is correct rather than silently broken.

### What this shows

- Embeddings beat lexical matching here (NDCG@10 +9.6% relative,
  zero-shot vs. TF-IDF) — expected, since NFCorpus queries are often
  natural-language questions that share little surface vocabulary with
  the relevant documents.
- In-domain fine-tuning gives a further, real but modest gain over the
  zero-shot embedding: NDCG@10 +4.1% relative (0.319 → 0.332), Recall@100
  +11.2% relative (0.315 → 0.350). This is a small, cheap fine-tune (7,253
  pairs, 3 epochs, no hyperparameter search) — the gain is believable in
  size but not large; a longer run, hard-negative mining, or more
  positives per query are the natural next steps, not a different
  architecture.
- No negative result here: every stage improved over the previous one.
  The interesting limitation is in Part 2.

## Part 2: does VecStream's ANN behavior change on real embeddings?

VecStream's own committed benchmark (`benchmarks/ann_benchmark.py` in the
[VecStream repo](https://github.com/torinriley/VecStream)) measures
recall@10 vs. `ef_search` on 1,000 synthetic 64-dim Gaussian vectors,
finding recall@10 = 0.768 at `ef_search=10`. Real sentence embeddings are
not isotropic Gaussian — they cluster by topic/semantics — so the ANN
recall/latency frontier should differ. `vecstream_benchmark.py` repeats
the same measurement methodology (recall of HNSW search vs. VecStream's
own exact cosine search, used as ground truth; `M=16`, `ef_construction=100`,
seed=42) on the two real 384-dim embedding sets from Part 1, over the
full NFCorpus corpus (3,633 vectors) and all 3,237 queries.

### Results

| `ef_search` | Recall@10 vs. exact (zero-shot emb.) | Recall@10 vs. exact (fine-tuned emb.) | p50 latency (ms) | QPS |
|---:|---:|---:|---:|---:|
| 10 | 0.868 | 0.857 | 0.43–0.50 | ~2,000 |
| 20 | 0.933 | 0.920 | 0.65–0.66 | ~1,500 |
| 40 | 0.973 | 0.960 | 1.00–1.03 | ~960 |
| 80 | 0.992 | 0.984 | 1.57–1.71 | ~600 |
| 160 | 0.999 | 0.995 | 2.62–2.81 | ~360 |

Full JSON: `vecstream_benchmark_minilm_zeroshot.json`,
`vecstream_benchmark_minilm_finetuned.json`. HNSW build time: 21.6s
(zero-shot) / 30.5s (fine-tuned) for 3,633 384-dim vectors, both on one
CPU core, pure Python/NumPy.

### What this shows

- **The hypothesis holds**: at `ef_search=10`, recall@10 vs. exact search
  is 0.857–0.868 here, vs. 0.768 in VecStream's own synthetic-Gaussian
  benchmark. Real semantic embeddings are more clusterable than isotropic
  Gaussian noise, so the graph needs less search effort to find true
  near-duplicates. This is a real, if expected, finding — and a genuine
  extension of VecStream's own benchmark suite, which had not been
  evaluated on real embedding data before.
- At this corpus size (3,633 vectors), VecStream's **exact** search is
  already fast (3.6–3.7ms p50, pure vectorized NumPy) — the ANN speed
  advantage over exact search is real but modest at full recall (~1.3x at
  `ef_search=160`, ~8x at `ef_search=10` but with ~13-14% recall loss).
  The approximate index earns its complexity at larger corpus sizes,
  where exact search stops being a vectorized one-shot matmul; this
  benchmark doesn't test that scale and shouldn't be read as a
  verdict on whether ANN is "worth it" in general, only on how its
  recall/latency shape differs between synthetic and real embeddings at
  this specific size.

## Reproduction

```bash
cd embed-retrieval
source ../.venv/bin/activate
python prepare_data.py
python tfidf_baseline.py
python embed_eval.py --model-path sentence-transformers/all-MiniLM-L6-v2 --cache-tag minilm_zeroshot
python finetune.py --epochs 3 --batch-size 32          # seed=1234
python embed_eval.py --model-path checkpoints/nfcorpus-finetuned-minilm --cache-tag minilm_finetuned
python vecstream_benchmark.py --cache-tag minilm_zeroshot
python vecstream_benchmark.py --cache-tag minilm_finetuned
```

Requires the [VecStream](https://github.com/torinriley/VecStream) repo
checked out at `~/Downloads/Code/VecStream` (it's run directly from its
own repository, not published as a package — `vecstream_benchmark.py`
adds that path to `sys.path` rather than vendoring its code).

## Limitations

- Single dataset (NFCorpus), single base model (MiniLM-L6-v2), single
  fine-tuning run (one seed, no hyperparameter sweep) — point estimates.
- Fine-tuning uses only in-batch negatives; no hard-negative mining, which
  typically gives a larger gain in the embedding-fine-tuning literature.
- The VecStream benchmark compares HNSW to VecStream's own exact search,
  not to a different library's HNSW (e.g. hnswlib/FAISS) — it measures
  "how much search effort does this implementation need," not "how does
  this implementation compare to the state of the art."
- Corpus size (3,633) is small enough that exact search is already fast;
  conclusions about ANN's latency advantage don't extrapolate to larger
  corpora without further testing.

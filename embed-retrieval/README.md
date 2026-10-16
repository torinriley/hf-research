# Embedding retrieval: fine-tuning + a real-embedding ANN benchmark

Two linked experiments on [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) (BEIR benchmark, 3,633 medical documents, 323 test queries):

1. Does in-domain contrastive fine-tuning of a small pretrained sentence
   embedding model improve retrieval over the off-the-shelf model, on this
   dataset? Baselines: TF-IDF (lexical) and zero-shot MiniLM.
2. Does [VecStream](https://github.com/torinriley/VecStream) (a from-scratch
   HNSW engine) hold up on real semantic embeddings rather than the
   synthetic Gaussian vectors its own README benchmarks against?

See [REPORT.md](REPORT.md) for full methodology and results once both
experiments are complete.

```
embed-retrieval/
├── prepare_data.py          # caches BeIR/nfcorpus + qrels locally
├── metrics.py                # Recall@k / NDCG@k against graded qrels
├── tfidf_baseline.py         # lexical baseline
├── embed_eval.py              # zero-shot / fine-tuned embedding retrieval eval
├── finetune.py                 # in-domain contrastive fine-tuning (MultipleNegativesRankingLoss)
├── vecstream_benchmark.py       # HNSW vs exact search on real embeddings
└── REPORT.md
```

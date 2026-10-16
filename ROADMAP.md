# HF Research Portfolio — Audit & Roadmap

## 1. Account audit (torinriley, as of 2026-10-16)

| Repo | Type | Status | Notes |
|---|---|---|---|
| [FTCVision-PyTorch](https://huggingface.co/torinriley/FTCVision-PyTorch) | model | public, 0 downloads, 1 like | Object detection for FTC robotics. No model card detail, no eval numbers. |
| [CRISPR-Efficiency](https://huggingface.co/torinriley/CRISPR-Efficiency) | model | public, 0 downloads | Bio domain, minimal documentation. |
| [FTCVision](https://huggingface.co/datasets/torinriley/FTCVision) | dataset | public, 53 downloads, 1 like | Paired with the model above. |
| [spell-correction](https://huggingface.co/datasets/torinriley/spell-correction) | dataset | public, 49 downloads, 2 likes | Char-level spelling correction pairs — no model trained on it yet. |
| `evolve0-pipeline` | dataset | **private**, pushed today | Belongs to the separate `Evolve01` project (self-improvement loop, negative result documented). Out of scope here per your instruction — not touching it. |
| Spaces | — | none | No interactive demos at all. |
| Collections | — | none | No curation/grouping of related work. |

**Takeaways:** two old models/datasets with no evaluation methodology, no baselines, no Spaces, no collections, nothing connecting the pieces. There's also an unused asset: `spell-correction` has downloads/likes but no model was ever trained and published against it.

**Environment:** MacBook, Apple M5, 10 CPU cores, 16GB RAM, no discrete GPU, no CUDA, `torch` not installed locally. MPS backend is available via Apple's Metal backend once torch is installed. Free ZeroGPU quota: 2400s available (tied to Spaces, not a purchase). No HF Jobs usage yet. This rules out large from-scratch pretraining — favors small models, fine-tuning/adaptation, distillation, quantization, and evaluation/systems work, which also matches current research trends more than brute-force scaling does.

## 2. Candidate projects (15, ranked)

Scored on technical depth, originality, feasibility on this hardware, reference-implementation quality, likelihood of a measurable result, relevance to ML engineering roles, artifact quality, cost, and differentiation vs. generic HF content.

1. **Finish the spell-correction model** — train a small char-level seq2seq/transformer on the existing `spell-correction` dataset, publish with real eval (accuracy, edit-distance reduction) against a baseline (Levenshtein-only correction). Closes an obvious gap, cheap, CPU-feasible, gives an immediate "complete pair" (dataset + model + eval).
2. **Small embedding model + VecStream retrieval benchmark** — fine-tune a small sentence-embedding model (contrastive, InfoNCE) on a public retrieval dataset, then benchmark recall@k and latency using your own VecStream HNSW engine vs. exact search and vs. a stock SBERT model. Directly combines ML + the systems work the brief flags as high-value, produces a dataset, model, benchmark, and write-up.
3. **Hybrid linear-attention distillation on a tiny GPT-2-scale model** — reproduce the Mamba-in-the-Llama / ICLR 2026 layer-selection idea at small scale: distill a subset of attention layers in a small pretrained transformer into linear-attention layers, measure perplexity retention vs. layers replaced and inference throughput gained. CPU/MPS feasible at small scale, directly engages the most active 2026 research thread.
4. **Low-bit PTQ study on sub-1B models** — systematically quantize 2–3 small open models to 8/4/2-bit and measure accuracy/perplexity degradation vs. bit-width and layer sensitivity, informed by the LieQ/pQuant line of work. Produces a reusable benchmark suite + report, cheap (inference-only).
5. **Reranker vs. retriever unification (E2Rank-style, small scale)** — train one small model jointly with contrastive + listwise objectives, compare against separate retriever+reranker pipeline on a small IR benchmark (e.g. a BEIR subset). Research-flavored, clear ablation story.
6. **Parameter-efficient fine-tuning comparison suite** — one base model, same task, LoRA vs. DoRA vs. (IA)³ vs. full fine-tune on CPU/MPS budget: compare final accuracy, trainable-param count, memory, wall-clock. Pure engineering-evaluation project, very reproducible, good "systematic comparison" artifact.
7. **Speculative decoding on a small model pair** — implement draft+verify decoding with a small/large model pair from the same family, measure wall-clock speedup vs. acceptance rate on CPU. Ties to inference-efficiency interests and Alpaca.swift-adjacent work.
8. **FTCVision v2** — revisit the existing object-detection model: add a real eval (mAP), compare against a baseline pretrained detector, write a proper model card, maybe a Space demo. Lower novelty but fixes a real documentation/credibility gap in the existing portfolio.
9. **Calibration/uncertainty benchmark for small classifiers** — measure calibration error (ECE) and behavior under label noise / distribution shift for a small fine-tuned classifier, before/after temperature scaling. Interpretability/evaluation-flavored, cheap.
10. **Agent evaluation harness (ties to Xysera direction, but a new standalone artifact, not touching Evolve01)** — a small, reproducible benchmark environment for scoring tool-use/agent behaviors on fixed tasks, published as a dataset + eval scripts. Infra-heavy, complements your Relay/agent interests without depending on Evolve01's code.
11. **Biological sequence embedding** — small contrastive embedding model over protein/DNA k-mers for a simple downstream task (e.g. family classification), benchmarked against k-mer baselines. Higher research risk, needs careful dataset vetting for licensing.
12. **Time-series foundation model mini-benchmark** — evaluate a couple of small open time-series models + a from-scratch baseline on a public forecasting benchmark, report MASE/CRPS. Moderate novelty, data is easy to source.
13. **KV-cache compression micro-study** — on a small model, compare cache eviction/quantization strategies for long-context inference, measure memory vs. quality tradeoff. Narrow but technically sharp; needs careful scoping to stay cheap.
14. **Dataset-contamination check tool** — build a small, reusable tool/Space that flags n-gram overlap between a benchmark and common pretraining corpora, apply it to 2-3 popular small-model benchmarks as a case study. Infra + methodology-critique angle, differentiated (most people don't publish this).
15. **Distillation data-efficiency study** — for a fixed small student model, vary the amount and curation quality of distillation data from a larger teacher, plot accuracy vs. data budget. Clean, cheap, classic "data efficiency" research question.

## 3. Recommendation

Start with **#1 (spell-correction) as a fast, low-risk warm-up** that fixes an existing portfolio gap end-to-end in under a day, then move to **#2 (embedding model + VecStream benchmark)** as the flagship project — it's the strongest "ML + systems" combination the brief calls out, produces 4 linked artifacts (dataset, model, benchmark code, Space demo) for one Collection, and is fully feasible on this machine. #3 and #6 are the next-best research-depth options once the pipeline/tooling from #1–#2 exists.

I have **not** created, trained, or published anything yet — this is the audit + proposal only, per your request. Tell me which project(s) to start executing, or say "go with your recommendation" and I'll begin #1.

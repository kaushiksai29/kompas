# Evaluation: Hybrid GraphRAG vs. Naive RAG

Corpus: USCIS immigration form instructions. Questions: **28** (hand-built, multi-hop). Settings: top_k=8, graph_hops=2.

| Metric | Naive RAG | Hybrid GraphRAG | Δ (Hybrid − Naive) |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.827 | 0.943 | +0.116 |
| Source-doc recall (gold form's doc retrieved) | 0.789 | 0.845 | +0.057 |
| Answer accuracy (gold forms in answer) | 0.902 | 0.967 | +0.065 |
| Citation precision (citations on gold docs) | 0.710 | 0.723 | +0.013 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.845 | 0.889 | +0.044 |
| Graph-only share of retrieved context | 0.000 | 0.500 | +0.500 |

### Multi-hop subset (20 questions)

| Metric | Naive | Hybrid | Δ |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.808 | 0.921 | +0.113 |
| Source-doc recall (gold form's doc retrieved) | 0.754 | 0.833 | +0.079 |
| Answer accuracy (gold forms in answer) | 0.863 | 0.954 | +0.092 |
| Citation precision (citations on gold docs) | 0.787 | 0.825 | +0.037 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.803 | 0.880 | +0.077 |
| Graph-only share of retrieved context | 0.000 | 0.500 | +0.500 |

### Refusal correctness (out-of-corpus)

Given 6 questions outside the ingested corpus, the system correctly declined **83%** of the time (5/6) instead of inventing an answer.

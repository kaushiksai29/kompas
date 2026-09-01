# Evaluation: Hybrid GraphRAG vs. Naive RAG

Corpus: USCIS immigration form instructions. Questions: **28** (hand-built, multi-hop). Settings: top_k=8, graph_hops=2.

| Metric | Naive RAG | Hybrid GraphRAG | Δ (Hybrid − Naive) |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.807 | 0.890 | +0.083 |
| Source-doc recall (gold form's doc retrieved) | 0.759 | 0.860 | +0.101 |
| Answer accuracy (gold forms in answer) | 0.949 | 0.979 | +0.030 |
| Citation precision (citations on gold docs) | 0.487 | 0.556 | +0.069 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.736 | 0.838 | +0.102 |
| Graph-only share of retrieved context | 0.000 | 0.500 | +0.500 |

### Multi-hop subset (20 questions)

| Metric | Naive | Hybrid | Δ |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.779 | 0.896 | +0.117 |
| Source-doc recall (gold form's doc retrieved) | 0.713 | 0.854 | +0.142 |
| Answer accuracy (gold forms in answer) | 0.929 | 0.971 | +0.042 |
| Citation precision (citations on gold docs) | 0.531 | 0.628 | +0.097 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.740 | 0.850 | +0.110 |
| Graph-only share of retrieved context | 0.000 | 0.500 | +0.500 |

### Refusal correctness (out-of-corpus)

Given 6 questions outside the ingested corpus, the system correctly declined **50%** of the time (3/6) instead of inventing an answer.

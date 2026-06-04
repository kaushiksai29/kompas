# Evaluation: Hybrid GraphRAG vs. Naive RAG

Corpus: USCIS immigration form instructions. Questions: **28** (hand-built, multi-hop). Settings: top_k=8, graph_hops=2.

| Metric | Naive RAG | Hybrid GraphRAG | Δ (Hybrid − Naive) |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.818 | 0.926 | +0.107 |
| Source-doc recall (gold form's doc retrieved) | 0.780 | 0.845 | +0.065 |
| Answer accuracy (gold forms in answer) | 0.899 | 0.926 | +0.027 |
| Citation precision (citations on gold docs) | 0.701 | 0.688 | -0.013 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.867 | 0.875 | +0.008 |

### Multi-hop subset (20 questions)

| Metric | Naive | Hybrid | Δ |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.796 | 0.896 | +0.100 |
| Source-doc recall (gold form's doc retrieved) | 0.742 | 0.833 | +0.092 |
| Answer accuracy (gold forms in answer) | 0.858 | 0.896 | +0.037 |
| Citation precision (citations on gold docs) | 0.775 | 0.759 | -0.016 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.874 | 0.870 | -0.004 |

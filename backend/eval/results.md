# Evaluation: Hybrid GraphRAG vs. Naive RAG

Corpus: USCIS immigration form instructions. Questions: **28** (hand-built, multi-hop). Settings: top_k=8, graph_hops=2.

| Metric | Naive RAG | Hybrid GraphRAG | Δ (Hybrid − Naive) |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.818 | 0.908 | +0.089 |
| Source-doc recall (gold form's doc retrieved) | 0.780 | 0.810 | +0.030 |
| Answer accuracy (gold forms in answer) | 0.890 | 0.961 | +0.071 |
| Citation precision (citations on gold docs) | 0.701 | 0.674 | -0.027 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.852 | 0.909 | +0.057 |

### Multi-hop subset (20 questions)

| Metric | Naive | Hybrid | Δ |
|---|---:|---:|---:|
| Retrieval recall@k (gold forms in context) | 0.796 | 0.921 | +0.125 |
| Source-doc recall (gold form's doc retrieved) | 0.742 | 0.783 | +0.042 |
| Answer accuracy (gold forms in answer) | 0.846 | 0.946 | +0.100 |
| Citation precision (citations on gold docs) | 0.775 | 0.744 | -0.031 |
| LLM-judge accuracy (0-1, answers question + forms) | 0.823 | 0.893 | +0.070 |

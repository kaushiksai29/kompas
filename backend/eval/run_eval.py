"""
Evaluation harness — Hybrid GraphRAG vs. Naive vector RAG.

For each hand-built multi-hop question we run retrieval + citation-grounded
generation in BOTH modes and measure:

  * retrieval_recall  — fraction of the question's gold forms that appear in the
                        retrieved context text (could the answer cite them?).
  * doc_recall        — fraction of gold forms whose OWN instruction document
                        contributed a retrieved chunk (directly shows the graph
                        pulling chunks from other documents).
  * answer_recall     — fraction of gold forms named in the generated answer
                        (a proxy for answer completeness / accuracy).
  * citation_precision— fraction of the answer's citations that point at a gold
                        document (are the citations on-topic?).
  * judge_accuracy    — an LLM-as-judge score in [0,1] rating whether the
                        generated answer correctly and completely addresses the
                        question and names the right forms (semantic accuracy,
                        not just exact form-id overlap).

The hybrid-minus-naive delta on these is the project's innovation claim.

Usage:
    .venv/Scripts/python.exe -m backend.eval.run_eval
    .venv/Scripts/python.exe -m backend.eval.run_eval --top-k 8 --hops 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.llm_client import create_fallback_client
from backend.database.vector_db import VectorDB
from backend.database.graph_db import GraphDB
from backend.search.retriever import HybridRetriever
from backend.search.generator import generate_answer

QA_PATH = Path(__file__).resolve().parent / "qa_set.json"
RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
RESULTS_MD = Path(__file__).resolve().parent / "results.md"

_FORM_IN_TEXT = re.compile(r"\b([A-Z]{1,3}-\d{2,4}[A-Z]?)\b")
_FORM_IN_FILE = re.compile(r"([a-z]{1,3})-?(\d{2,4})([a-z]?)", re.IGNORECASE)


def forms_in_text(text: str) -> set[str]:
    return {m.upper() for m in _FORM_IN_TEXT.findall(text or "")}


def form_of_filepath(fp: str) -> str | None:
    stem = re.split(r"[\\/]", fp or "")[-1].lower().rsplit(".", 1)[0]
    stem = stem[:-5] if stem.endswith("instr") else stem
    m = _FORM_IN_FILE.match(stem)
    if not m:
        return None
    p, n, s = m.groups()
    return f"{p.upper()}-{n}{s.upper()}"


def _frac(num: int, den: int) -> float:
    return num / den if den else 0.0


_SCORE_RE = re.compile(r"(?:score\D*)?(-?\d+(?:\.\d+)?)")


def _parse_score(text: str) -> float:
    """Robustly extract a 0.0–1.0 numeric score from an LLM judge response.

    Handles plain numbers, JSON-ish ``{"score": 0.8}``, percentages, and stray
    prose. Returns 0.0 if nothing parseable is found, and clamps to [0, 1].
    """
    if not text:
        return 0.0
    # Prefer a JSON "score" field if present.
    m = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if not m:
        m = _SCORE_RE.search(text)
    if not m:
        return 0.0
    try:
        val = float(m.group(1))
    except ValueError:
        return 0.0
    # A model may answer on a 0–100 or 0–10 scale; normalize the obvious cases.
    if val > 1.0:
        if val <= 10.0:
            val = val / 10.0
        else:
            val = val / 100.0
    return max(0.0, min(1.0, val))


_JUDGE_SYSTEM = (
    "You are a strict grader for a US-immigration question-answering system. "
    "You will be given a user question, the list of gold-standard USCIS forms a "
    "correct and complete answer must reference, and a candidate answer. "
    "Score from 0.0 to 1.0 how well the candidate answer correctly and "
    "completely addresses the question AND names the right forms. "
    "1.0 = fully correct and complete (addresses the question and names all the "
    "right forms with no wrong ones); 0.0 = wrong, irrelevant, or empty. "
    "Penalize missing required forms and naming incorrect forms. "
    'Respond with ONLY a JSON object of the form {"score": <number>} and nothing else.'
)


def judge_accuracy(question: str, gold_forms, generated_answer: str, client) -> float:
    """LLM-as-judge: score in [0,1] whether *generated_answer* correctly and
    completely answers *question* and names the *gold_forms*.

    Uses the Groq-backed fallback client. Any failure (e.g. provider error or
    unparseable output) degrades gracefully to 0.0 so the eval never crashes.
    """
    gold = ", ".join(sorted({g.upper() for g in gold_forms})) or "(none)"
    user = (
        f"Question:\n{question}\n\n"
        f"Gold-standard forms a correct answer must reference: {gold}\n\n"
        f"Candidate answer:\n{generated_answer or '(empty)'}\n\n"
        'Return ONLY {"score": <0.0-1.0>}.'
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    try:
        raw = client.generate_sync(messages=messages)
    except Exception as exc:  # noqa: BLE001
        print(f"    [judge] scoring failed: {str(exc)[:120]}", flush=True)
        return 0.0
    return _parse_score(raw)


def evaluate(retriever: HybridRetriever, client, questions, top_k: int, hops: int):
    rows = []
    for i, q in enumerate(questions):
        gold = {g.upper() for g in q["gold_forms"]}
        per_mode = {}
        for mode in ("hybrid", "naive"):
            use_graph = mode == "hybrid"
            res = retriever.retrieve(
                q["question"], top_k_vector=top_k, top_k_graph=top_k,
                graph_hops=hops, use_graph=use_graph,
            )
            chunks = res.merged_chunks
            ctx_text = " ".join(c.get("content", "") for c in chunks)
            retrieved_forms = forms_in_text(ctx_text)
            retrieved_docs = {form_of_filepath(c.get("filepath", "")) for c in chunks}
            retrieved_docs.discard(None)

            gen = generate_answer(q["question"], chunks, client,
                                  subgraph=res.subgraph if use_graph else None)
            answer = gen["answer"]
            answer_forms = forms_in_text(answer)
            cites = gen.get("citations", [])
            cite_docs = [form_of_filepath(c.get("filepath", "")) for c in cites]
            cite_hits = sum(1 for d in cite_docs if d in gold)

            judge = judge_accuracy(q["question"], gold, answer, client)

            per_mode[mode] = {
                "retrieval_recall": _frac(len(gold & retrieved_forms), len(gold)),
                "doc_recall": _frac(len(gold & retrieved_docs), len(gold)),
                "answer_recall": _frac(len(gold & answer_forms), len(gold)),
                "citation_precision": _frac(cite_hits, len(cite_docs)),
                "judge_accuracy": judge,
                "n_graph_chunks": len(res.graph_chunks),
                "subgraph_nodes": len(res.subgraph["nodes"]),
            }
        rows.append({"id": q["id"], "category": q["category"], "hops": q["hops"],
                     "gold_forms": sorted(gold), "modes": per_mode})
        h, n = per_mode["hybrid"], per_mode["naive"]
        print(f"  [{i+1:>2}/{len(questions)}] {q['id']:8} "
              f"ret H={h['retrieval_recall']:.2f}/N={n['retrieval_recall']:.2f}  "
              f"ans H={h['answer_recall']:.2f}/N={n['answer_recall']:.2f}  "
              f"judge H={h['judge_accuracy']:.2f}/N={n['judge_accuracy']:.2f}", flush=True)
    return rows


def aggregate(rows):
    metrics = ["retrieval_recall", "doc_recall", "answer_recall",
               "citation_precision", "judge_accuracy"]
    agg = {m: {"hybrid": 0.0, "naive": 0.0} for m in metrics}
    for r in rows:
        for m in metrics:
            agg[m]["hybrid"] += r["modes"]["hybrid"][m]
            agg[m]["naive"] += r["modes"]["naive"][m]
    n = len(rows)
    for m in metrics:
        agg[m]["hybrid"] /= n
        agg[m]["naive"] /= n
    return agg, metrics


def render_md(agg, metrics, rows, top_k, hops) -> str:
    n = len(rows)
    lines = [
        "# Evaluation: Hybrid GraphRAG vs. Naive RAG",
        "",
        f"Corpus: USCIS immigration form instructions. Questions: **{n}** "
        f"(hand-built, multi-hop). Settings: top_k={top_k}, graph_hops={hops}.",
        "",
        "| Metric | Naive RAG | Hybrid GraphRAG | Δ (Hybrid − Naive) |",
        "|---|---:|---:|---:|",
    ]
    label = {
        "retrieval_recall": "Retrieval recall@k (gold forms in context)",
        "doc_recall": "Source-doc recall (gold form's doc retrieved)",
        "answer_recall": "Answer accuracy (gold forms in answer)",
        "citation_precision": "Citation precision (citations on gold docs)",
        "judge_accuracy": "LLM-judge accuracy (0-1, answers question + forms)",
    }
    for m in metrics:
        nv, hy = agg[m]["naive"], agg[m]["hybrid"]
        lines.append(f"| {label[m]} | {nv:.3f} | {hy:.3f} | {hy - nv:+.3f} |")
    multi = [r for r in rows if r["hops"] == "multi"]
    if multi:
        lines += ["", f"### Multi-hop subset ({len(multi)} questions)", "",
                  "| Metric | Naive | Hybrid | Δ |", "|---|---:|---:|---:|"]
        for m in metrics:
            nv = sum(r["modes"]["naive"][m] for r in multi) / len(multi)
            hy = sum(r["modes"]["hybrid"][m] for r in multi) / len(multi)
            lines.append(f"| {label[m]} | {nv:.3f} | {hy:.3f} | {hy - nv:+.3f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hops", type=int, default=2)
    args = ap.parse_args()

    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    questions = qa["questions"]

    settings = get_settings()
    from backend.embeddings import embed_one, load
    load()
    emb = embed_one
    vdb = VectorDB(db_path=str(settings.resolved_sqlite_path), embedding_dim=384)
    gdb = GraphDB(db_path=str(settings.resolved_kuzu_path))
    client = create_fallback_client(settings)
    retriever = HybridRetriever(vdb, gdb, emb, client)

    print(f"Evaluating {len(questions)} questions (top_k={args.top_k}, hops={args.hops}) ...")
    t0 = time.time()
    rows = evaluate(retriever, client, questions, args.top_k, args.hops)
    agg, metrics = aggregate(rows)
    md = render_md(agg, metrics, rows, args.top_k, args.hops)

    RESULTS_JSON.write_text(json.dumps(
        {"settings": {"top_k": args.top_k, "hops": args.hops},
         "aggregate": agg, "rows": rows}, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(md, encoding="utf-8")

    print(f"\nDone in {time.time() - t0:.0f}s.\n")
    print(md)
    gdb.close(); vdb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

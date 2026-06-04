"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import QueryPanel from "./components/QueryPanel";
import AnswerPanel from "./components/AnswerPanel";
import StatsBar from "./components/StatsBar";

// 3D evidence model — hidden by default, lazy-loaded only when the lawyer
// chooses to show the map (and client-only, it touches WebGL/window).
const GraphViewer3D = dynamic(() => import("./components/GraphViewer3D"), {
  ssr: false,
});

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://graphrag-backend-production-f3da.up.railway.app";

export interface Citation {
  citation_id: string;
  chunk_id: string;
  filepath: string;
  heading_path: string;
  retrieval_source: string;
  preview: string;
  source_url?: string;
  page?: number;
}

export interface QueryResult {
  answer: string;
  citations: Citation[];
  subgraph: { nodes: any[]; edges: any[] };
  query_entities: any[];
  matched_entities: any[];
  chunks_used: number;
  retrieval_mode: string;
}

export default function Home() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"hybrid" | "naive">("hybrid");
  const [highlightedCitation, setHighlightedCitation] = useState<string | null>(null);
  const [stats, setStats] = useState<any>(null);
  // Evidence map hidden by default — a lawyer doesn't want it in their face.
  const [showMap, setShowMap] = useState(false);
  const answerRef = useRef<HTMLDivElement>(null);

  const handleQuery = useCallback(
    async (query: string) => {
      setLoading(true);
      setError(null);
      setResult(null);

      try {
        const response = await fetch(`${API_BASE}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            mode,
            top_k: 10,
            graph_hops: 2,
          }),
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const data: QueryResult = await response.json();
        setResult(data);

        // Scroll to answer
        setTimeout(() => {
          answerRef.current?.scrollIntoView({ behavior: "smooth" });
        }, 100);
      } catch (err: any) {
        setError(err.message || "Failed to connect to the server");
      } finally {
        setLoading(false);
      }
    },
    [mode]
  );

  // Fetch corpus stats on mount (shown in the empty state)
  useEffect(() => {
    fetch(`${API_BASE}/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, []);

  return (
    <main className="min-h-screen flex flex-col">
      {/* ── Header ────────────────────────────────────────────── */}
      <header className="reveal reveal-1 px-4 sm:px-6 py-4 sm:py-5 border-b-2 border-[var(--color-border-primary)]">
        <div className="max-w-7xl mx-auto flex items-end justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3.5">
            {/* KOMPAS compass mark */}
            <img
              src="/kompas-mark.png"
              alt="KOMPAS compass"
              width={44}
              height={44}
              className="w-11 h-11 shrink-0"
              style={{ mixBlendMode: "multiply" }}
            />
            <div>
              <h1 className="wordmark text-[26px] leading-none tracking-[0.06em]">
                KOMPAS
              </h1>
              <p className="docline mt-1.5">
                Find your direction through US immigration
              </p>
            </div>
          </div>

          {/* Search mode — plain words, not jargon */}
          <div className="flex items-center gap-2">
            <span className="docline hidden sm:inline mr-1">Search</span>
            <button
              onClick={() => setMode("hybrid")}
              data-active={mode === "hybrid"}
              className="stamp"
              title="Connects related forms so you don't miss a prerequisite"
            >
              Connected
            </button>
            <button
              onClick={() => setMode("naive")}
              data-active={mode === "naive"}
              className="stamp"
              title="Plain text search only"
            >
              Simple
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Content ──────────────────────────────────────── */}
      <div className="flex-1 flex flex-col max-w-7xl mx-auto w-full px-4 sm:px-6 py-5 sm:py-6 gap-6">
        {/* Query Input */}
        <div className="reveal reveal-2">
          <QueryPanel onSubmit={handleQuery} loading={loading} />
        </div>

        {/* Error */}
        {error && (
          <div
            className="glass-panel p-4 animate-fade-in border-l-4"
            style={{ borderLeftColor: "var(--color-accent-primary)" }}
          >
            <p className="text-sm flex items-center gap-2 text-[var(--color-text-secondary)]">
              <span className="docline text-[var(--color-accent-primary)]">
                Something went wrong
              </span>
              {error}
            </p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="space-y-4 animate-fade-in">
            <div className="shimmer h-6 w-48" />
            <div className="shimmer h-4 w-full" />
            <div className="shimmer h-4 w-3/4" />
            <div className="shimmer h-4 w-5/6" />
            <div className="shimmer h-32 w-full mt-4" />
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="space-y-6 animate-fade-in">
            {/* Stats Bar + evidence-map toggle.
                The map is hidden by default — a lawyer wants the answer, not a
                graph in their face. They opt into the 3D evidence model. */}
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div className="flex-1 min-w-[16rem]">
                <StatsBar result={result} mode={mode} />
              </div>
              <button
                onClick={() => setShowMap((v) => !v)}
                data-active={showMap}
                className="stamp shrink-0"
                title="See how the forms connect"
              >
                {showMap ? "Hide the map" : "Show the map ↗"}
              </button>
            </div>

            {/* Answer — full width by default */}
            <div ref={answerRef}>
              <AnswerPanel
                answer={result.answer}
                citations={result.citations}
                highlightedCitation={highlightedCitation}
                onCitationClick={setHighlightedCitation}
              />
            </div>

            {/* 3D evidence model — full width, only when opted in */}
            {showMap && (
              <div className="animate-fade-in">
                <GraphViewer3D subgraph={result.subgraph} />
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {!result && !loading && !error && (
          <div className="reveal reveal-3 flex-1 flex items-center justify-center py-14">
            <div className="max-w-2xl w-full">
              <div className="paper-panel px-5 sm:px-8 py-7 sm:py-9">
                <p className="docline">Start here</p>
                <h2 className="wordmark text-2xl mt-3 leading-tight">
                  Find the right forms, and what they require.
                </h2>
                <p className="mt-3 text-[15px] leading-relaxed text-[var(--color-text-secondary)]">
                  Ask about US immigration forms in your own words. KOMPAS points
                  you to the{" "}
                  <span className="text-[var(--color-text-primary)] font-semibold">
                    right forms
                  </span>
                  , the{" "}
                  <span className="text-[var(--color-text-primary)] font-semibold">
                    documents you will need
                  </span>
                  , and how they{" "}
                  <span className="text-[var(--color-accent-primary)] font-semibold">
                    connect
                  </span>{" "}
                  — and shows you exactly where each answer comes from in the
                  official USCIS instructions.
                </p>

                <p className="docline mt-7 mb-3 pt-4 border-t border-[var(--color-border-subtle)]">
                  Try one of these
                </p>
                <div className="flex flex-col gap-2">
                  {[
                    "I'm adjusting status through marriage — which forms and documents do I need?",
                    "Can I work and travel while my green card application is pending?",
                    "After my I-140 is approved, how do I get my green card from inside the US?",
                  ].map((q, i) => (
                    <button
                      key={q}
                      onClick={() => handleQuery(q)}
                      className="group text-left flex items-start gap-3 px-3 py-2.5 border border-[var(--color-border-subtle)] hover:border-[var(--color-accent-primary)] transition-colors cursor-pointer"
                      style={{ borderRadius: 2 }}
                    >
                      <span className="font-mono text-xs font-bold text-[var(--color-accent-primary)] mt-0.5">
                        №{String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="text-sm text-[var(--color-text-secondary)] group-hover:text-[var(--color-text-primary)]">
                        {q}
                      </span>
                    </button>
                  ))}
                </div>

                {/* Case metrics — confirms the record is loaded */}
                {stats &&
                  (stats.graph_db?.entities || stats.vector_db?.total_chunks) && (
                    <div className="flex flex-wrap gap-x-6 gap-y-1 mt-7 pt-4 border-t border-[var(--color-border-subtle)] font-mono text-[11px] text-[var(--color-text-muted)]">
                      {stats.vector_db?.total_chunks != null && (
                        <span>
                          <span className="text-[var(--color-text-primary)] font-bold">
                            {stats.vector_db.total_chunks}
                          </span>{" "}
                          PAGES
                        </span>
                      )}
                      {stats.graph_db?.entities != null && (
                        <span>
                          <span className="text-[var(--color-text-primary)] font-bold">
                            {stats.graph_db.entities}
                          </span>{" "}
                          TOPICS
                        </span>
                      )}
                      {stats.graph_db?.relations != null && (
                        <span>
                          <span className="text-[var(--color-accent-primary)] font-bold">
                            {stats.graph_db.relations}
                          </span>{" "}
                          CONNECTIONS
                        </span>
                      )}
                    </div>
                  )}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

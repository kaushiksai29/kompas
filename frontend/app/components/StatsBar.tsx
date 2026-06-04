"use client";

import type { QueryResult } from "../page";

interface StatsBarProps {
  result: QueryResult;
  mode: string;
}

export default function StatsBar({ result, mode }: StatsBarProps) {
  const ink = "text-[var(--color-text-primary)]";
  const stats = [
    {
      label: "Search",
      value: mode === "hybrid" ? "Connected" : "Simple",
      color: "text-[var(--color-text-primary)]",
    },
    { label: "Pages", value: String(result.chunks_used), color: ink },
    { label: "Sources", value: String(result.citations.length), color: ink },
    {
      label: "Topics",
      value: String(result.subgraph?.nodes?.length || 0),
      color: ink,
    },
    {
      label: "Connections",
      value: String(result.subgraph?.edges?.length || 0),
      color: ink,
    },
  ];

  // Count retrieval sources
  const vectorCount = result.citations.filter(
    (c) => c.retrieval_source === "vector"
  ).length;
  const graphCount = result.citations.filter(
    (c) => c.retrieval_source === "graph"
  ).length;
  const bothCount = result.citations.filter(
    (c) => c.retrieval_source === "both"
  ).length;

  return (
    <div className="glass-panel px-5 py-2.5 flex items-center justify-between flex-wrap gap-x-6 gap-y-2">
      {/* Case metrics ledger */}
      <div className="flex items-center gap-x-6 gap-y-1 flex-wrap">
        {stats.map((s) => (
          <div key={s.label} className="flex items-baseline gap-1.5">
            <span className="text-[10px] text-[var(--color-text-muted)] uppercase tracking-[0.14em] font-mono">
              {s.label}
            </span>
            <span className={`text-sm font-bold font-mono ${s.color}`}>
              {s.value}
            </span>
          </div>
        ))}
      </div>

      {/* Retrieval channel breakdown */}
      {mode === "hybrid" && (
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider">
          {vectorCount > 0 && (
            <span className="px-1.5 py-0.5 border border-[var(--color-border-primary)] text-[var(--color-text-muted)]">
              {vectorCount} vector
            </span>
          )}
          {graphCount > 0 && (
            <span className="px-1.5 py-0.5 border border-[var(--color-accent-primary)] text-[var(--color-accent-primary)]">
              {graphCount} graph
            </span>
          )}
          {bothCount > 0 && (
            <span className="px-1.5 py-0.5 border border-[var(--color-accent-secondary)] text-[var(--color-accent-secondary)]">
              {bothCount} both
            </span>
          )}
        </div>
      )}
    </div>
  );
}

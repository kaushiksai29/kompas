"use client";

import { useMemo, useRef, useCallback } from "react";
import dynamic from "next/dynamic";

// react-force-graph-3d touches `window`/WebGL on import, so it must be loaded
// client-side only — SSR would otherwise throw "window is not defined".
const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
  ssr: false,
  loading: () => (
    <div className="h-full flex items-center justify-center">
      <p className="docline animate-pulse">Rendering model…</p>
    </div>
  ),
});

interface GraphViewer3DProps {
  subgraph: { nodes: any[]; edges: any[] };
}

// Same earthy "case file" palette as GraphViewer.tsx so the 2D board and the
// 3D model read identically (oxblood form, ink-blue agency, olive, ochre…).
// Cold, low-chroma palette — nodes read as a precise instrument, not decoration.
const TYPE_COLORS: Record<string, string> = {
  form: "#20262b", // near-black slate — the anchor
  agency: "#3f5566", // cold steel
  benefit_or_status: "#4b5a51", // cold slate-green
  eligibility_condition: "#646570", // cold grey-violet
  evidence_or_document: "#4f6168", // cold teal-grey
  statute: "#5a554f", // cold taupe
  applicant_category: "#585f66", // cold grey
  fee: "#8b9197", // light cold grey
  chunk: "#8b9197",
};

const TYPE_DISPLAY: Record<string, string> = {
  form: "Form",
  agency: "Agency",
  benefit_or_status: "Benefit / Status",
  eligibility_condition: "Eligibility",
  evidence_or_document: "Evidence / Doc",
  statute: "Statute",
  applicant_category: "Applicant",
  fee: "Fee",
};

const OXBLOOD = "#20262b";
const PAPER_BG = "#eef0f1";

export default function GraphViewer3D({ subgraph }: GraphViewer3DProps) {
  const fgRef = useRef<any>(null);

  // react-force-graph expects { nodes: [{id,…}], links: [{source,target,…}] }.
  // Map the typed-edge subgraph into that shape and drop dangling links.
  const graphData = useMemo(() => {
    const rawNodes = subgraph?.nodes || [];
    const rawEdges = subgraph?.edges || [];
    const nodeIds = new Set(rawNodes.map((n) => n.id));

    const links = [...rawEdges]
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
      .sort((a, b) => (b.weight || 0) - (a.weight || 0))
      .slice(0, 80)
      .map((e) => ({
        source: e.source,
        target: e.target,
        predicate: (e.predicate || "").replace(/_/g, " "),
      }));

    // Only keep nodes that actually connect to something — the floating
    // singletons are what made the 3D view an unreadable dust cloud.
    const connected = new Set<string>();
    links.forEach((l) => {
      connected.add(l.source as string);
      connected.add(l.target as string);
    });

    const nodes = rawNodes
      .filter((n) => connected.has(n.id))
      .map((n) => {
        const type = n.type || "unknown";
        return {
          id: n.id,
          name: n.name || n.id,
          type,
          description: n.description || "",
          color: TYPE_COLORS[type] || "#8a7d65",
        };
      });

    return { nodes, links };
  }, [subgraph]);

  const nodeCount = subgraph?.nodes?.length || 0;
  const edgeCount = subgraph?.edges?.length || 0;

  const presentTypes = useMemo(() => {
    const s = new Set<string>();
    (subgraph?.nodes || []).forEach((n) => s.add(n.type || "unknown"));
    return s;
  }, [subgraph]);

  const nodeLabel = useCallback((node: any) => {
    const display = TYPE_DISPLAY[node.type] || String(node.type || "").replace(/_/g, " ");
    return `<div style="
        background:#ffffff;
        border:1px solid ${OXBLOOD};
        color:#211c15;
        padding:4px 8px;
        border-radius:2px;
        font-family:var(--font-space-mono),monospace;
        font-size:11px;
        max-width:240px;
      ">
        <strong>${node.name}</strong>
        <div style="color:${node.color};font-size:9px;letter-spacing:.05em;text-transform:uppercase;margin-top:2px;">${display}</div>
      </div>`;
  }, []);

  const linkLabel = useCallback((link: any) => {
    if (!link.predicate) return "";
    return `<div style="
        background:#ffffff;
        border:0.5px solid ${OXBLOOD};
        color:#463d2f;
        padding:2px 5px;
        border-radius:1px;
        font-family:var(--font-space-mono),monospace;
        font-size:9px;
        font-weight:700;
      ">${link.predicate}</div>`;
  }, []);

  if (nodeCount === 0) {
    return (
      <div className="glass-panel h-[500px] flex items-center justify-center">
        <div className="text-center space-y-2 px-6">
          <p className="docline">The map</p>
          <p className="font-display italic text-lg text-[var(--color-text-secondary)]">
            Nothing to show yet.
          </p>
          <p className="text-xs text-[var(--color-text-muted)] max-w-[15rem] mx-auto">
            Ask a question with{" "}
            <span className="text-[var(--color-accent-primary)] font-semibold">
              Connected
            </span>{" "}
            search to see how the forms link together.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel overflow-hidden h-[500px] relative">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-[var(--color-border-primary)] flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span
            className="w-2.5 h-2.5 bg-[var(--color-accent-primary)] inline-block"
            style={{ transform: "rotate(45deg)" }}
          />
          <h2 className="font-display text-base font-semibold text-[var(--color-text-primary)]">
            How the forms connect
          </h2>
        </div>
        <span className="docline">
          {nodeCount} nodes · {edgeCount} threads
        </span>
      </div>

      {/* 3D Graph */}
      <div className="h-[calc(100%-40px)] relative">
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData}
          backgroundColor={PAPER_BG}
          showNavInfo={false}
          nodeLabel={nodeLabel as any}
          nodeColor={(n: any) => n.color}
          nodeOpacity={1}
          nodeResolution={20}
          nodeRelSize={7}
          nodeVal={3}
          linkLabel={linkLabel as any}
          linkColor={() => OXBLOOD}
          linkOpacity={0.7}
          linkWidth={1.6}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkDirectionalArrowColor={() => OXBLOOD}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={2}
          linkDirectionalParticleColor={() => OXBLOOD}
          warmupTicks={60}
          cooldownTicks={120}
          onEngineStop={() => fgRef.current?.zoomToFit?.(500, 60)}
        />
      </div>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 glass-panel px-3 py-2 flex flex-wrap gap-x-3 gap-y-1 max-w-[280px]">
        {Object.entries(TYPE_DISPLAY)
          .filter(([k]) => presentTypes.has(k))
          .map(([type, label]) => (
            <span
              key={type}
              className="text-[9px] flex items-center gap-1 whitespace-nowrap"
              style={{ color: TYPE_COLORS[type] }}
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: TYPE_COLORS[type] }}
              />
              {label}
            </span>
          ))}
      </div>
    </div>
  );
}

"use client";

import { useMemo, useCallback, useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

interface GraphViewerProps {
  subgraph: { nodes: any[]; edges: any[] };
  highlightedCitation: string | null;
  onNodeClick: (nodeId: string) => void;
}

// Color map for immigration entity types — earthy "case file" palette
// (oxblood, ink-blue, olive, ochre, sepia) so nodes read on warm paper.
const TYPE_COLORS: Record<string, string> = {
  form: "#9e2b23",                // oxblood — the protagonist
  agency: "#294a63",             // ink blue
  benefit_or_status: "#5b6b39",  // olive
  eligibility_condition: "#9a6b1e", // ochre
  evidence_or_document: "#3f6661",  // muted teal
  statute: "#7a4b34",            // sepia brown
  applicant_category: "#6a4a52", // mauve-brown
  fee: "#8a7d65",                // faded sepia
  chunk: "#8a7d65",
};

const TYPE_LABELS: Record<string, string> = {
  form: "📄",
  agency: "🏢",
  benefit_or_status: "✅",
  eligibility_condition: "⚠️",
  evidence_or_document: "📋",
  statute: "📜",
  applicant_category: "👤",
  fee: "💵",
  chunk: "📑",
};

/* Human-readable names for the legend (avoids raw snake_case). */
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

/* Deterministic pseudo-random in [-1, 1] so the layout is stable between
   renders (important for screenshots / demos — no jitter on re-render). */
function jitter(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

function layoutNodes(rawNodes: any[]): Node[] {
  if (!rawNodes || rawNodes.length === 0) return [];

  // Force-directed-like layout using circular arrangement by type
  const byType: Record<string, any[]> = {};
  rawNodes.forEach((n) => {
    const t = n.type || "unknown";
    if (!byType[t]) byType[t] = [];
    byType[t].push(n);
  });

  const types = Object.keys(byType);
  const nodes: Node[] = [];
  const cx = 400, cy = 300;
  const outerRadius = 250;

  types.forEach((type, typeIdx) => {
    const angleStep = (2 * Math.PI) / types.length;
    const groupAngle = typeIdx * angleStep;
    const groupCx = cx + outerRadius * Math.cos(groupAngle);
    const groupCy = cy + outerRadius * Math.sin(groupAngle);

    const items = byType[type];
    const innerRadius = Math.min(80, items.length * 20);

    items.forEach((item, i) => {
      const a = items.length === 1
        ? groupAngle
        : groupAngle - 0.5 + (i / (items.length - 1));
      const x = groupCx + innerRadius * Math.cos(a) + jitter(typeIdx * 97 + i) * 12;
      const y = groupCy + innerRadius * Math.sin(a) + jitter(typeIdx * 131 + i + 7) * 12;
      
      const color = TYPE_COLORS[type] || "#475569";
      const emoji = TYPE_LABELS[type] || "•";
      const label = item.name || item.id;
      const truncatedLabel = label.length > 30 ? label.slice(0, 28) + "…" : label;

      nodes.push({
        id: item.id,
        position: { x, y },
        data: {
          label: `${emoji} ${truncatedLabel}`,
          fullName: label,
          type: type,
          description: item.description || "",
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: {
          background: `${color}20`,
          border: `1.5px solid ${color}`,
          borderRadius: "10px",
          padding: "6px 10px",
          fontSize: "10px",
          fontWeight: 500,
          color: color,
          maxWidth: "160px",
          whiteSpace: "nowrap" as const,
          overflow: "hidden",
          textOverflow: "ellipsis",
        },
      });
    });
  });

  return nodes;
}

function layoutEdges(rawEdges: any[]): Edge[] {
  if (!rawEdges) return [];

  // Only show top N edges to avoid visual clutter
  const sorted = [...rawEdges]
    .sort((a, b) => (b.weight || 0) - (a.weight || 0))
    .slice(0, 60);

  return sorted.map((e, i) => ({
    id: `e-${i}`,
    source: e.source,
    target: e.target,
    label: (e.predicate || "").replace(/_/g, " "),
    labelStyle: {
      fill: "#463d2f",
      fontSize: 9,
      fontWeight: 700,
      fontFamily: "var(--font-space-mono), monospace",
    },
    labelBgStyle: { fill: "#f4eede", fillOpacity: 0.92, stroke: "#9e2b23", strokeWidth: 0.5 },
    labelBgPadding: [4, 2] as [number, number],
    labelBgBorderRadius: 1,
    markerEnd: { type: "arrowclosed" as any, color: "#9e2b23" },
    animated: (e.weight || 0) > 2,
    style: {
      stroke: "#9e2b23",
      strokeOpacity: 0.5,
      strokeWidth: Math.min(3, 1 + (e.weight || 0) * 0.3),
    },
  }));
}

export default function GraphViewer({
  subgraph,
  highlightedCitation,
  onNodeClick,
}: GraphViewerProps) {
  const initialNodes = useMemo(
    () => layoutNodes(subgraph?.nodes || []),
    [subgraph]
  );
  const initialEdges = useMemo(
    () => layoutEdges(subgraph?.edges || []),
    [subgraph]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Re-sync React Flow state when a new query returns a different subgraph
  // (useNodesState/useEdgesState only seed from their initial value once).
  useEffect(() => {
    setNodes(initialNodes);
  }, [initialNodes, setNodes]);
  useEffect(() => {
    setEdges(initialEdges);
  }, [initialEdges, setEdges]);

  const nodeCount = subgraph?.nodes?.length || 0;
  const edgeCount = subgraph?.edges?.length || 0;

  // Which entity types actually appear, so the legend only lists relevant ones.
  const presentTypes = useMemo(() => {
    const s = new Set<string>();
    (subgraph?.nodes || []).forEach((n) => s.add(n.type || "unknown"));
    return s;
  }, [subgraph]);

  const handleNodeClick = useCallback(
    (_: any, node: Node) => {
      onNodeClick(node.id);
    },
    [onNodeClick]
  );

  if (nodeCount === 0) {
    return (
      <div className="glass-panel h-[500px] flex items-center justify-center">
        <div className="text-center space-y-2 px-6">
          <p className="docline">Evidence Board</p>
          <p className="font-display italic text-lg text-[var(--color-text-secondary)]">
            No threads pinned.
          </p>
          <p className="text-xs text-[var(--color-text-muted)] max-w-[15rem] mx-auto">
            Switch the method stamp to{" "}
            <span className="text-[var(--color-accent-primary)] font-semibold">
              Hybrid · Graph
            </span>{" "}
            to trace the typed connections between forms.
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
            Evidence Board
          </h2>
        </div>
        <span className="docline">
          {nodeCount} nodes · {edgeCount} threads
        </span>
      </div>

      {/* Graph */}
      <div className="h-[calc(100%-40px)]">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
          minZoom={0.3}
          maxZoom={2}
        >
          <Background color="rgba(33, 28, 21, 0.10)" gap={22} />
          <Controls
            showInteractive={false}
            style={{
              background: "var(--color-bg-secondary)",
              borderColor: "var(--color-border-primary)",
              borderRadius: "2px",
            }}
          />
          <MiniMap
            style={{
              background: "var(--color-bg-tertiary)",
              borderRadius: "2px",
            }}
            maskColor="rgba(236, 228, 210, 0.7)"
            nodeColor={(n) => TYPE_COLORS[String(n.data?.type || "")] || "#8a7d65"}
          />
        </ReactFlow>
      </div>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 glass-panel px-3 py-2 flex flex-wrap gap-x-3 gap-y-1 max-w-[280px]">
        {Object.entries(TYPE_LABELS)
          .filter(([k]) => k !== "chunk" && presentTypes.has(k))
          .map(([type]) => (
            <span
              key={type}
              className="text-[9px] flex items-center gap-1 whitespace-nowrap"
              style={{ color: TYPE_COLORS[type] }}
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: TYPE_COLORS[type] }}
              />
              {TYPE_DISPLAY[type] || type.replace(/_/g, " ")}
            </span>
          ))}
      </div>
    </div>
  );
}

"use client";

import { useMemo } from "react";
import type { Citation } from "../page";

interface AnswerPanelProps {
  answer: string;
  citations: Citation[];
  highlightedCitation: string | null;
  onCitationClick: (id: string | null) => void;
}

/** Derive a clean form label from a filepath (no raw C:\ path shown) */
function formLabel(filepath: string): string {
  const base = filepath.split(/[/\\]/).pop() || filepath;
  return base.replace(/\.[^.]+$/, "") || base;
}

/** Convert markdown-like text to basic HTML for rendering */
function renderMarkdown(text: string): string {
  let html = text
    // Headers
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Code
    .replace(/`(.+?)`/g, '<code>$1</code>')
    // Lists
    .replace(/^\* (.+)$/gm, '<li>$1</li>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li><span class="ledger-num">$1.</span>$2</li>')
    // Paragraphs (double newline)
    .replace(/\n\n/g, '</p><p>')
    // Single newlines within paragraphs
    .replace(/\n/g, '<br/>');

  // Wrap consecutive list items in <ul>
  html = html.replace(
    /(?:<li>[\s\S]*?<\/li>\s*)+/g,
    (match) => `<ul>${match}</ul>`
  );

  // Highlight citation tags [1], [2], etc.
  html = html.replace(
    /\[(\d+)\]/g,
    '<span class="citation-tag" data-citation="$1">$1</span>'
  );

  return `<p>${html}</p>`;
}

export default function AnswerPanel({
  answer,
  citations,
  highlightedCitation,
  onCitationClick,
}: AnswerPanelProps) {
  const renderedAnswer = useMemo(() => renderMarkdown(answer), [answer]);

  return (
    <div className="glass-panel overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-[var(--color-border-primary)] flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span
            className="w-2.5 h-2.5 bg-[var(--color-accent-primary)] inline-block"
            style={{ transform: "rotate(45deg)" }}
          />
          <h2 className="font-display text-base font-semibold text-[var(--color-text-primary)]">
            Answer
          </h2>
        </div>
        <span className="docline">{citations.length} sources</span>
      </div>

      {/* Answer Content */}
      <div
        className="px-5 py-4 answer-content text-sm leading-relaxed max-h-[60vh] overflow-y-auto"
        dangerouslySetInnerHTML={{ __html: renderedAnswer }}
        onClick={(e) => {
          const target = e.target as HTMLElement;
          if (target.classList.contains("citation-tag")) {
            const citationId = target.dataset.citation || null;
            onCitationClick(citationId === highlightedCitation ? null : citationId);
          }
        }}
      />

      {/* Citations Footer */}
      {citations.length > 0 && (
        <div className="px-5 py-3 border-t border-[var(--color-border-primary)] space-y-2 bg-[var(--color-bg-glass)]">
          <h3 className="docline">Sources</h3>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {citations.map((c, i) => (
              <div
                key={c.chunk_id || i}
                className={`flex items-start gap-2.5 p-2 text-xs transition-colors cursor-pointer border-l-2 ${
                  highlightedCitation === String(i + 1)
                    ? "border-[var(--color-accent-primary)] bg-[var(--color-bg-tertiary)]"
                    : "border-transparent hover:bg-[var(--color-bg-glass)]"
                }`}
                onClick={() =>
                  onCitationClick(
                    highlightedCitation === String(i + 1) ? null : String(i + 1)
                  )
                }
                id={`citation-${i + 1}`}
              >
                <span className="font-mono text-xs font-bold text-[var(--color-accent-primary)] shrink-0 mt-0.5">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className="min-w-0">
                  {(() => {
                    const label = c.heading_path || formLabel(c.filepath);
                    const href = c.page && c.source_url
                      ? `${c.source_url}#page=${c.page}`
                      : c.source_url;
                    const inner = (
                      <>
                        {label}
                        {c.page ? (
                          <span className="font-mono text-[var(--color-text-muted)] not-italic">
                            {" "}· p.{c.page}
                          </span>
                        ) : null}
                      </>
                    );
                    return href ? (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener"
                        onClick={(e) => e.stopPropagation()}
                        className="text-[var(--color-accent-primary)] font-semibold truncate block hover:underline"
                      >
                        {inner}
                        <span className="ml-1 text-[10px] align-baseline">↗</span>
                      </a>
                    ) : (
                      <p className="text-[var(--color-text-primary)] font-semibold truncate">
                        {inner}
                      </p>
                    );
                  })()}
                  <p className="text-[var(--color-text-muted)] truncate mt-0.5 italic">
                    {c.preview}
                  </p>
                  <div className="flex items-center gap-2 mt-1.5">
                    <span
                      className={`px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider border ${
                        c.retrieval_source === "graph"
                          ? "text-[var(--color-accent-primary)] border-[var(--color-accent-primary)]"
                          : c.retrieval_source === "both"
                          ? "text-[var(--color-accent-secondary)] border-[var(--color-accent-secondary)]"
                          : "text-[var(--color-text-muted)] border-[var(--color-border-primary)]"
                      }`}
                    >
                      {c.retrieval_source === "graph"
                        ? "connected"
                        : c.retrieval_source === "both"
                        ? "both"
                        : "text"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

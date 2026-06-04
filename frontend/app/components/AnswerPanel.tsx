"use client";

import { useMemo } from "react";
import type { Citation } from "../page";

interface AnswerPanelProps {
  answer: string;
  citations: Citation[];
  highlightedCitation: string | null;
  onCitationClick: (id: string | null) => void;
}

/** Official USCIS titles for APA-style references. */
const FORM_TITLES: Record<string, string> = {
  "I-129": "Instructions for Petition for a Nonimmigrant Worker",
  "I-130": "Instructions for Petition for Alien Relative",
  "I-130A": "Supplemental Information for Spouse Beneficiary",
  "I-131": "Instructions for Application for Travel Document",
  "I-140": "Instructions for Immigrant Petition for Alien Worker",
  "I-485":
    "Instructions for Application to Register Permanent Residence or Adjust Status",
  "I-693":
    "Instructions for Report of Immigration Medical Examination and Vaccination Record",
  "I-765": "Instructions for Application for Employment Authorization",
  "I-864": "Instructions for Affidavit of Support Under Section 213A of the INA",
  "I-907": "Instructions for Request for Premium Processing Service",
};

/** Parse the USCIS form id from a URL/filepath ("i-485instr.pdf" -> "I-485"). */
function formIdFrom(s: string): string | null {
  const base = (s || "").split(/[/\\]/).pop() || "";
  const m = base.toLowerCase().replace("instr", "").match(/([a-z]+)-?(\d+)([a-z]?)/);
  return m ? `${m[1].toUpperCase()}-${m[2]}${m[3].toUpperCase()}` : null;
}

/** APA-style reference text for a citation (URL is rendered separately). */
function apaReference(c: Citation): string {
  const id = formIdFrom(c.source_url || c.filepath);
  const title = (id && FORM_TITLES[id]) || "USCIS form instructions";
  const formNote = id ? ` (Form ${id})` : "";
  return `U.S. Citizenship and Immigration Services. (n.d.). ${title}${formNote}. U.S. Department of Homeland Security.`;
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
  // Strip the model's own trailing "Sources" list (it echoes raw file paths);
  // the panel renders proper APA citations below instead.
  const renderedAnswer = useMemo(() => {
    const cleaned = answer.replace(/\n+#{0,6}\s*sources?\b[\s\S]*$/i, "").trim();
    return renderMarkdown(cleaned);
  }, [answer]);

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
          <h3 className="docline">Sources (APA)</h3>
          <div className="space-y-2 max-h-80 overflow-y-auto">
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
                  <p className="text-[13px] leading-snug text-[var(--color-text-secondary)]">
                    {apaReference(c)}{" "}
                    {c.source_url ? (
                      <a
                        href={c.page ? `${c.source_url}#page=${c.page}` : c.source_url}
                        target="_blank"
                        rel="noopener"
                        onClick={(e) => e.stopPropagation()}
                        className="text-[var(--color-accent-primary)] underline break-all"
                      >
                        {c.source_url}
                        {c.page ? `#page=${c.page}` : ""} ↗
                      </a>
                    ) : null}
                  </p>
                  {c.heading_path ? (
                    <p className="text-[var(--color-text-muted)] mt-1">
                      Section: {c.heading_path}
                    </p>
                  ) : null}
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

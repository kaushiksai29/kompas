"use client";

import { useState, useRef, useEffect } from "react";

interface QueryPanelProps {
  onSubmit: (query: string) => void;
  loading: boolean;
}

export default function QueryPanel({ onSubmit, loading }: QueryPanelProps) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [query]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !loading) {
      onSubmit(query.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="glass-panel glow-border p-2 flex items-stretch gap-2">
        <span
          className="docline self-center pl-2 pr-1 hidden sm:block shrink-0"
          aria-hidden
        >
          Ask
        </span>
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder='Ask in plain words — e.g. "I married a US citizen. Which forms do I file?"'
          className="flex-1 bg-transparent px-3 py-2.5 text-[15px] text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] placeholder:italic focus:outline-none resize-none"
          rows={1}
          disabled={loading}
          id="query-input"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          style={{ borderRadius: 2 }}
          className="px-5 font-mono text-xs font-bold uppercase tracking-wider
                     text-[var(--color-bg-secondary)] bg-[var(--color-accent-primary)]
                     border border-[var(--color-accent-primary)]
                     disabled:opacity-30 disabled:cursor-not-allowed
                     hover:bg-[var(--color-accent-glow)] hover:border-[var(--color-accent-glow)]
                     active:translate-y-px transition-colors flex items-center gap-2"
          id="submit-button"
        >
          {loading ? (
            <>
              <svg
                className="animate-spin w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                />
              </svg>
              Searching…
            </>
          ) : (
            <>
              <svg
                className="w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              Search
            </>
          )}
        </button>
      </div>
    </form>
  );
}

"use client";

import { useEffect, useId, useRef, useState } from "react";
import { ApiError, streamChat, type Source } from "@/lib/api";
import { splitCitations } from "@/lib/citations";

type Turn = {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  cited: number[];
  status: "streaming" | "done" | "error" | "stopped";
  error?: string;
};

type ChatProps = {
  hasDocuments: boolean;
};

export function Chat({ hasDocuments }: ChatProps) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const controllerRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inputId = useId();
  const busy = turns.at(-1)?.status === "streaming";

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  function patch(id: string, change: (turn: Turn) => Partial<Turn>) {
    setTurns((current) => current.map((t) => (t.id === id ? { ...t, ...change(t) } : t)));
  }

  async function ask(text: string) {
    const id = crypto.randomUUID();
    const controller = new AbortController();
    controllerRef.current = controller;
    setTurns((current) => [
      ...current,
      { id, question: text, answer: "", sources: [], cited: [], status: "streaming" },
    ]);
    try {
      for await (const event of streamChat({ question: text }, controller.signal)) {
        if (event.type === "sources") patch(id, () => ({ sources: event.sources }));
        else if (event.type === "token") patch(id, (t) => ({ answer: t.answer + event.text }));
        else if (event.type === "done")
          patch(id, () => ({ answer: event.answer, cited: event.cited, status: "done" }));
        else patch(id, () => ({ status: "error", error: event.detail }));
      }
    } catch (error) {
      if (controller.signal.aborted) patch(id, () => ({ status: "stopped" }));
      else
        patch(id, () => ({
          status: "error",
          error: error instanceof ApiError ? error.message : "Could not reach the API.",
        }));
    }
  }

  function submit() {
    const text = question.trim();
    if (!text || busy) return;
    setQuestion("");
    void ask(text);
  }

  return (
    <section aria-label="Chat" className="flex min-h-[60vh] flex-col">
      <div className="flex-1 space-y-8 pb-6">
        {turns.length === 0 && (
          <div className="flex h-full flex-col justify-center gap-3 py-16">
            <h1 className="text-3xl font-semibold tracking-tight">Chat with your documents</h1>
            <p className="max-w-xl text-zinc-600 dark:text-zinc-400">
              {hasDocuments
                ? "Ask a question. Answers are grounded on your documents and cite the passages they come from."
                : "Upload a PDF, text or Markdown file on the left, then ask a question about it."}
            </p>
          </div>
        )}
        {turns.map((turn) => (
          <TurnView key={turn.id} turn={turn} />
        ))}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="sticky bottom-0 flex items-end gap-2 border-t border-zinc-200 bg-white pt-4 pb-2 dark:border-zinc-800 dark:bg-zinc-950"
      >
        <label htmlFor={inputId} className="sr-only">
          Question
        </label>
        <textarea
          id={inputId}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          rows={2}
          maxLength={2000}
          placeholder="Ask about your documents… (Enter to send, Shift+Enter for a new line)"
          className="min-h-12 flex-1 resize-y rounded-xl border border-zinc-300 bg-transparent px-4 py-3 text-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/30 dark:border-zinc-700"
        />
        {busy ? (
          <button
            type="button"
            onClick={() => controllerRef.current?.abort()}
            className="h-12 rounded-xl border border-zinc-300 px-4 text-sm font-medium dark:border-zinc-700"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!question.trim()}
            className="h-12 rounded-xl bg-blue-600 px-5 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-40"
          >
            Ask
          </button>
        )}
      </form>
    </section>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  const [openSource, setOpenSource] = useState<number | null>(null);
  const sourceIds = useId();
  const parts = splitCitations(turn.answer);
  const byNumber = new Map(turn.sources.map((s) => [s.number, s]));

  return (
    <article className="space-y-3">
      <p className="ml-auto w-fit max-w-[85%] rounded-2xl bg-zinc-100 px-4 py-2 text-sm whitespace-pre-wrap dark:bg-zinc-800">
        {turn.question}
      </p>

      <div
        className="text-[15px] leading-relaxed"
        aria-live="polite"
        aria-busy={turn.status === "streaming"}
      >
        {parts.map((part, index) =>
          part.type === "text" ? (
            <span key={index} className="whitespace-pre-wrap">
              {part.text}
            </span>
          ) : byNumber.has(part.number) ? (
            <button
              key={index}
              type="button"
              onClick={() => setOpenSource(openSource === part.number ? null : part.number)}
              aria-expanded={openSource === part.number}
              aria-controls={`${sourceIds}-${part.number}`}
              aria-label={`Show source ${part.number}`}
              className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-blue-100 px-1 align-text-top text-[11px] font-semibold text-blue-700 hover:bg-blue-200 dark:bg-blue-950 dark:text-blue-300"
            >
              {part.number}
            </button>
          ) : (
            <span key={index}>[{part.number}]</span>
          ),
        )}
        {turn.status === "streaming" && (
          <span
            aria-hidden="true"
            className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-zinc-400 align-middle"
          />
        )}
        {turn.status === "stopped" && <span className="ml-1 text-xs text-zinc-500">(stopped)</span>}
      </div>

      {turn.status === "error" && (
        <p
          role="alert"
          className="rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/40 dark:text-red-300"
        >
          {turn.error}
        </p>
      )}

      {turn.sources.length > 0 && (
        <ol className="space-y-2" aria-label="Sources">
          {turn.sources.map((source) => {
            const open = openSource === source.number;
            const cited = turn.cited.includes(source.number);
            return (
              <li
                key={source.number}
                id={`${sourceIds}-${source.number}`}
                className={`rounded-lg border text-sm ${
                  open
                    ? "border-blue-400 dark:border-blue-700"
                    : "border-zinc-200 dark:border-zinc-800"
                }`}
              >
                <button
                  type="button"
                  onClick={() => setOpenSource(open ? null : source.number)}
                  aria-expanded={open}
                  className="flex w-full items-center gap-2 px-3 py-2 text-left"
                >
                  <span className="font-mono text-xs text-blue-700 dark:text-blue-300">
                    [{source.number}]
                  </span>
                  <span className="truncate font-medium">{source.filename}</span>
                  <span className="text-xs text-zinc-500">p. {source.page}</span>
                  {cited && (
                    <span className="ml-auto rounded-full bg-emerald-100 px-2 text-[11px] text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                      cited
                    </span>
                  )}
                </button>
                {open && (
                  <p className="border-t border-zinc-200 px-3 py-2 whitespace-pre-wrap text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
                    {source.text}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </article>
  );
}

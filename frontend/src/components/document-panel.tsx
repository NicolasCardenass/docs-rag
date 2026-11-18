"use client";

import { useCallback, useEffect, useId, useState } from "react";
import {
  ApiError,
  deleteDocument,
  listDocuments,
  uploadDocument,
  type DocumentRecord,
} from "@/lib/api";
import { formatBytes, plural } from "@/lib/format";

const ACCEPTED = ".pdf,.txt,.md,.markdown";

function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "Could not reach the API. Is the backend running?";
}

type DocumentPanelProps = {
  /** Called whenever the list changes, so other components can react. */
  onDocumentsChange?: (documents: DocumentRecord[]) => void;
};

export function DocumentPanel({ onDocumentsChange }: DocumentPanelProps) {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputId = useId();

  const update = useCallback(
    (next: DocumentRecord[]) => {
      setDocuments(next);
      onDocumentsChange?.(next);
    },
    [onDocumentsChange],
  );

  useEffect(() => {
    const controller = new AbortController();
    listDocuments(controller.signal)
      .then(update)
      .catch((err: unknown) => {
        if (!controller.signal.aborted) setError(errorText(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [update]);

  async function upload(files: FileList | File[]) {
    setError(null);
    const list = Array.from(files);
    setUploading((current) => [...current, ...list.map((f) => f.name)]);
    const added: DocumentRecord[] = [];
    const failures: string[] = [];
    // Upload one at a time: indexing is CPU/API bound on the server anyway.
    for (const file of list) {
      try {
        added.push(await uploadDocument(file));
      } catch (err) {
        failures.push(`${file.name}: ${errorText(err)}`);
      } finally {
        setUploading((current) => current.filter((name) => name !== file.name));
      }
    }
    if (added.length) update([...added.reverse(), ...documents]);
    if (failures.length) setError(failures.join("\n"));
  }

  async function remove(document: DocumentRecord) {
    setError(null);
    try {
      await deleteDocument(document.id);
      update(documents.filter((d) => d.id !== document.id));
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <section aria-labelledby="documents-heading" className="flex flex-col gap-4">
      <h2 id="documents-heading" className="text-sm font-semibold tracking-wide uppercase">
        Documents
      </h2>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (event.dataTransfer.files.length) void upload(event.dataTransfer.files);
        }}
        className={`rounded-xl border-2 border-dashed p-6 text-center text-sm transition-colors ${
          dragging
            ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
            : "border-zinc-300 dark:border-zinc-700"
        }`}
      >
        <p className="text-zinc-600 dark:text-zinc-400">Drop files here or</p>
        <label
          htmlFor={inputId}
          className="mt-2 inline-flex cursor-pointer rounded-full bg-zinc-900 px-4 py-2 font-medium text-white focus-within:ring-2 focus-within:ring-blue-500 hover:bg-zinc-700 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
        >
          Choose files
        </label>
        <input
          id={inputId}
          type="file"
          accept={ACCEPTED}
          multiple
          className="sr-only"
          onChange={(event) => {
            if (event.target.files?.length) void upload(event.target.files);
            event.target.value = "";
          }}
        />
        <p className="mt-2 text-xs text-zinc-500">PDF, TXT or Markdown</p>
      </div>

      {error && (
        <p
          role="alert"
          className="rounded-lg bg-red-50 p-3 text-sm whitespace-pre-line text-red-700 dark:bg-red-950/40 dark:text-red-300"
        >
          {error}
        </p>
      )}

      <ul className="flex flex-col gap-2" aria-live="polite" aria-busy={loading || undefined}>
        {uploading.map((name) => (
          <li
            key={`uploading-${name}`}
            className="rounded-lg border border-zinc-200 p-3 text-sm text-zinc-500 dark:border-zinc-800"
          >
            Indexing {name}…
          </li>
        ))}
        {documents.map((document) => (
          <li
            key={document.id}
            className="flex items-start justify-between gap-3 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium" title={document.filename}>
                {document.filename}
              </p>
              <p className="mt-0.5 text-xs text-zinc-500">
                {plural(document.pages, "page")} · {plural(document.chunks, "chunk")} ·{" "}
                {formatBytes(document.size_bytes)}
              </p>
            </div>
            <button
              type="button"
              onClick={() => void remove(document)}
              aria-label={`Delete ${document.filename}`}
              className="shrink-0 rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-red-50 hover:text-red-700 dark:hover:bg-red-950/40 dark:hover:text-red-300"
            >
              Delete
            </button>
          </li>
        ))}
        {!loading && documents.length === 0 && uploading.length === 0 && (
          <li className="text-sm text-zinc-500">No documents yet.</li>
        )}
      </ul>
    </section>
  );
}

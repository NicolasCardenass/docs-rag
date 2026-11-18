import { parseSSE } from "./sse";

/**
 * Typed client for the Docs RAG backend API.
 * Types mirror the Pydantic models exposed in the API's OpenAPI schema.
 */

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

export type Health = {
  status: "ok";
  version: string;
};

export type DocumentRecord = {
  id: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  pages: number;
  chunks: number;
  created_at: string;
};

export type Source = {
  number: number;
  document_id: string;
  filename: string;
  page: number;
  text: string;
  score: number;
  cited?: boolean;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  model: string;
};

/** Error returned by the API, carrying the HTTP status and FastAPI's `detail`. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const { detail } = body as { detail: unknown };
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) return "Invalid request";
    }
  } catch {
    // Not JSON: fall through to the status text.
  }
  return response.statusText || `Request failed with status ${response.status}`;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response));
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>("/health", { signal });
}

export function listDocuments(signal?: AbortSignal): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/documents", { signal });
}

export function uploadDocument(file: File): Promise<DocumentRecord> {
  const body = new FormData();
  body.append("file", file);
  return request<DocumentRecord>("/documents", { method: "POST", body });
}

export function deleteDocument(id: string): Promise<void> {
  return request<void>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export type ChatStreamEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; answer: string; cited: number[]; model: string }
  | { type: "error"; detail: string };

export type ChatRequest = {
  question: string;
  document_ids?: string[];
  top_k?: number;
};

/** Ask a question and receive the answer as it is generated. */
export async function* streamChat(
  body: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new ApiError(response.status, await errorMessage(response));
  }
  for await (const { event, data } of parseSSE(response.body)) {
    const payload: unknown = JSON.parse(data);
    switch (event) {
      case "sources":
        yield { type: "sources", sources: payload as Source[] };
        break;
      case "token":
        yield { type: "token", text: (payload as { text: string }).text };
        break;
      case "done":
        yield { type: "done", ...(payload as { answer: string; cited: number[]; model: string }) };
        break;
      case "error":
        yield { type: "error", detail: (payload as { detail: string }).detail };
        break;
    }
  }
}

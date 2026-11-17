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

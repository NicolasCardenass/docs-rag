"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

type Status = { state: "checking" } | { state: "online"; version: string } | { state: "offline" };

const POLL_INTERVAL_MS = 15_000;

/** Small badge showing whether the backend API is reachable. */
export function ApiStatus() {
  const [status, setStatus] = useState<Status>({ state: "checking" });

  useEffect(() => {
    let controller = new AbortController();

    async function check() {
      controller.abort();
      controller = new AbortController();
      try {
        const health = await getHealth(controller.signal);
        setStatus({ state: "online", version: health.version });
      } catch (error) {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setStatus({ state: "offline" });
        }
      }
    }

    void check();
    const timer = setInterval(check, POLL_INTERVAL_MS);
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, []);

  const label =
    status.state === "online"
      ? `API online · v${status.version}`
      : status.state === "offline"
        ? "API offline"
        : "Checking API…";
  const dot =
    status.state === "online"
      ? "bg-emerald-500"
      : status.state === "offline"
        ? "bg-red-500"
        : "bg-zinc-400";

  return (
    <span
      role="status"
      className="inline-flex items-center gap-2 rounded-full border border-zinc-200 px-3 py-1 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-400"
    >
      <span aria-hidden="true" className={`size-2 rounded-full ${dot}`} />
      {label}
    </span>
  );
}

import type { Metadata } from "next";
import "@fontsource-variable/inter";
import "./globals.css";
import { ApiStatus } from "@/components/api-status";

export const metadata: Metadata = {
  title: "Docs RAG",
  description: "Chat with your documents using retrieval-augmented generation.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-white font-sans text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
        <header className="border-b border-zinc-200 dark:border-zinc-800">
          <div className="mx-auto flex h-14 w-full max-w-5xl items-center justify-between px-4">
            <span className="font-semibold tracking-tight">Docs RAG</span>
            <ApiStatus />
          </div>
        </header>
        <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 py-8">{children}</main>
      </body>
    </html>
  );
}

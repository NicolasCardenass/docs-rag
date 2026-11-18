import { DocumentPanel } from "@/components/document-panel";

export default function Home() {
  return (
    <div className="grid flex-1 gap-8 md:grid-cols-[18rem_1fr]">
      <aside>
        <DocumentPanel />
      </aside>
      <section className="flex flex-col justify-center gap-4">
        <h1 className="text-3xl font-semibold tracking-tight">Chat with your documents</h1>
        <p className="max-w-xl text-zinc-600 dark:text-zinc-400">
          Upload PDFs, text or Markdown files and ask questions. Answers are grounded on your
          documents and cite the passages they come from.
        </p>
      </section>
    </div>
  );
}

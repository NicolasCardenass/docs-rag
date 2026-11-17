export default function Home() {
  return (
    <section className="flex flex-1 flex-col justify-center gap-4">
      <h1 className="text-3xl font-semibold tracking-tight">Chat with your documents</h1>
      <p className="max-w-xl text-zinc-600 dark:text-zinc-400">
        Upload PDFs, text or Markdown files and ask questions. Answers are grounded on your
        documents and cite the passages they come from.
      </p>
    </section>
  );
}

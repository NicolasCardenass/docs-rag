"use client";

import { useCallback, useState } from "react";
import type { DocumentRecord } from "@/lib/api";
import { Chat } from "./chat";
import { DocumentPanel } from "./document-panel";

/** Two-column layout: documents on the left, chat on the right. */
export function Workspace() {
  const [documentCount, setDocumentCount] = useState(0);
  const onDocumentsChange = useCallback(
    (documents: DocumentRecord[]) => setDocumentCount(documents.length),
    [],
  );

  return (
    <div className="grid flex-1 gap-8 md:grid-cols-[18rem_1fr]">
      <aside>
        <DocumentPanel onDocumentsChange={onDocumentsChange} />
      </aside>
      <Chat hasDocuments={documentCount > 0} />
    </div>
  );
}

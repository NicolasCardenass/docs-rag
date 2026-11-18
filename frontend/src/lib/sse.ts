export type ServerSentEvent = {
  event: string;
  data: string;
};

/**
 * Parse a Server-Sent Events stream from a fetch response body.
 *
 * `EventSource` only supports GET requests, so for POST endpoints the body is
 * read manually: events are separated by a blank line and each has
 * `event:` and `data:` fields.
 */
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<ServerSentEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      // `stream: true` keeps multi-byte characters split across chunks intact.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseBlock(block);
        if (event) yield event;
      }
    }
    const last = parseBlock(buffer + decoder.decode());
    if (last) yield last;
  } finally {
    reader.releaseLock();
  }
}

function parseBlock(block: string): ServerSentEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

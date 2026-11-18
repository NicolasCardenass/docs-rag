export type AnswerPart = { type: "text"; text: string } | { type: "citation"; number: number };

/** Split an answer into plain text and `[n]` citation markers. */
export function splitCitations(answer: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  const pattern = /\[(\d+)\]/g;
  let last = 0;
  for (const match of answer.matchAll(pattern)) {
    if (match.index > last) parts.push({ type: "text", text: answer.slice(last, match.index) });
    parts.push({ type: "citation", number: Number(match[1]) });
    last = match.index + match[0].length;
  }
  if (last < answer.length) parts.push({ type: "text", text: answer.slice(last) });
  return parts;
}

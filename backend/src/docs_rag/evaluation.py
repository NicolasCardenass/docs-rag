"""Evaluate retrieval and answer quality on a small labelled question set.

Usage (from ``backend/``)::

    uv run python -m docs_rag.evaluation                    # default corpus and questions
    uv run python -m docs_rag.evaluation --top-k 3 --json report.json --min-hit-rate 0.8

The corpus is indexed into a temporary directory with the providers
configured in the environment (mock by default), so results are
reproducible and the real data directory is never touched.
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from docs_rag.config import Settings
from docs_rag.main import build_services

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "eval"


class EvalCase(BaseModel):
    question: str
    expected_source: str
    """File name of the document that contains the answer."""
    expected_keywords: list[str] = []


@dataclass(frozen=True, slots=True)
class CaseResult:
    question: str
    expected_source: str
    retrieved_sources: list[str]
    rank: int | None
    """1-based position of the first chunk from the expected source, if retrieved."""
    cited_expected: bool
    keyword_coverage: float
    answer: str


@dataclass(frozen=True, slots=True)
class Summary:
    cases: int
    top_k: int
    hit_rate: float
    mrr: float
    citation_accuracy: float
    keyword_coverage: float
    embedder: str
    model: str


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float:
    """Fraction of expected keywords that appear in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(keyword.lower() in text for keyword in keywords) / len(keywords)


def summarize(results: Sequence[CaseResult], top_k: int, embedder: str, model: str) -> Summary:
    count = len(results) or 1
    return Summary(
        cases=len(results),
        top_k=top_k,
        hit_rate=sum(r.rank is not None for r in results) / count,
        mrr=sum(1 / r.rank for r in results if r.rank) / count,
        citation_accuracy=sum(r.cited_expected for r in results) / count,
        keyword_coverage=sum(r.keyword_coverage for r in results) / count,
        embedder=embedder,
        model=model,
    )


def evaluate(
    corpus_dir: Path, cases: Sequence[EvalCase], settings: Settings, top_k: int
) -> tuple[list[CaseResult], Summary]:
    services = build_services(settings)
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file():
            services.documents.ingest(path.name, path.read_bytes())

    results: list[CaseResult] = []
    for case in cases:
        answer = services.chat.answer(case.question, top_k=top_k)
        filenames = [source.result.filename for source in answer.sources]
        rank = next(
            (i for i, name in enumerate(filenames, start=1) if name == case.expected_source), None
        )
        results.append(
            CaseResult(
                question=case.question,
                expected_source=case.expected_source,
                retrieved_sources=filenames,
                rank=rank,
                cited_expected=any(
                    s.cited and s.result.filename == case.expected_source for s in answer.sources
                ),
                keyword_coverage=keyword_coverage(answer.answer, case.expected_keywords),
                answer=answer.answer,
            )
        )

    return results, summarize(
        results, top_k, services.retriever.embedder_name, services.chat.model_name
    )


def print_report(results: Sequence[CaseResult], summary: Summary) -> None:
    print(f"Embedder: {summary.embedder} · Model: {summary.model} · top_k={summary.top_k}\n")
    print(f"{'#':>2}  {'rank':>4}  {'cited':>5}  {'kw':>4}  question")
    for number, r in enumerate(results, start=1):
        rank = str(r.rank) if r.rank else "-"
        cited = "yes" if r.cited_expected else "no"
        print(f"{number:>2}  {rank:>4}  {cited:>5}  {r.keyword_coverage:>4.0%}  {r.question}")
    print(
        f"\nHit rate@{summary.top_k}: {summary.hit_rate:.0%} · MRR: {summary.mrr:.2f} · "
        f"Citation accuracy: {summary.citation_accuracy:.0%} · "
        f"Keyword coverage: {summary.keyword_coverage:.0%}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_DIR / "corpus")
    parser.add_argument("--questions", type=Path, default=DEFAULT_DIR / "questions.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--json", type=Path, help="Write the full report to this file")
    parser.add_argument(
        "--min-hit-rate", type=float, default=0.0, help="Exit with an error below this hit rate"
    )
    args = parser.parse_args(argv)

    cases = TypeAdapter(list[EvalCase]).validate_json(args.questions.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(data_dir=Path(tmp))
        results, summary = evaluate(args.corpus, cases, settings, args.top_k)

    print_report(results, summary)
    if args.json:
        report = {"summary": asdict(summary), "cases": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary.hit_rate < args.min_hit_rate:
        print(f"\nHit rate below the minimum of {args.min_hit_rate:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

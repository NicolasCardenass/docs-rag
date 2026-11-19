import json
from pathlib import Path

import pytest

from docs_rag.config import Settings
from docs_rag.evaluation import (
    DEFAULT_DIR,
    CaseResult,
    EvalCase,
    evaluate,
    keyword_coverage,
    main,
    summarize,
)


def case_result(rank: int | None, cited: bool, coverage: float) -> CaseResult:
    return CaseResult("q", "a.md", ["a.md"], rank, cited, coverage, "answer")


def test_keyword_coverage_is_case_insensitive() -> None:
    assert keyword_coverage("Direct Current flows", ["direct current", "AC"]) == 0.5
    assert keyword_coverage("anything", []) == 1.0


def test_summary_metrics() -> None:
    results = [case_result(1, True, 1.0), case_result(2, False, 0.5), case_result(None, False, 0)]

    summary = summarize(results, top_k=3, embedder="e", model="m")

    assert summary.cases == 3
    assert summary.hit_rate == pytest.approx(2 / 3)
    assert summary.mrr == pytest.approx((1 + 0.5) / 3)
    assert summary.citation_accuracy == pytest.approx(1 / 3)
    assert summary.keyword_coverage == pytest.approx(0.5)


def test_bundled_question_set_retrieves_expected_sources(tmp_path: Path) -> None:
    cases = [
        EvalCase.model_validate(c) for c in json.loads((DEFAULT_DIR / "questions.json").read_text())
    ]
    settings = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]

    results, summary = evaluate(DEFAULT_DIR / "corpus", cases, settings, top_k=3)

    assert len(results) == len(cases)
    assert summary.hit_rate == 1.0
    assert summary.embedder.startswith("mock:")


def test_cli_writes_report_and_enforces_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # ignore any developer .env file
    report = tmp_path / "report.json"

    assert main(["--json", str(report), "--min-hit-rate", "0.5"]) == 0
    assert "Hit rate@4" in capsys.readouterr().out
    assert json.loads(report.read_text())["summary"]["cases"] == 6

    assert main(["--top-k", "1", "--min-hit-rate", "1.01"]) == 1

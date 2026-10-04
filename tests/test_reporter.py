from pathlib import Path

import pytest

from verifact.agents.reporter import build_reporter_prompt, run_reporter
from verifact.models import (
    ColumnProfile,
    DataProfile,
    PipelineContext,
    RepairReport,
    ValidationReport,
)


def _context() -> PipelineContext:
    profile = DataProfile(
        dataset_name="people",
        row_count=3,
        column_count=1,
        duplicate_row_count=0,
        total_null_count=0,
        columns=[
            ColumnProfile(
                name="age",
                dtype="object",
                null_count=0,
                null_pct=0.0,
                unique_count=3,
                unique_pct=100.0,
                sample_values=["25", "30", "35"],
                inferred_type="age",
            )
        ],
        summary="test profile",
    )
    validation = ValidationReport(
        passed=True,
        rules_applied=[],
        failure_count=0,
        failures=[],
        summary="test validation",
    )
    repair = RepairReport(
        total_repairs=0,
        rows_dropped=0,
        actions=[],
        output_path="data/processed/people_clean.csv",
        summary="test repair",
    )
    return PipelineContext(
        input_path="data/raw/people.csv",
        output_path="data/processed/people_clean.csv",
        profile=profile,
        validation=validation,
        repair=repair,
    )


def test_build_reporter_prompt_includes_dataset_name() -> None:
    """The prompt embeds the full context's JSON, including the dataset name."""
    prompt = build_reporter_prompt(_context())
    assert '"dataset_name": "people"' in prompt


def test_build_reporter_prompt_includes_instructions() -> None:
    """The prompt instructs the agent to generate a complete report."""
    prompt = build_reporter_prompt(_context())
    assert "Generate a complete data quality report" in prompt


@pytest.mark.llm
def test_run_reporter_writes_markdown_report_to_disk(tmp_path: Path) -> None:
    """The agent's markdown output is both returned and written to disk."""
    output_path = str(tmp_path / "reports" / "report.md")
    report_md = run_reporter(_context(), output_path)
    assert Path(output_path).read_text() == report_md
    assert "# Data Quality Report" in report_md

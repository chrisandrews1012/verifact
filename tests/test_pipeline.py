from pathlib import Path

import pandas as pd
import pytest

from verifact.pipeline import (
    DEFAULT_INPUT_PATH,
    resolve_input_path,
    run_pipeline,
)


def test_resolve_input_path_uses_provided_argument() -> None:
    """An explicit CLI argument is used as the input path."""
    assert resolve_input_path(["custom.csv"]) == "custom.csv"


def test_resolve_input_path_falls_back_to_default() -> None:
    """With no CLI argument, the default input path is used."""
    assert resolve_input_path([]) == DEFAULT_INPUT_PATH


@pytest.mark.llm
def test_run_pipeline_produces_a_full_context(tmp_path: Path) -> None:
    """Against a small synthetic dataset, the pipeline runs all four
    stages and writes both the repaired CSV and the markdown report."""
    input_path = str(tmp_path / "hr_messy.csv")
    output_path = str(tmp_path / "hr_messy_clean.csv")
    report_path = str(tmp_path / "report.md")
    df = pd.DataFrame(
        {
            "employee_id": ["e1", "e2", "e2", "e3"],
            "age": [25, 30, 30, None],
            "email": ["a@x.com", "b@x.com", "b@x.com", "not-an-email"],
        }
    )
    df.to_csv(input_path, index=False)

    context = run_pipeline(
        input_path=input_path, output_path=output_path, report_path=report_path
    )

    assert context.input_path == input_path
    assert context.profile.row_count == 4
    assert context.validation.failure_count == len(context.validation.failures)
    assert context.repair.total_repairs == len(context.repair.actions)
    assert Path(output_path).exists()
    assert Path(report_path).exists()

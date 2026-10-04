from pathlib import Path

import pandas as pd
import pytest

from verifact.agents.profiler import (
    build_missingness_report,
    build_profiler_prompt,
    extract_dataset_name,
    run_profiler,
)


def test_extract_dataset_name_strips_directory_and_extension() -> None:
    """The dataset name is the CSV filename without its path or extension."""
    assert extract_dataset_name("data/raw/hr_messy.csv") == "hr_messy"


def test_build_profiler_prompt_includes_dataset_name() -> None:
    """The prompt states the dataset name extracted from the csv path."""
    df = pd.DataFrame({"age": [25, 30, None], "name": ["Al", "Bo", "Cy"]})
    prompt = build_profiler_prompt("data/raw/people.csv", df)
    assert "people" in prompt


def test_build_profiler_prompt_includes_ground_truth_dataset_stats() -> None:
    """The prompt states the deterministic row/column counts explicitly."""
    df = pd.DataFrame({"age": [25, 30, None], "name": ["Al", "Bo", "Cy"]})
    prompt = build_profiler_prompt("data/raw/people.csv", df)
    assert "Row count: 3" in prompt
    assert "Column count: 2" in prompt


def test_build_profiler_prompt_includes_per_column_stats() -> None:
    """Per-column statistics for every column are included in the prompt."""
    df = pd.DataFrame({"age": [25, 30, None], "name": ["Al", "Bo", "Cy"]})
    prompt = build_profiler_prompt("data/raw/people.csv", df)
    assert "'column': 'age'" in prompt
    assert "'column': 'name'" in prompt


def test_build_missingness_report_classifies_mar_column() -> None:
    """A column whose missingness correlates with a numeric column is
    attached to the report as MAR with safe_to_impute True."""
    n = 60
    salary = list(range(n))
    bonus: list[float | None] = [None if i < n // 2 else float(i) for i in range(n)]
    df = pd.DataFrame({"salary": salary, "bonus": bonus})
    report = build_missingness_report(df)
    bonus_analysis = next(c for c in report.columns_analyzed if c.column == "bonus")
    assert bonus_analysis.mechanism == "MAR"
    assert bonus_analysis.safe_to_impute is True


def test_build_missingness_report_summary_mentions_dataset_level_test() -> None:
    """The report's summary includes the dataset-level MCAR test conclusion."""
    df = pd.DataFrame({"age": [25, 30, None], "score": [1, 2, 3]})
    report = build_missingness_report(df)
    assert "Dataset-level test" in report.summary


@pytest.mark.llm
def test_run_profiler_produces_a_valid_data_profile(tmp_path: Path) -> None:
    """Against a real small dataset, the agent's output matches the schema
    and its restated numbers agree with ground truth, and missingness is
    attached deterministically after the LLM call."""
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 45],
            "email": ["a@x.com", "b@x.com", "c@x.com", None],
        }
    )
    csv_path = str(tmp_path / "people.csv")
    df.to_csv(csv_path, index=False)

    profile = run_profiler(csv_path)
    assert profile.row_count == 4
    assert profile.column_count == 2
    assert {c.name for c in profile.columns} == {"age", "email"}
    assert profile.missingness is not None

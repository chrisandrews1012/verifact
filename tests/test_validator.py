import pytest

from verifact.agents.validator import build_validator_prompt, run_validator
from verifact.models import ColumnProfile, DataProfile


def _column_profile(name: str, inferred_type: str) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        null_count=0,
        null_pct=0.0,
        unique_count=1,
        unique_pct=100.0,
        sample_values=[],
        inferred_type=inferred_type,
    )


def _profile(
    columns: list[ColumnProfile], summary: str = "test profile"
) -> DataProfile:
    return DataProfile(
        dataset_name="test",
        row_count=4,
        column_count=len(columns),
        duplicate_row_count=0,
        total_null_count=0,
        columns=columns,
        summary=summary,
    )


def test_build_validator_prompt_includes_dataset_name() -> None:
    """The prompt embeds the profile's serialized JSON, including the dataset name."""
    profile = _profile([_column_profile("patient_id", "id")])
    prompt = build_validator_prompt(profile)
    assert '"dataset_name": "test"' in prompt


def test_build_validator_prompt_includes_instructions_to_infer_rules() -> None:
    """The prompt instructs the agent to infer rules from semantic types."""
    profile = _profile([_column_profile("patient_id", "id")])
    prompt = build_validator_prompt(profile)
    assert "Infer appropriate rules from the column semantic" in prompt


def test_build_validator_prompt_includes_column_inferred_types() -> None:
    """The serialized profile includes each column's inferred semantic type."""
    profile = _profile([_column_profile("email", "email")])
    prompt = build_validator_prompt(profile)
    assert '"inferred_type": "email"' in prompt


@pytest.mark.llm
def test_run_validator_produces_a_valid_validation_report() -> None:
    """Against a profile with a duplicate-prone id column, the agent's
    output matches the schema and flags at least one failure."""
    profile = _profile(
        [_column_profile("patient_id", "id"), _column_profile("age", "age")]
    )
    report = run_validator(profile)
    assert report.failure_count == len(report.failures)

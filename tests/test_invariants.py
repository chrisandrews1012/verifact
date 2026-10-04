import pandas as pd
import pytest

from verifact.invariants import (
    InvariantViolation,
    assert_invariants,
    check_profile_invariants,
    check_repair_invariants,
    check_validation_invariants,
)
from verifact.models import (
    ColumnProfile,
    DataProfile,
    RepairAction,
    RepairReport,
    ValidationFailure,
    ValidationReport,
    ValidationRule,
)


def _column_profile(
    name: str,
    null_count: int = 0,
    null_pct: float = 0.0,
    unique_count: int = 1,
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        null_count=null_count,
        null_pct=null_pct,
        unique_count=unique_count,
        unique_pct=100.0,
        sample_values=[],
        inferred_type="unknown",
    )


def test_check_profile_invariants_passes_for_accurate_profile() -> None:
    """A profile whose numbers match the dataframe produces no violations."""
    df = pd.DataFrame({"age": [25, None, 30]})
    profile = DataProfile(
        dataset_name="test",
        row_count=3,
        column_count=1,
        duplicate_row_count=0,
        total_null_count=1,
        columns=[_column_profile("age", null_count=1, null_pct=33.33, unique_count=2)],
        summary="test",
    )
    assert check_profile_invariants(profile, df) == []


def test_check_profile_invariants_flags_wrong_row_count() -> None:
    """A claimed row_count that doesn't match the dataframe is flagged."""
    df = pd.DataFrame({"age": [25, 30]})
    profile = DataProfile(
        dataset_name="test",
        row_count=99,
        column_count=1,
        duplicate_row_count=0,
        total_null_count=0,
        columns=[_column_profile("age", unique_count=2)],
        summary="test",
    )
    violations = check_profile_invariants(profile, df)
    assert any("row_count" in v for v in violations)


def test_check_profile_invariants_flags_unprofiled_column() -> None:
    """A column present in the dataframe but missing from the profile is flagged."""
    df = pd.DataFrame({"age": [25, 30], "email": ["a@x.com", "b@x.com"]})
    profile = DataProfile(
        dataset_name="test",
        row_count=2,
        column_count=2,
        duplicate_row_count=0,
        total_null_count=0,
        columns=[_column_profile("age", unique_count=2)],
        summary="test",
    )
    violations = check_profile_invariants(profile, df)
    assert any("email" in v for v in violations)


def test_check_validation_invariants_passes_for_consistent_report() -> None:
    """A report whose passed/failure_count are internally consistent passes."""
    profile = DataProfile(
        dataset_name="test",
        row_count=1,
        column_count=1,
        duplicate_row_count=0,
        total_null_count=0,
        columns=[_column_profile("age")],
        summary="test",
    )
    validation = ValidationReport(
        passed=True,
        rules_applied=[
            ValidationRule(column="age", rule_description="test", severity="info")
        ],
        failure_count=0,
        failures=[],
        summary="test",
    )
    assert check_validation_invariants(validation, profile) == []


def test_check_validation_invariants_flags_passed_true_with_critical_failure() -> None:
    """passed=True alongside a critical failure is a contradiction."""
    profile = DataProfile(
        dataset_name="test",
        row_count=1,
        column_count=1,
        duplicate_row_count=0,
        total_null_count=0,
        columns=[_column_profile("age")],
        summary="test",
    )
    validation = ValidationReport(
        passed=True,
        rules_applied=[],
        failure_count=1,
        failures=[
            ValidationFailure(
                column="age",
                rule="test rule",
                severity="critical",
                affected_rows=1,
                description="bad",
                suggested_fix="fix it",
            )
        ],
        summary="test",
    )
    violations = check_validation_invariants(validation, profile)
    assert any("passed=True" in v for v in violations)


def test_check_validation_invariants_flags_unknown_column_reference() -> None:
    """A failure referencing a column not in the profile is flagged."""
    profile = DataProfile(
        dataset_name="test",
        row_count=1,
        column_count=1,
        duplicate_row_count=0,
        total_null_count=0,
        columns=[_column_profile("age")],
        summary="test",
    )
    validation = ValidationReport(
        passed=False,
        rules_applied=[],
        failure_count=1,
        failures=[
            ValidationFailure(
                column="nonexistent",
                rule="test rule",
                severity="critical",
                affected_rows=1,
                description="bad",
                suggested_fix="fix it",
            )
        ],
        summary="test",
    )
    violations = check_validation_invariants(validation, profile)
    assert any("nonexistent" in v for v in violations)


def test_check_repair_invariants_passes_for_consistent_report() -> None:
    """A report whose rows_dropped matches the actual row delta passes."""
    input_df = pd.DataFrame({"age": [25, 25, 30]})
    output_df = pd.DataFrame({"age": [25, 30]})
    repair = RepairReport(
        total_repairs=1,
        rows_dropped=1,
        actions=[
            RepairAction(
                column="(all columns)",
                issue="duplicate rows",
                action_taken="dropped_rows",
                rows_affected=1,
                before_example="dup",
                after_example="removed",
                reason="test",
            )
        ],
        output_path="out.csv",
        summary="test",
    )
    assert check_repair_invariants(repair, input_df, output_df) == []


def test_check_repair_invariants_flags_wrong_rows_dropped() -> None:
    """A claimed rows_dropped that doesn't match the actual delta is flagged."""
    input_df = pd.DataFrame({"age": [25, 25, 30]})
    output_df = pd.DataFrame({"age": [25, 30]})
    repair = RepairReport(
        total_repairs=0,
        rows_dropped=99,
        actions=[],
        output_path="out.csv",
        summary="test",
    )
    violations = check_repair_invariants(repair, input_df, output_df)
    assert any("rows_dropped" in v for v in violations)


def test_assert_invariants_raises_with_violations() -> None:
    """A non-empty violations list raises InvariantViolation naming the agent."""
    with pytest.raises(InvariantViolation, match="Profiler"):
        assert_invariants(["something is wrong"], "Profiler")


def test_assert_invariants_is_a_noop_with_no_violations() -> None:
    """An empty violations list raises nothing."""
    assert_invariants([], "Profiler")

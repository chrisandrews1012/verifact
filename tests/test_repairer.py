from pathlib import Path

import pandas as pd
import pytest

from verifact.agents.repairer import apply_repairs, run_repairer
from verifact.models import (
    ColumnMissingness,
    ColumnProfile,
    DataProfile,
    MissingnessReport,
)


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
    columns: list[ColumnProfile], missingness: MissingnessReport | None = None
) -> DataProfile:
    return DataProfile(
        dataset_name="test",
        row_count=4,
        column_count=len(columns),
        duplicate_row_count=0,
        total_null_count=0,
        columns=columns,
        summary="test profile",
        missingness=missingness,
    )


def _missingness(column: str, safe_to_impute: bool) -> MissingnessReport:
    return MissingnessReport(
        dataset_mcar_conclusion="test",
        columns_analyzed=[
            ColumnMissingness(
                column=column,
                null_count=1,
                null_pct=25.0,
                mechanism="MNAR" if not safe_to_impute else "MCAR",
                confidence="high",
                evidence="test",
                safe_to_impute=safe_to_impute,
            )
        ],
        summary="test",
    )


def test_apply_repairs_drops_exact_duplicate_rows() -> None:
    """Exact duplicate rows are dropped before any per-column repairs."""
    df = pd.DataFrame({"name": ["Al", "Al", "Bo"]})
    profile = _profile([_column_profile("name", "text")])
    result_df, actions, rows_dropped, _ = apply_repairs(df, profile)
    assert len(result_df) == 2
    assert rows_dropped == 1
    assert any(a.action_taken == "dropped_rows" for a in actions)


def test_apply_repairs_cleans_currency_and_imputes_median() -> None:
    """Currency symbols are stripped and nulls are median-imputed."""
    df = pd.DataFrame({"salary": ["$1,000", "$2,000", None]})
    profile = _profile([_column_profile("salary", "currency")])
    result_df, actions, _, _ = apply_repairs(df, profile)
    assert result_df["salary"].isnull().sum() == 0
    action_types = {a.action_taken for a in actions}
    assert "reformatted" in action_types
    assert "imputed_median" in action_types


def test_apply_repairs_flags_out_of_range_age_and_imputes_median() -> None:
    """Ages outside 0-120 are nulled out and then median-imputed."""
    df = pd.DataFrame({"age": [25, -5, 150, 30]})
    profile = _profile([_column_profile("age", "age")])
    result_df, actions, _, _ = apply_repairs(df, profile)
    assert result_df["age"].isnull().sum() == 0
    action_types = {a.action_taken for a in actions}
    assert "flagged" in action_types
    assert "imputed_median" in action_types


def test_apply_repairs_escalates_email_issues_to_unresolved_without_fixing() -> None:
    """Null and malformed emails are escalated, never auto-repaired."""
    df = pd.DataFrame({"email": ["a@x.com", None, "not-an-email"]})
    profile = _profile([_column_profile("email", "email")])
    result_df, actions, _, unresolved = apply_repairs(df, profile)
    assert result_df["email"].isnull().sum() == 1
    assert not any(a.column == "email" for a in actions)
    assert len(unresolved) == 2


def test_apply_repairs_standardizes_non_iso_dates() -> None:
    """Dates not already in YYYY-MM-DD format are standardized."""
    df = pd.DataFrame({"signup": ["2024-01-15", "01/20/2024"]})
    profile = _profile([_column_profile("signup", "date")])
    result_df, actions, _, _ = apply_repairs(df, profile)
    assert result_df["signup"].tolist() == ["2024-01-15", "2024-01-20"]
    assert any(a.action_taken == "reformatted" for a in actions)


def test_apply_repairs_normalizes_casing_and_imputes_mode_for_categorical() -> None:
    """Minority-cased values are normalized to the majority casing, and
    nulls are mode-imputed."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "dept": ["engineering", "Engineering", "Engineering", None],
        }
    )
    profile = _profile(
        [_column_profile("id", "id"), _column_profile("dept", "categorical")]
    )
    result_df, actions, _, _ = apply_repairs(df, profile)
    assert result_df["dept"].isnull().sum() == 0
    assert (result_df["dept"] == "Engineering").all()
    action_types = {a.action_taken for a in actions}
    assert "reformatted" in action_types
    assert "imputed_mode" in action_types


def test_apply_repairs_leaves_consistent_abbreviations_untouched() -> None:
    """A consistently-used abbreviation like 'HR' is not mistaken for
    inconsistent casing just because it doesn't match Python's .title()."""
    df = pd.DataFrame({"id": [1, 2, 3, 4], "dept": ["HR", "HR", "Sales", "Sales"]})
    profile = _profile(
        [_column_profile("id", "id"), _column_profile("dept", "categorical")]
    )
    result_df, actions, _, _ = apply_repairs(df, profile)
    assert list(result_df["dept"]) == ["HR", "HR", "Sales", "Sales"]
    assert not any(a.column == "dept" for a in actions)


def test_apply_repairs_imputes_median_for_numeric_nulls() -> None:
    """Null values in a plain numeric column are median-imputed."""
    df = pd.DataFrame({"score": [10, 20, None]})
    profile = _profile([_column_profile("score", "numeric")])
    result_df, actions, _, _ = apply_repairs(df, profile)
    assert result_df["score"].isnull().sum() == 0
    assert any(a.action_taken == "imputed_median" for a in actions)


def test_apply_repairs_escalates_null_ids_without_fixing() -> None:
    """Null id values are escalated to unresolved, never fabricated."""
    df = pd.DataFrame({"patient_id": ["p1", None, "p3"]})
    profile = _profile([_column_profile("patient_id", "id")])
    result_df, actions, _, unresolved = apply_repairs(df, profile)
    assert result_df["patient_id"].isnull().sum() == 1
    assert not any(a.column == "patient_id" for a in actions)
    assert len(unresolved) == 1


def test_apply_repairs_imputes_mode_for_boolean_nulls() -> None:
    """Null values in a boolean column are mode-imputed."""
    df = pd.DataFrame({"active": [True, True, False, None]})
    profile = _profile([_column_profile("active", "boolean")])
    result_df, actions, _, _ = apply_repairs(df, profile)
    assert result_df["active"].isnull().sum() == 0
    assert any(a.action_taken == "imputed_mode" for a in actions)


def test_apply_repairs_escalates_sparse_columns_regardless_of_type() -> None:
    """A column over 50% null is escalated to unresolved untouched."""
    df = pd.DataFrame({"id": [1, 2, 3, 4], "notes": [None, None, None, "text"]})
    profile = _profile([_column_profile("id", "id"), _column_profile("notes", "text")])
    result_df, actions, _, unresolved = apply_repairs(df, profile)
    assert result_df["notes"].isnull().sum() == 3
    assert not any(a.column == "notes" for a in actions)
    assert len(unresolved) == 1


def test_apply_repairs_never_touches_unsafe_to_impute_columns() -> None:
    """A column the Profiler marked unsafe to impute is escalated, not repaired."""
    df = pd.DataFrame({"diagnosis": ["flu", None, "flu"]})
    profile = _profile(
        [_column_profile("diagnosis", "categorical")],
        missingness=_missingness("diagnosis", safe_to_impute=False),
    )
    result_df, actions, _, unresolved = apply_repairs(df, profile)
    assert result_df["diagnosis"].isnull().sum() == 1
    assert not any(a.column == "diagnosis" for a in actions)
    assert len(unresolved) == 1


@pytest.mark.llm
def test_run_repairer_produces_a_valid_repair_report(tmp_path: Path) -> None:
    """Against a dataset with a known duplicate row, the agent narrates
    the actual repairs into a schema-valid RepairReport."""
    input_path = str(tmp_path / "people.csv")
    output_path = str(tmp_path / "people_clean.csv")
    df = pd.DataFrame({"name": ["Al", "Al", "Bo"], "age": [25, 25, 30]})
    df.to_csv(input_path, index=False)

    profile = _profile([_column_profile("name", "text"), _column_profile("age", "age")])
    report = run_repairer(input_path, output_path, profile)
    assert report.total_repairs == len(report.actions)

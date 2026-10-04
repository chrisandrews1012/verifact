from evals.runner import (
    _score_column_types,
    _score_known_facts,
    _score_mnar_safety,
    _score_no_false_positives,
    _score_repair_coverage,
    _score_unresolved_coverage,
)
from verifact.models import (
    ColumnMissingness,
    ColumnProfile,
    DataProfile,
    MissingnessReport,
    RepairAction,
    RepairReport,
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
        row_count=10,
        column_count=len(columns),
        duplicate_row_count=2,
        total_null_count=3,
        columns=columns,
        summary="test profile",
        missingness=missingness,
    )


def _repair(
    actions: list[RepairAction],
    unresolved: list[str] | None = None,
    rows_dropped: int = 0,
) -> RepairReport:
    return RepairReport(
        total_repairs=len(actions),
        rows_dropped=rows_dropped,
        actions=actions,
        unresolved=unresolved or [],
        output_path="out.csv",
        summary="test repair",
    )


def _action(column: str, action_taken: str) -> RepairAction:
    return RepairAction(
        column=column,
        issue="test issue",
        action_taken=action_taken,  # type: ignore[arg-type]
        rows_affected=1,
        before_example="before",
        after_example="after",
        reason="test reason",
    )


def test_score_column_types_counts_correct_and_wrong() -> None:
    """Columns matching the expected type are correct; mismatches are reported."""
    profile = _profile(
        [_column_profile("age", "age"), _column_profile("email", "text")]
    )
    result = _score_column_types(profile, {"age": "age", "email": "email"})
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["wrong"] == {"email": {"expected": "email", "got": "text"}}


def test_score_column_types_full_score_when_all_match() -> None:
    """A perfect match scores 1.0 with no wrong entries."""
    profile = _profile([_column_profile("age", "age")])
    result = _score_column_types(profile, {"age": "age"})
    assert result["score"] == 1.0
    assert result["wrong"] == {}


def test_score_known_facts_passes_when_all_match() -> None:
    """Matching row/duplicate/null counts produce zero violations."""
    profile = _profile([_column_profile("age", "age")])
    result = _score_known_facts(
        profile, {"row_count": 10, "duplicate_row_count": 2, "total_null_count": 3}
    )
    assert result == {"violations": [], "passed": True}


def test_score_known_facts_flags_mismatched_row_count() -> None:
    """A claimed row_count that doesn't match the spec is a violation."""
    profile = _profile([_column_profile("age", "age")])
    result = _score_known_facts(profile, {"row_count": 999})
    assert result["passed"] is False
    assert any("row_count" in v for v in result["violations"])


def test_score_repair_coverage_detects_missing_column_repair() -> None:
    """A column the spec requires repaired, but wasn't, is reported missing."""
    repair = _repair([_action("age", "imputed_median")])
    result = _score_repair_coverage(
        repair, {"columns": {"age": ["x"], "salary": ["y"]}}
    )
    assert result["found"] == 1
    assert result["total"] == 2
    assert "'salary' was not repaired (expected: ['y'])" in result["missing"]


def test_score_repair_coverage_detects_missing_duplicate_removal() -> None:
    """A spec requiring duplicate-row removal with no matching action is flagged."""
    repair = _repair([_action("age", "imputed_median")])
    result = _score_repair_coverage(repair, {"duplicate_rows": True})
    assert "duplicate rows were not removed" in result["missing"]


def test_score_repair_coverage_recognizes_duplicate_removal() -> None:
    """A '(all columns)' action mentioning duplicates satisfies the requirement."""
    dup_action = RepairAction(
        column="(all columns)",
        issue="Duplicate rows",
        action_taken="dropped_rows",
        rows_affected=2,
        before_example="dup",
        after_example="removed",
        reason="test",
    )
    repair = _repair([dup_action], rows_dropped=2)
    result = _score_repair_coverage(repair, {"duplicate_rows": True})
    assert result["missing"] == []
    assert result["score"] == 1.0


def test_score_unresolved_coverage_finds_mentioned_column() -> None:
    """A column named in the unresolved text satisfies the requirement."""
    repair = _repair(
        [], unresolved=["'email' (email): 3 null values. Manual review required."]
    )
    result = _score_unresolved_coverage(repair, ["email"])
    assert result["found"] == 1
    assert result["missing"] == []


def test_score_unresolved_coverage_flags_column_never_mentioned() -> None:
    """A required column absent from the unresolved text is reported missing."""
    repair = _repair([], unresolved=["something else entirely"])
    result = _score_unresolved_coverage(repair, ["email"])
    assert "'email' not mentioned in unresolved issues" in result["missing"]


def test_score_no_false_positives_passes_on_truly_clean_repair() -> None:
    """A repair with no actions, no dropped rows, and nothing unresolved passes."""
    repair = _repair([])
    result = _score_no_false_positives(repair)
    assert result == {"violations": [], "passed": True}


def test_score_no_false_positives_flags_unexpected_column_repair() -> None:
    """A repair action applied to a column on supposedly clean data is a violation."""
    repair = _repair([_action("age", "imputed_median")])
    result = _score_no_false_positives(repair)
    assert result["passed"] is False
    assert any("age" in v for v in result["violations"])


def test_score_mnar_safety_passes_when_no_missingness_report() -> None:
    """A profile with no missingness analysis at all trivially passes."""
    profile = _profile([_column_profile("age", "age")])
    repair = _repair([])
    result = _score_mnar_safety(profile, repair)
    assert result == {"mnar_columns": [], "violations": [], "passed": True}


def test_score_mnar_safety_flags_an_imputed_mnar_column() -> None:
    """Imputing a column the Profiler classified as MNAR is a safety violation."""
    missingness = MissingnessReport(
        dataset_mcar_conclusion="test",
        columns_analyzed=[
            ColumnMissingness(
                column="diagnosis",
                null_count=5,
                null_pct=50.0,
                mechanism="MNAR",
                confidence="low",
                evidence="test",
                safe_to_impute=False,
            )
        ],
        summary="test",
    )
    profile = _profile([_column_profile("diagnosis", "categorical")], missingness)
    repair = _repair([_action("diagnosis", "imputed_mode")])
    result = _score_mnar_safety(profile, repair)
    assert result["passed"] is False
    assert "diagnosis" in result["mnar_columns"]


def test_score_mnar_safety_passes_when_mnar_column_left_untouched() -> None:
    """Leaving an MNAR column unrepaired (no imputation action) passes."""
    missingness = MissingnessReport(
        dataset_mcar_conclusion="test",
        columns_analyzed=[
            ColumnMissingness(
                column="diagnosis",
                null_count=5,
                null_pct=50.0,
                mechanism="MNAR",
                confidence="low",
                evidence="test",
                safe_to_impute=False,
            )
        ],
        summary="test",
    )
    profile = _profile([_column_profile("diagnosis", "categorical")], missingness)
    repair = _repair([])
    result = _score_mnar_safety(profile, repair)
    assert result["passed"] is True

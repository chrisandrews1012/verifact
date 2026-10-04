from verifact.models import (
    ColumnMissingness,
    ColumnProfile,
    DataProfile,
    MissingnessReport,
    PipelineContext,
    RepairAction,
    RepairReport,
    ValidationFailure,
    ValidationReport,
    ValidationRule,
)


def make_column_profile(**overrides: object) -> ColumnProfile:
    defaults: dict[str, object] = {
        "name": "age",
        "dtype": "int64",
        "null_count": 0,
        "null_pct": 0.0,
        "unique_count": 42,
        "unique_pct": 8.1,
        "sample_values": ["25", "31", "40"],
        "inferred_type": "age",
    }
    defaults.update(overrides)
    return ColumnProfile(**defaults)  # type: ignore[arg-type]


def test_column_profile_construction() -> None:
    """A ColumnProfile can be built with only its required fields."""
    profile = make_column_profile()
    assert profile.name == "age"
    assert profile.min_value is None
    assert profile.inferred_type == "age"


def test_data_profile_construction() -> None:
    """A DataProfile bundles its columns and defaults missingness to None."""
    profile = DataProfile(
        dataset_name="hr",
        row_count=520,
        column_count=1,
        duplicate_row_count=20,
        total_null_count=0,
        columns=[make_column_profile()],
        summary="Mostly clean HR data.",
    )
    assert profile.row_count == 520
    assert profile.missingness is None
    assert len(profile.columns) == 1


def test_missingness_report_construction() -> None:
    """A MissingnessReport carries per-column missingness analysis."""
    missingness = ColumnMissingness(
        column="email",
        null_count=18,
        null_pct=3.5,
        mechanism="MNAR",
        confidence="high",
        evidence="Missingness correlates with department = 'Sales'.",
        safe_to_impute=False,
    )
    report = MissingnessReport(
        dataset_mcar_conclusion="Not MCAR.",
        columns_analyzed=[missingness],
        summary="One column shows MNAR missingness.",
    )
    assert report.columns_analyzed[0].mechanism == "MNAR"
    assert report.dataset_mcar_pvalue is None


def test_validation_report_failure_count_is_corrected_from_failures() -> None:
    """failure_count is derived from len(failures), not the value passed in."""
    failure = ValidationFailure(
        column="age",
        rule="range check",
        severity="critical",
        affected_rows=3,
        description="3 values outside 0-120.",
        suggested_fix="Nullify then impute.",
    )
    report = ValidationReport(
        passed=False,
        rules_applied=[
            ValidationRule(
                column="age", rule_description="range check", severity="critical"
            )
        ],
        failure_count=999,
        failures=[failure],
        summary="One critical failure.",
    )
    assert report.failure_count == 1


def test_repair_report_total_repairs_is_corrected_from_actions() -> None:
    """total_repairs is derived from len(actions), not the value passed in."""
    action = RepairAction(
        column="salary",
        issue="Currency symbols present.",
        action_taken="reformatted",
        rows_affected=520,
        before_example="$50,000",
        after_example="50000.0",
        reason="Strip currency symbols before numeric analysis.",
    )
    report = RepairReport(
        total_repairs=999,
        rows_dropped=20,
        actions=[action],
        output_path="data/processed/hr_clean.csv",
        summary="One repair action applied.",
    )
    assert report.total_repairs == 1
    assert report.unresolved == []


def test_pipeline_context_bundles_stage_outputs() -> None:
    """PipelineContext holds the profile, validation, and repair results together."""
    profile = DataProfile(
        dataset_name="hr",
        row_count=520,
        column_count=1,
        duplicate_row_count=0,
        total_null_count=0,
        columns=[make_column_profile()],
        summary="Clean.",
    )
    validation = ValidationReport(
        passed=True,
        rules_applied=[],
        failure_count=0,
        failures=[],
        summary="No failures.",
    )
    repair = RepairReport(
        total_repairs=0,
        rows_dropped=0,
        actions=[],
        output_path="data/processed/hr_clean.csv",
        summary="Nothing to repair.",
    )
    context = PipelineContext(
        input_path="data/raw/hr.csv",
        output_path="data/processed/hr_clean.csv",
        profile=profile,
        validation=validation,
        repair=repair,
    )
    assert context.validation.passed is True
    assert context.repair.total_repairs == 0

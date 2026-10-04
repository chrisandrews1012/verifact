import pandas as pd

from verifact.models import DataProfile, RepairReport, ValidationReport

_PCT_TOLERANCE = 0.5


class InvariantViolation(Exception):
    pass


def check_profile_invariants(profile: DataProfile, df: pd.DataFrame) -> list[str]:
    violations: list[str] = []

    actual_rows = len(df)
    actual_cols = len(df.columns)
    actual_nulls = int(df.isnull().sum().sum())
    actual_dupes = int(df.duplicated().sum())

    if profile.row_count != actual_rows:
        violations.append(
            f"row_count claimed {profile.row_count}, actual {actual_rows}"
        )
    if profile.column_count != actual_cols:
        violations.append(
            f"column_count claimed {profile.column_count}, actual {actual_cols}"
        )
    if profile.total_null_count != actual_nulls:
        violations.append(
            f"total_null_count claimed {profile.total_null_count}, "
            f"actual {actual_nulls}"
        )
    if profile.duplicate_row_count != actual_dupes:
        violations.append(
            f"duplicate_row_count claimed {profile.duplicate_row_count}, "
            f"actual {actual_dupes}"
        )

    df_col_set = set(df.columns)
    profiled_col_set = {cp.name for cp in profile.columns}

    for cp in profile.columns:
        if cp.name not in df_col_set:
            violations.append(f"Profiled column '{cp.name}' does not exist in CSV")
            continue

        actual_null_count = int(df[cp.name].isnull().sum())
        if cp.null_count != actual_null_count:
            violations.append(
                f"'{cp.name}': null_count claimed {cp.null_count}, "
                f"actual {actual_null_count}"
            )

        actual_null_pct = round(float(df[cp.name].isnull().mean() * 100), 2)
        if abs(cp.null_pct - actual_null_pct) > _PCT_TOLERANCE:
            violations.append(
                f"'{cp.name}': null_pct claimed {cp.null_pct}, actual {actual_null_pct}"
            )

        actual_unique = int(df[cp.name].nunique())
        if cp.unique_count != actual_unique:
            violations.append(
                f"'{cp.name}': unique_count claimed {cp.unique_count}, "
                f"actual {actual_unique}"
            )

    for col in df.columns:
        if col not in profiled_col_set:
            violations.append(f"Column '{col}' in CSV was not profiled")

    return violations


def check_validation_invariants(
    validation: ValidationReport, profile: DataProfile
) -> list[str]:
    violations: list[str] = []

    if validation.failure_count != len(validation.failures):
        violations.append(
            f"failure_count={validation.failure_count} but "
            f"len(failures)={len(validation.failures)}"
        )

    critical_count = sum(1 for f in validation.failures if f.severity == "critical")
    if critical_count > 0 and validation.passed:
        violations.append(f"passed=True but {critical_count} critical failure(s) exist")
    if critical_count == 0 and not validation.passed:
        violations.append("passed=False but no critical failures were found")

    known_cols = {cp.name for cp in profile.columns} | {"(all columns)"}

    for failure in validation.failures:
        if failure.column not in known_cols:
            violations.append(
                f"ValidationFailure references unknown column '{failure.column}'"
            )

    for rule in validation.rules_applied:
        if rule.column not in known_cols:
            violations.append(
                f"ValidationRule references unknown column '{rule.column}'"
            )

    return violations


def check_repair_invariants(
    repair: RepairReport,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> list[str]:
    violations: list[str] = []

    if repair.total_repairs != len(repair.actions):
        violations.append(
            f"total_repairs={repair.total_repairs} but "
            f"len(actions)={len(repair.actions)}"
        )

    actual_delta = len(input_df) - len(output_df)
    if repair.rows_dropped != actual_delta:
        violations.append(
            f"rows_dropped claimed {repair.rows_dropped}, "
            f"actual row delta is {actual_delta}"
        )

    if repair.rows_dropped > len(input_df):
        violations.append(
            f"rows_dropped={repair.rows_dropped} exceeds input row count "
            f"{len(input_df)}"
        )

    known_cols = set(input_df.columns) | {"(all columns)"}
    for action in repair.actions:
        if action.column not in known_cols:
            violations.append(
                f"RepairAction references unknown column '{action.column}'"
            )

    return violations


def assert_invariants(violations: list[str], agent_name: str) -> None:
    if violations:
        bullet_list = "\n".join(f"  - {v}" for v in violations)
        raise InvariantViolation(
            f"{agent_name} invariant violations detected:\n{bullet_list}"
        )

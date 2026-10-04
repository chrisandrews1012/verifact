from typing import Literal

from pydantic import BaseModel, Field, model_validator


# Profiler output
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[str]
    min_value: str | None = None
    max_value: str | None = None
    mean_value: float | None = None
    std_value: float | None = None
    inferred_type: str = Field(
        description=(
            "Semantic type inferred from name and values. "
            "Examples: id, email, age, date, currency, categorical, "
            "numeric, text, boolean, unknown"
        )
    )


class ColumnMissingness(BaseModel):
    column: str
    null_count: int
    null_pct: float
    mechanism: Literal["MCAR", "MAR", "MNAR", "none"]
    confidence: Literal["high", "medium", "low"]
    evidence: str
    correlated_with: list[str] = Field(default_factory=list)
    safe_to_impute: bool


class MissingnessReport(BaseModel):
    dataset_mcar_pvalue: float | None = None
    dataset_mcar_conclusion: str
    columns_analyzed: list[ColumnMissingness]
    summary: str


class DataProfile(BaseModel):
    dataset_name: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    total_null_count: int
    columns: list[ColumnProfile]
    summary: str = Field(
        description=(
            "2-3 sentence plain English summary of overall data quality. "
            "What looks good, what looks problematic, what needs "
            "investigation."
        )
    )
    missingness: MissingnessReport | None = None


# Validator output
class ValidationRule(BaseModel):
    column: str
    rule_description: str
    severity: Literal["critical", "consideration", "info"]


class ValidationFailure(BaseModel):
    column: str
    rule: str
    severity: Literal["critical", "consideration", "info"]
    affected_rows: int
    description: str
    suggested_fix: str


class ValidationReport(BaseModel):
    passed: bool = Field(description="True only if zero critical failures were found")
    rules_applied: list[ValidationRule]
    failure_count: int
    failures: list[ValidationFailure]
    summary: str

    @model_validator(mode="after")
    def _fix_failure_count(self) -> "ValidationReport":
        self.failure_count = len(self.failures)
        return self


# Repairer output
class RepairAction(BaseModel):
    column: str
    issue: str
    action_taken: Literal[
        "imputed_mean",
        "imputed_mode",
        "imputed_median",
        "dropped_rows",
        "reformatted",
        "capped",
        "flagged",
        "skipped",
    ]
    rows_affected: int
    before_example: str
    after_example: str
    reason: str = Field(
        description="Why this repair action was chosen over alternatives"
    )


class RepairReport(BaseModel):
    total_repairs: int
    rows_dropped: int
    actions: list[RepairAction]
    unresolved: list[str] = Field(
        default_factory=list,
        description="Issues the repairer flagged but could not automatically fix",
    )
    output_path: str
    summary: str

    @model_validator(mode="after")
    def _fix_total_repairs(self) -> "RepairReport":
        self.total_repairs = len(self.actions)
        return self


# Pipeline context (passed to Reporter)
class PipelineContext(BaseModel):
    input_path: str
    output_path: str
    profile: DataProfile
    validation: ValidationReport
    repair: RepairReport

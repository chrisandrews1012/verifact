from pathlib import Path

import pandas as pd

from verifact.tools import (
    analyze_missingness,
    build_missingness_summary,
    clean_numeric_string,
    count_invalid_emails,
    count_non_numeric,
    count_non_standard_dates,
    detect_case_inconsistency,
    get_column_stats,
    get_dataset_stats,
    is_numeric_string,
    is_valid_date,
    is_valid_email,
    load_dataframe,
    normalize_case_inconsistency,
    save_dataframe,
    standardize_case,
    standardize_date,
)


def test_load_and_save_dataframe_round_trip(tmp_path: Path) -> None:
    """A dataframe saved to a nested path can be loaded back with the same data."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = str(tmp_path / "nested" / "out.csv")
    save_dataframe(df, path)
    loaded = load_dataframe(path)
    assert list(loaded["a"]) == [1, 2]
    assert list(loaded["b"]) == ["x", "y"]


def test_get_column_stats_numeric_column() -> None:
    """Numeric columns get min/max/mean/std alongside the shared stats."""
    df = pd.DataFrame({"age": [10, 20, 30, None]})
    stats = get_column_stats(df, "age")
    assert stats["null_count"] == 1
    assert stats["unique_count"] == 3
    assert stats["mean_value"] == 20.0


def test_get_column_stats_non_numeric_column_omits_numeric_fields() -> None:
    """Non-numeric columns don't get min/max/mean/std keys at all."""
    df = pd.DataFrame({"category": ["a", "b", "a", None]})
    stats = get_column_stats(df, "category")
    assert "mean_value" not in stats


def test_get_dataset_stats() -> None:
    """Dataset stats report row/column counts, duplicates, and total nulls."""
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"]})
    stats = get_dataset_stats(df)
    assert stats["row_count"] == 4
    assert stats["column_count"] == 2
    assert stats["duplicate_row_count"] == 1
    assert stats["total_null_count"] == 1


def test_is_valid_email_accepts_standard_format() -> None:
    """A well-formed email address passes validation."""
    assert is_valid_email("a@x.com") is True


def test_is_valid_email_rejects_missing_at_sign() -> None:
    """A string without an @ sign fails validation."""
    assert is_valid_email("not-an-email") is False


def test_is_numeric_string_accepts_currency_formatting() -> None:
    """A currency-formatted string with symbols and commas is still numeric."""
    assert is_numeric_string("$1,000.00") is True


def test_is_numeric_string_rejects_free_text() -> None:
    """Free text that isn't a number fails."""
    assert is_numeric_string("not a number") is False


def test_is_valid_date_accepts_common_format() -> None:
    """A standard date string parses successfully."""
    assert is_valid_date("2024-01-15") is True


def test_is_valid_date_rejects_garbage() -> None:
    """Non-date text fails to parse."""
    assert is_valid_date("not a date") is False


def test_clean_numeric_string_strips_currency_symbols_and_commas() -> None:
    """Currency symbols and thousands separators are stripped, leaving a float."""
    assert clean_numeric_string("$1,234.50") == 1234.50


def test_clean_numeric_string_returns_none_for_unparseable_value() -> None:
    """A value that can't be converted to a float returns None."""
    assert clean_numeric_string("not a number") is None


def test_standardize_date_converts_to_iso_format() -> None:
    """A date in a different format is standardized to YYYY-MM-DD."""
    assert standardize_date("01/15/2024") == "2024-01-15"


def test_standardize_case_title_style() -> None:
    """Title-casing capitalizes each word."""
    assert standardize_case("new york", "title") == "New York"


def test_count_invalid_emails_counts_only_malformed_non_null_values() -> None:
    """Only non-null values that fail email validation are counted."""
    series = pd.Series(["a@x.com", "not-an-email", None])
    assert count_invalid_emails(series) == 1


def test_count_non_numeric_counts_unparseable_values() -> None:
    """Values that can't be parsed as numbers are counted, nulls are not."""
    series = pd.Series(["100", "not a number", None])
    assert count_non_numeric(series) == 1


def test_count_non_standard_dates_counts_non_iso_values() -> None:
    """Values not already in YYYY-MM-DD format are counted."""
    series = pd.Series(["2024-01-15", "01/15/2024"])
    assert count_non_standard_dates(series) == 1


def test_detect_case_inconsistency_counts_minority_casing_variants() -> None:
    """A value spelled differently than the majority casing for its
    category is counted."""
    series = pd.Series(["Engineering", "engineering", "Sales"])
    assert detect_case_inconsistency(series) == 1


def test_detect_case_inconsistency_ignores_consistent_abbreviations() -> None:
    """A value like 'HR' used consistently isn't flagged just because it
    differs from its own Python .title()-cased form ('Hr')."""
    series = pd.Series(["HR", "HR", "Sales", "Sales"])
    assert detect_case_inconsistency(series) == 0


def test_normalize_case_inconsistency_fixes_minority_casing_only() -> None:
    """Minority-cased values are normalized to the majority casing within
    their category; a consistently-used abbreviation is left untouched."""
    series = pd.Series(["HR", "Engineering", "engineering", "Engineering"])
    result = normalize_case_inconsistency(series)
    assert list(result) == ["HR", "Engineering", "Engineering", "Engineering"]


def test_analyze_missingness_detects_mar_correlation_with_numeric_column() -> None:
    """A column whose missingness correlates with a numeric column is MAR."""
    n = 60
    salary = list(range(n))
    bonus: list[float | None] = [None if i < n // 2 else float(i) for i in range(n)]
    df = pd.DataFrame({"salary": salary, "bonus": bonus})
    result = analyze_missingness(df)
    bonus_result = next(c for c in result["columns"] if c["column"] == "bonus")
    assert bonus_result["mechanism"] == "MAR"
    assert bonus_result["safe_to_impute"] is True


def test_analyze_missingness_flags_high_uncorrelated_nulls_as_mnar() -> None:
    """A column with a high null rate and no correlate is classified MNAR."""
    n = 40
    df = pd.DataFrame(
        {
            "steady": list(range(n)),
            "sensitive": [None if i % 2 == 0 else "value" for i in range(n)],
        }
    )
    result = analyze_missingness(df)
    sensitive_result = next(c for c in result["columns"] if c["column"] == "sensitive")
    assert sensitive_result["mechanism"] == "MNAR"
    assert sensitive_result["safe_to_impute"] is False


def test_analyze_missingness_returns_empty_columns_when_no_nulls() -> None:
    """A dataset with no nulls at all produces no per-column results."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = analyze_missingness(df)
    assert result["columns"] == []


def test_build_missingness_summary_reports_no_missing_data() -> None:
    """An empty columns list produces the no-missing-data message."""
    raw = {"columns": [], "dataset_mcar_conclusion": "n/a"}
    summary = build_missingness_summary(raw)
    assert summary == "No missing data detected."


def test_build_missingness_summary_mentions_mnar_columns() -> None:
    """A column classified as MNAR is called out by name in the summary."""
    raw = {
        "columns": [
            {"column": "notes", "mechanism": "MNAR"},
        ],
        "dataset_mcar_conclusion": "not MCAR",
    }
    summary = build_missingness_summary(raw)
    assert "notes" in summary
    assert "MNAR" in summary

import os
import re
from typing import Any

import numpy as np
import pandas as pd

# I/O


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


# Column statistics


def get_column_stats(df: pd.DataFrame, col: str) -> dict[str, object]:
    series = df[col]
    n = len(series)
    non_null = series.dropna()

    stats: dict[str, object] = {
        "dtype": str(series.dtype),
        "null_count": int(series.isnull().sum()),
        "null_pct": round(float(series.isnull().mean() * 100), 2),
        "unique_count": int(series.nunique()),
        "unique_pct": round(series.nunique() / n * 100, 2) if n > 0 else 0.0,
        "sample_values": [
            str(v)
            for v in non_null.sample(
                min(5, non_null.shape[0]), random_state=42
            ).tolist()
        ],
    }

    if pd.api.types.is_numeric_dtype(series) and not series.isnull().all():
        stats["min_value"] = str(round(float(series.min()), 4))
        stats["max_value"] = str(round(float(series.max()), 4))
        stats["mean_value"] = round(float(series.mean()), 4)
        stats["std_value"] = round(float(series.std()), 4)

    return stats


def get_dataset_stats(df: pd.DataFrame) -> dict[str, int]:
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
        "total_null_count": int(df.isnull().sum().sum()),
    }


# Validators

_EMAIL_PATTERN = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"


def is_valid_email(value: object) -> bool:
    return bool(re.match(_EMAIL_PATTERN, str(value)))


def is_numeric_string(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        float(str(value).replace("$", "").replace(",", "").replace("£", "").strip())
        return True
    except ValueError:
        return False


def is_valid_date(value: object) -> bool:
    from dateutil import parser as dateparser

    try:
        dateparser.parse(str(value))
        return True
    except Exception:
        return False


# Cleaners


def clean_numeric_string(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(
            str(value)
            .replace("$", "")
            .replace("£", "")
            .replace("€", "")
            .replace(",", "")
            .strip()
        )
    except ValueError:
        return None


def standardize_date(value: object) -> str | None:
    from dateutil import parser as dateparser

    try:
        return str(dateparser.parse(str(value)).date())
    except Exception:
        return None


def standardize_case(value: Any, style: str = "title") -> Any:
    if pd.isna(value):
        return value
    if style == "title":
        return str(value).title()
    elif style == "upper":
        return str(value).upper()
    else:
        return str(value).lower()


# Repair helpers


def count_invalid_emails(series: pd.Series) -> int:
    return int((~series.dropna().apply(is_valid_email)).sum())


def count_non_numeric(series: pd.Series) -> int:
    return int((~series.dropna().apply(is_numeric_string)).sum())


def count_non_standard_dates(series: pd.Series) -> int:
    pattern = r"^\d{4}-\d{2}-\d{2}$"
    return int((~series.dropna().astype(str).str.match(pattern)).sum())


def _dominant_casing_by_key(series: pd.Series) -> dict[str, str]:
    from collections import Counter

    groups: dict[str, Counter[str]] = {}
    for value in series.dropna():
        key = str(value).lower()
        groups.setdefault(key, Counter())[str(value)] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in groups.items()}


def detect_case_inconsistency(series: pd.Series) -> int:
    dominant = _dominant_casing_by_key(series)
    mismatched = series.dropna().apply(lambda x: str(x) != dominant[str(x).lower()])
    return int(mismatched.sum())


def normalize_case_inconsistency(series: pd.Series) -> pd.Series:
    dominant = _dominant_casing_by_key(series)
    return series.apply(  # type: ignore[no-any-return]
        lambda x: dominant[str(x).lower()] if pd.notna(x) else x
    )


# Missingness mechanism detection


def analyze_missingness(df: pd.DataFrame) -> dict[str, object]:
    from scipy import stats as scipy_stats

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_with_nulls = [c for c in df.columns if df[c].isnull().any()]
    column_results = []

    for col in cols_with_nulls:
        null_pct = float(df[col].isnull().mean())
        missing_indicator = df[col].isnull().astype(int)
        correlated_with = []

        for other in numeric_cols:
            if other == col:
                continue
            valid = df[other].notna()
            if valid.sum() < 20:
                continue
            corr, pval = scipy_stats.pointbiserialr(
                missing_indicator[valid], df[other][valid]
            )
            if pval < 0.05 and abs(corr) > 0.1:
                correlated_with.append(other)

        if correlated_with:
            mechanism = "MAR"
            confidence = "high" if len(correlated_with) >= 2 else "medium"
            evidence = (
                f"Missingness significantly correlated with: "
                f"{', '.join(correlated_with)}. Imputation is valid but "
                f"conditioning on these columns improves accuracy."
            )
            safe_to_impute = True
        elif null_pct > 0.20:
            mechanism = "MNAR"
            confidence = "low"
            evidence = (
                f"{null_pct:.0%} missing with no detectable correlation to "
                f"other columns. Values may be absent because of the value "
                f"itself. Imputation would introduce systematic bias."
            )
            safe_to_impute = False
        else:
            mechanism = "MCAR"
            confidence = "medium"
            evidence = (
                f"Low null rate ({null_pct:.0%}) with no significant "
                f"correlation to other columns. Safe to impute."
            )
            safe_to_impute = True

        column_results.append(
            {
                "column": col,
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(null_pct * 100, 2),
                "mechanism": mechanism,
                "confidence": confidence,
                "evidence": evidence,
                "correlated_with": correlated_with,
                "safe_to_impute": safe_to_impute,
            }
        )

    mcar_pvalue = None
    mcar_conclusion = "insufficient numeric columns for test"
    if len(numeric_cols) >= 2 and df[numeric_cols].isnull().any().any():
        try:
            mcar_pvalue, mcar_conclusion = _little_mcar_test(df[numeric_cols])
        except Exception:
            mcar_conclusion = "test could not be computed"

    return {
        "columns": column_results,
        "dataset_mcar_pvalue": mcar_pvalue,
        "dataset_mcar_conclusion": mcar_conclusion,
    }


def _little_mcar_test(df: pd.DataFrame) -> tuple[float | None, str]:
    from scipy import stats as scipy_stats

    miss_matrix = df.isnull().astype(int)
    pattern_series = miss_matrix.apply(tuple, axis=1)
    unique_patterns = pattern_series.unique()

    if len(unique_patterns) == 1:
        return (
            1.0,
            "MCAR: single missingness pattern (all rows have same observed columns)",
        )

    overall_means = df.mean()
    overall_cov = df.cov()
    d_sq = 0.0
    dof = 0

    for pattern in unique_patterns:
        mask = pattern_series == pattern
        n_k = int(mask.sum())
        if n_k < 2:
            continue
        observed_cols = [c for c, m in zip(df.columns, pattern, strict=True) if m == 0]
        if not observed_cols:
            continue
        group_means = df.loc[mask, observed_cols].mean()
        diff = (group_means - overall_means[observed_cols]).to_numpy()
        sub_cov = overall_cov.loc[observed_cols, observed_cols].to_numpy()
        try:
            cov_inv = np.linalg.pinv(sub_cov)
            d_sq += n_k * float(diff @ cov_inv @ diff)
            dof += len(observed_cols)
        except Exception:
            continue

    if dof == 0:
        return (None, "could not compute: no valid patterns")

    p_value = round(float(1 - scipy_stats.chi2.cdf(d_sq, df=dof)), 4)
    if p_value > 0.05:
        conclusion = f"MCAR (p={p_value} > 0.05): fail to reject null hypothesis"
    else:
        conclusion = (
            f"not MCAR (p={p_value} <= 0.05): missingness is likely MAR or MNAR"
        )

    return (p_value, conclusion)


def build_missingness_summary(raw: dict[str, object]) -> str:
    cols = raw["columns"]
    assert isinstance(cols, list)
    if not cols:
        return "No missing data detected."

    mnar = [c["column"] for c in cols if c["mechanism"] == "MNAR"]
    mar = [c["column"] for c in cols if c["mechanism"] == "MAR"]
    mcar = [c["column"] for c in cols if c["mechanism"] == "MCAR"]
    parts = []
    if mnar:
        parts.append(
            f"{len(mnar)} column(s) flagged as potentially MNAR "
            f"({', '.join(mnar)}). Imputation not recommended."
        )
    if mar:
        parts.append(
            f"{len(mar)} column(s) classified as MAR "
            f"({', '.join(mar)}). Safe to impute."
        )
    if mcar:
        parts.append(f"{len(mcar)} column(s) classified as MCAR. Safe to impute.")
    parts.append(f"Dataset-level test: {raw['dataset_mcar_conclusion']}.")
    return " ".join(parts)

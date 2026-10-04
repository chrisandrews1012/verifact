import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from verifact.models import ColumnMissingness, DataProfile, MissingnessReport
from verifact.tools import (
    analyze_missingness,
    build_missingness_summary,
    get_column_stats,
    get_dataset_stats,
    load_dataframe,
)

SYSTEM_PROMPT = """
You are a data profiling agent. You receive pre-computed statistics for a
dataset and produce a structured DataProfile object.

Your most important job is to infer a semantic type for each column using
the column name, dtype, and sample values. Use one of:

  id          - unique identifier (uuid, integer key, record id)
  email       - email address
  phone       - phone number
  age         - age in years
  date        - any date or datetime value
  currency    - monetary amount (may contain symbols like $, £)
  categorical - low-cardinality text (status, department, gender, country)
  numeric     - continuous number with no special meaning (score, count)
  boolean     - true/false or yes/no or 0/1
  text        - free-form text (name, description, notes)
  unknown     - cannot determine

In the summary field, write 2-3 sentences in plain English describing
overall data quality. What looks good, what looks problematic, and
what should be investigated.

Be precise. Only use the statistics provided. Do not guess or invent values.
"""

profiler_agent = Agent(output_type=DataProfile, system_prompt=SYSTEM_PROMPT)


def extract_dataset_name(csv_path: str) -> str:
    return csv_path.split("/")[-1].replace(".csv", "")


def build_profiler_prompt(csv_path: str, df: pd.DataFrame) -> str:
    dataset_stats = get_dataset_stats(df)
    column_stats = [{"column": col, **get_column_stats(df, col)} for col in df.columns]
    dataset_name = extract_dataset_name(csv_path)

    return f"""
    Profile this dataset and return a complete DataProfile.

    Dataset name: {dataset_name}
    Row count: {dataset_stats["row_count"]}
    Column count: {dataset_stats["column_count"]}
    Duplicate rows: {dataset_stats["duplicate_row_count"]}
    Total nulls: {dataset_stats["total_null_count"]}

    Per-column statistics:
    {column_stats}

    For each column, infer the most appropriate semantic type based on
    the column name and sample values.
    """


def build_missingness_report(df: pd.DataFrame) -> MissingnessReport:
    miss_raw = analyze_missingness(df)
    columns_raw = miss_raw["columns"]
    assert isinstance(columns_raw, list)

    return MissingnessReport(
        dataset_mcar_pvalue=miss_raw["dataset_mcar_pvalue"],  # type: ignore[arg-type]
        dataset_mcar_conclusion=miss_raw["dataset_mcar_conclusion"],  # type: ignore[arg-type]
        columns_analyzed=[ColumnMissingness(**c) for c in columns_raw],
        summary=build_missingness_summary(miss_raw),
    )


def run_profiler(csv_path: str) -> DataProfile:
    df = load_dataframe(csv_path)
    prompt = build_profiler_prompt(csv_path, df)

    result = profiler_agent.run_sync(
        prompt, model=AnthropicModel("claude-haiku-4-5-20251001")
    )
    profile = result.output
    profile.missingness = build_missingness_report(df)

    return profile

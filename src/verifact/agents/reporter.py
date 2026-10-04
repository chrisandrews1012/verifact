import os

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from verifact.models import PipelineContext

SYSTEM_PROMPT = """
You are a data quality reporter. You receive a complete pipeline context
and produce a professional markdown report.

Structure the report exactly as follows:

# Data Quality Report: {dataset_name}

## Executive Summary
3-4 sentences in plain English. No jargon. Suitable for a non-technical
stakeholder. Summarise what was found and what was done. Italicize the
dataset name when you reference it, e.g. *round32_data*.

## Dataset Overview
A markdown table with: row count, column count, duplicate rows,
total nulls, columns profiled.
For total nulls: use only the raw null count from the profile as it
existed before any repairs were applied.

## Column Profile
A markdown table with one row per column showing: name, inferred type,
null %, unique %, and any notable observations.

## Missingness Analysis
State the dataset-level Little's MCAR test result in one sentence.
Then produce a markdown table with one row per column that had missing values:
column | mechanism | confidence | null % | evidence
End with one sentence explaining what any MNAR finding means for
downstream users of this dataset.

## Validation Findings
A markdown table of failures only (rules that did not pass).
Columns: column, severity, description, suggested fix.
Severity values: critical, consideration, info. Plain text, no emoji.
If there are no failures, write "No issues found."

## Repairs Applied
A markdown table of: column, issue, action taken, rows affected,
before example, after example.

## Unresolved Issues
Bullet list of issues that could not be automatically repaired
and require manual review. If none, write "None. All issues resolved."

## Before You Use This Data
A short paragraph reminding the reader that automated repairs involve
judgment calls. Imputed values are statistical estimates, not ground truth.
For any high-stakes use (financial analysis, medical data, legal records,
model training) the repairs applied should be reviewed manually before
the cleaned dataset is treated as reliable. Point the reader to the
Repairs Applied section for a full list of what was changed.

## Before vs After
A comparison table showing key metrics before and after repair:
row count, null count, duplicate count.
For null count before: use only the raw null count from the profile, not
including any nulls created during repair.
If any nulls were created during repair (e.g. from string-to-numeric
coercion), add a plain-English note directly below the table explaining
how many additional nulls appeared during processing and why, e.g.
"2 additional nulls appeared during repair when column X64PD was converted
from text to numeric. Entries that could not be parsed as numbers were set
to null and then imputed."

## Recommendations
3-5 bullet points for the data owner on how to prevent these issues
at the source.

Be concise. Use markdown tables throughout. Do not add any preamble
or postamble outside the report structure above.
Do not use dashes in sentences. Use colons or periods instead.
"""

reporter_agent = Agent(output_type=str, system_prompt=SYSTEM_PROMPT)


def build_reporter_prompt(context: PipelineContext) -> str:
    return f"""
    Generate a complete data quality report from this pipeline context:

    {context.model_dump_json(indent=2)}
    """


def run_reporter(
    context: PipelineContext,
    output_path: str = "docs/data_quality_report.md",
) -> str:
    prompt = build_reporter_prompt(context)
    result = reporter_agent.run_sync(
        prompt, model=AnthropicModel("claude-haiku-4-5-20251001")
    )
    report_md = result.output

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_md)

    return report_md

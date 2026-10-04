import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from verifact.agents.profiler import run_profiler
from verifact.agents.repairer import run_repairer
from verifact.agents.reporter import run_reporter
from verifact.agents.validator import run_validator
from verifact.invariants import (
    InvariantViolation,
    assert_invariants,
    check_profile_invariants,
    check_repair_invariants,
    check_validation_invariants,
)
from verifact.models import PipelineContext
from verifact.tools import load_dataframe

load_dotenv()
console = Console()
logger = logging.getLogger("verifact")

if not logger.handlers:
    Path("logs").mkdir(exist_ok=True)
    _handler = logging.FileHandler("logs/pipeline.log")
    _handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

DEFAULT_INPUT_PATH = "data/raw/hr_messy.csv"
DEFAULT_OUTPUT_PATH = "data/processed/cleaned_data.csv"
DEFAULT_REPORT_PATH = "docs/data_quality_report.md"

T = TypeVar("T")


def run_pipeline(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineContext:
    def emit(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            console.print(msg)

    def run_with_progress(fn: Callable[[], T], task_label: str) -> T:
        if progress_callback:
            return fn()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as p:
            task = p.add_task(task_label, total=None)
            result = fn()
            p.remove_task(task)
            return result

    if not progress_callback:
        console.print(
            Panel.fit(
                "[bold cyan]Multi-Agent Data Quality Pipeline[/bold cyan]\n"
                f"Input:  {input_path}\n"
                f"Output: {output_path}\n"
                f"Report: {report_path}",
                border_style="cyan",
            )
        )

    input_df = load_dataframe(input_path)
    pipeline_start = time.monotonic()
    logger.info("Pipeline started: input=%s", input_path)

    # Agent 1: Profile
    t = time.monotonic()
    try:
        profile = run_with_progress(
            lambda: run_profiler(input_path),
            "[cyan]Agent 1/4: Profiling dataset...",
        )
        assert_invariants(check_profile_invariants(profile, input_df), "Profiler")
    except InvariantViolation:
        logger.error("Profiler invariant violation", exc_info=True)
        raise
    logger.info(
        "Profiler: rows=%d cols=%d duplicates=%d (%.1fs)",
        profile.row_count,
        profile.column_count,
        profile.duplicate_row_count,
        time.monotonic() - t,
    )
    emit(
        f"Profiler complete: {profile.row_count} rows, "
        f"{profile.column_count} columns, {profile.duplicate_row_count} duplicates"
    )

    # Agent 2: Validate
    t = time.monotonic()
    try:
        validation = run_with_progress(
            lambda: run_validator(profile),
            "[cyan]Agent 2/4: Validating dataset...",
        )
        assert_invariants(check_validation_invariants(validation, profile), "Validator")
    except InvariantViolation:
        logger.error("Validator invariant violation", exc_info=True)
        raise
    status = "Critical issues found" if not validation.passed else "No critical issues"
    logger.info(
        "Validator: %s | rules=%d findings=%d (%.1fs)",
        status,
        len(validation.rules_applied),
        validation.failure_count,
        time.monotonic() - t,
    )
    emit(
        f"Validator complete: {status} | "
        f"{len(validation.rules_applied)} rules applied | "
        f"{validation.failure_count} findings"
    )

    # Agent 3: Repair
    t = time.monotonic()
    try:
        repair = run_with_progress(
            lambda: run_repairer(input_path, output_path, profile),
            "[cyan]Agent 3/4: Repairing dataset...",
        )
        output_df = load_dataframe(output_path)
        assert_invariants(
            check_repair_invariants(repair, input_df, output_df), "Repairer"
        )
    except InvariantViolation:
        logger.error("Repairer invariant violation", exc_info=True)
        raise
    logger.info(
        "Repairer: repairs=%d rows_dropped=%d unresolved=%d (%.1fs)",
        repair.total_repairs,
        repair.rows_dropped,
        len(repair.unresolved),
        time.monotonic() - t,
    )
    emit(
        f"Repairer complete: {repair.total_repairs} repairs, "
        f"{repair.rows_dropped} rows dropped, {len(repair.unresolved)} unresolved"
    )

    # Agent 4: Report
    t = time.monotonic()
    context = PipelineContext(
        input_path=input_path,
        output_path=output_path,
        profile=profile,
        validation=validation,
        repair=repair,
    )
    run_with_progress(
        lambda: run_reporter(context, report_path),
        "[cyan]Agent 4/4: Writing report...",
    )
    logger.info("Reporter: report=%s (%.1fs)", report_path, time.monotonic() - t)
    emit(f"Reporter complete: report saved to {report_path}")

    logger.info("Pipeline complete: total=%.1fs", time.monotonic() - pipeline_start)

    if not progress_callback:
        console.print(
            Panel.fit(
                "[bold green]Pipeline complete.[/bold green]\n"
                f"Cleaned data: {output_path}\n"
                f"Report:       {report_path}",
                border_style="green",
            )
        )

    return context


def resolve_input_path(argv: list[str]) -> str:
    return argv[0] if argv else DEFAULT_INPUT_PATH


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    run_pipeline(input_path=resolve_input_path(args))


if __name__ == "__main__":
    main()

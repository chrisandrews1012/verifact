"""
Eval runner for the data quality pipeline.

Runs the full pipeline against a known dataset and scores its behavior against
a spec that defines expected column types, required repairs, forbidden behaviors,
and known ground-truth facts.

Usage:
    uv run python -m evals.runner                    # runs all specs in evals/expected/
    uv run python -m evals.runner hr_expected.json    # runs one spec by filename
"""

import json
import sys
from pathlib import Path

from verifact.invariants import InvariantViolation
from verifact.models import DataProfile, RepairReport
from verifact.pipeline import run_pipeline

REPO_ROOT = Path(__file__).parent.parent

EVAL_MODEL = "claude-haiku-4-5-20251001"


# Scoring helpers


def _score_column_types(profile: DataProfile, expected_types: dict[str, str]) -> dict:
    actual = {cp.name: cp.inferred_type for cp in profile.columns}
    correct = sum(
        1 for col, expected in expected_types.items() if actual.get(col) == expected
    )
    total = len(expected_types)
    wrong = {
        col: {"expected": expected, "got": actual.get(col, "missing")}
        for col, expected in expected_types.items()
        if actual.get(col) != expected
    }
    return {
        "correct": correct,
        "total": total,
        "score": correct / total if total else 1.0,
        "wrong": wrong,
    }


def _score_known_facts(profile: DataProfile, known_facts: dict) -> dict:
    violations = []
    if (
        known_facts.get("row_count") is not None
        and profile.row_count != known_facts["row_count"]
    ):
        violations.append(
            f"row_count: claimed {profile.row_count}, "
            f"expected {known_facts['row_count']}"
        )
    if (
        known_facts.get("duplicate_row_count") is not None
        and profile.duplicate_row_count != known_facts["duplicate_row_count"]
    ):
        violations.append(
            f"duplicate_row_count: claimed {profile.duplicate_row_count}, "
            f"expected {known_facts['duplicate_row_count']}"
        )
    if (
        known_facts.get("total_null_count") is not None
        and profile.total_null_count != known_facts["total_null_count"]
    ):
        violations.append(
            f"total_null_count: claimed {profile.total_null_count}, "
            f"expected {known_facts['total_null_count']}"
        )
    return {"violations": violations, "passed": len(violations) == 0}


def _score_repair_coverage(repair: RepairReport, must_repair: dict) -> dict:
    missing = []
    repaired_cols = {a.column for a in repair.actions}

    if must_repair.get("duplicate_rows"):
        dup_actions = [
            a
            for a in repair.actions
            if a.column == "(all columns)" and "duplicate" in a.issue.lower()
        ]
        if not dup_actions:
            missing.append("duplicate rows were not removed")

    for col, issue_types in must_repair.get("columns", {}).items():
        if col not in repaired_cols:
            missing.append(f"'{col}' was not repaired (expected: {issue_types})")

    found = (
        len(must_repair.get("columns", {}))
        + (1 if must_repair.get("duplicate_rows") else 0)
        - len(missing)
    )
    total = len(must_repair.get("columns", {})) + (
        1 if must_repair.get("duplicate_rows") else 0
    )
    return {
        "found": found,
        "total": total,
        "score": found / total if total else 1.0,
        "missing": missing,
    }


def _score_unresolved_coverage(
    repair: RepairReport, must_flag_unresolved: list[str]
) -> dict:
    missing = []
    unresolved_text = " ".join(repair.unresolved).lower()
    for col in must_flag_unresolved:
        if col.lower() not in unresolved_text:
            missing.append(f"'{col}' not mentioned in unresolved issues")
    found = len(must_flag_unresolved) - len(missing)
    total = len(must_flag_unresolved)
    return {
        "found": found,
        "total": total,
        "score": found / total if total else 1.0,
        "missing": missing,
    }


def _score_no_false_positives(repair: RepairReport) -> dict:
    """Check that the repairer applied zero actions on a clean dataset."""
    violations = []
    col_actions = [a for a in repair.actions if a.column != "(all columns)"]
    if repair.rows_dropped > 0:
        violations.append(
            f"rows_dropped={repair.rows_dropped} on a dataset with no duplicates"
        )
    if col_actions:
        violations.append(
            f"{len(col_actions)} column repair(s) applied to clean data: "
            + ", ".join(f"'{a.column}' ({a.action_taken})" for a in col_actions)
        )
    if repair.unresolved:
        violations.append(
            f"{len(repair.unresolved)} issue(s) escalated to unresolved on clean data"
        )
    return {"violations": violations, "passed": len(violations) == 0}


def _score_mnar_safety(profile: DataProfile, repair: RepairReport) -> dict:
    if not profile.missingness:
        return {"mnar_columns": [], "violations": [], "passed": True}

    mnar_cols = {
        cm.column
        for cm in profile.missingness.columns_analyzed
        if cm.mechanism == "MNAR"
    }
    imputation_actions = {"imputed_mean", "imputed_mode", "imputed_median"}
    violations = [
        f"MNAR column '{a.column}' was imputed with {a.action_taken}"
        for a in repair.actions
        if a.column in mnar_cols and a.action_taken in imputation_actions
    ]
    return {
        "mnar_columns": list(mnar_cols),
        "violations": violations,
        "passed": len(violations) == 0,
    }


# Report rendering


def _pct(score: float) -> str:
    return f"{score * 100:.0f}%"


def _write_scorecard_md(all_results: dict, results_dir: Path) -> None:
    from datetime import datetime

    lines = ["# Eval Scorecard", ""]
    lines += [f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  "]
    lines += [f"**Model:** {EVAL_MODEL}", ""]

    lines += ["## Summary", ""]
    lines += [
        "| Dataset | Factual Accuracy | Semantic Types | Repair Coverage | "
        "Unresolved | MNAR Safety | No False Positives | Overall |"
    ]
    lines += ["|---|---|---|---|---|---|---|---|"]

    for spec_name, r in all_results.items():
        dataset = spec_name.replace("_expected.json", "")
        facts = "PASS" if r["known_facts"]["passed"] else "FAIL"
        ct = r["column_types"]
        types = f"{ct['correct']}/{ct['total']} ({_pct(ct['score'])})"
        repair = f"{r['repair_coverage']['found']}/{r['repair_coverage']['total']}"
        unres = (
            f"{r['unresolved_coverage']['found']}/{r['unresolved_coverage']['total']}"
            if r["unresolved_coverage"]["total"] > 0
            else "—"
        )
        mnar = (
            ("PASS" if r["mnar_safety"]["passed"] else "FAIL")
            if r["mnar_safety"]["mnar_columns"]
            else "N/A"
        )
        no_fp = (
            ("PASS" if r["no_false_positives"]["passed"] else "FAIL")
            if r.get("no_false_positives") is not None
            else "—"
        )
        overall = (
            "PASS"
            if (
                r["known_facts"]["passed"]
                and r["column_types"]["score"] == 1.0
                and r["repair_coverage"]["score"] == 1.0
                and r["unresolved_coverage"]["score"] == 1.0
                and r["mnar_safety"]["passed"]
                and (
                    r.get("no_false_positives") is None
                    or r["no_false_positives"]["passed"]
                )
            )
            else "FAIL"
        )
        lines.append(
            f"| {dataset} | {facts} | {types} | {repair} | {unres} | "
            f"{mnar} | {no_fp} | {overall} |"
        )

    failures = {
        name: r
        for name, r in all_results.items()
        if not r["known_facts"]["passed"]
        or r["column_types"]["score"] < 1.0
        or r["repair_coverage"]["score"] < 1.0
        or r["unresolved_coverage"]["score"] < 1.0
        or not r["mnar_safety"]["passed"]
        or (
            r.get("no_false_positives") is not None
            and not r["no_false_positives"]["passed"]
        )
    }

    if failures:
        lines += ["", "## Failures", ""]
        for spec_name, r in failures.items():
            lines += [f"### {spec_name}", ""]
            if not r["known_facts"]["passed"]:
                lines += ["**Factual accuracy**"]
                for v in r["known_facts"]["violations"]:
                    lines += [f"- {v}"]
                lines.append("")
            if r["column_types"]["score"] < 1.0:
                ct = r["column_types"]
                lines += [f"**Semantic type accuracy: {ct['correct']}/{ct['total']}**"]
                for col, diff in r["column_types"]["wrong"].items():
                    lines += [
                        f"- `{col}`: expected `{diff['expected']}`, got `{diff['got']}`"
                    ]
                lines.append("")
            if r["repair_coverage"]["score"] < 1.0:
                lines += ["**Repair coverage**"]
                for m in r["repair_coverage"]["missing"]:
                    lines += [f"- {m}"]
                lines.append("")
            if r["unresolved_coverage"]["score"] < 1.0:
                lines += ["**Unresolved coverage**"]
                for m in r["unresolved_coverage"]["missing"]:
                    lines += [f"- {m}"]
                lines.append("")
            if not r["mnar_safety"]["passed"]:
                lines += ["**MNAR safety**"]
                for v in r["mnar_safety"]["violations"]:
                    lines += [f"- {v}"]
                lines.append("")
            no_fp = r.get("no_false_positives")
            if no_fp is not None and not no_fp["passed"]:
                lines += ["**No false positives**"]
                for v in no_fp["violations"]:
                    lines += [f"- {v}"]
                lines.append("")
    else:
        lines += ["", "All checks passed."]

    results_dir.mkdir(exist_ok=True)
    out = results_dir / "scorecard.md"
    out.write_text("\n".join(lines))
    print(f"\nScorecard written to {out}")


def _print_scorecard(spec_name: str, results: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Eval: {spec_name}")
    print(f"{'=' * 60}")

    facts = results["known_facts"]
    status = "PASS" if facts["passed"] else "FAIL"
    print(f"\n[{status}] Factual accuracy")
    if facts["violations"]:
        for v in facts["violations"]:
            print(f"       - {v}")

    types = results["column_types"]
    print(
        f"\n[{'PASS' if types['score'] == 1.0 else 'FAIL'}] Semantic type accuracy: "
        f"{types['correct']}/{types['total']} ({_pct(types['score'])})"
    )
    if types["wrong"]:
        for col, diff in types["wrong"].items():
            print(f"       - '{col}': expected {diff['expected']}, got {diff['got']}")

    repair = results["repair_coverage"]
    print(
        f"\n[{'PASS' if repair['score'] == 1.0 else 'FAIL'}] Repair coverage: "
        f"{repair['found']}/{repair['total']} ({_pct(repair['score'])})"
    )
    if repair["missing"]:
        for m in repair["missing"]:
            print(f"       - {m}")

    unresolved = results["unresolved_coverage"]
    if unresolved["total"] > 0:
        unresolved_label = "PASS" if unresolved["score"] == 1.0 else "FAIL"
        print(
            f"\n[{unresolved_label}] Unresolved coverage: "
            f"{unresolved['found']}/{unresolved['total']} ({_pct(unresolved['score'])})"
        )
        if unresolved["missing"]:
            for m in unresolved["missing"]:
                print(f"       - {m}")

    mnar = results["mnar_safety"]
    mnar_label = "PASS" if mnar["passed"] else "FAIL"
    if mnar["mnar_columns"]:
        print(
            f"\n[{mnar_label}] MNAR safety (columns: {', '.join(mnar['mnar_columns'])})"
        )
    else:
        print("\n[N/A] MNAR safety: no MNAR columns detected in this run")
    if mnar["violations"]:
        for v in mnar["violations"]:
            print(f"       - {v}")

    no_fp = results.get("no_false_positives")
    if no_fp is not None:
        label = "PASS" if no_fp["passed"] else "FAIL"
        print(f"\n[{label}] No false positives (zero repairs on clean data)")
        for v in no_fp["violations"]:
            print(f"       - {v}")

    all_passed = (
        facts["passed"]
        and types["score"] == 1.0
        and repair["score"] == 1.0
        and unresolved["score"] == 1.0
        and mnar["passed"]
        and (no_fp is None or no_fp["passed"])
    )
    print(f"\n{'  Overall: PASS' if all_passed else '  Overall: FAIL'}")
    print(f"{'=' * 60}\n")


def run_eval(spec_path: Path) -> dict:
    with open(spec_path) as f:
        spec = json.load(f)

    dataset_path = str(REPO_ROOT / spec["dataset_path"])
    stem = Path(spec["dataset_path"]).stem
    results_dir = REPO_ROOT / "evals" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = str(results_dir / f"{stem}_cleaned.csv")
    report_path = str(results_dir / f"{stem}_report.md")

    print(f"\nRunning pipeline on: {spec['dataset_path']}")
    try:
        context = run_pipeline(
            input_path=dataset_path,
            output_path=output_path,
            report_path=report_path,
        )
    except InvariantViolation as e:
        results = {
            "known_facts": {"passed": False, "violations": [str(e)]},
            "column_types": {"correct": 0, "total": 0, "score": 0.0, "wrong": {}},
            "repair_coverage": {"found": 0, "total": 0, "score": 1.0, "missing": []},
            "unresolved_coverage": {
                "found": 0,
                "total": 0,
                "score": 1.0,
                "missing": [],
            },
            "mnar_safety": {"mnar_columns": [], "violations": [], "passed": True},
            "no_false_positives": None,
        }
        _print_scorecard(spec_path.name, results)
        return results

    results = {
        "known_facts": _score_known_facts(context.profile, spec.get("known_facts", {})),
        "column_types": _score_column_types(
            context.profile, spec.get("expected_column_types", {})
        ),
        "repair_coverage": _score_repair_coverage(
            context.repair, spec.get("must_repair", {})
        ),
        "unresolved_coverage": _score_unresolved_coverage(
            context.repair, spec.get("must_flag_unresolved", [])
        ),
        "mnar_safety": (
            _score_mnar_safety(context.profile, context.repair)
            if spec.get("must_not_impute_mnar")
            else {"mnar_columns": [], "violations": [], "passed": True}
        ),
        "no_false_positives": (
            _score_no_false_positives(context.repair)
            if spec.get("expect_zero_repairs")
            else None
        ),
    }

    _print_scorecard(spec_path.name, results)
    return results


if __name__ == "__main__":
    specs_dir = Path(__file__).parent / "expected"

    if len(sys.argv) > 1:
        targets = [specs_dir / sys.argv[1]]
    else:
        targets = sorted(specs_dir.glob("*.json"))

    if not targets:
        print("No eval specs found in evals/expected/")
        sys.exit(1)

    all_results = {}
    for spec_path in targets:
        all_results[spec_path.name] = run_eval(spec_path)

    _write_scorecard_md(all_results, REPO_ROOT / "evals" / "results")

    any_failed = any(
        not r["known_facts"]["passed"]
        or r["column_types"]["score"] < 1.0
        or r["repair_coverage"]["score"] < 1.0
        or r["unresolved_coverage"]["score"] < 1.0
        or not r["mnar_safety"]["passed"]
        or (
            r["no_false_positives"] is not None
            and not r["no_false_positives"]["passed"]
        )
        for r in all_results.values()
    )
    sys.exit(1 if any_failed else 0)

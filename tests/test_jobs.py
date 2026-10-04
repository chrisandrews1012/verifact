from verifact.jobs import (
    append_event,
    complete_job,
    create_job,
    fail_job,
    get_job,
    set_running,
)


def test_create_job_starts_pending_with_no_events() -> None:
    """A freshly created job is pending with an empty event log."""
    job_id = create_job()
    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "pending"
    assert job["events"] == []


def test_append_event_adds_to_the_job_events_list() -> None:
    """Appended events accumulate in order."""
    job_id = create_job()
    append_event(job_id, "Profiler complete")
    append_event(job_id, "Validator complete")
    job = get_job(job_id)
    assert job is not None
    assert job["events"] == ["Profiler complete", "Validator complete"]


def test_set_running_updates_status() -> None:
    """Marking a job running updates its status field."""
    job_id = create_job()
    set_running(job_id)
    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "running"


def test_complete_job_sets_status_and_paths() -> None:
    """Completing a job records its status and output locations."""
    job_id = create_job()
    complete_job(job_id, "out.csv", "report.md", "/tmp/xyz")
    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "complete"
    assert job["output_path"] == "out.csv"
    assert job["report_path"] == "report.md"
    assert job["tmp_dir"] == "/tmp/xyz"


def test_fail_job_sets_status_and_error_message() -> None:
    """Failing a job records its status and error message."""
    job_id = create_job()
    fail_job(job_id, "something went wrong")
    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "error"
    assert job["error_message"] == "something went wrong"


def test_get_job_returns_none_for_unknown_id() -> None:
    """Looking up a job id that was never created returns None."""
    assert get_job("does-not-exist") is None

import time

import pytest
from fastapi.testclient import TestClient

from verifact.jobs import create_job
from verifact.server import app

client = TestClient(app)


def test_index_serves_the_frontend() -> None:
    """The root path serves the static HTML upload page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_run_rejects_non_csv_upload() -> None:
    """Uploading a non-CSV file is rejected with a 400."""
    response = client.post(
        "/run", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_stream_unknown_job_returns_404() -> None:
    """Streaming progress for a job id that doesn't exist returns 404."""
    response = client.get("/stream/does-not-exist")
    assert response.status_code == 404


def test_report_unknown_job_returns_404() -> None:
    """Fetching the report for an unknown job id returns 404."""
    response = client.get("/report/does-not-exist")
    assert response.status_code == 404


def test_report_not_yet_complete_returns_404() -> None:
    """Fetching the report for a still-pending job returns 404."""
    job_id = create_job()
    response = client.get(f"/report/{job_id}")
    assert response.status_code == 404


def test_download_unknown_job_returns_404() -> None:
    """Downloading the output for an unknown job id returns 404."""
    response = client.get("/download/does-not-exist")
    assert response.status_code == 404


def test_download_not_yet_complete_returns_404() -> None:
    """Downloading the output for a still-pending job returns 404."""
    job_id = create_job()
    response = client.get(f"/download/{job_id}")
    assert response.status_code == 404


@pytest.mark.llm
def test_run_end_to_end_completes_and_serves_report_and_download() -> None:
    """Uploading a real CSV runs the pipeline in the background, and the
    report and cleaned CSV both become available once it finishes."""
    csv_bytes = b"employee_id,age\ne1,25\ne2,30\ne2,30\n"
    response = client.post(
        "/run", files={"file": ("people.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    from verifact.jobs import get_job

    for _ in range(120):
        job = get_job(job_id)
        assert job is not None
        if job["status"] in ("complete", "error"):
            break
        time.sleep(1)

    assert job["status"] == "complete"

    report_response = client.get(f"/report/{job_id}")
    assert report_response.status_code == 200

    download_response = client.get(f"/download/{job_id}")
    assert download_response.status_code == 200

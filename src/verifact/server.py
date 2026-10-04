import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import markdown as md
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from verifact.jobs import (
    append_event,
    complete_job,
    create_job,
    fail_job,
    get_job,
    set_running,
)
from verifact.pipeline import run_pipeline

app = FastAPI()
_executor = ThreadPoolExecutor(max_workers=4)

STATIC_DIR = Path(__file__).parent.parent.parent / "static"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/run")
async def run(file: UploadFile) -> dict[str, str]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, file.filename)
    output_path = os.path.join(tmp_dir, "cleaned.csv")
    report_path = os.path.join(tmp_dir, "report.md")

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    job_id = create_job()

    def run_job() -> None:
        set_running(job_id)
        try:
            run_pipeline(
                input_path=input_path,
                output_path=output_path,
                report_path=report_path,
                progress_callback=lambda msg: append_event(job_id, msg),
            )
            complete_job(job_id, output_path, report_path, tmp_dir)
        except Exception as e:
            fail_job(job_id, str(e))

    asyncio.get_running_loop().run_in_executor(_executor, run_job)

    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        sent = 0
        while True:
            job = get_job(job_id)
            assert job is not None
            events = job["events"]

            while sent < len(events):
                yield f"data: {events[sent]}\n\n"
                sent += 1

            if job["status"] in ("complete", "error"):
                if job["status"] == "error":
                    yield f"data: error: {job['error_message']}\n\n"
                yield "data: done\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/report/{job_id}")
async def report(job_id: str) -> HTMLResponse:
    job = get_job(job_id)
    if not job or job["status"] != "complete" or not job["report_path"]:
        raise HTTPException(status_code=404, detail="Report not ready.")

    with open(job["report_path"]) as f:
        content = f.read()

    return HTMLResponse(md.markdown(content, extensions=["tables"]))


@app.get("/download/{job_id}")
async def download(job_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    job = get_job(job_id)
    if not job or job["status"] != "complete" or not job["output_path"]:
        raise HTTPException(status_code=404, detail="File not ready.")

    if job["tmp_dir"]:
        background_tasks.add_task(shutil.rmtree, job["tmp_dir"], True)

    return FileResponse(
        job["output_path"],
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned.csv"},
    )

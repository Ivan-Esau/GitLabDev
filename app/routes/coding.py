from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
import asyncio

from ..coding_agent import CodingJob, run_coding_job

router = APIRouter()

class StartJobRequest(BaseModel):
    session_id: str = "default"
    source_project: str
    fork_namespace: str | None = None
    model: str | None = None
    max_issue: int | None = None

@router.post("/coding/start")
async def start_job(req: StartJobRequest, request: Request):
    app = request.app
    job = CodingJob(
        sid=req.session_id,
        src_project=req.source_project,
        fork_namespace=req.fork_namespace,
        model=req.model,
        max_issue=req.max_issue,
    )
    # init status
    app.state.coding_status[req.session_id] = {"state": "starting", "msg": "initializing"}
    asyncio.create_task(run_coding_job(app, job))
    return {"status": "started"}

@router.get("/coding/status")
async def coding_status(request: Request, sid: str = Query("default")):
    app = request.app
    return app.state.coding_status.get(sid, {"state": "idle"})

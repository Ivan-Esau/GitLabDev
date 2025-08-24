import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from agent import run_agent
from app_restructured.infrastructure.mcp.mcp_client import call_mcp_tool, update_context
from app_restructured.core.config import DEFAULT_MODEL

def _set_status(app: FastAPI, sid: str, **kwargs):
    st = app.state.coding_status.setdefault(sid, {})
    st.update(kwargs)

class CodingJob:
    def __init__(self, sid: str, src_project: str, fork_namespace: Optional[str] = None,
                 model: Optional[str] = None, max_issue: Optional[int] = None):
        self.sid = sid
        self.src_project = src_project
        self.fork_namespace = fork_namespace
        self.model = model or DEFAULT_MODEL
        self.max_issue = max_issue

async def run_coding_job(app: FastAPI, job: CodingJob):
    sid = job.sid
    try:
        _set_status(app, sid, state="resolving_project")
        ranked = await call_mcp_tool(app, "resolve_project", {"project_ref": job.src_project, "limit": 5})
        if not ranked:
            _set_status(app, sid, state="error", msg="source project not found")
            return
        src_id = ranked[0]["id"]

        _set_status(app, sid, state="forking", msg="creating fork")
        fork = await call_mcp_tool(app, "fork_project", {"project_ref": src_id, "namespace": job.fork_namespace})
        fork_id = fork["id"] if isinstance(fork, dict) else fork

        _set_status(app, sid, state="waiting_fork")
        await call_mcp_tool(app, "wait_for_fork", {"fork_id": fork_id, "poll_interval": 2, "timeout": 300})

        _set_status(app, sid, state="list_issues")
        issues = await call_mcp_tool(app, "list_issues", {"project_ref": src_id, "state": "opened", "limit": 200})
        if job.max_issue:
            issues = issues[:job.max_issue]

        _set_status(app, sid, state="clone_issues", total=len(issues))
        id_map = {}
        for idx, iss in enumerate(issues, start=1):
            new_issue = await call_mcp_tool(app, "create_issue", {
                "project_ref": fork_id,
                "title": iss.get("title"),
                "description": f"(Forked from {src_id}#{iss.get('iid')})\n\n" + (iss.get("description") or "")
            })
            id_map[iss["iid"]] = new_issue.get("iid")
            _set_status(app, sid, state="clone_issues", done=idx, total=len(issues))

        # Per issue implement
        for i_idx, iss in enumerate(issues, start=1):
            branch = f"issue-{iss['iid']}"
            _set_status(app, sid, state="working_issue", issue=iss["iid"], branch=branch, progress=f"{i_idx}/{len(issues)}")

            await call_mcp_tool(app, "create_branch", {"project_ref": fork_id, "branch": branch, "ref": None})

            ctx_text = (f"Arbeite an Issue #{iss['iid']}: {iss['title']}\n"
                        f"Beschreibung:\n{iss.get('description','')}\n"
                        f"Branch: {branch}")
            await run_agent(app, f"[PLAN]\n{ctx_text}\nPlane Änderungen und nenne betroffene Dateien.",
                            max_steps=6, model=job.model, sid=sid)

            success = False
            for attempt in range(3):
                _set_status(app, sid, state="implement", attempt=attempt+1, issue=iss['iid'])
                await run_agent(app, f"[IMPLEMENT]\nArbeite die geplanten Änderungen ab und commite auf Branch {branch}.",
                                max_steps=8, model=job.model, sid=sid)

                pip = await call_mcp_tool(app, "trigger_pipeline", {"project_ref": fork_id, "ref": branch})
                pid = pip.get("id") if isinstance(pip, dict) else pip

                _set_status(app, sid, state="pipeline_wait", pipeline=pid, issue=iss['iid'])
                pstat = await call_mcp_tool(app, "wait_for_pipeline", {
                    "project_ref": fork_id, "pipeline_id": pid, "timeout_sec": 1200
                })
                if pstat.get("status") == "success":
                    success = True
                    break
                else:
                    # Debug run
                    await run_agent(app,
                                    f"[DEBUG]\nTests fehlgeschlagen. Analysiere Logs und fixen. Pipeline-Status: {pstat}",
                                    max_steps=10, model=job.model, sid=sid)

            _set_status(app, sid, state="create_mr", issue=iss['iid'])
            await call_mcp_tool(app, "create_merge_request", {
                "project_ref": fork_id,
                "source_branch": branch,
                "target_branch": "main",
                "title": f"Fix Issue #{iss['iid']}: {iss['title']}",
                "description": "Automated by Coding-Agent"
            })
            _set_status(app, sid, state="issue_done", issue=iss['iid'], success=success)

        _set_status(app, sid, state="done", msg="All issues processed", fork_id=fork_id)
    except Exception as e:
        _set_status(app, sid, state="error", msg=str(e))

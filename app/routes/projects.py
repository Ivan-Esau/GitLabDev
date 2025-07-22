import json
from fastapi import APIRouter, HTTPException, Request
from ..schemas import SelectProjectRequest
from ..mcp_client import call_mcp_tool, update_context

router = APIRouter()

@router.get("/projects")
async def get_projects(request: Request):
    app = request.app
    if app.state.cached_projects is None:
        result = await call_mcp_tool(app, "list_projects", {"limit": 200})
        app.state.cached_projects = result
    return {"projects": app.state.cached_projects}

@router.post("/select_project")
async def select_project(req: SelectProjectRequest, request: Request):
    app = request.app
    sid = req.session_id
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    ranked = await call_mcp_tool(app, "resolve_project", {"project_ref": req.project_ref, "limit": 5})
    if not ranked:
        raise HTTPException(404, "Projekt nicht gefunden")

    proj_id = ranked[0]["id"]
    desc = await call_mcp_tool(app, "describe_project", {"project_ref": proj_id, "tree_limit": 200})
    if isinstance(desc, str):
        try:
            desc = json.loads(desc)
        except Exception:
            desc = {"raw": desc}

    update_context(app, sid, "describe_project", desc)
    ctx["project_index_ready"] = True

    conv.append({"role": "user", "parts": [{"text": f"Aktuelles Projekt gesetzt: {desc.get('name')} (ID {proj_id})."}]})

    return {"project": desc}

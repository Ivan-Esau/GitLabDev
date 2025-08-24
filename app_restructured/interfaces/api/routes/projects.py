import json
from typing import Union

from fastapi import APIRouter, HTTPException, Request, Body
from app_restructured.interfaces.api.schemas.schemas import SelectProjectRequest
from app_restructured.infrastructure.mcp.mcp_client import call_mcp_tool, update_context

router = APIRouter()

@router.get("/projects")
async def get_projects(request: Request):
    app = request.app
    if getattr(app.state, "cached_projects", None) is None:
        result = await call_mcp_tool(app, "list_projects", {"limit": 200})
        app.state.cached_projects = result
    return {"projects": app.state.cached_projects}

@router.post("/select_project")
async def select_project(
    request: Request,
    req: SelectProjectRequest = Body(...)
):
    """
    Erwartet JSON: { "project_ref": <id|name>, "session_id": "..." }
    """
    app = request.app
    sid = req.session_id
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    ranked = await call_mcp_tool(app, "resolve_project", {"project_ref": str(req.project_ref), "limit": 5})
    if not ranked:
        raise HTTPException(404, "Projekt nicht gefunden")

    proj_id = ranked[0]["id"]
    desc = await call_mcp_tool(app, "describe_project", {"project_ref": proj_id, "tree_limit": 200})
    if isinstance(desc, str):
        try:
            desc = json.loads(desc)
        except Exception:
            desc = {"raw": desc}

    desc.setdefault("id", proj_id)
    desc.setdefault("name", desc.get("name") or desc.get("path") or ranked[0]["name"])
    desc.setdefault("path", desc.get("path") or ranked[0]["path"])
    desc.setdefault("web_url", desc.get("web_url") or ranked[0]["web_url"])
    desc.setdefault("default_branch", desc.get("default_branch") or "main")
    desc.setdefault("root_tree", desc.get("root_tree", []))

    update_context(app, sid, "describe_project", desc)
    ctx["project_index_ready"] = True

    conv.append({"role": "user", "parts": [{"text": f"Aktuelles Projekt gesetzt: {desc.get('name')} (ID {proj_id})."}]})

    return {"project": desc}

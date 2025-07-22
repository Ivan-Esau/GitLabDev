import asyncio
from typing import Any, Dict, List
from contextlib import AsyncExitStack

from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from .config import server_params
from .utils import try_json, sanitize_schema, norm, projects_md_table

# ---------- Startup / Shutdown ----------
async def init_mcp(app: FastAPI):
    stack = AsyncExitStack()
    await stack.__aenter__()
    app.state.stack = stack

    read, write = await stack.enter_async_context(stdio_client(server_params))
    session = ClientSession(read, write)
    await stack.enter_async_context(session)

    await asyncio.wait_for(session.initialize(), timeout=30)

    tools_resp = await session.list_tools()
    mcp_tools = [t for t in tools_resp.tools if t.name != "gemini_generate"]

    app.state.session = session
    app.state.lock = asyncio.Lock()
    app.state.mcp_tools = mcp_tools
    app.state.gemini_tool_block = build_gemini_tool_block(mcp_tools)

    app.state.conversations = {}
    app.state.contexts = {}
    app.state.cached_projects = None

async def close_mcp(app: FastAPI):
    await app.state.stack.aclose()

# ---------- Tools ----------
def build_gemini_tool_block(mcp_tools) -> List[Dict[str, Any]]:
    fns = []
    for t in mcp_tools:
        raw = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        schema = sanitize_schema(raw)
        fns.append({
            "name": t.name,
            "description": t.description or "",
            "parameters": schema,
        })
    return [{"functionDeclarations": fns}]

async def call_mcp_tool(app: FastAPI, name: str, args: Dict[str, Any]) -> Any:
    async with app.state.lock:
        res = await app.state.session.call_tool(name, args)

    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return try_json(sc.get("result", sc))

    if getattr(res, "content", None):
        part = res.content[0]
        if hasattr(part, "text"):
            return try_json(part.text)
        if hasattr(part, "data"):
            return try_json(part.data)

    return try_json(str(res))

# ---------- Context Utils ----------
def update_context(app: FastAPI, sid: str, tool_name: str, result: Any) -> None:
    ctx = app.state.contexts.setdefault(sid, {})
    try:
        if tool_name in ("resolve_project", "describe_project"):
            if isinstance(result, list) and result:
                p = result[0]
            elif isinstance(result, dict):
                p = result
            else:
                return
            pid = p.get("id")
            pname = p.get("name") or p.get("path")
            dbranch = p.get("default_branch")
            if pid:
                ctx["current_project_id"] = pid
                ctx["project_ref"] = pid
            if pname:
                ctx["current_project_name"] = pname
            if dbranch:
                ctx["default_branch"] = dbranch

        elif tool_name == "list_projects" and isinstance(result, list):
            aliases = ctx.setdefault("project_aliases", {})
            ctx["projects"] = result
            for proj in result:
                name = proj.get("name", "")
                path = proj.get("path", "")
                pid = proj.get("id")
                for raw in (name, path, name.split("/")[-1], path.split("/")[-1]):
                    if raw:
                        aliases[raw.lower()] = pid
                        aliases[norm(raw)] = pid
    except Exception:
        pass

def autofill_args(app: FastAPI, sid: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    ctx = app.state.contexts.get(sid, {})
    out = dict(args)

    if "project_ref" not in out and name not in ("resolve_project",):
        pr = ctx.get("project_ref") or ctx.get("current_project_id") or ctx.get("current_project_name")
        if pr is not None:
            out["project_ref"] = pr

    if "project_ref" not in out and name not in ("resolve_project",):
        aliases = ctx.get("project_aliases", {})
        if aliases:
            last_user_text = ""
            for msg in reversed(app.state.conversations.get(sid, [])):
                if msg["role"] == "user":
                    last_user_text = msg["parts"][0].get("text", "")
                    break
            lut_norm = norm(last_user_text)
            for alias_norm, pid in aliases.items():
                if norm(alias_norm) and norm(alias_norm) in lut_norm:
                    out["project_ref"] = pid
                    break

    if "ref" in out and not out["ref"]:
        out["ref"] = ctx.get("default_branch", "main")
    if "branch" in out and not out["branch"]:
        out["branch"] = ctx.get("default_branch", "main")

    return out

async def ensure_project_index(app: FastAPI, sid: str, conv: List[Dict[str, Any]]) -> None:
    ctx = app.state.contexts.setdefault(sid, {})
    if ctx.get("project_index_ready"):
        return
    if app.state.cached_projects is None:
        result = await call_mcp_tool(app, "list_projects", {"limit": 200})
        app.state.cached_projects = result
    else:
        result = app.state.cached_projects
    update_context(app, sid, "list_projects", result)
    table = projects_md_table(result)
    conv.append({"role": "user", "parts": [{"text": "Projektindex (verwende diese IDs/Bezeichnungen):\n" + table}]})
    ctx["project_index_ready"] = True

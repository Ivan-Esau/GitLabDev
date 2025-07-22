# backend.py – Gemini + MCP: Projektwahl, Kontext-Pinning, Loop-Bremse

import os
import sys
import asyncio
import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
load_dotenv()
log = logging.getLogger("uvicorn.error")

app = FastAPI()

ROOT = Path(__file__).parent
SERVER_PATH = ROOT / "gitlab_mcp.py"
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(SERVER_PATH)],
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    working_directory=str(ROOT),
)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_TOOL_MODE = os.getenv("GEMINI_TOOL_MODE", "AUTO")  # AUTO | ANY | NONE

SYSTEM_PROMPT = """
Du hast GitLab-Tools. Regeln:

A) Projektbezug
- Ist ein Projekt (ID/Name/Pfad) bereits im Kontext, nutze es direkt.
- Sonst: resolve_project(<name/pfad>) (toleriert Tippfehler). Bei eindeutigem Treffer (score ≥ 0.6) ohne Rückfrage verwenden.
- Für Inhalte (README, Dateien, Default-Branch): describe_project(<id>).

B) Dateien/Ordner
- Kennst du den default_branch nicht? -> describe_project.
- Struktur: list_repo_tree(path?, ref=default_branch, recursive=true)
- Datei: get_file(file_path, ref=default_branch)
- Schreiben: upsert_file(file_path, branch, content, commit_message)

C) Issues / MRs / CI
- Stelle sicher, dass project_ref gesetzt ist (siehe A).
- Verwende passende list_*/create_*/merge_*/trigger_* Tools.

D) Kontextpflege
- Nutze Fakten aus dem Verlauf (project_id, default_branch, aliases).
- Frage nicht erneut nach bereits bekannten Infos.

Antwortstil:
- Kompaktes Markdown (Listen, Tabellen, Codeblöcke).
- Kein Tool nötig? Antworte direkt.
"""

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: Optional[str] = None
    max_steps: int = 8
    reset: bool = False

class ToolCall(BaseModel):
    arguments: Dict[str, Any] = {}

class SelectProjectRequest(BaseModel):
    project_ref: str
    session_id: str = "default"

# --------------------------------------------------------------------------- #
# Startup / Shutdown
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def startup() -> None:
    stack = AsyncExitStack()
    await stack.__aenter__()
    app.state.stack = stack

    read, write = await stack.enter_async_context(stdio_client(server_params))
    session = ClientSession(read, write)
    await stack.enter_async_context(session)

    try:
        await asyncio.wait_for(session.initialize(), timeout=30)
    except asyncio.TimeoutError as e:
        raise RuntimeError("MCP init timed out") from e

    tools_resp = await session.list_tools()
    mcp_tools = [t for t in tools_resp.tools if t.name != "gemini_generate"]

    app.state.session = session
    app.state.lock = asyncio.Lock()
    app.state.mcp_tools = mcp_tools
    app.state.gemini_tool_block = _build_gemini_tool_block(mcp_tools)

    # Session Memory
    app.state.conversations: Dict[str, List[Dict[str, Any]]] = {}
    app.state.contexts: Dict[str, Dict[str, Any]] = {}
    app.state.cached_projects: Optional[List[Dict[str, Any]]] = None  # global Cache

@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.stack.aclose()

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sanitize_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        clean: Dict[str, Any] = {}

        t = schema.get("type")
        if t in {"object", "array", "string", "integer", "number", "boolean"}:
            clean["type"] = t

        if props:
            clean["properties"] = {k: _sanitize_schema(v) for k, v in props.items()}

        if isinstance(schema.get("required"), list):
            req = [r for r in schema["required"] if r in props]
            if req:
                clean["required"] = req

        if "items" in schema:
            clean["items"] = _sanitize_schema(schema["items"])

        if isinstance(schema.get("enum"), list):
            clean["enum"] = schema["enum"]
        if isinstance(schema.get("description"), str):
            clean["description"] = schema["description"]

        return clean or {"type": "object", "properties": {}}
    if isinstance(schema, list):
        return [_sanitize_schema(x) for x in schema]
    return schema

def _build_gemini_tool_block(mcp_tools) -> List[Dict[str, Any]]:
    fns = []
    for t in mcp_tools:
        raw = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
        schema = _sanitize_schema(raw)
        fns.append({
            "name": t.name,
            "description": t.description or "",
            "parameters": schema,
        })
    return [{"functionDeclarations": fns}]

def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

async def _call_mcp_tool(name: str, args: Dict[str, Any]) -> Any:
    async with app.state.lock:
        res = await app.state.session.call_tool(name, args)

    def _try_json(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return x
        return x

    if getattr(res, "structuredContent", None):
        sc = res.structuredContent
        return _try_json(sc.get("result", sc))

    if getattr(res, "content", None):
        part = res.content[0]
        if hasattr(part, "text"):
            return _try_json(part.text)
        if hasattr(part, "data"):
            return _try_json(part.data)

    return _try_json(str(res))

async def _gemini_call(messages: List[Dict[str, Any]], tools_block, model: Optional[str]) -> Dict[str, Any]:
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": messages,
        "tools": tools_block,
        "toolConfig": {"functionCallingConfig": {"mode": GEMINI_TOOL_MODE}},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model or DEFAULT_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
    if resp.status_code >= 400:
        log.error("Gemini 400: %s", resp.text)
        resp.raise_for_status()
    return resp.json()

def _extract_function_calls(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "functionCall" in p:
                calls.append(p["functionCall"])
    return calls

def _extract_text(resp_json: Dict[str, Any]) -> Optional[str]:
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "text" in p:
                return p["text"]
    return None

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _update_context(sid: str, tool_name: str, result: Any) -> None:
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
                        aliases[_norm(raw)] = pid
    except Exception:
        pass

def _autofill_args(sid: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    ctx = app.state.contexts.get(sid, {})
    out = dict(args)

    # auto project_ref
    if "project_ref" not in out and name not in ("resolve_project",):
        pr = ctx.get("project_ref") or ctx.get("current_project_id") or ctx.get("current_project_name")
        if pr is not None:
            out["project_ref"] = pr

    # fuzzy alias match
    if "project_ref" not in out and name not in ("resolve_project",):
        aliases = ctx.get("project_aliases", {})
        if aliases:
            last_user_text = ""
            for msg in reversed(app.state.conversations.get(sid, [])):
                if msg["role"] == "user":
                    last_user_text = msg["parts"][0].get("text", "")
                    break
            lut_norm = _norm(last_user_text)
            for alias_norm, pid in aliases.items():
                if isinstance(alias_norm, str) and _norm(alias_norm) and _norm(alias_norm) in lut_norm:
                    out["project_ref"] = pid
                    break

    # defaults for ref/branch
    if "ref" in out and not out["ref"]:
        out["ref"] = ctx.get("default_branch", "main")
    if "branch" in out and not out["branch"]:
        out["branch"] = ctx.get("default_branch", "main")

    return out

def _projects_md_table(projects: List[Dict[str, Any]]) -> str:
    lines = ["| ID | Name | Pfad | URL |", "|---|------|------|-----|"]
    for p in projects:
        lines.append(f"| {p.get('id')} | {p.get('name')} | {p.get('path')} | {p.get('web_url')} |")
    return "\n".join(lines)

async def _ensure_project_index(sid: str, conv: List[Dict[str, Any]]) -> None:
    ctx = app.state.contexts.setdefault(sid, {})
    if ctx.get("project_index_ready"):
        return
    if app.state.cached_projects is None:
        result = await _call_mcp_tool("list_projects", {"limit": 200})
        app.state.cached_projects = result
    else:
        result = app.state.cached_projects
    _update_context(sid, "list_projects", result)
    table = _projects_md_table(result)
    conv.append({"role": "user", "parts": [{"text": "Projektindex (verwende diese IDs/Bezeichnungen):\n" + table}]})
    ctx["project_index_ready"] = True

# --------------------------------------------------------------------------- #
# Agent Loop
# --------------------------------------------------------------------------- #
async def run_agent(user_msg: str, max_steps: int, model: Optional[str], sid: str) -> str:
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    # Nur Index injizieren, wenn kein Projekt gesetzt
    if not ctx.get("project_ref"):
        await _ensure_project_index(sid, conv)

    conv.append({"role": "user", "parts": [{"text": user_msg}]})

    seen_calls: set[str] = set()
    tool_only_turns = 0
    TOOL_TURN_LIMIT = 3
    SKIP_WHEN_PROJECT_KNOWN = {"list_projects", "resolve_project"}

    for _ in range(max_steps):
        resp = await _gemini_call(conv, app.state.gemini_tool_block, model)
        parts = resp["candidates"][0]["content"]["parts"]
        conv.append({"role": "model", "parts": parts})

        calls = _extract_function_calls(resp)
        if not calls:
            return _extract_text(resp) or json.dumps(resp)

        response_parts = []
        for call in calls:
            name = call["name"]

            # Skip unnötige Tools
            if ctx.get("project_ref") and name in SKIP_WHEN_PROJECT_KNOWN:
                continue

            sig = f"{name}:{json.dumps(call.get('args', {}), sort_keys=True)}"
            if sig in seen_calls:
                continue
            seen_calls.add(sig)

            args = _autofill_args(sid, name, call.get("args", {}) or {})
            try:
                result = await _call_mcp_tool(name, args)
            except Exception as e:
                result = {"error": str(e)}
            _update_context(sid, name, result)

            response_parts.append({
                "functionResponse": {
                    "name": name,
                    "response": {"result": result}
                }
            })

        if not response_parts:
            # Keine neuen Antworten -> zwinge Abschluss
            conv.append({"role": "user", "parts": [{"text": "Beantworte jetzt final, ohne Tools."}]})
            continue

        conv.append({"role": "user", "parts": response_parts})
        tool_only_turns += 1
        if tool_only_turns >= TOOL_TURN_LIMIT:
            conv.append({"role": "user", "parts": [{"text": "Fasse alles zusammen und antworte endgültig."}]})
            tool_only_turns = 0

    return "[Agent aborted: max_steps reached]"

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.reset:
        app.state.conversations[req.session_id] = []
        app.state.contexts[req.session_id] = {}
    try:
        answer = await run_agent(req.message, req.max_steps, req.model, req.session_id)
        return JSONResponse({"answer": answer})
    except httpx.HTTPStatusError as e:
        return {"error": e.response.text}
    except Exception as e:
        log.exception("chat failed")
        raise HTTPException(500, str(e))

@app.get("/api/projects")
async def get_projects():
    if app.state.cached_projects is None:
        result = await _call_mcp_tool("list_projects", {"limit": 200})
        app.state.cached_projects = result
    return {"projects": app.state.cached_projects}

@app.post("/api/select_project")
async def select_project(req: SelectProjectRequest):
    sid = req.session_id
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    ranked = await _call_mcp_tool("resolve_project", {"project_ref": req.project_ref, "limit": 5})
    if not ranked:
        raise HTTPException(404, "Projekt nicht gefunden")

    proj_id = ranked[0]["id"]
    desc = await _call_mcp_tool("describe_project", {"project_ref": proj_id, "tree_limit": 200})
    if isinstance(desc, str):
        try:
            desc = json.loads(desc)
        except Exception:
            desc = {"raw": desc}

    _update_context(sid, "describe_project", desc)
    ctx["project_index_ready"] = True  # Index nicht erneut injizieren

    conv.append({"role": "user", "parts": [{"text": f"Aktuelles Projekt gesetzt: {desc.get('name')} (ID {proj_id})."}]})

    return {"project": desc}

@app.get("/api/debug/conv")
def debug_conv(sid: str = Query("default")):
    return app.state.conversations.get(sid, [])

@app.get("/api/debug/context")
def debug_context(sid: str = Query("default")):
    return app.state.contexts.get(sid, {})

@app.post("/api/tool/{tool}")
async def call_tool(tool: str, body: ToolCall = Body(...)):
    try:
        data = await _call_mcp_tool(tool, body.arguments)
        return {"result": data}
    except Exception as e:
        log.exception("tool endpoint failed")
        raise HTTPException(500, str(e))

# --------------------------------------------------------------------------- #
# Static Frontend
# --------------------------------------------------------------------------- #
STATIC = ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))

import json
from typing import Any, Dict, List, Optional, Set

import httpx
from fastapi import FastAPI

from .config import GEMINI_API_KEY, DEFAULT_MODEL, GEMINI_TOOL_MODE, SYSTEM_PROMPT
from .mcp_client import call_mcp_tool, update_context, autofill_args, ensure_project_index

async def gemini_call(messages, tools_block, model: Optional[str]) -> Dict[str, Any]:
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
    resp.raise_for_status()
    return resp.json()

def extract_function_calls(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = []
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "functionCall" in p:
                calls.append(p["functionCall"])
    return calls

def extract_text(resp_json: Dict[str, Any]) -> Optional[str]:
    for cand in resp_json.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if "text" in p:
                return p["text"]
    return None

async def run_agent(app: FastAPI, user_msg: str, max_steps: int, model: Optional[str], sid: str) -> str:
    conv = app.state.conversations.setdefault(sid, [])
    ctx = app.state.contexts.setdefault(sid, {})

    if not ctx.get("project_ref"):
        await ensure_project_index(app, sid, conv)

    conv.append({"role": "user", "parts": [{"text": user_msg}]})

    seen_calls: Set[str] = set()
